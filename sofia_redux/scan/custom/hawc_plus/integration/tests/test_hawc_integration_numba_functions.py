# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.hawc_plus.integration import \
    hawc_integration_numba_functions as hnf


def test_find_inconsistencies():
    n_frames, n_channels = 19, 6
    frame_valid = np.full(n_frames, True)
    frame_data = np.ones((n_frames, n_channels))
    frame_weights = np.ones(n_frames)
    modeling_frames = np.full(n_frames, False)
    frame_parms = np.ones(n_frames)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    exclude_sample_flag = 1
    channel_indices = np.arange(n_channels)
    channel_parms = np.ones(n_channels)
    min_jump_level_frames = 4
    jump_flag = 2
    fix_each = True
    fix_subarray = np.full(1, True)
    has_jumps = np.full(n_channels, True)
    subarray = np.zeros(n_channels, dtype=int)
    jump_counter = np.zeros((n_frames, n_channels), dtype=int)
    drift_size = 5
    flag_before = 0
    flag_after = 0
    for i in range(1, 20, drift_size):
        jump_counter[i:] += 1
    frame_valid[0] = False
    jump_counter[1] = 0  # To account for first invalid frame

    inconsistencies = hnf.find_inconsistencies(
        frame_valid, frame_data, frame_weights, modeling_frames, frame_parms,
        sample_flags, exclude_sample_flag, channel_indices, channel_parms,
        min_jump_level_frames, jump_flag, fix_each, fix_subarray, has_jumps,
        subarray, jump_counter, drift_size, flag_before=flag_before,
        flag_after=flag_after)
    assert np.allclose(inconsistencies, [4] * n_channels)

    fd_expect = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    fd_expect = np.asarray(fd_expect)[:, None]
    assert np.allclose(frame_data, fd_expect)

    sf_expect = [0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2]
    sf_expect = np.asarray(sf_expect)[:, None]
    assert np.allclose(sample_flags, sf_expect)

    assert np.allclose(
        frame_parms,
        [1, 1, 1, 1, 1, 1, 2.5, 2.5, 2.5, 2.5,
         1., 2.5, 2.5, 2.5, 2.5, 1, 1, 1, 1])
    assert np.allclose(channel_parms, [3] * n_channels)


def test_fix_jumps():
    n_frames, n_channels = 10, 6
    channel_indices = np.arange(n_channels)  # just the first three channels
    frame_valid = np.full(n_frames, True)
    frame_data = np.arange(n_frames * n_channels).reshape(
        (n_channels, n_frames)).T.copy()
    frame_weights = np.ones(n_frames)
    modeling_frames = np.full(n_frames, False)
    frame_parms = np.ones(n_frames)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    exclude_sample_flag = 1
    channel_parms = np.ones(n_channels)
    min_jump_level_frames = 9
    jump_flag = 2
    subarray = channel_indices // 3
    jump_counter = np.zeros((n_frames, n_channels), dtype=int)
    has_jumps = (channel_indices % 2) == 0
    fix_subarray = np.asarray([True, False])
    jump_counter[2:, has_jumps] = 1
    jump_counter[8:, has_jumps] = 2

    d0 = frame_data.copy()
    s0 = sample_flags.copy()
    frame_valid[0] = False

    fix_each = False  # Do not fix jumps
    no_jumps = hnf.fix_jumps(
        frame_valid, frame_data, frame_weights, modeling_frames, frame_parms,
        sample_flags, exclude_sample_flag, channel_indices, channel_parms,
        min_jump_level_frames, jump_flag, fix_each, fix_subarray, has_jumps,
        subarray, jump_counter)
    assert no_jumps.all()
    assert np.allclose(frame_data, d0) and np.allclose(sample_flags, s0)

    fix_each = True  # Fix jumps
    no_jumps = hnf.fix_jumps(
        frame_valid, frame_data, frame_weights, modeling_frames, frame_parms,
        sample_flags, exclude_sample_flag, channel_indices, channel_parms,
        min_jump_level_frames, jump_flag, fix_each, fix_subarray, has_jumps,
        subarray, jump_counter, flag_before=1, flag_after=1)

    assert np.allclose(no_jumps, [False, True, False, True, True, True])
    assert np.allclose(frame_data, d0)
    assert np.allclose(sample_flags[:, no_jumps], 0)
    assert np.allclose(sample_flags[1:, ~no_jumps], 2)

    sample_flags = s0.copy()
    frame_data = d0.copy()
    min_jump_level_frames = 3
    no_jumps = hnf.fix_jumps(
        frame_valid, frame_data, frame_weights, modeling_frames, frame_parms,
        sample_flags, exclude_sample_flag, channel_indices, channel_parms,
        min_jump_level_frames, jump_flag, fix_each, fix_subarray, has_jumps,
        subarray, jump_counter, flag_before=0, flag_after=0, start_frame=0,
        end_frame=n_frames)
    assert np.allclose(no_jumps, [False, True, False, True, True, True])
    assert np.allclose(frame_data[:, no_jumps], d0[:, no_jumps])
    assert np.allclose(sample_flags[:, no_jumps], 0)
    assert np.allclose(frame_data[:, ~no_jumps],
                       [[0, 20],
                        [1, 21],
                        [-2, -2],
                        [-1, -1],
                        [0, 0],
                        [0, 0],
                        [1, 1],
                        [2, 2],
                        [8, 28],
                        [9, 29]])
    assert np.allclose(sample_flags[:, ~no_jumps],
                       np.array([0, 2, 0, 0, 0, 0, 0, 0, 2, 2])[:, None])


def test_fix_block():
    n_frames = 10
    frame_valid = np.full(n_frames, True)
    frame_data = np.arange(n_frames)[:, None].astype(float) + 1
    frame_weights = np.ones(n_frames)
    modeling_frames = np.full(n_frames, False)
    frame_parms = np.ones(n_frames)
    sample_flags = np.zeros((n_frames, 1), dtype=int)
    exclude_sample_flag = 0
    channel = 0
    channel_parms = np.ones(1)
    jump_flag = 1

    # Check flagging
    min_jump_level_frames = 9
    d0 = frame_data.copy()
    hnf.fix_block(1, 9, frame_valid, frame_data, frame_weights,
                  modeling_frames, frame_parms, sample_flags,
                  exclude_sample_flag, channel, channel_parms,
                  min_jump_level_frames, jump_flag)
    assert np.allclose(sample_flags[:, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert np.allclose(frame_data, d0)
    assert np.allclose(frame_parms, 1)
    assert np.allclose(channel_parms, 1)

    # Check levelling
    sample_flags.fill(0)
    min_jump_level_frames = 8
    hnf.fix_block(1, 9, frame_valid, frame_data, frame_weights,
                  modeling_frames, frame_parms, sample_flags,
                  exclude_sample_flag, channel, channel_parms,
                  min_jump_level_frames, jump_flag)
    assert np.allclose(frame_data[:, 0],
                       [1, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 10])
    assert np.allclose(sample_flags, 0)
    assert np.allclose(frame_parms, [1] + ([1.125] * 8) + [1])
    assert np.allclose(channel_parms, [2])


def test_flag_block():
    n_frames = 10
    frame_valid = np.full(n_frames, True)
    frame_valid[:3] = False
    sample_flags = np.zeros((n_frames, 1), dtype=int)
    jump_flag = 2
    channel = 0
    hnf.flag_block(1, 9, frame_valid, sample_flags, jump_flag, channel)
    assert np.allclose(sample_flags[:, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 2, 0])


def test_level_block():
    n_frames, n_channels = 10, 1
    frame_data = np.arange(10).astype(float)[:, None] + 1
    frame_weights = np.ones(n_frames)
    frame_valid = np.full(n_frames, True)
    modeling_frames = np.full(n_frames, False)
    frame_parms = np.ones(n_frames)
    channel_parms = np.ones(n_channels)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    exclude_sample_flag = 1
    channel = 0

    cp = channel_parms.copy()
    fp = frame_parms.copy()
    d = frame_data.copy()

    hnf.level_block(0, 10, frame_valid, d, frame_weights, modeling_frames,
                    fp, sample_flags, exclude_sample_flag, channel, cp)

    assert np.allclose(d, frame_data - 5.5)
    assert np.allclose(cp, [2.0])
    assert np.allclose(fp, [1.1] * 10)

    cp = channel_parms.copy()
    fp = frame_parms.copy()
    d = frame_data.copy()
    v = frame_valid.copy()
    w = frame_weights.copy()
    m = modeling_frames.copy()
    s = sample_flags.copy()
    v[1] = False
    s[2, 0] = 1
    w[3] = 0.0
    m[4] = True

    hnf.level_block(1, 9, v, d, w, m, fp, s, exclude_sample_flag, channel, cp)
    assert np.allclose(d[:, 0],
                       [1, 2, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 10])
    assert np.allclose(fp, [1, 1, 1, 1, 1, 1.25, 1.25, 1.25, 1.25, 1])
    assert np.allclose(cp, [2])

    d0, fp0, cp0 = d.copy(), fp.copy(), cp.copy()
    hnf.level_block(
        1, 9, v, d, w * 0, m, fp, s, exclude_sample_flag, channel, cp)
    assert np.allclose(d, d0) and np.allclose(fp, fp0) and np.allclose(cp, cp0)


def test_correct_jumps():
    n_frames, n_channels = 10, 5
    frame_data = np.zeros((n_frames, n_channels))  # zero baseline test
    frame_valid = np.full(n_frames, True)
    frame_valid[0] = False
    jump_counter = frame_data.astype(int)  # All zeros at this point
    # Just look at channels 2-4
    channel_indices = np.array([2, 3, 4])
    channel_jumps = np.ones(3)
    channel_jumps[0] = 0
    jump_range = 16  # jumps for wrap... (actual range is +- jump_range/2)

    jump_counter[:, 2] = 1
    jump_counter[4:, 3] = 1
    jump_counter[7:, 3] = 2
    jump_counter[5:, 4] = 10  # should wrap
    jump_counter[8:, 4] = -11  # should wrap in other direction

    # Check nothing happens with no valid frames
    invalid = np.full(n_frames, False)
    hnf.correct_jumps(frame_data, invalid, jump_counter, channel_indices,
                      channel_jumps, jump_range)

    assert np.allclose(frame_data, 0)

    hnf.correct_jumps(frame_data, frame_valid, jump_counter, channel_indices,
                      channel_jumps, jump_range)

    assert np.allclose(frame_data[:, :3], 0)
    assert np.allclose(frame_data[:, 3],
                       [0, 0, 0, 0, -1, -1, -1, -2, -2, -2])
    assert np.allclose(frame_data[:, 4],
                       [0, 0, 0, 0, 0, 6, 6, 6, -5, -5])


def test_flag_zeroed_channels():
    n_frames, n_channels = 10, 5
    frame_data = np.zeros((n_frames, n_channels))
    frame_valid = np.full(n_frames, True)
    frame_valid[0] = False
    frame_data[0] = 1  # First frame is non-zero but invalid
    channel_indices = np.arange(n_channels)
    discard_flag = 1
    channel_flags = np.full(n_channels, discard_flag | 2)  # 3
    # Channels 2 and 3 are non-zero
    frame_data[:, 2] = 2
    frame_data[:, 3] = 3
    channel_flags[4] = 2  # Application of discard flag
    hnf.flag_zeroed_channels(frame_data, frame_valid, channel_indices,
                             channel_flags, discard_flag)
    assert np.allclose(channel_flags, [3, 3, 2, 2, 3])


def test_check_jumps():
    n_frames, n_channels = 10, 5
    jump_counter = np.zeros((n_frames, n_channels), dtype=int)
    channel_indices = np.arange(n_channels)
    frame_valid = np.full(10, True)
    has_jumps = np.full(n_channels, False)
    jump_counter += np.arange(n_channels)[None]
    start_counter = jump_counter[0, :]
    # Have jumps in the 2nd and 3rd channel
    jump_counter[3:, 2] += 2
    jump_counter[7:, 3] += 5
    n_jumps = hnf.check_jumps(start_counter, jump_counter, frame_valid,
                              has_jumps, channel_indices)
    assert n_jumps == 2
    assert np.allclose(has_jumps, [0, 0, 1, 1, 0])
