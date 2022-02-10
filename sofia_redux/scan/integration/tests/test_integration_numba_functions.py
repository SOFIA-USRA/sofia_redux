# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.integration import integration_numba_functions as nf


def make_frame_data(nframe, nchannel):
    frame_dependents = np.full(nframe, 1.0)
    frame_weight_flag = 1
    frame_dof_flag = 2
    channel_weights = np.full(nchannel, 1.0)
    channel_indices = np.arange(nchannel)
    channel_flags = np.full(nchannel, 0)
    time_weight_flag = 3
    sample_flags = np.full((nframe, nchannel), 0)

    # uniform, valid frames
    frame_data = np.full((nframe, nchannel), 1.0)
    frame_valid = np.full(nframe, True)
    frame_flags = np.full(nframe, 0)
    frame_dof = np.full(nframe, 1.0)
    frame_weight = np.full(nframe, 1.0)

    return (frame_data, frame_dof, frame_weight,
            frame_valid, frame_dependents, frame_flags,
            frame_weight_flag, frame_dof_flag,
            channel_weights, channel_indices, channel_flags,
            time_weight_flag, sample_flags)


def test_determine_time_weights():
    nframe = 10
    nchannel = 20
    block_size = 2

    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)

    nf.determine_time_weights(block_size, frame_data, frame_dof, frame_weight,
                              frame_valid, frame_dependents, frame_flags,
                              frame_weight_flag, frame_dof_flag,
                              channel_weights, channel_indices, channel_flags,
                              time_weight_flag, sample_flags)

    # no change to weights
    assert np.allclose(frame_weight, 1.0)
    assert np.allclose(np.sum(frame_weight), nframe)
    assert np.all(frame_flags == 0)
    assert np.all(frame_dof == 1 - 1 / nchannel)

    # set frame data to increasing values
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_weight = np.full(nframe, 1.0)
    nf.determine_time_weights(block_size, frame_data, frame_dof, frame_weight,
                              frame_valid, frame_dependents, frame_flags,
                              frame_weight_flag, frame_dof_flag,
                              channel_weights, channel_indices, channel_flags,
                              time_weight_flag, sample_flags)

    # output weights are higher at the beginning, since values are lower,
    # normalized over number of frames
    assert frame_weight[0] > frame_weight[-1]
    assert np.allclose(np.sum(frame_weight), nframe)
    assert np.all(frame_flags == 0)
    assert np.all(frame_dof == 1 - 1 / nchannel)

    # set some invalid frames
    frame_weight = np.full(nframe, 0.0)
    frame_valid[0:3] = False

    nf.determine_time_weights(block_size, frame_data, frame_dof, frame_weight,
                              frame_valid, frame_dependents, frame_flags,
                              frame_weight_flag, frame_dof_flag,
                              channel_weights, channel_indices, channel_flags,
                              time_weight_flag, sample_flags)
    assert np.all(frame_weight[:3] == 0)
    assert np.all(frame_weight[3:] > 0)
    assert np.allclose(np.sum(frame_weight), nframe - 3)

    # set all invalid
    frame_weight = np.full(nframe, 0.0)
    frame_valid[:] = False

    nf.determine_time_weights(block_size, frame_data, frame_dof, frame_weight,
                              frame_valid, frame_dependents, frame_flags,
                              frame_weight_flag, frame_dof_flag,
                              channel_weights, channel_indices, channel_flags,
                              time_weight_flag, sample_flags)
    assert np.all(frame_weight == 0)
    assert np.allclose(np.sum(frame_weight), 0)

    # make initial dependents too high: all flagged for DOF
    frame_weight = np.full(nframe, 1.0)
    frame_valid[:] = True
    frame_dependents = np.full(nframe, 400.0)
    nf.determine_time_weights(block_size, frame_data, frame_dof, frame_weight,
                              frame_valid, frame_dependents, frame_flags,
                              frame_weight_flag, frame_dof_flag,
                              channel_weights, channel_indices, channel_flags,
                              time_weight_flag, sample_flags)
    assert np.all(frame_weight == 0)
    assert np.all(frame_flags == frame_dof_flag)


def test_get_time_weights_by_block():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)

    # sum over all frames
    frame_sum, frame_weight_sum = nf.get_time_weights_by_block(
        frame_data, frame_dof, frame_weight,
        frame_valid, frame_dependents, frame_flags,
        frame_weight_flag, frame_dof_flag,
        channel_weights, channel_indices, channel_flags,
        time_weight_flag, sample_flags)

    assert np.allclose(frame_sum, nframe)
    assert np.allclose(frame_weight_sum, nframe - nframe / nchannel)
    assert np.all(frame_flags == 0)

    # set all channel_weights to 0
    channel_weights *= 0

    frame_sum, frame_weight_sum = nf.get_time_weights_by_block(
        frame_data, frame_dof, frame_weight,
        frame_valid, frame_dependents, frame_flags,
        frame_weight_flag, frame_dof_flag,
        channel_weights, channel_indices, channel_flags,
        time_weight_flag, sample_flags)

    # relative weights are not reduced
    assert np.allclose(frame_weight_sum, 10.0)

    # reset weights, set channel/sample flags instead: values are skipped
    channel_weights += 1.0
    channel_flags[:3] = 1
    sample_flags[:, 3:5] = 1

    frame_sum, frame_weight_sum = nf.get_time_weights_by_block(
        frame_data, frame_dof, frame_weight,
        frame_valid, frame_dependents, frame_flags,
        frame_weight_flag, frame_dof_flag,
        channel_weights, channel_indices, channel_flags,
        time_weight_flag, sample_flags)

    # relative weights ignore flagged channels
    assert np.allclose(frame_sum, nframe)
    assert np.allclose(frame_weight_sum, nframe - nframe / (nchannel - 5))


def test_robust_channel_weights():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)

    channel_var_sum, channel_var_weight = nf.robust_channel_weights(
        frame_data, frame_weight, sample_flags, frame_valid,
        channel_indices)

    assert channel_var_sum.shape == (nchannel,)
    assert channel_var_weight.shape == (nchannel,)
    assert np.allclose(channel_var_sum, 1 / 0.454937)
    assert np.allclose(channel_var_weight, nframe)

    # flag some frames, modify data to increasing values
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_valid[:3] = False

    channel_var_sum, channel_var_weight = nf.robust_channel_weights(
        frame_data, frame_weight, sample_flags, frame_valid,
        channel_indices)
    assert channel_var_sum[-1] > channel_var_sum[0]
    assert np.allclose(channel_var_weight, nframe - 3)

    # zero weights
    frame_weight *= 0
    channel_var_sum, channel_var_weight = nf.robust_channel_weights(
        frame_data, frame_weight, sample_flags, frame_valid,
        channel_indices)
    assert np.allclose(channel_var_sum, 0)
    assert np.allclose(channel_var_weight, 0)


def test_differential_channel_weights():
    nframe = 10
    nchannel = 20
    frame_delta = 1
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)

    channel_var_sum, channel_var_weight = nf.differential_channel_weights(
        frame_data, frame_weight, sample_flags,
        frame_valid, channel_indices, frame_delta)

    assert channel_var_sum.shape == (nchannel,)
    assert channel_var_weight.shape == (nchannel,)
    # expect 0, since frame data is uniform
    assert np.allclose(channel_var_sum, 0)
    assert np.allclose(channel_var_weight, nframe - 1)

    # flag some frames, modify data to increasing values
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_valid[-3:] = False
    sample_flags[:, 0:2] = 1
    sample_flags[-4:-3, :] = 1

    channel_var_sum, channel_var_weight = nf.differential_channel_weights(
        frame_data, frame_weight, sample_flags,
        frame_valid, channel_indices, frame_delta)
    # flagged channels are 0
    assert np.allclose(channel_var_sum[:2], 0)
    assert np.allclose(channel_var_weight[:2], 0)
    # otherwise expect uniform variance,
    # since frame data increases monotonically
    assert np.allclose(channel_var_sum[2:], 2000)
    assert np.allclose(channel_var_weight[2:], nframe - 5)

    # zero weights for all
    frame_weight *= 0
    channel_var_sum, channel_var_weight = nf.differential_channel_weights(
        frame_data, frame_weight, sample_flags,
        frame_valid, channel_indices, frame_delta)
    assert np.allclose(channel_var_sum, 0)
    assert np.allclose(channel_var_weight, 0)


def test_rms_channel_weights():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)

    channel_var_sum, channel_var_weight = nf.rms_channel_weights(
        frame_data, frame_weight, frame_valid, sample_flags,
        channel_indices)

    assert channel_var_sum.shape == (nchannel,)
    assert channel_var_weight.shape == (nchannel,)
    assert np.allclose(channel_var_sum, nframe)
    assert np.allclose(channel_var_weight, nframe)

    # flag some frames, modify data to increasing values
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_valid[-3:] = False
    sample_flags[:, 0:2] = 1
    sample_flags[-4:-3, :] = 1
    frame_data[:, -1] = np.nan

    channel_var_sum, channel_var_weight = nf.rms_channel_weights(
        frame_data, frame_weight, frame_valid, sample_flags,
        channel_indices)
    # flagged channels are 0
    assert np.allclose(channel_var_sum[:2], 0)
    assert np.allclose(channel_var_weight[:2], 0)
    assert np.allclose(channel_var_sum[-1], 0)
    assert np.allclose(channel_var_weight[-1], 0)
    # otherwise expect increasing variance
    assert channel_var_sum[-2] > channel_var_sum[2]
    assert np.allclose(channel_var_weight[2:-1], nframe - 4)

    # zero weights for all
    frame_weight *= 0
    channel_var_sum, channel_var_weight = nf.rms_channel_weights(
        frame_data, frame_weight, frame_valid, sample_flags,
        channel_indices)
    assert np.allclose(channel_var_sum, 0)
    assert np.allclose(channel_var_weight, 0)


def test_set_weights_from_var_stats():
    nframe = 5
    nchannel = 21
    channel_indices = np.arange(nchannel)
    var_sum = channel_indices * 0.1
    var_weight = np.full(nchannel, nframe, dtype=float)

    base_dependents = np.full(nchannel, 1.0)
    base_dof = np.full(nchannel, 0.0)
    base_variance = np.full(nchannel, 0.0)
    base_weight = np.full(nchannel, 0.0)

    nf.set_weights_from_var_stats(channel_indices, var_sum, var_weight,
                                  base_dependents, base_dof, base_variance,
                                  base_weight)
    assert np.allclose(base_dependents, 1.0)
    assert np.allclose(base_dof, 0.8)
    assert base_variance[-1] > base_variance[0]
    assert np.allclose(base_weight[0], 0)
    assert np.allclose(base_weight[1:], base_dof[1:] / base_variance[1:])

    # dependents too high: dof set to 0
    base_dependents = np.full(nchannel, 200.0)
    base_dof = np.full(nchannel, 1.0)
    base_variance = np.full(nchannel, 0.0)
    base_weight = np.full(nchannel, 0.0)
    nf.set_weights_from_var_stats(channel_indices, var_sum, var_weight,
                                  base_dependents, base_dof, base_variance,
                                  base_weight)
    assert np.allclose(base_dof, 0.0)
    assert np.allclose(base_weight, 0.0)

    # zero weights: none updated
    var_weight *= 0
    base_dof = np.full(nchannel, 0.0)
    base_variance = np.full(nchannel, 0.0)
    base_weight = np.full(nchannel, 0.0)
    nf.set_weights_from_var_stats(channel_indices, var_sum, var_weight,
                                  base_dependents, base_dof, base_variance,
                                  base_weight)
    assert np.allclose(base_dof, 0.0)
    assert np.allclose(base_weight, 0)


def test_despike_neighbouring():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    channel_level = np.full(nchannel, 1.0)
    delta = 1
    spike_flag = 1
    exclude_flag = 2

    # set sample flags to spike: cleared out before proceeding
    sample_flags += spike_flag

    nflag = nf.despike_neighbouring(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        delta, spike_flag, exclude_flag)
    # no spikes in uniform data
    assert nflag == 0
    assert np.all(sample_flags == 0)

    # make a spike
    frame_data[nframe // 2, nchannel // 2] = 3.0
    nflag = nf.despike_neighbouring(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        delta, spike_flag, exclude_flag)
    # spike is identified and flagged, so is the one + delta
    assert nflag == 1
    assert sample_flags[nframe // 2, nchannel // 2] == 1
    assert sample_flags[nframe // 2 + delta, nchannel // 2] == 1
    assert np.sum(sample_flags) == 2

    # invalidate the spike frame
    frame_valid[nframe // 2] = False
    nflag = nf.despike_neighbouring(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        delta, spike_flag, exclude_flag)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 0
    assert np.sum(sample_flags) == 0

    # same if valid but excluded
    frame_valid[nframe // 2] = True
    sample_flags[nframe // 2, nchannel // 2] = exclude_flag
    nflag = nf.despike_neighbouring(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        delta, spike_flag, exclude_flag)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 2
    assert np.sum(sample_flags) == 2

    # same if zero weight
    sample_flags[nframe // 2, nchannel // 2] = 0
    frame_weight[nframe // 2] = 0
    nflag = nf.despike_neighbouring(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        delta, spike_flag, exclude_flag)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 0
    assert np.sum(sample_flags) == 0

    # set frame level at delta to exonerate spike
    frame_weight[nframe // 2] = 1
    frame_weight[nframe // 2 + delta] = .001
    nflag = nf.despike_neighbouring(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        delta, spike_flag, exclude_flag)
    # spike is identified, delta position is not flagged
    assert nflag == 1
    assert sample_flags[nframe // 2, nchannel // 2] == 1
    assert sample_flags[nframe // 2 + delta, nchannel // 2] == 0
    assert np.sum(sample_flags) == 1


def test_despike_absolute():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    channel_level = np.full(nchannel, 1.0)
    spike_flag = 1
    exclude_flag = 2

    # set sample flags to spike: will be cleared out
    sample_flags += spike_flag

    nflag = nf.despike_absolute(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, exclude_flag)
    # no spikes in uniform data
    assert nflag == 0
    assert np.all(sample_flags == 0)

    # make a spike
    frame_data[nframe // 2, nchannel // 2] = 3.0
    nflag = nf.despike_absolute(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, exclude_flag)
    # spike is identified and flagged
    assert nflag == 1
    assert sample_flags[nframe // 2, nchannel // 2] == 1
    assert np.sum(sample_flags) == 1

    # invalidate the spike frame
    frame_valid[nframe // 2] = False
    sample_flags[nframe // 2, nchannel // 2] = 0
    nflag = nf.despike_absolute(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, exclude_flag)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 0
    assert np.sum(sample_flags) == 0

    # same if valid but excluded
    frame_valid[nframe // 2] = True
    sample_flags[nframe // 2, nchannel // 2] = exclude_flag | spike_flag
    nflag = nf.despike_absolute(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, exclude_flag)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 2
    assert np.sum(sample_flags) == 2

    # if zero weight, the frame is flagged in all channels
    sample_flags[nframe // 2, nchannel // 2] = 0
    frame_weight[nframe // 2] = 0
    nflag = nf.despike_absolute(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, exclude_flag)
    assert nflag == nchannel
    assert np.all(sample_flags[nframe // 2, :] == 1)
    assert np.sum(sample_flags) == nchannel


def test_despike_gradual():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    channel_level = np.full(nchannel, 1.0)
    spike_flag = 1
    exclude_flag = 2
    source_blank_flag = 3
    channel_gain = np.full(nchannel, 1.0)
    depth = 0.5

    # set sample flags to spike: will be cleared out
    sample_flags += spike_flag

    nflag = nf.despike_gradual(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, source_blank_flag, exclude_flag,
        channel_gain, depth)
    # no spikes in uniform data
    assert nflag == 0
    assert np.all(sample_flags == 0)

    # make a spike
    frame_data[nframe // 2, nchannel // 2] = 3.0
    nflag = nf.despike_gradual(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, source_blank_flag, exclude_flag,
        channel_gain, depth)
    # spike is identified and flagged
    assert nflag == 1
    assert sample_flags[nframe // 2, nchannel // 2] == 1
    assert np.sum(sample_flags) == 1

    # invalidate the spike frame
    frame_valid[nframe // 2] = False
    sample_flags[nframe // 2, nchannel // 2] = 0
    nflag = nf.despike_gradual(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, source_blank_flag, exclude_flag,
        channel_gain, depth)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 0
    assert np.sum(sample_flags) == 0

    # same if valid but excluded
    frame_valid[nframe // 2] = True
    sample_flags[nframe // 2, nchannel // 2] = exclude_flag | spike_flag
    nflag = nf.despike_gradual(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, source_blank_flag, exclude_flag,
        channel_gain, depth)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 2
    assert np.sum(sample_flags) == 2

    # if zero weight, the frame is flagged in all channels
    sample_flags[nframe // 2, nchannel // 2] = 0
    frame_weight[nframe // 2] = 0
    nflag = nf.despike_gradual(
        frame_data, sample_flags, channel_indices,
        frame_weight, frame_valid, channel_level,
        spike_flag, source_blank_flag, exclude_flag,
        channel_gain, depth)
    assert nflag == nchannel
    assert np.all(sample_flags[nframe // 2, :] == 1)
    assert np.sum(sample_flags) == nchannel


def test_despike_multi_resolution():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    sample_weight = np.full((nframe, nchannel), 1.0)
    spike_flag = 1
    level = 4.0
    max_block_size = 2
    timestream_data = frame_data.copy()

    # set sample flags to spike: will be cleared out
    sample_flags += spike_flag

    nflag = nf.despike_multi_resolution(
        timestream_data, sample_weight, sample_flags,
        channel_indices, frame_valid, level,
        spike_flag, max_block_size)
    # no spikes in uniform data
    assert nflag == 0
    assert np.all(sample_flags == 0)

    # make a spike
    timestream_data[nframe // 2, nchannel // 2] = 20.0
    nflag = nf.despike_multi_resolution(
        timestream_data, sample_weight, sample_flags,
        channel_indices, frame_valid, level,
        spike_flag, max_block_size)
    # spike is identified and flagged, plus adjacent neighbor
    assert nflag == 2
    assert sample_flags[nframe // 2, nchannel // 2] == 1
    assert np.sum(sample_flags) == 2

    # invalidate the spike frame
    timestream_data = frame_data.copy()
    timestream_data[nframe // 2, nchannel // 2] = 20.0
    frame_valid[nframe // 2] = False
    sample_flags[nframe // 2, nchannel // 2] = 0
    nflag = nf.despike_multi_resolution(
        timestream_data, sample_weight, sample_flags,
        channel_indices, frame_valid, level,
        spike_flag, max_block_size)
    # spike is not identified
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 0
    assert np.sum(sample_flags) == 0

    # if zero weight over all resolutions, the frame is not flagged
    timestream_data = frame_data.copy()
    timestream_data[nframe // 2, nchannel // 2] = 20.0
    sample_flags[nframe // 2, nchannel // 2] = 0
    sample_weight[:, nchannel // 2] = 0
    nflag = nf.despike_multi_resolution(
        timestream_data, sample_weight, sample_flags,
        channel_indices, frame_valid, level,
        spike_flag, max_block_size)
    assert nflag == 0
    assert sample_flags[nframe // 2, nchannel // 2] == 0
    assert np.sum(sample_flags) == 0


def test_flagged_channels_per_frame():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    flag = 1

    # unflagged data
    result = nf.flagged_channels_per_frame(
        sample_flags, flag, frame_valid, channel_indices)
    assert result.shape == (nframe,)
    assert np.all(result == 0)

    # flag a couple channels/frames
    sample_flags[0:4, 0:4] = flag
    result = nf.flagged_channels_per_frame(
        sample_flags, flag, frame_valid, channel_indices)
    assert np.all(result[0:4] == 4)
    assert np.all(result[4:] == 0)

    # invalidate frames: not marked as flagged
    frame_valid[0:2] = False
    result = nf.flagged_channels_per_frame(
        sample_flags, flag, frame_valid, channel_indices)
    assert np.all(result[0:2] == 0)
    assert np.all(result[2:4] == 4)
    assert np.all(result[4:] == 0)


def test_flagged_frames_per_channel():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    flag = 1

    # unflagged data
    result = nf.flagged_frames_per_channel(
        sample_flags, flag, frame_valid, channel_indices)
    assert result.shape == (nchannel,)
    assert np.all(result == 0)

    # flag a couple channels/frames
    sample_flags[0:4, 0:4] = flag
    result = nf.flagged_frames_per_channel(
        sample_flags, flag, frame_valid, channel_indices)
    assert np.all(result[0:4] == 4)
    assert np.all(result[4:] == 0)

    # invalidate frames: not marked as flagged
    frame_valid[0:2] = False
    result = nf.flagged_frames_per_channel(
        sample_flags, flag, frame_valid, channel_indices)
    assert np.all(result[0:4] == 2)
    assert np.all(result[4:] == 0)


def test_frame_block_expand_flag():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    flag = 1
    block_size = 7

    # no effect for unflagged data
    nf.frame_block_expand_flag(
        sample_flags, frame_valid, flag, block_size, channel_indices)
    assert np.all(sample_flags == 0)

    # flag a couple channels/frames
    sample_flags[0:4, 0:4] = flag
    nf.frame_block_expand_flag(
        sample_flags, frame_valid, flag, block_size, channel_indices)
    assert np.all(sample_flags[0:block_size, 0:4] == 1)
    assert np.all(sample_flags[block_size:, 0:4] == 0)
    assert np.all(sample_flags[:, 4:] == 0)

    # invalidate frames: not expanded, not modified
    sample_flags *= 0
    sample_flags[0:4, 0:4] = flag
    sample_flags[-2:, -3:] = flag
    frame_valid[-2:] = False
    nf.frame_block_expand_flag(
        sample_flags, frame_valid, flag, block_size, channel_indices)
    assert np.all(sample_flags[0:block_size, 0:4] == 1)
    assert np.all(sample_flags[block_size:, 0:4] == 0)
    assert np.all(sample_flags[:, 4:-3] == 0)
    assert np.all(sample_flags[-2:, -3:] == 1)
    assert np.all(sample_flags[:-2, -3:] == 0)


def test_next_weight_transit():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    level = 0.5

    # weights are all 1.0, so always above level 0.5: first frame returned
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag)
    assert result == 0
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag,
        start_frame=2)
    assert result == 2

    # none below level, so return last
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag,
        above=False)
    assert result == -1

    # set half weights below, half above
    frame_weight[:nframe // 2] = 0.4
    frame_weight[nframe // 2:] = 0.6
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag)
    assert result == nframe // 2
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag,
        above=False)
    assert result == 0

    # set some invalid, 0 weight frames, and flagged frames
    frame_weight[0] = 0.0
    frame_valid[nframe // 2] = False
    frame_flags[nframe // 2 + 1] = time_weight_flag
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag)
    assert result == nframe // 2 + 2
    result = nf.next_weight_transit(
        frame_weight, level, frame_valid, frame_flags, time_weight_flag,
        above=False)
    assert result == 1


def test_get_mean_frame_level():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    modeling_frames = np.full(nframe, False)

    # flat data: same answer for median and mean
    mval, mweight = nf.get_mean_frame_level(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, channel_indices, robust=False)
    assert mval.shape == (nchannel,)
    assert mweight.shape == (nchannel,)
    assert np.allclose(mval, 1.0)
    assert np.allclose(mweight, nframe)
    mval, mweight = nf.get_mean_frame_level(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, channel_indices, robust=True)
    assert np.allclose(mval, 1.0)
    assert np.allclose(mweight, nframe)

    # modify data: mean and median give different values
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_weight = np.arange(nframe, dtype=float) / nframe
    mval, mweight = nf.get_mean_frame_level(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, channel_indices, robust=False)
    assert np.allclose(mval[0], 126.6667)
    assert np.allclose(mval[-1], 145.6667)
    assert np.allclose(mval, 130, atol=20)
    assert np.allclose(mweight, 4.5)
    t1 = mval.copy()

    mval, mweight = nf.get_mean_frame_level(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, channel_indices, robust=True)
    assert np.allclose(mval[0], 138.85057)
    assert np.allclose(mval[-1], 157.85057)
    assert np.allclose(mval, 140, atol=20)
    assert np.allclose(mweight, 4.5)
    t2 = mval.copy()

    # invalidate/flag some frames/channels
    frame_valid[-1] = False
    sample_flags[:, -2] = 1
    mval, mweight = nf.get_mean_frame_level(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, channel_indices, robust=False)
    # all mean values are lowered, ignoring the highest frame
    assert np.all(mval < t1)
    assert np.allclose(mweight[:-2], 3.6)
    # fully flagged channels are 0
    assert np.allclose(mval[-2], 0)
    assert np.allclose(mweight[-2], 0)

    mval, mweight = nf.get_mean_frame_level(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, channel_indices, robust=True)
    assert np.all(mval[:-2] < t2[:-2])
    assert np.allclose(mweight[:-2], 3.6)
    # fully flagged channel is NaN for robust method
    assert np.isnan(mval[-2])
    assert np.allclose(mweight[-2], 3.6)


def test_remove_channel_drifts():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    modeling_frames = np.full(nframe, False)
    drift_frame_size = 3
    channel_filtering = np.full(nchannel, 1.0)
    channel_dependents = np.full(nchannel, 1.0)

    offset, weight = nf.remove_channel_drifts(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, drift_frame_size, channel_filtering,
        frame_dependents, channel_dependents, channel_indices)
    assert offset.shape == (nchannel,)
    assert weight.shape == (nchannel,)

    # flat data, average offsets all the same
    assert np.allclose(offset, 1)
    assert np.allclose(weight, 10)
    # frame data updated to remove mean
    assert np.allclose(frame_data, 0.0)

    # modify data
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_weight = np.arange(nframe, dtype=float) / nframe
    offset, weight = nf.remove_channel_drifts(
        frame_data, frame_weight, frame_valid, modeling_frames,
        sample_flags, drift_frame_size, channel_filtering,
        frame_dependents, channel_dependents, channel_indices)

    # modified value, weight reported
    assert np.allclose(offset, 130, atol=20)
    assert np.allclose(weight, 4.5)
    assert np.allclose(frame_data, 0, atol=40)


def test_level():
    nframe = 10
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    modeling_frames = np.full(nframe, False)
    channel_filtering = np.full(nchannel, 1.0)
    offset = np.full(nchannel, 0.0)
    offset_weight = np.full(nchannel, 10.0)

    t1 = frame_data.copy()
    t2 = frame_dependents.copy()

    nf.level(frame_data, frame_weight, frame_valid, modeling_frames,
             sample_flags, channel_indices, 0, nframe, offset,
             offset_weight, frame_dependents, channel_filtering)
    # no change with zero offset, but dependents updated
    assert np.allclose(frame_data, 1)
    assert np.allclose(frame_dependents, 3)

    # set offset: should be subtracted
    offset += 0.5
    frame_data = t1.copy()
    frame_dependents = t2.copy()
    nf.level(frame_data, frame_weight, frame_valid, modeling_frames,
             sample_flags, channel_indices, 0, nframe, offset,
             offset_weight, frame_dependents, channel_filtering)
    assert np.allclose(frame_data, 0.5)
    assert np.allclose(frame_dependents, 3)

    # flag some frames
    frame_data = t1.copy()
    frame_dependents = t2.copy()
    frame_valid[-1] = False
    sample_flags[:, -2] = 1
    modeling_frames[0] = True
    nf.level(frame_data, frame_weight, frame_valid, modeling_frames,
             sample_flags, channel_indices, 0, nframe, offset,
             offset_weight, frame_dependents, channel_filtering)
    assert np.allclose(frame_data[0:-1, :], 0.5)
    assert np.allclose(frame_data[-1, :], 1)
    assert np.allclose(frame_dependents[1:-1], 2.9)
    assert np.allclose(frame_dependents[0], 1)
    assert np.allclose(frame_dependents[-1], 1)

    # zero weights: offsets applied, no change to dependents
    frame_data = t1.copy()
    frame_dependents = t2.copy()
    frame_valid[:] = True
    frame_weight *= 0
    offset_weight *= 0
    nf.level(frame_data, frame_weight, frame_valid, modeling_frames,
             sample_flags, channel_indices, 0, nframe, offset,
             offset_weight, frame_dependents, channel_filtering)
    assert np.allclose(frame_data, 0.5)
    assert np.allclose(frame_dependents, 1)


def test_apply_drifts_to_channel_data():
    nframe = 30
    nchannel = 20
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    offset = np.full(nchannel, 0.0)
    average_drifts = np.full(nchannel, 1.5)
    inconsistencies = np.full(nchannel, 0)
    hardware_gain = np.full(nchannel, 0.9)
    filter_time_scale = np.full(nchannel, 0.0)
    source_filtering = np.full(nchannel, 1.0)
    scale = 1.0
    crossing_time = 1.0
    stage = True
    update = True

    # no inconsistencies for flat data
    intchan, totint = nf.apply_drifts_to_channel_data(
        channel_indices, offset, average_drifts, inconsistencies,
        hardware_gain, filter_time_scale, source_filtering,
        scale, crossing_time, stage, update)
    assert intchan == 0
    assert totint == 0
    # no zero division errors in unreasonable data;
    # source filtering is set to 0 for 0 time scale
    assert np.allclose(filter_time_scale, 0)
    assert np.allclose(source_filtering, 0)
    # offsets updated with gain * drifts
    assert np.allclose(offset, hardware_gain * average_drifts)

    # add inconsistencies, fix input time scales
    inconsistencies = np.arange(nchannel)
    filter_time_scale += 2.0
    source_filtering += 1.0
    intchan, totint = nf.apply_drifts_to_channel_data(
        channel_indices, offset, average_drifts, inconsistencies,
        hardware_gain, filter_time_scale, source_filtering,
        scale, crossing_time, stage, update)
    assert intchan == nchannel - 1
    assert totint == np.sum(inconsistencies)
    # filter time scale corrected to instrument scale,
    # source filtering unchanged
    assert np.allclose(filter_time_scale, 1)
    assert np.allclose(source_filtering, 1)

    # with update=False, filter_time_scale and source_filtering
    # not updated
    filter_time_scale *= 0
    nf.apply_drifts_to_channel_data(
        channel_indices, offset, average_drifts, inconsistencies,
        hardware_gain, filter_time_scale, source_filtering,
        scale, crossing_time, stage, False)
    assert np.allclose(filter_time_scale, 0)
    assert np.allclose(source_filtering, 1)
    filter_time_scale += 2.0
    nf.apply_drifts_to_channel_data(
        channel_indices, offset, average_drifts, inconsistencies,
        hardware_gain, filter_time_scale, source_filtering,
        scale, crossing_time, stage, False)
    assert np.allclose(filter_time_scale, 2)
    assert np.allclose(source_filtering, 1)

    # if not detector_stage, offsets updated with drifts without gain
    offset = np.full(nchannel, 0.0)
    nf.apply_drifts_to_channel_data(
        channel_indices, offset, average_drifts, inconsistencies,
        hardware_gain, filter_time_scale, source_filtering,
        scale, crossing_time, False, False)
    assert np.allclose(offset, average_drifts)


def test_detector_stage():
    nframe = 11
    nchannel = 5
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    t1 = frame_data.copy()
    frame_valid = np.full(nframe, True)
    channel_indices = np.arange(nchannel)
    channel_hardware_gain = np.full(nchannel, 0.9)

    nf.detector_stage(frame_data, frame_valid, channel_indices,
                      channel_hardware_gain)
    assert np.allclose(frame_data, t1 / channel_hardware_gain)

    # invalid frames are skipped
    frame_data = t1.copy()
    frame_valid[-3:] = False
    nf.detector_stage(frame_data, frame_valid, channel_indices,
                      channel_hardware_gain)
    assert np.allclose(frame_data[:-3], t1[:-3] / channel_hardware_gain)
    assert np.allclose(frame_data[-3:], t1[-3:])

    # readout stage inverts operation
    nf.readout_stage(frame_data, frame_valid, channel_indices,
                     channel_hardware_gain)
    assert np.allclose(frame_data, t1)


def test_search_corners():
    nframe = 11
    nchannel = 5
    sample_coordinates = np.arange(2 * nframe * nchannel,
                                   dtype=float).reshape((2, nframe, nchannel))
    frame_valid = np.full(nframe, True)
    channel_indices = np.arange(nchannel)
    sample_flags = np.full((nframe, nchannel), 0)
    skip_flag = 1

    rng = nf.search_corners(sample_coordinates, frame_valid, channel_indices,
                            sample_flags, skip_flag)
    assert np.allclose(rng, [[0, 54], [55, 109]])
    assert np.all(sample_flags == 0)

    # invalid frames are skipped
    frame_valid[0] = False
    frame_valid[-1] = False
    rng = nf.search_corners(sample_coordinates, frame_valid, channel_indices,
                            sample_flags, skip_flag)
    assert np.allclose(rng, [[5, 49], [60, 104]])
    assert np.all(sample_flags == 0)

    # bad values are flagged
    sample_coordinates[0, 1, :] = np.nan
    sample_coordinates[1, -2, :] = np.inf
    rng = nf.search_corners(sample_coordinates, frame_valid, channel_indices,
                            sample_flags, skip_flag)
    assert np.allclose(rng, [[10, 44], [65, 99]])
    assert np.all(sample_flags[0, :] == 0)
    assert np.all(sample_flags[1, :] == 1)
    assert np.all(sample_flags[2:-2, :] == 0)
    assert np.all(sample_flags[-2, :] == 1)


def test_get_weighted_timestream():
    nframe = 11
    nchannel = 5
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)

    data, weight = nf.get_weighted_timestream(
        frame_data, sample_flags, frame_valid,
        frame_weight, channel_indices)
    assert np.allclose(data, frame_data)
    assert np.allclose(weight, 1.0)

    # non-trivial data
    frame_data = np.arange(nframe * nchannel,
                           dtype=float).reshape(nframe, nchannel)
    frame_weight = np.arange(nframe, dtype=float) / nframe
    frame_valid[0] = False
    sample_flags[1, :] = 1
    data, weight = nf.get_weighted_timestream(
        frame_data, sample_flags, frame_valid,
        frame_weight, channel_indices)
    assert np.allclose(data[2:], frame_data[2:] * frame_weight[2:, None])
    assert np.allclose(weight[2:], frame_weight[2:, None])
    assert np.allclose(data[:2], 0)
    assert np.allclose(weight[:2], 0)


def test_calculate_2d_velocities():
    ns = 21
    coordinates = np.arange(2 * ns, dtype=float).reshape((2, ns))
    dt = 1.2

    v = nf.calculate_2d_velocities(coordinates, dt)
    assert np.allclose(v, 1 / dt)

    # add some invalid values: invalidates one frame before/after
    coordinates[0, :3] = np.nan
    coordinates[1, -3:] = np.nan
    v = nf.calculate_2d_velocities(coordinates, dt)
    assert np.allclose(v[:, 4:-4], 1 / dt)
    assert np.all(np.isnan(v[0, :4]))
    assert np.all(np.isnan(v[1, -4:]))


def test_calculate_2d_accelerations():
    ns = 21
    coordinates = np.arange(2 * ns, dtype=float).reshape((2, ns))
    dt = 1.2

    # no acceleration for constant speed
    v = nf.calculate_2d_accelerations(coordinates, dt)
    assert np.allclose(v, 0)

    # add some invalid values: invalidates one frame before/after
    coordinates[0, :3] = np.nan
    coordinates[1, -3:] = np.nan
    v = nf.calculate_2d_accelerations(coordinates, dt)
    assert np.allclose(v[:, 4:-4], 0)
    assert np.all(np.isnan(v[0, :4]))
    assert np.all(np.isnan(v[1, -4:]))

    # non trivial speeds
    coordinates = coordinates**2
    v = nf.calculate_2d_accelerations(coordinates, dt)
    assert not np.allclose(v[:, 4:-4], 0)
    assert np.all(np.isnan(v[0, :4]))
    assert np.all(np.isnan(v[1, -4:]))


def test_classify_scanning_speeds():
    ns = 21
    speeds = np.arange(ns, dtype=float)
    min_s = 2
    max_s = ns - 3
    valid = np.full(ns, True)
    strict = True

    # keeps indexes in range, cuts others
    keep, cut, flag = nf.classify_scanning_speeds(
        speeds, min_s, max_s, valid, strict)
    assert np.allclose(keep, np.arange(min_s, max_s + 1))
    assert np.allclose(cut, np.hstack([np.arange(min_s),
                                       np.arange(max_s + 1, ns)]))
    assert flag.size == 0
    assert np.all(valid[min_s:max_s + 1])
    assert not np.any(valid[:min_s])
    assert not np.any(valid[max_s + 1:])

    # without strict, flags out of range
    strict = False
    valid[:] = True
    keep, cut, flag = nf.classify_scanning_speeds(
        speeds, min_s, max_s, valid, strict)
    assert np.allclose(keep, np.arange(min_s, max_s + 1))
    assert cut.size == 0
    assert np.allclose(flag, np.hstack([np.arange(min_s),
                                        np.arange(max_s + 1, ns)]))

    # invalidate all: all empty arrays
    valid[:] = False
    keep, cut, flag = nf.classify_scanning_speeds(
        speeds, min_s, max_s, valid, strict)
    assert keep.size == 0
    assert cut.size == 0
    assert flag.size == 0

    # invalid values are marked and cut, regardless of strict
    valid[:] = True
    speeds = np.arange(ns, dtype=float)
    speeds[0] = np.nan
    speeds[1] = np.inf
    speeds[2] = -np.inf
    keep, cut, flag = nf.classify_scanning_speeds(
        speeds, 0, ns, valid, False)
    assert np.allclose(keep, np.arange(3, ns))
    assert np.allclose(cut, [0, 1, 2])
    assert flag.size == 0
    assert np.all(valid[3:])
    assert not np.any(valid[:3])


def test_smooth_positions():
    ns = 21
    coordinates = np.arange(2 * ns, dtype=float).reshape((2, ns))
    bin_size = 3
    result = nf.smooth_positions(coordinates, bin_size)
    assert np.all(np.isnan(result[:, 0]))
    assert np.all(np.isnan(result[:, -1]))
    assert np.allclose(result[:, 1:-1], coordinates[:, 1:-1])

    # add a noisy point
    coordinates[:, 4] *= 2
    result = nf.smooth_positions(coordinates, bin_size)
    assert np.all(np.isnan(result[:, 0]))
    assert np.all(np.isnan(result[:, -1]))
    assert np.allclose(result[:, 1:3], coordinates[:, 1:3])
    assert np.allclose(result[:, 6:-1], coordinates[:, 6:-1])
    assert np.allclose(result[:, 3:6], [[4.333333, 5.333333, 6.333333],
                                        [32.33333, 33.33333, 34.33333]])

    # add invalid points: below uses previous points,
    # above uses later points
    coordinates[0, 0] = np.nan
    coordinates[1, 4] = np.nan
    result = nf.smooth_positions(coordinates, bin_size)
    assert np.all(np.isnan(result[:, 0]))
    assert np.all(np.isnan(result[:, -1]))
    assert np.allclose(result[:, 6:-1], coordinates[:, 6:-1])
    assert np.allclose(result[:, 1:3], [[1.5, 2],
                                        [22.5, 23]])
    assert np.allclose(result[:, 3:6], [[2.5, 4, 5.5],
                                        [23.5, 25, 26.5]])

    # binsize < 2: return coords as is
    result = nf.smooth_positions(coordinates, 1)
    assert result is coordinates


def test_get_covariance():
    nframe = 11
    nchannel = 3
    (frame_data, frame_dof, frame_weight,
     frame_valid, frame_dependents, frame_flags,
     frame_weight_flag, frame_dof_flag,
     channel_weights, channel_indices, channel_flags,
     time_weight_flag, sample_flags) = make_frame_data(nframe, nchannel)
    source_flag = 1

    result = nf.get_covariance(frame_data, frame_valid, frame_weight,
                               channel_flags, channel_weights,
                               sample_flags, frame_flags, source_flag)
    assert np.allclose(result, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    # add some flags
    channel_flags[0] = 1
    frame_valid[0] = False
    sample_flags[1, 1] = 1
    result = nf.get_covariance(frame_data, frame_valid, frame_weight,
                               channel_flags, channel_weights,
                               sample_flags, frame_flags, source_flag)
    assert np.allclose(result, [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    frame_valid[:] = True
    channel_flags[:] = 0

    channel_flags[2] = 1
    sample_flags[4, 0] = 1
    result = nf.get_covariance(frame_data, frame_valid, frame_weight,
                               channel_flags, channel_weights,
                               sample_flags, frame_flags, source_flag)
    assert np.allclose(result, [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    channel_flags[:] = 0
    sample_flags[:] = 0

    frame_flags[1] = 1
    result = nf.get_covariance(frame_data, frame_valid, frame_weight,
                               channel_flags, channel_weights,
                               sample_flags, frame_flags, source_flag)
    assert np.allclose(result, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_get_full_covariance_matrix():
    covariance = np.ones((10, 10), dtype=float)
    fixed_indices = np.arange(10) + 1
    result = nf.get_full_covariance_matrix(covariance, fixed_indices)
    assert result.shape == (11, 11)
    assert np.all(np.isnan(result[0, :]))
    assert np.all(np.isnan(result[:, 0]))
    assert np.allclose(result[1:, 1:], covariance)


def test_get_partial_covariance_matrix():
    covariance = np.ones((10, 10), dtype=float)
    fixed_indices = np.arange(5) + 1
    result = nf.get_partial_covariance_matrix(covariance, fixed_indices)
    assert result.shape == (5, 5)
    assert np.allclose(result, 1)


def test_downsample_frame_data():
    nframe = 11
    nchannel = 5
    data = np.arange(nframe * nchannel,
                     dtype=float).reshape((nframe, nchannel))
    window = np.full(5, 1.0)
    result = nf.downsample_frame_data(data, window)
    assert result.shape == (3, 5)
    assert np.allclose(result[0], data[2])
    assert np.allclose(result[1], data[4])
    assert np.allclose(result[2], data[6])

    valid = np.full(nframe, True)
    valid[1] = False
    result = nf.downsample_frame_data(data, window, valid=valid)
    assert np.all(np.isnan(result[0]))
    assert np.allclose(result[1], data[4])
    assert np.allclose(result[2], data[6])

    data[0, -1] = np.nan
    result = nf.downsample_frame_data(data, window)
    assert np.allclose(result[0, :-1], data[2, :-1])
    assert np.isnan(result[0, -1])
    assert np.allclose(result[1], data[4])
    assert np.allclose(result[2], data[6])


def test_downsample_frame_flags():
    nframe = 11
    nchannel = 5
    flags = np.arange(nframe * nchannel).reshape((nframe, nchannel))
    window = np.full(5, 1.0)
    result = nf.downsample_frame_flags(flags, window)
    assert result.shape == (3, 5)
    assert np.allclose(result[0], 31)
    assert np.allclose(result[1], [31, 31, 63, 63, 63])
    assert np.allclose(result[2], 63)

    valid = np.full(nframe, True)
    valid[1] = False
    result = nf.downsample_frame_flags(flags, window, valid=valid)
    assert np.all(result[0] == -1)
    assert np.allclose(result[1], [31, 31, 63, 63, 63])
    assert np.allclose(result[2], 63)


def test_get_downsample_start_indices():
    nframe = 11
    valid = np.full(nframe, True)
    valid[1] = False
    window = np.full(5, 1.0)

    idx, dvalid = nf.get_downsample_start_indices(valid, window, 1)
    assert np.allclose(idx, np.arange(6))
    assert np.allclose(dvalid, [False, False, True, True, True, True])


def test_get_valid_downsampling_frames():
    nframe = 11
    valid = np.full(nframe, True)
    valid[1] = False
    idx = np.arange(nframe)
    window_size = 5

    dvalid = nf.get_valid_downsampling_frames(valid, idx, window_size)
    assert np.allclose(dvalid, [False, False, True, True, True, True,
                                True, False, False, False, False])
