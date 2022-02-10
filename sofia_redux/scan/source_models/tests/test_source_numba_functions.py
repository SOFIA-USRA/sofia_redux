# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.source_models.source_numba_functions import (
    calculate_coupling_increment, get_sample_points, blank_sample_values,
    flag_out_of_range_coupling, sync_map_samples, get_delta_sync_parms,
    flag_outside, validate_pixel_indices, add_skydip_frames)


@pytest.fixture
def zero_data():
    n_frames, n_channels = 11, 9
    frame_data = np.zeros((n_frames, n_channels))
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    channel_indices = np.arange(n_channels)
    frame_valid = np.full(n_frames, True)
    frame_weight = np.ones(n_frames)

    # Very simple mapping: frames map to x, channels map to y
    ny, nx = n_channels, n_frames
    blank_map = np.zeros((ny, nx))
    map_indices = np.empty((2, n_frames, n_channels), dtype=int)
    map_indices[0] = np.arange(n_frames)[:, None]  # x
    map_indices[1] = np.arange(n_channels)[None]  # y

    return (frame_data, sample_flags, frame_valid, frame_weight,
            channel_indices, blank_map, map_indices)


def test_calculate_coupling_increment(zero_data):
    (frame_data, sample_flags, frame_valid, frame_weight,
     channel_indices, blank_map, map_indices) = zero_data

    n_frames, n_channels = frame_data.shape
    map_values = blank_map.copy()
    base_values = map_values.copy()
    map_noise = base_values.copy()
    base_values.fill(1.0)
    map_values.fill(2.0)
    map_noise.fill(0.1)
    sync_gains = np.full(n_channels, 1.0)
    source_gains = np.full(n_channels, 2.0)
    frame_data.fill(5.0)
    frame_valid[0] = False
    frame_weight[1] = 0.0
    frame_gains = np.ones(n_frames)
    frame_gains[2] = 0.0
    sample_flags[3, 2] = 1  # frame, channel
    map_noise[2, 4] = 0.0  # channel, frame
    map_noise[2, 5] = 1e-6  # High S2N
    sample_flags[:, 3] = 1  # Blank out channel 3
    min_s2n = 0.0
    max_s2n = 1e3
    exclude_flag = 1
    map_indices[0, 6, 0] = -1

    increment = calculate_coupling_increment(
        map_indices, base_values, map_values, map_noise, sync_gains,
        source_gains, frame_data, frame_weight, frame_gains, frame_valid,
        sample_flags, channel_indices, min_s2n, max_s2n, exclude_flag)

    expected = np.full(n_channels, 0.5)
    expected[3] = 0
    assert np.allclose(increment, expected)


def test_get_sample_points(zero_data):
    (frame_data, sample_flags, valid_frames, frame_weights,
     channel_indices, blank_map, map_indices) = zero_data
    n_frames, n_channels = frame_data.shape
    frame_gains = np.full(n_frames, 2.0)
    source_gains = np.full(n_channels, 3.0)
    channel_variance = np.full(n_channels, 5.0)
    frame_data.fill(7.0)

    valid_frames[0] = False
    frame_gains[1] = 0.0
    frame_weights[2] = 0.0
    frame_gains[3] = np.nan
    source_gains[0] = 0.0
    channel_variance[1] = 0.0
    sample_flags[4, 4] = 1
    exclude_sample_flag = 1

    bad_mask = np.full((n_frames, n_channels), False)
    bad_mask[:4] = True
    bad_mask[:, :2] = True
    bad_mask[4, 4] = True

    pixel_map = map_indices[..., :1].copy()
    for mapping in [map_indices, pixel_map]:

        n, data, gains, weights, indices = get_sample_points(
            frame_data, frame_gains, frame_weights, source_gains,
            channel_variance, valid_frames, mapping,
            channel_indices, sample_flags, exclude_sample_flag)

        assert n == 7
        expected = np.full(bad_mask.shape, 7.0)
        expected[bad_mask] = np.nan
        assert np.allclose(expected, data, equal_nan=True)

        expected[~bad_mask] = 6
        assert np.allclose(expected, gains, equal_nan=True)

        expected.fill(0.2)
        expected[bad_mask] = 0
        assert np.allclose(expected, weights)

        expected = map_indices.copy()
        expected[:, bad_mask] = -1
        if mapping is pixel_map:
            expected[1, ~bad_mask] = 0

        assert np.allclose(expected, indices)


def test_blank_sample_values(zero_data):
    (sample_data, sample_flags, valid_frames, frame_weights,
     channel_indices, blank_map, sample_indices) = zero_data

    n_dimensions = 2
    sample_gains = sample_data.copy()
    sample_weights = np.ones(sample_gains.shape)
    frame = 0
    channel_index = 1
    inds0 = sample_indices.copy()

    blank_sample_values(frame, channel_index, n_dimensions, sample_data,
                        sample_gains, sample_weights, sample_indices)

    mask = np.full(sample_data.shape, False)
    mask[frame, channel_index] = True
    expected = np.zeros(sample_data.shape)
    expected[mask] = np.nan
    assert np.allclose(sample_data, expected, equal_nan=True)
    assert np.allclose(sample_gains, expected, equal_nan=True)
    expected = np.ones(sample_data.shape)
    expected[mask] = 0
    assert np.allclose(sample_weights, expected)
    expected = inds0
    expected[:, mask] = -1
    assert np.allclose(sample_indices, expected)


def test_flag_out_of_range_coupling():
    n_channels = 10
    channel_indices = np.arange(n_channels)
    coupling_values = np.arange(10, dtype=float)
    min_coupling = 2.0
    max_coupling = 7.0
    flags = np.zeros(n_channels, dtype=int)
    flags[5] = 1
    blind_flag = 2

    flag_out_of_range_coupling(channel_indices, coupling_values, min_coupling,
                               max_coupling, flags, blind_flag)
    assert np.allclose(flags, [2, 2, 0, 0, 0, 1, 0, 0, 2, 2])


def test_sync_map_samples(zero_data):
    (frame_data, sample_flags, frame_valid, frame_weight,
     channel_indices, map_values, map_indices) = zero_data

    n_frames, n_channels = frame_data.shape
    frame_gains = np.arange(n_frames, dtype=float)

    map_valid = np.full(map_values.shape, True)
    map_masked = np.full(map_values.shape, False)
    base_values = map_values.copy()
    base_valid = map_valid.copy()
    source_gains = np.full(n_channels, 2.0)
    sync_gains = np.ones(n_channels)
    sample_blank_flag = 1

    # frame_gains[0] is 0
    frame_valid[1] = False
    map_indices[0, 2, 0] = -1
    # Check map validity in channel 1
    map_valid[1, 4] = False
    base_valid[1, 5] = False
    map_masked[1, 6] = True
    sample_flags[:, 2] = 3  # Check for unflagging

    map_values.fill(2.0)
    base_values.fill(1.0)

    pixel_indices = map_indices[..., :1].copy()

    for mapping in [map_indices, pixel_indices]:

        d = frame_data.copy()
        s = sample_flags.copy()

        sync_map_samples(d, frame_valid, frame_gains, channel_indices,
                         map_values, map_valid, map_masked, mapping,
                         base_values, base_valid, source_gains, sync_gains,
                         s, sample_blank_flag)

        if mapping is map_indices:
            assert np.allclose(d,
                               [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, -6, -6, -6, -6, -6, -6, -6, -6],
                                [-9, -9, -9, -9, -9, -9, -9, -9, -9],
                                [-12, 0, -12, -12, -12, -12, -12, -12, -12],
                                [-15, -20, -15, -15, -15, -15, -15, -15, -15],
                                [-18, -18, -18, -18, -18, -18, -18, -18, -18],
                                [-21, -21, -21, -21, -21, -21, -21, -21, -21],
                                [-24, -24, -24, -24, -24, -24, -24, -24, -24],
                                [-27, -27, -27, -27, -27, -27, -27, -27, -27],
                                [-30, -30, -30, -30, -30, -30, -30, -30, -30]]
                               )
            assert np.allclose(s[:, 0], 0)
            assert np.allclose(s[:, 3:], 0)
            assert np.allclose(s[:, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            assert np.allclose(s[:, 2], [2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        else:
            assert np.allclose(d,
                               [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [-9, -9, -9, -9, -9, -9, -9, -9, -9],
                                [-12, -12, -12, -12, -12, -12, -12, -12, -12],
                                [-15, -15, -15, -15, -15, -15, -15, -15, -15],
                                [-18, -18, -18, -18, -18, -18, -18, -18, -18],
                                [-21, -21, -21, -21, -21, -21, -21, -21, -21],
                                [-24, -24, -24, -24, -24, -24, -24, -24, -24],
                                [-27, -27, -27, -27, -27, -27, -27, -27, -27],
                                [-30, -30, -30, -30, -30, -30, -30, -30, -30]]
                               )
            assert np.allclose(s[:, :2], 0)
            assert np.allclose(s[:, 3:], 0)
            assert np.allclose(s[:, 2], [2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2])


def test_get_delta_sync_parms():
    n_frames, n_channels = 10, 5
    channel_source_gains = np.arange(n_channels, dtype=float)
    channel_indices = np.arange(n_channels)
    channel_flags = np.zeros(n_channels, dtype=int)
    channel_variance = np.full(n_channels, 3.0)
    frame_weight = np.ones(n_frames)
    frame_source_gains = np.arange(n_frames, dtype=float)
    frame_valid = np.full(n_frames, True)
    frame_flags = np.zeros(n_frames, dtype=int)
    source_flag = 1
    n_points = 44.0

    channel_flags[0] = 1
    frame_valid[0] = False
    frame_flags[1] = 1

    fdp, cdp = get_delta_sync_parms(channel_source_gains, channel_indices,
                                    channel_flags, channel_variance,
                                    frame_weight, frame_source_gains,
                                    frame_valid, frame_flags, source_flag,
                                    n_points)
    assert np.allclose(fdp, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.allclose(cdp * 15, [0, 22, 88, 198, 352])


def test_flag_outside():
    n_frames, n_channels = 4, 5
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    valid_frames = np.full(n_frames, True)
    channel_indices = np.arange(n_channels)
    skip_flag = 1
    # x = 0->5, y = 1->3
    map_range = np.asarray([[0, 5], [1, 3.0]])
    sample_coordinates = np.full((2, n_frames, n_channels), 2.0)
    sample_coordinates[0, :, 0] = -1
    sample_coordinates[0, :, 1] = 6
    sample_coordinates[1, :, 2] = 0
    sample_coordinates[1, :, 3] = 4
    valid_frames[0] = False
    flag_outside(sample_coordinates, valid_frames, channel_indices,
                 sample_flags, skip_flag, map_range)

    assert np.allclose(sample_flags[0], 0)
    assert np.allclose(sample_flags[:, -1], 0)
    assert np.allclose(sample_flags[1:, :-1], 1)


def test_validate_pixel_indices():
    n_frames, n_channels = 6, 5
    indices = np.full((2, n_frames, n_channels), 2)
    x_size = 3
    y_size = 4
    valid_frame = np.full(n_frames, True)

    indices[0, :, 0] = -2
    indices[0, :, 1] = 4
    indices[1, :, 2] = -3
    indices[1, :, 3] = 5
    i0 = indices.copy()

    bad = validate_pixel_indices(indices, x_size, y_size)
    assert bad == 24
    assert np.allclose(indices[..., :-1], -1)
    assert np.allclose(indices[..., -1], 2)

    expected = indices.copy()
    indices = i0.copy()
    valid_frame[0] = False
    bad = validate_pixel_indices(indices, x_size, y_size,
                                 valid_frame=valid_frame)
    assert bad == 20
    assert np.allclose(indices, expected)


def test_add_skydip_frames():
    n_bins = 5
    n_frames = 12
    data_bins = np.arange(n_frames) // 2
    frame_valid = np.full(n_frames, True)
    data = np.zeros(n_bins)
    weight = np.zeros(n_bins)
    frame_weights = np.full(n_frames, 2.0)
    signal_values = np.arange(n_frames, dtype=float) + 1
    signal_weights = np.full(n_frames, 3.0)

    frame_valid[0] = False
    frame_weights[2] = 0.0
    signal_weights[3] = 0.0

    add_skydip_frames(data, weight, signal_values, signal_weights,
                      frame_weights, frame_valid, data_bins)

    assert np.allclose(data, [12, 0, 66, 90, 114])
    assert np.allclose(weight, [6, 0, 12, 12, 12])
