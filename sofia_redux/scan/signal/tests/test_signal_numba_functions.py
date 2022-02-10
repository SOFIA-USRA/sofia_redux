# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.signal.signal_numba_functions import (
    get_signal_variance, get_ml_correlated, get_robust_correlated,
    resync_gains, apply_gain_increments, calculate_filtering,
    differentiate_signal, differentiate_weighted_signal, integrate_signal,
    integrate_weighted_signal, add_drifts, level, remove_drifts,
    get_covariance, get_ml_gain_increment, get_robust_gain_increment,
    synchronize_gains, prepare_frame_temp_fields)


def test_get_signal_variance():
    values = np.arange(10, dtype=float)
    variance = get_signal_variance(values)
    assert variance == 28.5
    weights = np.ones(10)
    weights[5] = 0.0
    values[1] = np.nan
    variance = get_signal_variance(values, weights=weights)
    assert variance == 32.375
    weights.fill(0.0)
    assert get_signal_variance(values, weights=weights) == 0


def test_get_ml_correlated():
    n_frames, n_channels = 10, 2
    frame_data = np.arange(20, dtype=float).reshape((2, 10)).T.copy()

    gains = np.full(n_channels, 0.5)
    cw = np.full(n_channels, 2.0)
    frame_weights = np.full(n_frames, 1.0)
    channel_wg = cw * gains
    channel_wg2 = channel_wg * gains
    channel_indices = np.arange(n_channels)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    frame_valid = np.full(n_frames, True)
    resolution = 0  # Gets set to 1

    gi, gw = get_ml_correlated(frame_data, frame_weights, frame_valid,
                               channel_indices, channel_wg, channel_wg2,
                               sample_flags, resolution)

    assert np.allclose(gi, np.arange(10, 30, 2))
    assert np.allclose(gw, 1)

    resolution = 2
    gi, gw = get_ml_correlated(frame_data, frame_weights, frame_valid,
                               channel_indices, channel_wg, channel_wg2,
                               sample_flags, resolution)
    assert np.allclose(gi, np.arange(11, 30, 4))
    assert np.allclose(gw, 2)

    frame_valid[0] = False
    frame_weights[1] = 0.0
    sample_flags[2, 0] = 1
    sample_flags[:, 1] = 1

    gi, gw = get_ml_correlated(frame_data, frame_weights, frame_valid,
                               channel_indices, channel_wg, channel_wg2,
                               sample_flags, resolution)

    assert np.allclose(gi, [0, 6, 9, 13, 17])
    assert np.allclose(gw, [0, 0.5, 1, 1, 1])


def test_get_robust_correlated():
    n_frames, n_channels = 18, 2
    frame_data = np.arange(n_frames * n_channels, dtype=float).reshape(
        (n_channels, n_frames)).T.copy()
    frame_weights = np.full(n_frames, 1.0)
    frame_valid = np.full(n_frames, True)
    channel_indices = np.arange(n_channels)
    channel_g = np.full(n_channels, 0.5)
    cw = np.full(n_channels, 2.0)
    channel_wg2 = cw * channel_g * channel_g
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    resolution = 0  # Gets set to 1
    max_dependence = 0.25  # The default

    gi, gw = get_robust_correlated(frame_data, frame_weights, frame_valid,
                                   channel_indices, channel_g, channel_wg2,
                                   sample_flags, resolution,
                                   max_dependence=max_dependence)
    assert np.allclose(gi, np.arange(18, 53, 2))
    assert np.allclose(gw, 1)

    resolution = 3
    gi, gw = get_robust_correlated(frame_data, frame_weights, frame_valid,
                                   channel_indices, channel_g, channel_wg2,
                                   sample_flags, resolution,
                                   max_dependence=max_dependence)
    assert np.allclose(gi, np.arange(4, 35, 6))
    assert np.allclose(gw, 3)

    frame_valid[0] = False
    frame_weights[1] = 0.0
    channel_g[1] = 0.0
    sample_flags[3:7, 0] = 1
    gi, gw = get_robust_correlated(frame_data, frame_weights, frame_valid,
                                   channel_indices, channel_g, channel_wg2,
                                   sample_flags, resolution,
                                   max_dependence=max_dependence)
    assert np.allclose(gi, [4, 0, 15, 20, 26, 32])
    assert np.allclose(gw, [0.5, 0, 1, 1.5, 1.5, 1.5])


def test_resync_gains():
    n_frames, n_channels = 10, 3
    frame_data = np.zeros((n_frames, n_channels))
    resolution = 1
    signal_values = np.arange(n_frames // resolution, dtype=float)
    channel_indices = np.arange(n_channels)
    delta_gains = 0.5 * (channel_indices + 1)
    frame_valid = np.full(n_frames, True)
    resolution = 0  # Gets set to 1

    resync_gains(frame_data, signal_values, resolution, delta_gains,
                 channel_indices, frame_valid)

    assert np.allclose(frame_data[:, 0], -np.arange(n_frames) * 0.5)
    assert np.allclose(frame_data[:, 1], -np.arange(n_frames))
    assert np.allclose(frame_data[:, 2], -np.arange(n_frames) * 1.5)

    frame_data.fill(0.0)
    resolution = 2
    signal_values = np.arange(n_frames // resolution, dtype=float)
    resync_gains(frame_data, signal_values, resolution, delta_gains,
                 channel_indices, frame_valid)
    assert np.allclose(frame_data[:, 0], -0.5 * (np.arange(n_frames) // 2))
    assert np.allclose(frame_data[:, 1], -(np.arange(n_frames) // 2))
    assert np.allclose(frame_data[:, 2], -1.5 * (np.arange(n_frames) // 2))

    frame_data.fill(0.0)
    frame_valid[3] = False
    delta_gains[0] = 0.0
    resync_gains(frame_data, signal_values, resolution, delta_gains,
                 channel_indices, frame_valid)
    assert np.allclose(frame_data[:, 0], 0)
    assert np.allclose(frame_data[:, 1],
                       [0, 0, -1, 0, -2, -2, -3, -3, -4, -4])
    assert np.allclose(frame_data[:, 2],
                       [0, 0, -1.5, 0, -3, -3, -4.5, -4.5, -6, -6])


def test_apply_gain_increments():
    n_frames, n_channels = 10, 3
    frame_data = np.zeros((n_frames, n_channels))
    frame_weight = np.ones(n_frames)
    frame_valid = np.full(n_frames, True)
    modeling_frames = np.full(n_frames, False)
    frame_dependents = np.zeros(n_frames)
    channel_dependents = np.zeros(n_channels)
    channel_indices = np.arange(n_channels)
    channel_g = (channel_indices + 1) * 0.5
    filtering = 0.5
    channel_weights = 1.0
    channel_fwg2 = filtering * channel_weights * (channel_g ** 2)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)

    # resolution dependent values
    resolution = 1
    n_signal = n_frames // resolution
    signal_values = np.zeros(n_signal)
    signal_weights = np.zeros(n_signal)
    increment = np.arange(n_signal, dtype=float)
    increment_weight = np.arange(n_signal, dtype=float)

    resolution = 0  # Gets set to 1
    apply_gain_increments(frame_data, frame_weight, frame_valid,
                          modeling_frames, frame_dependents, channel_g,
                          channel_fwg2, channel_indices, channel_dependents,
                          sample_flags, signal_values, signal_weights,
                          resolution, increment, increment_weight)

    assert np.allclose(frame_data[:, 0], -0.5 * np.arange(n_frames))
    assert np.allclose(frame_data[:, 1], -np.arange(n_frames))
    assert np.allclose(frame_data[:, 2], -1.5 * np.arange(n_frames))
    assert np.allclose(
        frame_dependents,
        np.array([0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280]) / 1440)
    assert np.allclose(channel_dependents,
                       [0.35362103, 1.41448413, 3.18258929])
    assert np.allclose(signal_values, increment)
    assert np.allclose(signal_weights, increment_weight)

    for x in [frame_data, frame_dependents, channel_dependents]:
        x.fill(0.0)

    resolution = 2
    frame_valid[0] = False
    modeling_frames[1] = True
    sample_flags[2, 0] = 1
    frame_weight[3] = 0.0
    channel_fwg2[2] = 0.0
    n_signal = n_frames // resolution
    signal_values = np.zeros(n_signal)
    signal_weights = np.zeros(n_signal)
    increment = np.arange(n_signal, dtype=float) + 1
    increment_weight = np.arange(n_signal, dtype=float) + 1
    apply_gain_increments(frame_data, frame_weight, frame_valid,
                          modeling_frames, frame_dependents, channel_g,
                          channel_fwg2, channel_indices, channel_dependents,
                          sample_flags, signal_values, signal_weights,
                          resolution, increment, increment_weight)

    assert np.allclose(frame_data[:, 0],
                       [0, -0.5, -1, -1, -1.5, -1.5, -2, -2, -2.5, -2.5])
    assert np.allclose(frame_data[:, 1],
                       [0, -1, -2, -2, -3, -3, -4, -4, -5, -5])
    assert np.allclose(frame_data[:, 2],
                       [0, -1.5, -3, -3, -4.5, -4.5, -6, -6, -7.5, -7.5])
    assert np.allclose(frame_dependents,
                       [0, 0, 0.25, 0, 0.20833333,
                        0.20833333, 0.15625, 0.15625, 0.125, 0.125], atol=1e-6)
    assert np.allclose(channel_dependents,
                       [0.19583333, 1.03333333, 0], atol=1e-6)
    assert np.allclose(signal_values, increment)
    assert np.allclose(signal_weights, increment_weight)


def test_calculate_filtering():
    n_channels = 10
    overlaps = np.empty((n_channels, n_channels))
    for i in range(n_channels):  # Symmetric
        for j in range(i, n_channels):
            ij = i + j
            overlaps[i, j] = overlaps[j, i] = ij

    overlaps[5, 6] = overlaps[6, 5] = 0.0

    overlaps /= 20
    channel_source_filtering = np.ones(n_channels)
    signal_source_filtering = np.full(n_channels, 2.0)
    channel_dependents = np.full(n_channels, 0.1)
    n_parms = 2.0
    channel_valid = np.full(n_channels, True)
    channel_valid[1] = False
    signal_source_filtering[3] = 0.0
    channel_source_filtering[4] = np.nan
    channel_dependents[5] = 4
    channel_indices = np.arange(n_channels)
    cf, sf = calculate_filtering(channel_indices, channel_dependents, overlaps,
                                 channel_valid, n_parms,
                                 channel_source_filtering,
                                 signal_source_filtering)

    assert np.allclose(
        cf, [0.17625, 1, 0.06125, 0.0075, 0, 0, 0.38125, 0, 0, 0])
    assert np.allclose(
        sf, [0.3525, 2, 0.1225, 0.0075, 0, 0, 0.7625, 0, 0, 0])


def test_differentiate_signal():
    x = np.arange(10, dtype=float)
    differentiate_signal(x)
    assert np.allclose(x, 1)
    x = np.arange(0, 20, 2, dtype=float)
    differentiate_signal(x, dt=4.0)
    assert np.allclose(x, 0.5)


def test_differentiate_weighted_signal():
    x = np.arange(10, dtype=float)
    weights = np.ones(10)
    weights[3:6] = 0.0
    differentiate_weighted_signal(x, weights)
    assert np.allclose(x, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
    assert np.allclose(weights, [0.5, 1, 0.5, 0, 0, 0, 0.5, 1, 1, 1])


def test_integrate_signal():
    x0 = np.arange(1, 10, dtype=float)
    x = x0.copy()
    integrate_signal(x, dt=2)
    assert np.allclose(x, x0 ** 2)
    x = x0.copy()
    integrate_signal(x, dt=4)
    assert np.allclose(x, 2 * x0 ** 2)


def test_integrate_weighted_signal():
    x0 = np.arange(1, 11, dtype=float)
    x = x0.copy()
    weights = np.full(10, 2.0)
    weights[4:7] = 0.0
    integrate_weighted_signal(x, weights, dt=2.0)
    assert np.allclose(x, x0 ** 2)
    assert np.allclose(weights, [0, 4, 4, 4, 0, 0, 0, 0, 4, 4])


def test_add_drifts():
    x = np.zeros(20)
    drifts = np.arange(4, dtype=float)
    drift_length = 6
    add_drifts(x, drifts, drift_length)
    assert np.allclose(x, [0] * 6 + [1] * 6 + [2] * 6 + [3] * 2)


def test_level():
    x = np.arange(20, dtype=float)
    start_frame = 20  # signal index 10
    end_frame = 40  # signal index 20
    resolution = 2
    x0 = x.copy()
    mn = level(x, start_frame, end_frame, resolution)  # No weights, not robust
    assert np.isclose(mn, np.mean(x0[10:20]))
    assert np.allclose(x[:10], x0[:10])
    assert np.allclose(x[10:], x0[10:] - mn)

    x = x0.copy()
    weights = np.ones(x.size)
    mn = level(x, 0, 40, resolution, weights=weights, robust=True)
    assert mn == 9
    assert np.allclose(x, x0 - 9)


def test_remove_drifts():
    signal_values = np.arange(21, dtype=float)
    drifts = np.zeros(7)
    n_frames = 9
    resolution = 3
    integration_size = drifts.size * n_frames - 1
    remove_drifts(signal_values, drifts, n_frames,
                  resolution, integration_size)

    assert np.allclose(drifts, np.arange(1, 20, 3))
    assert np.allclose(signal_values, [-1, 0, 1] * 7)


def test_get_covariance():
    n_frames, n_channels = 10, 5
    frame_data = np.arange(50, dtype=float).reshape((n_channels, n_frames)
                                                    ).T.copy()
    frame_valid = np.full(n_frames, True)
    signal_values = np.full(n_frames, 2.0)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    channel_weights = np.full(n_channels, 0.5)
    channel_indices = np.arange(n_channels)

    frame_valid[0] = False
    signal_values[1] = np.nan
    channel_weights[0] = 0.0
    sample_flags[2, 2] = 1

    c = get_covariance(signal_values, frame_data, frame_valid, channel_indices,
                       channel_weights, sample_flags)
    assert np.isclose(c, 1.991495581, atol=1e-6)


def test_get_ml_gain_increment():
    n_frames, n_channels = 10, 3
    frame_data = np.arange(30, dtype=float).reshape((n_channels, n_frames)
                                                    ).T.copy()
    signal_wc = np.full(n_frames, 0.5)
    signal_wc2 = signal_wc ** 2
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    channel_indices = np.arange(n_channels)
    valid_frames = np.full(n_frames, True)

    valid_frames[0] = False
    signal_wc[1] = signal_wc2[1] = 0.0
    sample_flags[2, 2] = 1
    dc, dw = get_ml_gain_increment(frame_data, signal_wc, signal_wc2,
                                   sample_flags, channel_indices, valid_frames)
    assert np.allclose(dc, [11, 31, 52])
    assert np.allclose(dw, [2, 2, 1.75])


def test_get_robust_gain_increment():
    n_frames, n_channels = 10, 3
    frame_data = np.arange(30, dtype=float).reshape((n_channels, n_frames)
                                                    ).T.copy()
    signal_wc = np.full(n_frames, 0.5)
    signal_wc2 = signal_wc ** 2
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    channel_indices = np.arange(n_channels)
    valid_frames = np.full(n_frames, True)

    valid_frames[0] = False
    signal_wc[1] = signal_wc2[1] = 0.0
    sample_flags[:, 2] = 1

    dc, dw = get_robust_gain_increment(frame_data, signal_wc, signal_wc2,
                                       sample_flags, channel_indices,
                                       valid_frames)

    assert np.allclose(dc, [10, 30, 0])
    assert np.allclose(dw, [2, 2, 0])


def test_synchronize_gains():
    n_frames, n_channels = 10, 3
    frame_data = np.zeros((n_frames, n_channels))
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    frame_valid = np.full(n_frames, True)
    modeling_frames = np.full(n_frames, False)
    channel_indices = np.arange(n_channels)
    delta_gains = np.arange(n_channels, dtype=float) + 1
    frame_wc2 = np.arange(n_frames, dtype=float)
    channel_wc2 = np.arange(n_channels, dtype=float)
    signal_values = np.full(n_frames, 0.5)
    frame_parms = np.zeros(n_frames)
    channel_parms = np.zeros(n_channels)
    frame_valid[0] = False
    modeling_frames[1] = True
    sample_flags[2, 1] = 1
    synchronize_gains(frame_data, sample_flags, frame_valid,
                      modeling_frames, channel_indices, delta_gains,
                      frame_wc2, channel_wc2, signal_values,
                      frame_parms, channel_parms)
    assert np.allclose(frame_data[0], 0) and np.allclose(frame_data[:, 0], 0)
    assert np.allclose(frame_parms, [0, 0, 1, 4.5, 6, 7.5, 9, 10.5, 12, 13.5])
    assert np.allclose(channel_parms, [0, 2 / 3, 2 / 3])


def test_prepare_temp_fields():
    n_frames = 6
    signal_values = np.arange(n_frames, dtype=float)
    frame_weights = np.full(n_frames, 0.5)
    frame_valid = np.full(n_frames, True)
    frame_modeling = np.full(n_frames, False)
    frame_c = np.empty(n_frames)
    frame_wc = np.empty(n_frames)
    frame_wc2 = np.empty(n_frames)
    signal_values[0] = np.nan
    frame_valid[1] = False
    frame_modeling[2] = True
    frame_weights[3] = np.nan
    prepare_frame_temp_fields(signal_values, frame_weights, frame_valid,
                              frame_modeling, frame_c, frame_wc, frame_wc2)

    assert np.allclose(frame_c, [0, 0, 2, 3, 4, 5])
    assert np.allclose(frame_wc, [0, 0, 0, 0, 2, 2.5])
    assert np.allclose(frame_wc2, [0, 0, 0, 0, 8, 12.5])
