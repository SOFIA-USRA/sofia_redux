# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.filters_numba_functions import (
    load_timestream, remove_rejection_from_frames, apply_rejection_to_parms,
    dft_filter_channels, dft_filter_frames, dft_filter_frequency_channel,
    level_for_channels, level, level_1d, resample, accumulate_profiles,
    calculate_channel_point_responses, calculate_varied_point_response,
    calc_mean_amplitudes, whiten_profile, add_frame_parms,
    expand_rejection_filter, harmonize_rejection_filter)


@pytest.fixture
def data_ramp_200():
    n_frames = 200
    nf = 256  # pow2ceil(200)
    data = np.arange(nf, dtype=float)
    data[n_frames:] = 0.0
    return data, n_frames, 256  # (pow2ceil(200))


@pytest.fixture
def gaussian_512():
    n_frames = 512
    nf2 = n_frames // 2
    x = np.arange(n_frames) - nf2
    data = np.exp(-0.5 * (x / 16) ** 2)
    return data


def test_load_timestream():
    n_frames, n_channels = 10, 5
    frame_data = np.arange(n_frames * n_channels, dtype=float).reshape(
        (n_channels, n_frames)).T.copy()

    frame_weights = np.full(n_frames, 2.0)
    frame_valid = np.full(n_frames, True)
    modeling_frames = np.full(n_frames, False)
    channel_indices = np.arange(n_channels)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    timestream = np.empty((n_channels, n_frames))
    points = np.empty(n_channels)
    frame_valid[0] = False
    modeling_frames[1] = True
    sample_flags[2, 2] = 1
    frame_data[2, 3] = np.nan
    load_timestream(frame_data, frame_weights, frame_valid, modeling_frames,
                    channel_indices, sample_flags, timestream, points)

    assert np.allclose(points, [16, 16, 14, 14, 16])
    assert np.allclose(timestream[points == 16],
                       [0, 0, -7, -5, -3, -1, 1, 3, 5, 7])
    assert np.allclose(timestream[points == 14],
                       [0, 0, 0, -6, -4, -2, 0, 2, 4, 6])

    frame_valid.fill(False)
    load_timestream(frame_data, frame_weights, frame_valid, modeling_frames,
                    channel_indices, sample_flags, timestream, points)
    assert np.allclose(points, 0)
    assert np.allclose(timestream, 0)


def test_remove_rejection_from_frames():
    n_frames, n_channels = 10, 5
    channel_indices = np.arange(3)
    frame_valid = np.full(n_frames, True)
    frame_valid[0] = False
    frame_data = np.zeros((n_frames, n_channels))
    rejected_signal = np.arange((n_frames * channel_indices.size)).reshape(
        (channel_indices.size, n_frames))
    remove_rejection_from_frames(frame_data, frame_valid, channel_indices,
                                 rejected_signal)

    assert np.allclose(frame_data[0], 0)
    assert np.allclose(frame_data[:, 3:], 0)
    expected = -rejected_signal.T
    expected[0] = 0
    assert np.allclose(frame_data[:, :3], expected)


def test_apply_rejection_to_parms():
    n_frames, n_channels = 10, 5
    frame_valid = np.full(n_frames, True)
    frame_weight = np.full(n_frames, 2.0)
    frame_parms = np.zeros(n_frames)
    dp = np.arange(n_channels, dtype=float) + 1
    channel_indices = np.arange(n_channels)
    sample_flag = np.zeros((n_frames, n_channels), dtype=int)

    frame_valid[0] = False
    frame_weight[1] = 0.0
    sample_flag[2, 3] = 1

    apply_rejection_to_parms(frame_valid, frame_weight, frame_parms, dp,
                             channel_indices, sample_flag)

    assert np.allclose(frame_parms, [0, 0, 22, 30, 30, 30, 30, 30, 30, 30])


def test_dft_filter_frequency_channel(data_ramp_200):
    data, n_frames, nf = data_ramp_200
    data = np.arange(nf, dtype=float)
    data[n_frames:] = 0
    rejection_value = 0.5
    rejected = np.zeros(nf)
    dft_filter_frequency_channel(data, 0, rejection_value, rejected, n_frames)
    assert np.allclose(rejected[n_frames:], 0)
    toggle = np.ones(n_frames)
    toggle[(np.arange(n_frames) % 2) == 0] *= -1
    expected = toggle * 0.390625
    assert np.allclose(rejected[:n_frames], expected)

    rejected = np.zeros(nf)
    dft_filter_frequency_channel(data, 8, rejection_value, rejected, n_frames)

    # sinusoidal
    expected = rejected[:32]
    assert not np.allclose(expected, 0)
    for i in range(5):
        i1 = i * 32
        i2 = i1 + 32
        assert np.allclose(rejected[i1:i2], expected)


def test_dft_filter_frames(data_ramp_200):
    data, n_frames, nf = data_ramp_200
    filtered = data.copy()

    rejection = np.zeros(nf + 1)
    rejection[8] = 1.0
    rejection[16] = 1.0
    dft_filter_frames(filtered, rejection, n_frames)

    rpt = 32
    rpt_count = n_frames // 32
    expected = filtered[:rpt]
    assert not np.allclose(expected, 0)
    for i in range(1, rpt_count - 1):
        i1 = i * rpt
        i2 = i1 + rpt
        assert np.allclose(filtered[i1:i2], expected)


def test_dft_filter_channels(data_ramp_200):
    data, n_frames, nf = data_ramp_200
    n_channels = 5
    frame_data = data[None] + np.arange(n_channels)[:, None]
    frame_data[:, n_frames:] = 0
    filtered = frame_data.copy()
    rejection = np.zeros(nf + 1)
    rejection[8] = 1.0
    rejection[16] = 1.0

    # Should just be an offset difference due to the added constant
    dft_filter_channels(filtered, rejection, n_frames)
    diff = filtered[1] - filtered[0]
    assert not np.allclose(diff, 0)
    for i in range(2, n_channels):
        d = filtered[i] - filtered[i - 1]
        assert np.allclose(d, diff)

    # Check it's periodic
    rpt = 32
    rpt_count = n_frames // 32
    expected = filtered[:, :rpt]
    for i in range(1, rpt_count - 1):
        i1 = i * rpt
        i2 = i1 + rpt
        assert np.allclose(filtered[:, i1:i2], expected)


def test_level_for_channels():
    n_frames, n_channels = 10, 5
    signal = np.arange(n_frames)[None] + np.arange(n_channels)[:, None]
    signal = signal.astype(float)
    sample_flag = np.zeros((n_frames, n_channels), dtype=int)
    valid_frame = np.full(n_frames, True)
    modeling_frame = np.full(n_frames, False)
    channel_indices = np.arange(n_channels)

    valid_frame[0] = False
    modeling_frame[1] = True
    sample_flag[2, 3] = 1
    s0 = signal.copy()
    level_for_channels(signal, valid_frame, modeling_frame, sample_flag,
                       channel_indices)

    expected = np.linspace(-5.5, 3.5, 10)
    expected = expected[None] - np.asarray([0, 0, 0, 0.5, 0])[:, None]
    assert np.allclose(signal, expected)

    signal = s0.copy()
    valid_frame.fill(False)
    level_for_channels(signal, valid_frame, modeling_frame, sample_flag,
                       channel_indices)
    assert np.allclose(signal, 0)


def test_level_1d():
    data = np.arange(10, dtype=float)
    level_1d(data, 10)
    assert np.allclose(data, np.linspace(-4.5, 4.5, 10))
    data[1] = np.nan
    level_1d(data, 10)
    assert np.allclose(data, 0)


def test_level():
    data = np.arange(10)[None] + np.arange(3, dtype=float)[:, None]
    level(data, 10)
    assert np.allclose(data, np.linspace(-4.5, 4.5, 10)[None])


def test_resample():
    n_frames, n_channels = 10, 5
    old = np.arange(n_frames, dtype=float)[None] + np.arange(
        n_channels)[:, None]
    new = old.copy()
    resample(old, new)
    assert np.allclose(old, new)

    half_frames = n_frames // 2
    new = np.empty((n_channels, half_frames))
    resample(old, new)

    expected = np.linspace(0.5, 8.5, 5)[None] + np.arange(n_channels)[:, None]
    assert np.allclose(new, expected)


def test_accumulate_profiles():
    n_channels, n_freq = 5, 100
    channel_profiles = np.random.random((n_channels, n_freq))
    profiles = np.random.random((n_channels, n_freq))
    expected = channel_profiles * profiles
    channel_indices = np.arange(n_channels)
    accumulate_profiles(profiles, channel_profiles, channel_indices)
    assert np.allclose(channel_profiles, expected)
    assert np.allclose(profiles, expected)


def test_calculate_varied_point_response():
    source_profile = np.arange(100, dtype=float)
    response = np.full(100, 2.0)
    source_norm = 100
    min_fch = 50
    point_response = calculate_varied_point_response(
        min_fch, source_profile, response, source_norm)
    assert point_response == 86.75


def test_calculate_channel_point_response():
    n_channels, n_freq = 5, 100
    source_profile = np.arange(n_freq, dtype=float)
    profiles = np.ones((n_channels, n_freq)) + np.arange(n_channels)[:, None]
    channel_indices = np.arange(n_channels)
    source_norm = 100
    min_fch = 50
    point_response = calculate_channel_point_responses(
        min_fch, source_profile, profiles, channel_indices, source_norm)
    assert np.allclose(point_response,
                       [49.5, 86.75, 124, 161.25, 198.5])


def test_calc_mean_amplitudes(gaussian_512):

    data = gaussian_512
    n_frames = data.size
    nf2 = n_frames // 2

    n_channels = 2
    data = data[None] + np.zeros(n_channels)[:, None]
    data[0, 0] += 0.1  # non-zero Nyquist in first channel
    spectrum = np.fft.rfft(data, axis=-1)
    channel_indices = np.arange(n_channels)

    # Check a standard window size of 1
    amplitudes = np.empty((n_channels, nf2))
    amplitude_weights = np.empty((n_channels, nf2))
    calc_mean_amplitudes(amplitudes, amplitude_weights, spectrum, 1,
                         channel_indices)

    assert np.allclose(
        amplitudes[0, :5],
        [39.240349, 39.240349, 37.22984045, 33.6182123, 29.56202417],
        atol=1e-6)
    assert np.allclose(
        amplitudes[1, :5],
        [39.340349, 39.340349, 37.12984045, 33.7182123, 29.46202417],
        atol=1e-6)
    assert np.allclose(amplitudes[:, -1], [0.11547005, 0], atol=1e-6)

    expected = np.full((n_channels, nf2), 2)
    i0 = np.array([1] * 4), np.array([64, 128, 192, 245])
    expected[i0] = 0
    i1 = (np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
          [16, 64, 128, 192, 240, 245, 16, 125, 131, 240, 246, 250, 251,
           252, 255])
    expected[i1] = 1
    expected[0, 255] = 3
    assert np.allclose(amplitude_weights, expected)

    # Now downsample by 2
    amplitudes = np.empty((n_channels, nf2 // 2))
    amplitude_weights = np.empty((n_channels, nf2 // 2))
    calc_mean_amplitudes(amplitudes, amplitude_weights, spectrum, 2,
                         channel_indices)
    assert np.allclose(
        amplitudes[0, :5],
        [38.24830721, 35.47002409, 27.22588296, 17.96660618, 10.19580295],
        atol=1e-6)
    assert np.allclose(
        amplitudes[1, :5],
        [38.25106607, 35.46507361, 27.21708013, 17.95396325, 10.17929514],
        atol=1e-6)
    assert np.allclose(amplitudes[:, -1], [0.1095445115, 0], atol=1e-6)
    expected = np.full((n_channels, nf2 // 2), 4)
    i2 = np.array([1] * 5), np.array([32, 64, 96, 122, 125])
    expected[i2] = 2
    i3 = (np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
          np.array([8, 32, 64, 96, 120, 122, 8, 62, 65, 120, 123, 126, 127]))
    expected[i3] = 3
    expected[0, 127] = 5
    assert np.allclose(amplitude_weights, expected)


def test_whiten_profile(gaussian_512):
    data = gaussian_512
    n_frames = data.size
    nf2 = n_frames // 2
    n_channels = 2
    channel_indices = np.arange(n_channels)
    data = data[None] + np.zeros(n_channels)[:, None]
    spectrum = np.fft.rfft(data, axis=-1)
    amplitudes = np.empty((n_channels, nf2))
    amplitude_weights = np.empty((n_channels, nf2))
    calc_mean_amplitudes(amplitudes, amplitude_weights, spectrum, 1,
                         channel_indices)

    profiles = np.zeros((n_channels, nf2))
    channel_profiles = profiles.copy()

    white_from = 100
    white_to = 200
    filter_level = 1.0
    significance = 3.0
    one_over_f_bin = 1
    white_noise_bin = 2

    a0 = amplitudes.copy()
    w0 = amplitude_weights.copy()

    f1 = whiten_profile(a0 * 0, w0, profiles,
                        channel_profiles, white_from, white_to, filter_level,
                        significance, one_over_f_bin, white_noise_bin,
                        channel_indices)
    assert np.isnan(f1).all()
    assert np.allclose(w0, 0)
    assert np.allclose(channel_profiles, 0)
    assert np.allclose(profiles, 0)

    a0 = amplitudes.copy()
    w0 = amplitude_weights.copy()
    f1 = whiten_profile(a0, w0, profiles,
                        channel_profiles, white_from, white_to, filter_level,
                        significance, one_over_f_bin, white_noise_bin,
                        channel_indices)
    assert np.allclose(f1, 1)
    assert np.allclose(a0[:, 0], amplitudes[:, 0])
    assert np.allclose(a0[:, 1:], 0)
    assert np.allclose(profiles, 1)
    assert np.allclose(channel_profiles, 0)
    profiles.fill(0)

    a0 = np.ones_like(amplitudes)
    w0 = np.ones_like(amplitudes)
    profiles = a0.copy()
    channel_profiles = a0.copy()

    f1 = whiten_profile(a0, w0, profiles,
                        channel_profiles, white_from, white_to, filter_level,
                        significance, one_over_f_bin, white_noise_bin,
                        channel_indices)
    assert np.allclose(f1, 1)
    assert np.allclose(a0, 1)
    assert np.allclose(w0, 4)
    assert np.allclose(profiles, 1)
    assert np.allclose(channel_profiles, 1)

    profiles.fill(0)
    a0 = amplitudes.copy()
    w0 = amplitude_weights.copy()
    channel_profiles = a0.copy()

    f1 = whiten_profile(a0, w0, profiles,
                        channel_profiles, white_from, white_to, filter_level,
                        significance, one_over_f_bin, white_noise_bin,
                        channel_indices)
    assert np.allclose(f1, 1.05953456, atol=1e-6)
    assert np.allclose(channel_profiles, amplitudes)
    assert not np.allclose(amplitude_weights, w0)
    assert np.allclose(a0[:, 0], amplitudes[:, 0])
    assert np.allclose(a0[:, 1:], 0)


def test_add_frame_parms():
    n_frames, n_channels = 10, 5
    rejected = np.arange(n_channels, dtype=float)
    weights = np.ones(n_frames)
    frame_valid = np.full(n_frames, True)
    modeling_frames = np.full(n_frames, False)
    frame_parms = np.zeros(n_frames)
    sample_flags = np.zeros((n_frames, n_channels), dtype=int)
    points = np.full(n_channels, n_frames - 4, dtype=float)
    channel_indices = np.arange(n_channels)
    frame_valid[0] = False
    modeling_frames[1] = True
    sample_flags[2, 2] = 1
    weights[3] = 0.0
    points[4] = 0.0
    add_frame_parms(rejected, points, weights, frame_valid, modeling_frames,
                    frame_parms, sample_flags, channel_indices)

    assert np.allclose(frame_parms, [0, 0, 2 / 3, 0, 1, 1, 1, 1, 1, 1])


def test_expand_rejection_filter():
    reject = np.full(20, False)
    reject[1] = True
    reject[10] = True
    reject[19] = True
    df = 2.0
    half_width = 3.0  # should round to a delta of 2
    r0 = reject.copy()
    expand_rejection_filter(reject, half_width, df)
    assert np.allclose(
        reject, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    reject = r0.copy()
    half_width = 2.0  # should round to a delta of 1
    expand_rejection_filter(reject, half_width, df)
    assert np.allclose(
        reject, [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

    reject = r0.copy()
    half_width = 0.5  # should round to a delta of 0
    expand_rejection_filter(reject, half_width, df)
    assert np.allclose(reject, r0)


def test_harmonize_rejection_filter():
    reject = np.full(100, False)
    reject[0] = True
    odd_harmonics_only = False
    r0 = reject.copy()
    harmonize_rejection_filter(reject, 4, odd_harmonics_only)
    assert np.allclose(reject, r0)

    reject.fill(False)
    reject[1] = True
    harmonize_rejection_filter(reject, 4, odd_harmonics_only)
    assert not reject[0] and reject[1:5].all() and not reject[5:].any()

    reject.fill(False)
    reject[2] = True
    harmonize_rejection_filter(reject, 4, odd_harmonics_only)
    assert np.allclose(np.nonzero(reject)[0], [2, 4, 6, 8])

    reject.fill(False)
    reject[3] = True
    odd_harmonics_only = True
    harmonize_rejection_filter(reject, 4, odd_harmonics_only)
    assert np.allclose(np.nonzero(reject)[0], [3, 9])

    reject.fill(False)
    reject[40] = True  # Cannot hit third harmonic
    harmonize_rejection_filter(reject, 4, odd_harmonics_only)
    assert np.allclose(np.nonzero(reject)[0], 40)
