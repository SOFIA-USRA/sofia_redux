# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.frames.frames_numba_functions import (
    add_dependents, validate_frames, downsample_data)
from sofia_redux.scan.flags.mounts import Mount


def test_add_dependents():
    n_frames = 10
    dependents = np.zeros(n_frames)
    frame_valid = np.full(n_frames, True)
    frame_valid[0] = False
    add_dependents(dependents, 1.0, frame_valid)
    assert dependents[0] == 0 and np.allclose(dependents[1:], 1)
    add_dependents(dependents, 2.0, frame_valid, subtract=True)
    assert dependents[0] == 0 and np.allclose(dependents[1:], -1)

    dp = np.arange(10, dtype=float)
    frame_valid.fill(True)
    dependents.fill(0)
    add_dependents(dependents, dp, frame_valid, start_frame=2, end_frame=9)
    assert np.allclose(dependents[:2], 0) and dependents[-1] == 0
    assert np.allclose(dependents[2:9], dp[2:9])
    add_dependents(dependents, dp, frame_valid, start_frame=2, end_frame=9,
                   subtract=True)
    assert np.allclose(dependents, 0)


def test_validate_frames():
    angle = np.linspace(0, np.pi / 2, 4)  # 0, 30, 60, 90
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    native_sin_lat = sin_a.copy()
    native_cos_lat = cos_a.copy()
    validated = np.full(4, False)
    valid = np.full(4, True)
    has_telescope_info = np.full(4, True)
    left_nasmyth = Mount.LEFT_NASMYTH.value
    right_nasmyth = Mount.RIGHT_NASMYTH.value
    prime_focus = Mount.PRIME_FOCUS.value

    mount = prime_focus
    validate_frames(valid, cos_a, sin_a, native_sin_lat, native_cos_lat,
                    validated, has_telescope_info, mount, left_nasmyth,
                    right_nasmyth)
    assert valid.all()
    assert validated.all()

    valid[0] = False
    validated[:2] = False
    validate_frames(valid, cos_a, sin_a, native_sin_lat, native_cos_lat,
                    validated, has_telescope_info, mount, left_nasmyth,
                    right_nasmyth)
    assert not valid[0] and valid[1:].all()
    assert validated.all()

    valid.fill(True)
    validated.fill(False)
    cos_a[:2] = np.nan
    sin_a[:2] = np.nan
    has_telescope_info[0] = False

    validate_frames(valid, cos_a, sin_a, native_sin_lat, native_cos_lat,
                    validated, has_telescope_info, mount, left_nasmyth,
                    right_nasmyth)
    assert np.isnan(cos_a[0]) and np.isnan(sin_a[0])
    assert cos_a[1] == 1 and sin_a[1] == 0

    cos_a = native_cos_lat.copy()
    sin_a = native_sin_lat.copy()
    valid.fill(True)
    validated.fill(False)
    has_telescope_info.fill(True)
    cos_a[1] = np.nan
    sin_a[1] = np.nan
    mount = left_nasmyth
    validate_frames(valid, cos_a, sin_a, native_sin_lat, native_cos_lat,
                    validated, has_telescope_info, mount, left_nasmyth,
                    right_nasmyth)
    assert cos_a[1] == native_cos_lat[1]
    assert sin_a[1] == -native_sin_lat[1]

    cos_a = native_cos_lat.copy()
    sin_a = native_sin_lat.copy()
    valid.fill(True)
    validated.fill(False)
    has_telescope_info.fill(True)
    cos_a[1] = np.nan
    sin_a[1] = np.nan
    mount = right_nasmyth
    validate_frames(valid, cos_a, sin_a, native_sin_lat, native_cos_lat,
                    validated, has_telescope_info, mount, left_nasmyth,
                    right_nasmyth)
    assert cos_a[1] == native_cos_lat[1]
    assert sin_a[1] == native_sin_lat[1]


def test_downsample_data():
    n_frames = 21
    n_channels = 2
    data = np.zeros((n_frames, n_channels))
    sample_flag = np.zeros((n_frames, n_channels), dtype=int)
    window = np.array([0.25, 0.5, 0.25])
    n_window = window.size
    low_resolution_frames = n_frames // n_window
    valid = np.full(low_resolution_frames, True)
    start_indices = np.arange(0, 21, n_window)

    # First channel
    data[0, 0] = 1
    data[5:7, 0] = 1
    data[12:15, 0] = 1
    data[20, 0] = 1

    # Second channel
    data[10, 1] = 1
    sample_flag[9:12, 1] = 2 ** (np.arange(3))

    ld, lf = downsample_data(data, sample_flag, valid, window, start_indices)
    assert np.allclose(ld[:, 0], [0.25, 0.25, 0.25, 0, 1, 0, 0.25])
    assert np.allclose(ld[:, 1], [0, 0, 0, 0.5, 0, 0, 0])
    assert np.allclose(lf[:, 0], 0)
    assert np.allclose(lf[:, 1], [0, 0, 0, 7, 0, 0, 0])

    valid[0] = False
    window[0] = 0.0
    ld, lf = downsample_data(data, sample_flag, valid, window, start_indices)
    assert np.isnan(ld[0]).all()
    assert np.allclose(ld[1:, 0], [0.25, 0, 0, 0.75, 0, 0.25])
    assert np.allclose(ld[1:, 1], [0, 0, 0.5, 0, 0, 0])
    assert np.allclose(lf[:, 0], 0)
    assert np.allclose(lf[:, 1], [0, 0, 0, 6, 0, 0, 0])
