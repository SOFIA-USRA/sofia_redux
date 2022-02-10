# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.hawc_plus.frames.hawc_plus_frame_numba_functions \
    import validate, dark_correct, downsample_hwp_angle


def test_validate():
    n = 20
    valid = np.full(n, True)
    valid[0] = False
    validated = np.full(n, False)
    validated[1] = True
    status = np.zeros(n, dtype=int)
    chop_length = np.full(n, 9.0)
    chop_length[2] = 12.0
    chopping = True
    use_between_scans = True
    normal_observing_flag = 0
    between_scan_flag = 1
    transit_tolerance = 1.0  # arcseconds
    chopper_amplitude = 10.0  # arcseconds
    check_coordinates = np.full(n, True)
    non_sidereal = True
    equatorial_null = np.full(n, False)
    equatorial_null[3] = True
    equatorial_nan = np.full(n, False)
    equatorial_nan[4] = True
    object_null = np.full(n, False)
    object_null[5] = True
    horizontal_nan = np.full(n, False)
    horizontal_nan[6] = True
    chopper_nan = np.full(n, False)
    chopper_nan[7] = True
    lst = np.zeros(n)
    lst[8] = np.nan
    site_lon = np.zeros(n)
    site_lon[9] = np.nan
    site_lat = np.zeros(n)
    site_lat[10] = np.nan
    telescope_vpa = np.zeros(n)
    telescope_vpa[11] = np.nan
    instrument_vpa = np.zeros(n)
    instrument_vpa[12] = np.nan
    instrument_vpa[13] = np.nan
    check_coordinates[13] = False
    status[14] = 2

    valid0 = valid.copy()
    validate(
        valid, validated, status, chop_length, chopping, use_between_scans,
        normal_observing_flag, between_scan_flag, transit_tolerance,
        chopper_amplitude, check_coordinates, non_sidereal,
        equatorial_null, equatorial_nan, object_null, horizontal_nan,
        chopper_nan, lst, site_lon, site_lat, telescope_vpa,
        instrument_vpa)

    expected = np.full(n, False)
    expected[np.array([1, 13, 15, 16, 17, 18, 19])] = True
    assert validated[:2].all()
    assert not validated[2:].any()

    assert np.allclose(valid, expected)

    valid = valid0.copy()
    between_scan_flag = 0
    validated.fill(False)
    validate(
        valid, validated, status, chop_length, chopping, use_between_scans,
        normal_observing_flag, between_scan_flag, transit_tolerance,
        chopper_amplitude, check_coordinates, non_sidereal,
        equatorial_null, equatorial_nan, object_null, horizontal_nan,
        chopper_nan, lst, site_lon, site_lat, telescope_vpa,
        instrument_vpa)
    assert not valid.any()


def test_dark_correct():
    n_frames = 10
    n_channels = 5
    data = np.full((n_frames, n_channels), 10.0)
    squid_channel = 4
    data[:, squid_channel] = 1.0
    expected = data.copy()
    channel_indices = np.arange(4)
    squid_indices = np.full(4, 4)
    squid_indices[-1] = -1
    valid_frame = np.full(n_frames, True)
    valid_frame[0] = False
    dark_correct(data, valid_frame, channel_indices, squid_indices)

    expected[1:, :3] = 9
    assert np.allclose(expected, data)


def test_downsample_hwp_angle():
    window = np.array([0, 0.25, 0.5, 0.25, 0])
    hwp_angle = np.arange(100).astype(float)
    n = 20
    start_indices = np.arange(n) * 5
    valid = np.full(n, True)
    valid[5] = False
    low_res = downsample_hwp_angle(hwp_angle, start_indices, valid, window)
    expected = np.arange(0, 100, 5) + 2.0
    expected[5] = np.nan
    assert np.allclose(low_res, expected, equal_nan=True)

    low_res = downsample_hwp_angle(hwp_angle, start_indices + 2, valid, window)
    expected += 2
    expected[-1] = np.nan
    assert np.allclose(low_res, expected, equal_nan=True)
