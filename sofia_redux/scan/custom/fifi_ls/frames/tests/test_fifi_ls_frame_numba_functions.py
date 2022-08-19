# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.fifi_ls.frames.fifi_ls_frame_numba_functions \
    import (validate, get_relative_frame_weights)


def test_validate():
    n_frames = 20
    validated = np.full(n_frames, False)
    validated[0] = True
    valid = np.full(n_frames, True)
    valid[:2] = False
    weight = np.ones(n_frames)
    weight[2] = 0
    check_coordinates = np.full(n_frames, True)
    check_coordinates[3] = False
    equatorial_null = np.full(n_frames, False)
    equatorial_null[3:5] = True
    equatorial_nan = np.full(n_frames, False)
    equatorial_nan[5] = True
    horizontal_nan = np.full(n_frames, False)
    horizontal_nan[6] = True
    lst = np.ones(n_frames)
    lst[7] = np.nan
    site_lon = np.ones(n_frames)
    site_lon[8] = np.nan
    site_lat = np.ones(n_frames)
    site_lat[9] = np.nan
    telescope_vpa = np.ones(n_frames)
    telescope_vpa[10] = np.nan
    instrument_vpa = np.ones(n_frames)
    instrument_vpa[11] = np.nan
    chopper_nan = np.full(n_frames, False)
    chopper_nan[12] = True

    validate(valid=valid,
             validated=validated,
             weight=weight,
             check_coordinates=check_coordinates,
             equatorial_null=equatorial_null,
             equatorial_nan=equatorial_nan,
             horizontal_nan=horizontal_nan,
             chopper_nan=chopper_nan,
             lst=lst,
             site_lon=site_lon,
             site_lat=site_lat,
             telescope_vpa=telescope_vpa,
             instrument_vpa=instrument_vpa)
    assert np.allclose(
        valid, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    assert validated[:2].all()
    assert not validated[2:].any()


def test_get_relative_frame_weights():
    variance = np.ones((1, 1))
    wt = get_relative_frame_weights(variance)
    assert wt[0] == 0

    rand = np.random.RandomState(0)
    variance = rand.random((20, 10))
    wt = get_relative_frame_weights(variance)
    assert wt.shape == (20,)
    assert np.isclose(np.median(wt), 1.0, atol=0.1)

