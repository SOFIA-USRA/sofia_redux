# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000, B1950
from sofia_redux.scan.coordinate_systems.epoch.precession import Precession
from sofia_redux.scan.coordinate_systems.epoch.precession_numba_functions \
    import (precess_single, precess_times)


@pytest.fixture
def b2j():
    return Precession(B1950, J2000)


@pytest.fixture
def p_matrix(b2j):
    return b2j.p.copy()


def test_precess_single(p_matrix):
    ra = np.arange(5).astype(float) + 1
    dec = np.arange(5).astype(float) + 1
    cos_lat = np.cos(dec)
    sin_lat = np.sin(dec)
    precess_single(p_matrix, ra, dec, cos_lat, sin_lat)
    assert np.allclose(
        ra,
        [1.01759688, -1.14009078, -0.13051171, 0.86532802, -1.25636616])
    assert np.allclose(
        dec,
        [1.00258871, 1.14361761, 0.1464057, -0.85524471, -1.28174491])


def test_precess_times(p_matrix):
    p = np.empty((5, 3, 3))
    p[:] = p_matrix.copy()
    ra = np.arange(5).astype(float) + 1
    dec = np.arange(5).astype(float) + 1
    cos_lat = np.cos(dec)
    sin_lat = np.sin(dec)
    precess_times(p, ra, dec, cos_lat, sin_lat)
    assert np.allclose(
        ra,
        [1.01759688, -1.14009078, -0.13051171, 0.86532802, -1.25636616])
    assert np.allclose(
        dec,
        [1.00258871, 1.14361761, 0.1464057, -0.85524471, -1.28174491])
