# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.source_models.maps.significance_map import \
    SignificanceMap
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


@pytest.fixture
def obs2d():
    data = np.full((5, 5), 2.0)
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 1.0)
    return o


@pytest.fixture
def significance(obs2d):
    return SignificanceMap(observation=obs2d.copy())


def test_init(obs2d):
    s = SignificanceMap(obs2d)
    assert np.allclose(s.data, 8)


def test_set_default_unit(significance):
    s = significance.copy()
    s.unit = 1 * units.Unit('Jy')
    s.set_default_unit()
    assert s.unit == 1 * units.dimensionless_unscaled


def test_set_unit(significance):
    s = significance.copy()
    s.set_unit('K')
    assert s.unit == 1 * units.dimensionless_unscaled


def test_data(significance):
    s = significance.copy()
    assert np.allclose(s.data, 8)
    s.basis.basis._data = None
    assert s.data is None
    s = significance.copy()
    s.basis.weight._data = None
    assert s.data is None
    s.basis.weight = None
    assert s.data is None

    s = significance.copy()
    data = s.data * 2
    s.data = data
    assert np.allclose(s.data, data)
    assert np.allclose(s.basis.data, 4)
    s2 = significance.copy()
    s2.data = s
    assert s2 == s
    assert s2.basis == s.basis
