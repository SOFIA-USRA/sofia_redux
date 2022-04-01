# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.source_models.maps.exposure_map import ExposureMap
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


@pytest.fixture
def obs2d():
    data = np.zeros((5, 6), dtype=float)
    data[2, 3] = 1
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 0.5)
    return o


@pytest.fixture
def exposure(obs2d):
    return ExposureMap(observation=obs2d.copy())


def test_init(obs2d):
    o = obs2d.copy()
    e = ExposureMap(o)
    assert np.allclose(e.data, o.exposure.data)


def test_data(exposure):
    e = exposure.copy()
    assert e.data is e.basis.exposure.data
    data = e.data.copy()
    data += 1
    e.data = data
    assert np.allclose(e.data, 1.5)
    assert np.allclose(e.basis.exposure.data, 1.5)
    e.basis.exposure = None
    assert e.data is None


def test_discard(exposure):
    e = exposure.copy()
    basis = e.basis
    e.discard()
    assert np.allclose(e.data, 0)
    assert np.all(np.isnan(basis.data))
