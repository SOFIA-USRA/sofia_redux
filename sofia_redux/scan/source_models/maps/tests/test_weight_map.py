# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.source_models.maps.weight_map import WeightMap
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


@pytest.fixture
def obs2d():
    data = np.full((5, 5), 2.0)
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 9.0)
    o.exposure.data = np.full(o.shape, 1.0)
    return o


@pytest.fixture
def weight_map(obs2d):
    return WeightMap(observation=obs2d.copy())


def test_init(obs2d):
    w = WeightMap(obs2d.copy())
    assert np.allclose(w.data, 9)


def test_data(weight_map):
    w = weight_map.copy()
    assert np.allclose(w.data, 9)
    w.basis.weight = None
    assert w.data is None
    w = weight_map.copy()
    weights = w.data.copy()
    weights += 1
    w.data = weights
    assert np.allclose(w.data, 10)
    assert np.allclose(w.data, w.basis.weight.data)


def test_discard(weight_map):
    w = weight_map.copy()
    w.discard()
    assert np.allclose(w.data, 0)
    assert np.all(np.isnan(w.basis.data))
