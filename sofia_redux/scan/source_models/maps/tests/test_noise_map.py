# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.source_models.maps.noise_map import NoiseMap
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


@pytest.fixture
def obs2d():
    data = np.zeros((5, 6), dtype=float)
    data[2, 3] = 1
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 4.0)
    o.exposure.data = np.full(o.shape, 1.0)
    return o


@pytest.fixture
def noise(obs2d):
    return NoiseMap(observation=obs2d.copy())


def test_init(obs2d):
    n = NoiseMap(obs2d)
    assert np.allclose(n.data, 0.5)


def test_data(noise):
    n = noise.copy()
    w = n.basis.weight.data
    w[0, 0] = 0.0
    n.basis.weight.data = w
    assert n.data[0, 0] == 0
    assert np.allclose(n.data.ravel()[1:], 0.5)
    n.basis.weight = None
    assert n.data is None

    n = noise.copy()
    noise_values = n.data
    assert isinstance(noise_values, np.ndarray)
    noise_values.fill(0.25)
    noise_values[0, 0] = 0.0
    n.data = noise_values
    assert np.allclose(n.data, noise_values)
    assert n.basis.weight.data[0, 0] == 0
    assert np.allclose(n.basis.weight.data.ravel()[1:], 16)
    n2 = n.copy()
    n2.data = np.full(n2.shape, 2.0)
    n.data = n2
    assert np.allclose(n.data, n2.data)


def test_set_default_unit(noise):
    n = noise.copy()
    unit = 1 * units.Unit('m')
    n.unit = unit
    assert n.unit == unit
    n.set_default_unit()
    assert n.unit == 1 * units.Unit('Jy')
