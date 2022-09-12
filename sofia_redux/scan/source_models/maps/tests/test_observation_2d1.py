# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.source_models.maps.image_2d1 import Image2D1
from sofia_redux.scan.source_models.maps.observation_2d1 import Observation2D1


arcsec = units.Unit('arcsec')
um = units.Unit('um')


@pytest.fixture
def obs2d1():
    data = np.zeros((10, 11, 12), dtype=float)
    data[5, 6, 7] = 1
    o = Observation2D1(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 0.5)
    o.filter_fwhm = Coordinate2D1(xy=[1, 1] * arcsec, z=1.0 * um)
    o.grid.resolution = Coordinate2D1(
        [1, 1] * arcsec, 1 * um)
    return o


def test_init():
    o = Observation2D1(shape=(10, 11, 12))
    assert o.shape == (10, 11, 12)
    assert isinstance(o.weight, Image2D1) and o.weight.shape == (10, 11, 12)
    assert isinstance(o.exposure, Image2D1) and o.exposure.shape == (
        10, 11, 12)


def test_copy(obs2d1):
    o = obs2d1
    o2 = obs2d1.copy()
    assert o == o2 and o is not o2


def test_to_weight_image(obs2d1):
    image = obs2d1.to_weight_image(None)
    assert isinstance(image, Image2D1) and image.shape == (10, 11, 12)
    image = obs2d1.to_weight_image(np.zeros((10, 11, 12)))
    assert isinstance(image, Image2D1) and image.shape == (10, 11, 12)


def test_crop(obs2d1):
    o = obs2d1.copy()
    ranges = Coordinate2D1([[1, 3], [1, 4], [1, 5]])
    o.crop(ranges)
    assert o.shape == (5, 4, 3)


def test_filter_correct(obs2d1):
    o = obs2d1.copy()
    underlying_fwhm = Coordinate2D1(xy=[2, 2] * arcsec, z=1.0 * um)
    o.data += 1.0
    o.filter_blanking = 7.0
    o.filter_correct(underlying_fwhm)
    assert np.isclose(o.data.max(), 5)
    assert np.isclose(o.data.min(), 2)


def test_fft_filter_above(obs2d1):
    o = obs2d1.copy()
    fwhm = Coordinate2D1(xy=[1, 1] * arcsec, z=0.5 * um)
    o.fft_filter_above(fwhm)
    assert np.allclose(o.data, 0, atol=0.5)
