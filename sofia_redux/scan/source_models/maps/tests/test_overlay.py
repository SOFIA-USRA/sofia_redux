# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.source_models.maps.overlay import Overlay


ud = units.dimensionless_unscaled


@pytest.fixture
def ones():
    data = np.ones((8, 9))  # x = 9 pixels, y = 8 pixels
    image = Image2D(data=data, unit='Jy')
    return image


@pytest.fixture
def ones_overlay(ones):
    image = ones.copy()
    overlay = Overlay(data=image)
    return overlay


def test_init(ones):
    image = ones.copy()
    overlay = Overlay()
    assert overlay.data is None
    assert overlay.unit == 1 * ud

    overlay = Overlay(data=image)
    assert np.allclose(overlay.data, ones.data)
    assert overlay.unit == 1 * units.Unit('Jy')


def test_copy(ones_overlay):
    overlay = ones_overlay
    o2 = overlay.copy()
    assert o2 == overlay and o2 is not overlay
    o2 = overlay.copy(with_contents=False)
    assert o2 != overlay


def test_data(ones_overlay):
    overlay = ones_overlay.copy()
    d0 = overlay.data.copy()
    assert np.allclose(d0, 1)
    d2 = d0 + 1
    overlay.data = d2
    assert np.allclose(overlay.data, d2)
    assert np.allclose(overlay.basis.data, d2)


def test_flag(ones_overlay):
    overlay = ones_overlay.copy()
    flag = overlay.flag.copy()
    assert np.allclose(flag, 0)
    flag[0, 0] = 1
    overlay.flag = flag
    assert np.allclose(overlay.flag, flag)
    assert np.allclose(overlay.basis.flag, flag)


def test_valid(ones_overlay):
    overlay = ones_overlay.copy()
    assert np.all(overlay.valid)
    data = overlay.data.copy()
    data[0, 0] = np.nan
    overlay.data = data
    valid = overlay.valid
    assert not valid[0, 0]
    assert np.all(valid.ravel()[1:])


def test_blanking_value(ones_overlay):
    overlay = ones_overlay.copy()
    assert np.isnan(overlay.blanking_value)
    overlay.blanking_value = 1
    assert overlay.blanking_value == 1
    assert not np.any(overlay.valid)


def test_fixed_index(ones_overlay):
    overlay = ones_overlay.copy()
    fixed_index = overlay.fixed_index.copy()
    assert fixed_index.shape == overlay.shape
    assert np.allclose(fixed_index.ravel(), np.arange(fixed_index.size))

    overlay.fixed_index = fixed_index + 1
    assert np.allclose(overlay.fixed_index.ravel(),
                       np.arange(fixed_index.size) + 1)


def test_dtype(ones_overlay):
    overlay = ones_overlay.copy()
    assert overlay.dtype == float
    overlay.dtype = int
    assert overlay.dtype == int
    assert overlay.basis.dtype == int


def test_shape(ones_overlay):
    overlay = ones_overlay.copy()
    assert overlay.shape == (8, 9)
    overlay.shape = (4, 5)
    assert overlay.shape == (4, 5)
    assert np.allclose(overlay.data, 0)


def test_size(ones_overlay):
    assert ones_overlay.size == 72


def test_ndim(ones_overlay):
    assert ones_overlay.ndim == 2


def test_set_basis(ones_overlay):
    overlay = ones_overlay.copy()
    basis = ones_overlay.basis.copy()
    basis.scale(2)
    overlay.set_basis(basis)
    assert np.allclose(overlay.data, 2)


def test_set_data(ones_overlay):
    overlay = ones_overlay.copy()
    data = np.random.random((4, 5))
    overlay.set_data(data)
    assert np.allclose(overlay.data, data)


def test_set_data_shape(ones_overlay):
    overlay = ones_overlay.copy()
    overlay.set_data_shape((4, 5))
    assert overlay.shape == (4, 5)
    assert np.allclose(overlay.data, 0)


def test_destroy(ones_overlay):
    overlay = ones_overlay.copy()
    overlay.destroy()
    assert overlay.shape == (0, 0)


def test_crop(ones_overlay):
    overlay = ones_overlay.copy()
    ranges = np.array([[4, 6], [3, 7]])
    overlay.crop(ranges)
    assert overlay.shape == (4, 2)
