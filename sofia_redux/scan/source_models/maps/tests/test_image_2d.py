# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.source_models.maps.image_2d import Image2D


@pytest.fixture
def ones():
    data = np.ones((9, 10))
    return Image2D(data=data, unit='Jy')


@pytest.fixture
def image_hdu(ones):
    data = ones.data
    header = fits.Header()
    header['BUNIT'] = 'K'
    header['EXTNAME'] = 'image2d'
    hdu = fits.ImageHDU(data=data, header=header)
    return hdu


def test_init(ones):
    data = ones.data.copy()
    img = Image2D()
    assert img.data is None
    img = Image2D(data=data, unit='K')
    assert np.allclose(img.data, data)
    assert img.unit == 1 * units.Unit('K')
    img = Image2D(x_size=4, y_size=5)
    assert np.allclose(img.data, np.zeros((5, 4)))


def test_core(ones):
    img = ones.copy()
    assert img.core is img.data


def test_ndim(ones):
    assert ones.ndim == 2
    assert Image2D().ndim == 2


def test_add_proprietary_unit(ones):
    img = ones.copy()
    img.add_proprietary_unit()
    assert img == ones  # No change


def test_size_x(ones):
    assert ones.size_x() == 10


def test_size_y(ones):
    assert ones.size_y() == 9


def test_copy(ones):
    img = ones
    img2 = img.copy(with_contents=True)
    assert img2.history[-1] == 'pasted new content: 10x9'
    assert img == img2 and img is not img2
    img = img2.copy(with_contents=False)
    assert np.allclose(img.data, 0) and img.shape == img2.shape


def test_set_data_size():
    img = Image2D()
    img.set_data_size(3, 4)
    assert img.shape == (4, 3)


def test_set_data():
    img = Image2D()
    data = np.random.random((3, 4))
    img.set_data(data)
    assert img.history == ['set new image 4x3 2D float64 (no copy)']
    assert np.allclose(img.data, data)


def test_new_image(ones):
    img = ones.copy()
    new = img.new_image()
    assert new.shape == img.shape and np.allclose(new.data, 0)


def test_get_image(ones):
    img = ones.copy()
    img2 = img.get_image()
    assert img == img2 and img is not img2


def test_read_hdul(ones, image_hdu):
    hdu = image_hdu
    hdul = fits.HDUList()
    hdul.append(hdu)
    img = Image2D.read_hdul(hdul, 0)
    assert np.allclose(img.data, ones.data)
    assert img.unit == 1 * units.Unit('K')
    assert img.id == 'image2d'


def test_numpy_to_fits():
    img = Image2D()
    coordinates = np.arange(2)
    c = img.numpy_to_fits(coordinates)
    assert c.x == 1 and c.y == 0
