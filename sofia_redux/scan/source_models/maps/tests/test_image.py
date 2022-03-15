# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.source_models.maps.image import Image
from sofia_redux.scan.utilities.range import Range


ud = units.dimensionless_unscaled


@pytest.fixture
def ones():
    shape = (10, 11)
    data = np.ones(shape)
    img = Image(data=data, unit='Jy')
    return img


@pytest.fixture
def ascending():
    shape = (9, 10)
    data = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
    img = Image(data=data, unit='Jy')
    img.set_id('ascending')
    return img


@pytest.fixture
def image_hdu(ascending):
    data = ascending.data
    header = fits.Header()
    header['BUNIT'] = 'Jy'
    header['EXTNAME'] = 'foo'
    hdu = fits.ImageHDU(data=data, header=header)
    return hdu


def test_init():
    img = Image()
    assert img.id == ''
    assert img.data is None
    assert img.unit == 1 * ud


def test_copy(ones):
    img = ones.copy()
    img2 = img.copy()
    assert img2 == img and img2 is not img
    img2 = img.copy(with_contents=False)
    assert img2 != img
    assert np.allclose(img2.data, 0)


def test_eq(ones):
    img = ones.copy()
    assert img == img
    assert img != 1
    img2 = img.copy()
    assert img2 == img
    img2.id = 'foo'
    assert img2 != img


def test_new_image(ones):
    img = ones.copy()
    new = img.new_image()
    assert new.shape == img.shape
    assert np.allclose(new.data, 0)
    assert new.unit == img.unit


def test_set_id(ones):
    img = ones.copy()
    img.set_id(1)
    assert img.id == '1'


def test_get_id(ones):
    img = ones.copy()
    img.set_id(1)
    assert img.get_id() == img.id


def test_destroy(ones):
    img = ones.copy()
    img.destroy()
    assert img.size == 0
    assert img.history == []


def test_renew(ones):
    img = ones.copy()
    img.renew()
    assert np.allclose(img.data, 0)


def test_set_data(ones):
    img = ones.copy()
    img.set_data(np.arange(10, dtype=float).reshape(2, 5))
    assert img.shape == (2, 5)
    assert img.history == ['set new image 5x2 2D float64']


def test_transpose(ascending):
    img = ascending.copy()
    d1 = img.data
    img.transpose()
    assert np.allclose(img.data.T, d1)
    assert img.history[-1] == 'transposed'
    img = Image()
    img1 = img.copy()
    img.transpose()
    assert img == img1  # No difference with no data


def test_get_image(ascending):
    img = ascending.copy()
    assert img.dtype == float
    img2 = img.get_image()
    assert img2 == img and img2.dtype == img.dtype
    assert np.isnan(img.blanking_value) and np.isnan(img2.blanking_value)

    img2 = img.get_image(dtype=int, blanking_value=0)
    assert img2.dtype == int and img2.blanking_value == 0
    valid = img2.valid
    assert not valid[0, 0] and np.all(valid.ravel()[1:])
    assert img2.history[-1] == 'pasted new content: 10x9'


def test_get_valid_data(ascending):
    img = ascending.copy()
    d0 = img.data
    img.set_blanking_level(0.0)
    assert np.allclose(img.get_valid_data(), d0)
    data = img.get_valid_data(default=np.nan)
    assert np.isnan(data[0, 0])
    assert np.allclose(data.ravel()[1:], d0.ravel()[1:])
    img = Image()
    assert img.get_valid_data() is None


def test_crop(ascending):
    img = Image()
    img0 = img.copy()

    ranges = np.array([[5, 7], [6, 8]])
    img.crop(ranges)
    assert img == img0

    img = ascending.copy()
    with pytest.raises(ValueError) as err:
        img.crop(None)
    assert 'Received None' in str(err.value)

    img.crop(ranges)
    assert np.allclose(img.data, [[65, 66], [75, 76]])
    assert img.history == ['set new image 2x2 2D float64']

    no_change = np.array([[0, 2], [0, 2]])
    img.crop(no_change)
    assert img.history == ['set new image 2x2 2D float64']  # No change


def test_auto_crop(ascending):
    img = Image()
    img.auto_crop()
    assert img.data is None
    img = ascending.copy()
    assert isinstance(img.data, np.ndarray)
    d0 = img.data.copy()
    img.auto_crop()  # No change
    assert np.allclose(img.data, d0)
    data = d0.copy()
    data[0] = np.nan  # Remove first column
    img.data = data
    img.auto_crop()
    assert np.allclose(img.data, d0[1:])


def test_edit_header(ascending):
    img = ascending.copy()
    header = fits.Header()
    img.edit_header(header)
    assert header['EXTNAME'] == 'ascending'
    assert header['BUNIT'] == 'Jy'


def test_parse_header():
    img = Image()
    header = fits.Header()
    header['BUNIT'] = 'Jy'
    header['EXTNAME'] = 'foo'
    header['HISTORY'] = 'bar'
    img.parse_header(header)
    assert img.unit == 1 * units.Unit('Jy')
    assert img.id == 'foo'
    assert img.history == ['bar']


def test_read_hdu(ascending, image_hdu):
    hdu = image_hdu
    img = Image()
    img.read_hdu(hdu)
    assert np.allclose(img.data, ascending.data)
    assert img.unit == 1 * units.Unit('Jy')
    assert img.id == 'foo'


def test_read_hdul(ascending, image_hdu):
    hdu = image_hdu
    hdul = fits.HDUList()
    hdul.append(hdu)
    img = Image.read_hdul(hdul, 0)
    assert np.allclose(img.data, ascending.data)
    assert img.unit == 1 * units.Unit('Jy')
    assert img.id == 'foo'


def test_get_asymmetry(ones):
    img = ones.copy()
    data = img.data.copy()
    grid = FlatGrid2D()
    center_index = Coordinate2D([5.0, 5.0])
    radial_range = Range(0.0, 4.0)
    data.fill(0)
    data[4:6, 4:8] = 1.0  # Asymmetric about (5, 5) in x but not y
    img.data = data
    angle = 0 * units.Unit('degree')
    asymmetry, rms = img.get_asymmetry(grid, center_index, angle, radial_range)
    assert np.isclose(asymmetry,  0.361803, atol=1e-6)
    assert np.isclose(rms, 0.625)
    center_index = Coordinate2D([6.0, 5.0])
    asymmetry, rms = img.get_asymmetry(grid, center_index, angle, radial_range)
    assert np.isclose(asymmetry,  -0.111803, atol=1e-6)
    assert np.isclose(rms, 0.625)
    angle = 45 * units.Unit('degree')
    asymmetry, rms = img.get_asymmetry(grid, center_index, angle, radial_range)
    assert np.isclose(asymmetry,  -0.331974, atol=1e-6)
    assert np.isclose(rms, 0.618718, atol=1e-6)
    img.clear()  # zero data
    asymmetry, rms = img.get_asymmetry(grid, center_index, angle, radial_range)
    assert asymmetry == 0 and rms == 0


def test_get_asymmetry_2d(ones):
    img = ones.copy()
    img.clear()
    data = img.data
    data[4:6, 4:8] = 1.0
    img.data = data
    angle = 30 * units.Unit('degree')
    grid = FlatGrid2D()
    center_index = Coordinate2D([5.0, 5.0])
    radial_range = Range(0.0, 4.0)
    asymmetry = img.get_asymmetry_2d(grid, center_index, angle, radial_range)
    assert str(asymmetry) == (
        'Asymmetry: x = 13.449% +- 62.187%, y = -49.066% +- 61.555%')

    img.clear()
    asymmetry = img.get_asymmetry_2d(grid, center_index, angle, radial_range)
    assert str(asymmetry) == (
        'Asymmetry: x = 0.000% +- 0.000%, y = 0.000% +- 0.000%')
