# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.source_models.maps.image_2d1 import Image2D1
from sofia_redux.scan.utilities.range import Range


def test_init():
    data = np.zeros((3, 4, 5))
    image = Image2D1(data=data)
    assert image.shape == (3, 4, 5)
    image = Image2D1(x_size=5, y_size=6, z_size=7)
    assert image.shape == (7, 6, 5)


def test_ndim():
    assert Image2D1().ndim == 3


def test_size_x():
    assert Image2D1(x_size=5, y_size=6, z_size=7).size_x() == 5


def test_size_y():
    assert Image2D1(x_size=5, y_size=6, z_size=7).size_y() == 6


def test_size_z():
    assert Image2D1(x_size=5, y_size=6, z_size=7).size_z() == 7


def test_copy():
    image = Image2D1(data=np.arange(60).reshape((3, 4, 5)))
    image2 = image.copy()
    assert image == image2 and image is not image2


def test_set_data_size():
    image = Image2D1()
    image.set_data_size(3, 4, 5)
    assert image.shape == (5, 4, 3)


def test_set_data():
    image = Image2D1()
    image.set_data(np.arange(60).reshape((3, 4, 5)))
    assert 'set new image 5x4x3 2D1 float64 (no copy)' in image.history
    assert image.shape == (3, 4, 5)


def test_new_image():
    image = Image2D1().new_image()
    assert isinstance(image, Image2D1)


def test_get_image():
    image = Image2D1(data=np.zeros((3, 4, 5)))
    im2 = image.get_image()
    assert im2 == image and im2 is not image


def test_get_asymmetry():
    data = np.zeros((3, 7, 7))
    data[0, 2:5, 3] = 1
    data[1, 2:5, 2:5] = 1
    data[2, 3, 2:5] = 1
    image = Image2D1(data=data)
    center_index = Coordinate2D([3, 3])
    grid = FlatGrid2D()
    angle = 0 * units.Unit('degree')
    radial_range = Range(0, 5)
    asymmetry, rms = image.get_asymmetry(grid, center_index, angle,
                                         radial_range)
    assert np.allclose(asymmetry, [1/3, 1/9, 1/3])
    assert np.allclose(rms, [5/3, 5/9, 5/3])


def test_get_asymmetry_2d():
    data = np.zeros((3, 7, 7))
    data[0, 2:5, 3] = 1
    data[1, 3, 2:5] = 1
    image = Image2D1(data=data)
    center_index = Coordinate2D([3, 3])
    grid = FlatGrid2D()
    angle = 0 * units.Unit('degree')
    radial_range = Range(0, 5)
    asymmetry = image.get_asymmetry_2d(grid, center_index, angle, radial_range)
    assert np.allclose(asymmetry.x, [1/3, 1/3, 0])
    assert np.allclose(asymmetry.y, 0)
