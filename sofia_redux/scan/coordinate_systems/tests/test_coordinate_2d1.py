# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1


arcsec = units.Unit('arcsec')
um = units.Unit('um')


def test_class():
    assert Coordinate2D1.default_dimensions == 2


def test_init():
    c = Coordinate2D1(xy_unit='arcsec', z_unit='um')
    assert isinstance(c.xy_coordinates, Coordinate2D)
    assert c.xy_coordinates.size == 0
    assert c.xy_coordinates.unit == 'arcsec'
    assert isinstance(c.z_coordinates, Coordinate1D)
    assert c.z_coordinates.size == 0
    assert c.z_coordinates.unit == 'um'

    xy = Coordinate2D([1, 2])
    z = Coordinate1D(3)
    c = Coordinate2D1(xy, z)
    assert c.x == 1 and c.y == 2 and c.z == 3

    c = Coordinate2D1([1, 2, 3] * arcsec)
    assert c.x == 1 * arcsec and c.y == 2 * arcsec and c.z == 3 * arcsec

    c2 = Coordinate2D1(c)
    assert c2 == c

    c3 = Coordinate3D([1, 2, 3] * um)
    c = Coordinate2D1(c3)
    assert c.x == 1 * um and c.y == 2 * um and c.z == 3 * um


def test_eq():
    c = Coordinate2D1([1, 2, 3])
    assert c != Coordinate2D()
    c2 = c.copy()
    assert c2 == c
    c2.x = 5
    assert c2 != c
    c2.z = 5
    assert c2 != c


def test_empty_copy():
    c = Coordinate2D1([1, 2, 3])
    c2 = c.empty_copy()
    assert c2.xy_coordinates.size == 0
    assert c2.z_coordinates.size == 0


def test_copy():
    c = Coordinate2D1([1, 2, 3])
    c2 = c.copy()
    assert c2 is not c and c2 == c


def test_ndim():
    assert Coordinate2D1().ndim == (2, 1)


def test_x():
    c = Coordinate2D1([1, 2, 3])
    assert c.x == 1
    c.x = 2
    assert c.x == 2


def test_y():
    c = Coordinate2D1([1, 2, 3])
    assert c.y == 2
    c.y = 3
    assert c.y == 3


def test_z():
    c = Coordinate2D1([1, 2, 3])
    assert c.z == 3
    c.z = 4
    assert c.z == 4


def test_xy_unit():
    c = Coordinate2D1([1, 2, 3])
    c.xy_unit = 'arcsec'
    assert c.xy_unit == 'arcsec'
    assert c.x == 1 * arcsec


def test_z_unit():
    c = Coordinate2D1([1, 2, 3])
    c.z_unit = 'um'
    assert c.z == 3 * um
    assert c.z_unit == 'um'


def test_max():
    xy = np.arange(10).reshape((2, 5))
    z = np.arange(3)
    c = Coordinate2D1(xy, z)
    c_max = c.max
    assert c_max.x == 4 and c_max.y == 9 and c_max.z == 2


def test_min():
    xy = np.arange(10).reshape((2, 5)) + 1
    z = np.arange(3) + 2
    c = Coordinate2D1(xy, z)
    c_min = c.min
    assert c_min.x == 1 and c_min.y == 6 and c_min.z == 2


def test_span():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(3))
    span = c.span
    assert span.x == 4 and span.y == 4 and span.z == 2


def test_length():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(3))
    assert np.allclose(c.length,
                       [5, 6.08276253, 7.28010989, 8.54400375, 9.8488578],
                       atol=1e-4)


def test_singular():
    c = Coordinate2D1([1, 2, 3])
    assert c.singular
    c.z = np.arange(2)
    assert not c.singular
    c = Coordinate2D1([1, 2, 3])
    c.x = np.arange(2)
    assert not c.singular


def test_str():
    c = Coordinate2D1()
    assert str(c) == 'x=Empty, y=Empty, z=Empty'
    c = Coordinate2D1([1, 2, 3])
    assert str(c) == 'x=1.0 y=2.0, z=3.0'


def test_repr():
    c = Coordinate2D1()
    s = repr(c)
    assert s.startswith('x=Empty, y=Empty, z=Empty')
    assert 'Coordinate2D1' in s


def test_len():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(3))
    assert len(c) == 15


def test_get_indices():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c2 = c.get_indices((1, 2))
    assert c2.x == 1 and c2.y == 6 and c2.z == 2


def test_set_singular():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.set_singular()
    assert c.is_null() and len(c) == 1


def test_copy_coordinates():
    c = Coordinate2D1()
    c2 = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.copy_coordinates(c2)
    assert c == c2


def test_set_x():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.set_x(np.arange(5, 10))
    assert np.allclose(c.x, c.y)


def test_set_y():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.set_y(np.arange(5))
    assert np.allclose(c.x, c.y)


def test_set_z():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.set_z(1)
    assert c.z == 1 and np.allclose(c.x, np.arange(5))


def test_set():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.set([1, 2, 3])
    assert c.x == 1 and c.y == 2 and c.z == 3


def test_add():
    c = Coordinate2D1([1, 2, 3])
    c.add(c)
    assert c.x == 2 and c.y == 4 and c.z == 6


def test_add_x():
    c = Coordinate2D1([1, 2, 3])
    c.add_x(2)
    assert c.x == 3


def test_add_y():
    c = Coordinate2D1([1, 2, 3])
    c.add_y(2)
    assert c.y == 4


def test_add_z():
    c = Coordinate2D1([1, 2, 3])
    c.add_z(2)
    assert c.z == 5


def test_subtract():
    c = Coordinate2D1([1, 2, 3])
    c.subtract(c)
    assert c.is_null()


def test_subtract_x():
    c = Coordinate2D1([1, 2, 3])
    c.subtract_x(2)
    assert c.x == -1


def test_subtract_y():
    c = Coordinate2D1([1, 2, 3])
    c.subtract_y(3)
    assert c.y == -1


def test_subtract_z():
    c = Coordinate2D1([1, 2, 3])
    c.subtract_z(4)
    assert c.z == -1


def test_scale():
    c = Coordinate2D1([1, 2, 3])
    c.scale(c)
    assert c.x == 1 and c.y == 4 and c.z == 9
    c.scale([2, 3, 4])
    assert c.x == 2 and c.y == 12 and c.z == 36
    c.scale(0.5)
    assert c.x == 1 and c.y == 6 and c.z == 18


def test_scale_x():
    c = Coordinate2D1([2, 2, 3])
    c.scale_x(5)
    assert c.x == 10


def test_scale_y():
    c = Coordinate2D1([1, 2, 3])
    c.scale_y(5)
    assert c.y == 10


def test_scale_z():
    c = Coordinate2D1([1, 2, 3])
    c.scale_z(5)
    assert c.z == 15


def test_invert():
    c = Coordinate2D1([1, 2, 3])
    c.invert()
    assert c.x == -1 and c.y == -2 and c.z == -3


def test_invert_x():
    c = Coordinate2D1([1, 2, 3])
    c.invert_x()
    assert c.x == -1 and c.y == 2 and c.z == 3


def test_invert_y():
    c = Coordinate2D1([1, 2, 3])
    c.invert_y()
    assert c.x == 1 and c.y == -2 and c.z == 3


def test_invert_z():
    c = Coordinate2D1([1, 2, 3])
    c.invert_z()
    assert c.x == 1 and c.y == 2 and c.z == -3


def test_parse_header():
    c = Coordinate2D1()
    h = fits.Header()
    h['X1'] = 1.0
    h['X2'] = 2.0
    h['X3'] = 3.0
    c.parse_header(h, 'X')
    assert c.x == 1 and c.y == 2 and c.z == 3


def test_edit_header():
    c = Coordinate2D1([1, 2, 3])
    h = fits.Header()
    c.edit_header(h, 'X')
    assert h['X1'] == 1 and h['X2'] == 2 and h['X3'] == 3


def test_broadcast_to():
    c = Coordinate2D1([1, 2, 3])
    c.broadcast_to((2, 3))
    assert c.xy_coordinates.shape == (2, 3) and np.allclose(c.x, 1)
    assert np.allclose(c.y, 2)
    assert c.z_coordinates.shape == (2, 3) and np.allclose(c.z, 3)


def test_convert_indices():
    c = Coordinate2D1()
    assert c.convert_indices(None) == (None, None)
    assert c.convert_indices(slice(1, 5)) == (slice(1, 5), None)
    inds = c.convert_indices(np.arange(5))
    assert np.allclose(inds[0], np.arange(5))
    assert inds[1] is None
    assert c.convert_indices((1, 2)) == (1, 2)
    with pytest.raises(ValueError) as err:
        _ = c.convert_indices([1, 2, 3])
    assert "Could not convert indices" in str(err.value)


def test_nan():
    c = Coordinate2D1([1, 2, 3])
    c.nan()
    assert c.is_nan() == (True, True)


def test_zero():
    c = Coordinate2D1([1, 2, 3])
    c.zero()
    assert c.is_null() == (True, True)


def test_is_nan():
    c = Coordinate2D1([1, 2, 3])
    assert c.is_nan() == (False, False)
    c.xy_coordinates.nan()
    assert c.is_nan() == (True, False)
    c.z_coordinates.nan()
    assert c.is_nan() == (True, True)


def test_is_null():
    c = Coordinate2D1([1, 2, 3])
    assert c.is_null() == (False, False)
    c.xy_coordinates.zero()
    assert c.is_null() == (True, False)
    c.z_coordinates.zero()
    assert c.is_null() == (True, True)


def test_is_finite():
    c = Coordinate2D1([1, 2, 3])
    assert c.is_finite() == (True, True)
    c.nan()
    assert c.is_finite() == (False, False)


def test_is_infinite():
    c = Coordinate2D1([1, 2, 3])
    assert c.is_infinite() == (False, False)
    c.xy_coordinates.fill(np.inf)
    assert c.is_infinite() == (True, False)
    c.z_coordinates.fill(np.inf)
    assert c.is_infinite() == (True, True)


def test_insert_blanks():
    c = Coordinate2D1(np.arange(10).reshape((2, 5)), np.arange(5))
    c.insert_blanks((2, 3))
    assert np.allclose(c.x, [0, 1, np.nan, 2, 3, 4], equal_nan=True)
    assert np.allclose(c.y, [5, 6, np.nan, 7, 8, 9], equal_nan=True)
    assert np.allclose(c.z, [0, 1, 2, np.nan, 3, 4], equal_nan=True)


def test_merge():
    c = Coordinate2D1(np.arange(10).reshape(2, 5), np.arange(5))
    c2 = Coordinate2D1([[1, 2], [3, 4]], [5, 6])
    c.merge(c2)
    assert np.allclose(c.x, [0, 1, 2, 3, 4, 1, 2])
    assert np.allclose(c.y, [5, 6, 7, 8, 9, 3, 4])
    assert np.allclose(c.z, [0, 1, 2, 3, 4, 5, 6])


def test_paste():
    c = Coordinate2D1(np.arange(10).reshape(2, 5), np.arange(5))
    c.paste(Coordinate2D1([9, 9, 9]), (1, 2))
    assert np.allclose(c.x, [0, 9, 2, 3, 4])
    assert np.allclose(c.y, [5, 9, 7, 8, 9])
    assert np.allclose(c.z, [0, 1, 9, 3, 4])


def test_shift():
    c = Coordinate2D1(np.arange(10).reshape(2, 5), np.arange(5))
    c.shift((2, 1))
    assert np.allclose(c.x, [np.nan, np.nan, 0, 1, 2], equal_nan=True)
    assert np.allclose(c.y, [np.nan, np.nan, 5, 6, 7], equal_nan=True)
    assert np.allclose(c.z, [np.nan, 0, 1, 2, 3], equal_nan=True)


def test_mean():
    c = Coordinate2D1(np.arange(10).reshape(2, 5), np.arange(5))
    m = c.mean()
    assert m.x == 2 and m.y == 7 and m.z == 2


def test_to_coordinate_3d():
    c = Coordinate2D1()
    with pytest.raises(ValueError) as err:
        c.to_coordinate_3d()
    assert 'populated coordinates' in str(err.value)

    c = Coordinate2D1([1, 1] * arcsec, 2)
    with pytest.raises(ValueError) as err:
        c.to_coordinate_3d()
    assert 'units are not convertable' in str(err.value)

    c = Coordinate2D1([1, 2, 3])
    c3 = c.to_coordinate_3d()
    assert isinstance(c3, Coordinate3D)
    assert c3.x == 1 and c3.y == 2 and c3.z == 3

    x = np.arange(10) * arcsec
    y = x.copy()
    z = np.arange(5) * arcsec
    c = Coordinate2D1([x, y, z])
    c3 = c.to_coordinate_3d()
    assert np.allclose(c3.x, (np.arange(50) % 10) * arcsec)
    assert np.allclose(c3.y, c3.x)
    assert np.allclose(c3.z, (np.arange(50) // 10) * arcsec)
    c = Coordinate2D1([x.value, y.value, z.value])
    c3 = c.to_coordinate_3d()
    assert np.allclose(c3.x, np.arange(50) % 10)
    assert np.allclose(c3.y, c3.x)
    assert np.allclose(c3.z, np.arange(50) // 10)
