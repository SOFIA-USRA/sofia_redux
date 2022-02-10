# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D


@pytest.fixture
def c2d():
    """
    Return a test Coordinate2D.

    Returns
    -------
    Coordinate2D
    """
    xy = np.arange(10).reshape(2, 5).astype(float)
    c = Coordinate2D(xy, unit='degree')
    return c


def test_empty_copy(c2d):
    c = c2d.empty_copy()
    assert isinstance(c, Coordinate2D)
    assert c.coordinates is None
    assert c.unit == 'degree'


def test_copy(c2d):
    c = c2d.copy()
    c2 = c.copy()
    assert c2 is not c and c == c2


def test_ndim():
    c = Coordinate2D()
    assert c.ndim == 2


def test_x():
    c = Coordinate2D()
    assert c.x is None
    c.x = 1.0
    assert c.x == 1.0
    assert c.y == 0
    c.x = np.arange(10)
    assert np.allclose(c.x, np.arange(10))
    assert np.allclose(c.y, 0)


def test_y():
    c = Coordinate2D()
    assert c.y is None
    c.y = 2.0
    assert c.y == 2.0
    assert c.x == 0
    c.y = np.arange(10)
    assert np.allclose(c.y, np.arange(10))
    assert np.allclose(c.x, 0)


def test_max(c2d):
    m = c2d.max
    assert m.x.value == 4 and m.x.unit == 'degree'
    assert m.y.value == 9 and m.y.unit == 'degree'
    assert m.singular


def test_min(c2d):
    m = c2d.min
    assert m.x.value == 0 and m.x.unit == 'degree'
    assert m.y.value == 5 and m.y.unit == 'degree'
    assert m.singular


def test_span(c2d):
    s = c2d.span
    assert s.x.value == 4 and s.y.value == 4 and s.unit == 'degree'
    assert s.singular


def test_length(c2d):
    length = c2d.length
    assert np.allclose(length, np.hypot(*c2d.coordinates))


def test_singular(c2d):
    c = Coordinate2D()
    assert c.singular
    c = Coordinate2D([1, 2])
    assert c.singular
    assert not c2d.singular


def test_str(c2d):
    assert str(c2d) == 'x=0.0 deg->4.0 deg y=5.0 deg->9.0 deg'
    assert str(Coordinate2D()) == 'Empty coordinates'
    assert str(Coordinate2D([1, 2])) == 'x=1.0 y=2.0'


def test_repr(c2d):
    s = repr(c2d)
    assert 'x=0.0 deg->4.0 deg y=5.0 deg->9.0 deg' in s
    assert 'Coordinate2D object' in s


def test_getitem(c2d):
    c = c2d.copy()
    c1 = c[1]
    assert c1.ndim == 2 and c1.singular
    assert np.allclose(c1.coordinates.value, [1, 6])
    assert isinstance(c, Coordinate2D)
    c2 = c[1:3]
    assert not c2.singular
    assert np.allclose(c2.coordinates.value, [[1, 2], [6, 7]])


def test_get_indices(c2d):
    c = c2d.copy()
    c2 = c.get_indices(np.arange(3))
    assert isinstance(c2, Coordinate2D)
    assert c2.size == 3 and c2.ndim == 2
    assert np.allclose(c2.coordinates.value, [[0, 1, 2], [5, 6, 7]])


def test_set_singular(c2d):
    c = c2d.copy()
    c.set_singular(empty=True)
    assert c.singular and c.coordinates.shape == (2,)
    c.set_singular(empty=False)
    assert c.unit == 'degree' and np.allclose(c.coordinates, 0)


def test_copy_coordinates(c2d):
    c = Coordinate2D()
    c0 = c2d.copy()
    c.copy_coordinates(c0)
    assert c.coordinates is not c0.coordinates
    assert c == c0
    c.copy_coordinates(Coordinate2D())
    assert c.coordinates is None


def test_set_x():
    c = Coordinate2D()
    x = np.arange(5)
    c.set_x(x, copy=False)
    assert c.shape == (5,)
    assert c.coordinates[0] is not x  # because different shape array
    c.set_x(np.stack([x, x]), copy=True)
    assert c.shape == (2, 5) and c.coordinates.shape == (2, 2, 5)


def test_set_y():
    c = Coordinate2D()
    y = np.arange(4)
    c.set_y(y, copy=False)
    assert c.shape == (4,)
    assert c.coordinates[1] is not y  # because different shape array
    c.set_y(np.stack([y, y]), copy=True)
    assert c.shape == (2, 4) and c.coordinates.shape == (2, 2, 4)


def test_set():
    c = Coordinate2D()
    xy = np.arange(12).reshape((2, 3, 2))
    c.set(xy)
    assert c.shape == (3, 2)
    assert np.allclose(xy, c.coordinates)

    c.set((1 * units.Unit('m'), 2 * units.Unit('m')))
    assert c.unit == 'm'
    assert c.singular and np.allclose(c.coordinates.value, [1, 2])


def test_add_x(c2d):
    c = c2d.copy()
    x1 = 1 * c.unit
    c.add_x(x1)
    assert np.allclose(c.x.value, np.arange(5) + 1)
    c.add_x(-c.x)
    assert np.allclose(c.x, 0)


def test_subtract_x(c2d):
    c = c2d.copy()
    x1 = 1 * c.unit
    c.subtract_x(x1)
    assert np.allclose(c.x.value, np.arange(5) - 1)
    c.subtract_x(c.x)
    assert np.allclose(c.x, 0)


def test_add_y(c2d):
    c = c2d.copy()
    y1 = 1 * c.unit
    c.add_y(y1)
    assert np.allclose(c.y.value, np.arange(5) + 6)
    c.add_y(-c.y)
    assert np.allclose(c.y, 0)


def test_subtract_y(c2d):
    c = c2d.copy()
    y1 = 1 * c.unit
    c.subtract_y(y1)
    assert np.allclose(c.y.value, np.arange(5) + 4)
    c.subtract_y(c.y)
    assert np.allclose(c.y, 0)


def test_scale(c2d):
    c = c2d.copy()
    xy = c.coordinates.copy()
    c.scale(2)
    assert np.allclose(c.coordinates, xy * 2)
    half = Coordinate2D([0.5, 0.5])
    c.scale(half)
    assert np.allclose(c.coordinates, xy)

    c.scale(120 * units.Unit('arcmin'))
    assert np.allclose(c.coordinates, xy * 2)


def test_scale_x(c2d):
    c = c2d.copy()
    c.scale_x(2)
    assert np.allclose(c.x.value, np.arange(5) * 2)
    c.scale_x(30 * units.Unit('arcmin'))
    assert np.allclose(c.x.value, np.arange(5))


def test_scale_y(c2d):
    c = c2d.copy()
    c.scale_y(2)
    assert np.allclose(c.y.value, np.arange(10, 20, 2))
    c.scale_y(30 * units.Unit('arcmin'))
    assert np.allclose(c.y.value, np.arange(5, 10))


def test_invert_x(c2d):
    c = c2d.copy()
    c.invert_x()
    assert np.allclose(c.x.value, -np.arange(5))
    assert np.allclose(c.y.value, np.arange(5, 10))


def test_invert_y(c2d):
    c = c2d.copy()
    c.invert_y()
    assert np.allclose(c.x.value, np.arange(5))
    assert np.allclose(c.y.value, -np.arange(5, 10))


def test_invert(c2d):
    c = c2d.copy()
    c.invert()
    assert np.allclose(c.x.value, -np.arange(5))
    assert np.allclose(c.y.value, -np.arange(5, 10))


def test_add(c2d):
    c = c2d.copy()
    c.add(c)
    assert np.allclose(c.x.value, np.arange(5) * 2)
    assert np.allclose(c.y.value, np.arange(5, 10) * 2)


def test_subtract(c2d):
    c = c2d.copy()
    c.subtract(c)
    assert np.allclose(c.coordinates.value, 0)


def test_rotate(c2d):
    c = c2d.copy()
    c.rotate(90 * units.Unit('degree'))
    assert np.allclose(c.x.value, -np.arange(5, 10))
    assert np.allclose(c.y.value, np.arange(5))
    c.rotate(-np.pi / 2)
    assert np.allclose(c.x.value, np.arange(5))
    assert np.allclose(c.y.value, np.arange(5, 10))


def test_rotate_offsets(c2d):
    c = c2d.copy()
    angle = 180 * ((np.arange(5) % 2) - 0.5) * units.Unit('degree')
    offsets = c.coordinates.copy()
    c.rotate_offsets(offsets, angle)
    assert np.allclose(offsets[0].value, [5, -6, 7, -8, 9])
    assert np.allclose(offsets[1].value, [0, 1, -2, 3, -4])
    expected = offsets.copy()
    c = c2d.copy()
    c.rotate_offsets(c, angle)
    assert np.allclose(c.coordinates, expected)

    c = Coordinate2D()
    c.rotate_offsets(c, np.pi)
    assert c.coordinates is None


def test_angle(c2d):
    c = c2d.copy()
    a = c.angle().to('degree')
    assert np.allclose(a.value,
                       [90, 80.53767779, 74.0546041, 69.44395478, 66.03751103],
                       atol=1e-4)
    a = c.angle(center=Coordinate2D(c.mean())).to('degree')
    assert np.allclose(a.value, [-135, -135, 0, 45, 45])

    a = Coordinate2D([0, 1]).angle().to('degree')
    assert a.value == 90


def test_parse_header():
    h = fits.Header()
    c = Coordinate2D()
    c.parse_header(h, 'X')
    assert c.is_null() and c.singular
    c.parse_header(h, 'X', default=Coordinate2D([1, 2]), alt=None)
    assert c.singular and c.x == 1 and c.y == 2
    h['X1'] = 1.5
    h['X2'] = 2.5
    c.parse_header(h, 'X')
    assert c.singular and c.x == 1.5 and c.y == 2.5
    c.parse_header(h, 'Y', default=np.arange(2))
    assert np.allclose(c.coordinates, [0, 1])


def test_edit_header(c2d):
    h = fits.Header()
    c = c2d.copy()
    assert not c.singular
    c.edit_header(h, 'X')
    assert len(h) == 0

    c = Coordinate2D([1, 2])
    c.edit_header(h, 'X')
    assert h['X1'] == 1.0
    assert h['X2'] == 2.0
    assert h.comments['X1'] == 'The reference x coordinate.'
    assert h.comments['X2'] == 'The reference y coordinate.'
    c = Coordinate2D([1, 2], unit='arcsec')
    c.edit_header(h, 'X')
    assert h['X1'] == 1.0
    assert h['X2'] == 2.0
    assert h.comments['X1'] == 'The reference x coordinate (arcsec).'
    assert h.comments['X2'] == 'The reference y coordinate (arcsec).'
