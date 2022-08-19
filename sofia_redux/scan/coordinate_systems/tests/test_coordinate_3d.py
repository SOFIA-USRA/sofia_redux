# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D


def test_class():
    assert Coordinate3D.default_dimensions == 3


def test_empty_copy():
    c = Coordinate3D([1, 2, 3])
    c2 = c.empty_copy()
    assert c2.size == 0


def test_copy():
    c = Coordinate3D([1, 2, 3])
    c2 = c.copy()
    assert c2 == c and c2 is not c


def test_ndim():
    assert Coordinate3D().ndim == 3


def test_xyz():
    c = Coordinate3D()
    assert c.x is None and c.y is None and c.z is None
    c = Coordinate3D([1, 2, 3])
    assert c.x == 1 and c.y == 2 and c.z == 3
    c.x = 4
    c.y = 5
    c.z = 6
    assert c.x == 4 and c.y == 5 and c.z == 6


def test_max():
    c = Coordinate3D(np.random.random((3, 5)))
    assert np.allclose(c.max.coordinates, np.max(c.coordinates, axis=1))


def test_min():
    c = Coordinate3D(np.random.random((3, 5)))
    assert np.allclose(c.min.coordinates, np.min(c.coordinates, axis=1))


def test_span():
    c = Coordinate3D(np.random.random((3, 5)))
    assert np.allclose(c.span.coordinates, np.ptp(c.coordinates, axis=1))


def test_length():
    assert Coordinate3D().length is None
    c = Coordinate3D(np.random.random((3, 5)))
    assert np.allclose(c.length, np.linalg.norm(c.coordinates, axis=0))


def test_singular():
    assert Coordinate3D().singular
    assert Coordinate3D([1, 2, 3]).singular
    assert not Coordinate3D(np.random.random((3, 5))).singular


def test_str():
    assert str(Coordinate3D()) == 'Empty coordinates'
    assert str(Coordinate3D([1, 2, 3])) == 'x=1.0 y=2.0 z=3.0'
    assert str(Coordinate3D(np.arange(6).reshape((3, 2)))) == (
        'x=0.0->1.0 y=2.0->3.0 z=4.0->5.0')


def test_repr():
    s = repr(Coordinate3D())
    assert s.startswith('Empty coordinates') and 'Coordinate3D' in s


def test_getitem():
    c = Coordinate3D(np.arange(6).reshape((3, 2)))
    c1 = c[1]
    assert c1.x == 1 and c1.y == 3 and c1.z == 5


def test_get_indices():
    c = Coordinate3D(np.arange(6).reshape((3, 2)))
    assert c[...] == c


def test_set_singular():
    c0 = Coordinate3D(np.arange(6).reshape((3, 2)), unit='arcsec')
    c = c0.copy()
    c.set_singular(empty=False)
    assert c.singular and c.is_null()
    c = c0.copy()
    c.set_singular(empty=True)
    assert c.singular and c.coordinates.unit == 'arcsec'


def test_copy_coordinates():
    c = Coordinate3D()
    c1 = Coordinate3D([1, 2, 3])
    c.copy_coordinates(c1)
    assert c == c1
    c = c1.copy()
    c1.broadcast_to((2, 4))
    c.copy_coordinates(c1)
    assert c == c1
    c.copy_coordinates(Coordinate3D())
    assert c.coordinates is None


def test_set_x():
    c = Coordinate3D()
    x = np.arange(3, dtype=float)
    c.set_x(x, copy=False)
    assert np.allclose(c.x, x)
    c.set_x(x, copy=True)
    assert np.allclose(c.x, x) and c.x is not x


def test_set_y():
    c = Coordinate3D()
    x = np.arange(3, dtype=float)
    c.set_y(x, copy=False)
    assert np.allclose(c.y, x)
    c.set_y(x, copy=True)
    assert np.allclose(c.y, x) and c.y is not x


def test_set_z():
    c = Coordinate3D()
    x = np.arange(3, dtype=float)
    c.set_z(x, copy=False)
    assert np.allclose(c.z, x)
    c.set_z(x, copy=True)
    assert np.allclose(c.z, x) and c.z is not x


def test_set():
    c = Coordinate3D()
    c.set([1, 2, 3])
    assert c.x == 1 and c.y == 2 and c.z == 3


def test_add_x():
    c = Coordinate3D([1, 2, 3])
    c.add_x(1)
    assert c.x == 2 and c.y == 2 and c.z == 3


def test_subtract_x():
    c = Coordinate3D([1, 2, 3])
    c.subtract_x(1)
    assert c.x == 0 and c.y == 2 and c.z == 3


def test_add_y():
    c = Coordinate3D([1, 2, 3])
    c.add_y(1)
    assert c.x == 1 and c.y == 3 and c.z == 3


def test_subtract_y():
    c = Coordinate3D([1, 2, 3])
    c.subtract_y(1)
    assert c.x == 1 and c.y == 1 and c.z == 3


def test_add_z():
    c = Coordinate3D([1, 2, 3])
    c.add_z(1)
    assert c.x == 1 and c.y == 2 and c.z == 4


def test_subtract_z():
    c = Coordinate3D([1, 2, 3])
    c.subtract_z(1)
    assert c.x == 1 and c.y == 2 and c.z == 2


def test_scale():
    c = Coordinate3D([1, 2, 3])
    c.scale(c)
    assert c.x == 1 and c.y == 4 and c.z == 9
    c.scale(2)
    assert c.x == 2 and c.y == 8 and c.z == 18


def test_scale_x():
    c = Coordinate3D([1, 2, 3])
    c.scale_x(2)
    assert c.x == 2 and c.y == 2 and c.z == 3


def test_scale_y():
    c = Coordinate3D([1, 2, 3])
    c.scale_y(2)
    assert c.x == 1 and c.y == 4 and c.z == 3


def test_scale_z():
    c = Coordinate3D([1, 2, 3])
    c.scale_z(2)
    assert c.x == 1 and c.y == 2 and c.z == 6


def test_invert_x():
    c = Coordinate3D([1, 2, 3])
    c.invert_x()
    assert c.x == -1 and c.y == 2 and c.z == 3


def test_invert_y():
    c = Coordinate3D([1, 2, 3])
    c.invert_y()
    assert c.x == 1 and c.y == -2 and c.z == 3


def test_invert_z():
    c = Coordinate3D([1, 2, 3])
    c.invert_z()
    assert c.x == 1 and c.y == 2 and c.z == -3


def test_invert():
    c = Coordinate3D([1, 2, 3])
    c.invert()
    assert c.x == -1 and c.y == -2 and c.z == -3


def test_add():
    c = Coordinate3D([1, 2, 3])
    c.add(Coordinate3D([2, 3, 4]))
    assert c.x == 3 and c.y == 5 and c.z == 7


def test_subtract():
    c = Coordinate3D([1, 2, 3])
    c.subtract(Coordinate3D([2, 3, 4]))
    assert c.x == -1 and c.y == -1 and c.z == -1


def test_rotate():
    c = Coordinate3D([1, 2, 3])
    c.rotate(90 * units.Unit('degree'))
    assert np.isclose(c.x, -2)
    assert np.isclose(c.y, 1)
    assert c.z == 3


def test_rotate_offsets():
    c0 = Coordinate3D([1, 1, 1])
    c = c0.copy()
    angle = 90 * units.Unit('degree')
    c.rotate_offsets(c, angle, axis=0)
    assert np.isclose(c.x, 1) and np.isclose(c.y, -1) and c.z == 1
    c.rotate_offsets(c, angle, axis='y')
    assert np.isclose(c.x, 1) and np.isclose(c.y, -1) and np.isclose(c.z, -1)
    c.rotate_offsets(c, angle, axis=2)
    assert np.isclose(c.x, 1) and np.isclose(c.y, 1) and np.isclose(c.z, -1)
    c = Coordinate3D()
    c.rotate_offsets(c, angle)
    assert c.coordinates is None
    c = c0.copy()
    c.rotate_offsets(c, angle, axis=4)
    assert c == c0

    c = Coordinate3D(np.arange(6).reshape((3, 2)))
    c.rotate_offsets(c, angle)
    assert np.allclose(c.x, [-2, -3])
    assert np.allclose(c.y, [0, 1])
    assert np.allclose(c.z, [4, 5])

    x = np.arange(6).reshape((3, 2))
    c.rotate_offsets(x, angle)
    assert isinstance(c.coordinates, np.ndarray)
    assert np.allclose(c.coordinates, x)


def test_theta():
    c = Coordinate3D([1, 1, 1])
    assert np.isclose(c.theta(), 35.26439 * units.Unit('degree'), atol=1e-4)
    assert np.isclose(c.theta(center=Coordinate3D([0, 1, 0])),
                      45 * units.Unit('degree'), atol=1e-4)


def test_phi():
    c = Coordinate3D([1, 1, 1])
    assert np.isclose(c.phi(), 45 * units.Unit('degree'))
    assert np.isclose(c.phi(center=Coordinate3D([0, 1, 0])), 0)


def test_parse_header():
    h = fits.Header()
    c = Coordinate3D()
    h['X1'] = 1
    h['X2'] = 2
    h['X3'] = 3
    c.parse_header(h, 'X', alt=None)
    assert c.x == 1 and c.y == 2 and c.z == 3
    del h['X3']
    c.parse_header(h, 'X', default=np.arange(4, 7))
    assert c.x == 1 and c.y == 2 and c.z == 6
    c.parse_header(h, 'X', default=Coordinate3D([5, 6, 7]))
    assert c.x == 1 and c.y == 2 and c.z == 7


def test_edit_header():
    h = fits.Header()
    c = Coordinate3D(np.arange(6).reshape((3, 2)))
    c.edit_header(h, 'X')
    assert len(h) == 0
    c = Coordinate3D([1, 2, 3])
    c.edit_header(h, 'X')
    assert h['X1'] == 1 and h['X2'] == 2 and h['X3'] == 3
    assert h.comments['X1'] == 'The reference x coordinate.'
    assert h.comments['X2'] == 'The reference y coordinate.'
    assert h.comments['X3'] == 'The reference z coordinate.'

    c.change_unit('arcsec')
    c.edit_header(h, 'Y')
    assert h['Y1'] == 1 and h['Y2'] == 2 and h['Y3'] == 3
    assert h.comments['Y1'] == 'The reference x coordinate (arcsec).'
    assert h.comments['Y2'] == 'The reference y coordinate (arcsec).'
    assert h.comments['Y3'] == 'The reference z coordinate (arcsec).'

