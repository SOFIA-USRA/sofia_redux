# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D


@pytest.fixture
def simple():
    return Coordinate1D(np.arange(5))


def test_class():
    assert Coordinate1D.default_dimensions == 1


def test_empty_copy(simple):
    c2 = simple.empty_copy()
    assert isinstance(c2, Coordinate1D)
    assert c2.size == 0


def test_copy(simple):
    c = simple
    c2 = c.copy()
    assert c == c2
    assert c is not c2


def test_ndim(simple):
    assert simple.ndim == 1


def test_size(simple):
    assert simple.size == 5
    assert Coordinate1D().size == 0
    assert Coordinate1D(1).size == 1


def test_shape(simple):
    assert Coordinate1D().shape == ()
    assert simple.shape == (5,)


def test_x(simple):
    assert np.allclose(simple.x, np.arange(5))
    c = Coordinate1D()
    assert c.x is None
    c.x = np.arange(5)
    assert c == simple


def test_max(simple):
    assert simple.max == Coordinate1D(4)


def test_min(simple):
    assert simple.min == Coordinate1D(0)


def test_span(simple):
    assert simple.span == Coordinate1D(4)


def test_length():
    c = Coordinate1D(-2)
    assert c.length == 2


def test_singular():
    assert Coordinate1D().singular
    assert Coordinate1D(1).singular
    assert Coordinate1D(np.asarray(1)).singular
    assert not Coordinate1D(np.arange(2)).singular


def test_str(simple):
    assert str(Coordinate1D()) == 'Empty coordinates'
    assert str(Coordinate1D(1)) == 'x=1.0'
    assert str(simple) == 'x=0.0->4.0'


def test_repr():
    s = repr(Coordinate1D())
    assert s.startswith('Empty coordinates <')
    assert 'Coordinate1D object' in s


def test_getitem(simple):
    assert simple[1] == Coordinate1D(1)
    assert simple[np.asarray(1)] == Coordinate1D(1)
    assert simple[2:4] == Coordinate1D([2, 3])
    with pytest.raises(KeyError) as err:
        _ = Coordinate1D(1)[0]
    assert 'singular coordinates' in str(err.value)
    c = Coordinate1D()
    assert c[0] == c


def test_set_shape():
    c = Coordinate1D()
    c.set_shape((3, 4))
    assert c.coordinates.shape == (3, 4)


def test_set_singular():
    c = Coordinate1D(np.arange(2), unit='arcsec')
    c.set_singular()
    assert c == Coordinate1D(0, unit='arcsec')


def test_copy_coordinates(simple):
    c = Coordinate1D()
    c.copy_coordinates(simple)
    assert c == simple
    c.copy_coordinates(Coordinate1D())
    assert c.size == 0


def test_set_x():
    c = Coordinate1D(np.arange(3))
    c.set_x(None)
    x = np.arange(4, dtype=float)
    c.set_x(x, copy=False)
    assert c.x is x
    c.set_x(x, copy=True)
    assert np.allclose(c.x, x) and c.x is not x
    x = np.arange(4)
    c.set_x(x, copy=False)
    assert c.x is not x
    c.set_x(1)
    assert c.x == 1


def test_set():
    c = Coordinate1D(np.arange(3))
    x = np.random.rand(3, 4)
    c.set(x, copy=False)
    assert c.x is x


def test_broadcast_to():
    c = Coordinate1D()
    c.broadcast_to((3, 4))
    assert c.shape == ()
    c = Coordinate1D(np.arange(2))
    c.broadcast_to((3, 4))
    assert c.shape == (2,)
    c = Coordinate1D(1)
    c.broadcast_to((3, 4))
    assert np.allclose(c.x, np.ones((3, 4)))
    c = Coordinate1D(1)
    c.broadcast_to(np.empty((2, 3)))
    assert np.allclose(c.x, np.ones((2, 3)))
    c = Coordinate1D(1)
    c.broadcast_to('a')
    assert c.coordinates == 1
    c.broadcast_to(())
    assert c.coordinates == 1


def test_add_x():
    c = Coordinate1D(1)
    x = np.random.rand(3, 4)
    c.add_x(x)
    assert np.allclose(c.x, x + 1)


def test_subtract_x():
    c = Coordinate1D(1)
    x = np.random.rand(3, 4)
    c.subtract_x(x)
    assert np.allclose(c.x, 1 - x)


def test_scale():
    c = Coordinate1D(2)
    c.scale(2)
    assert c.x == 4
    c.scale(np.asarray([1, 2, 3]))
    assert np.allclose(c.x, [4, 8, 12])
    c.scale(Coordinate1D(2))
    assert np.allclose(c.x, [8, 16, 24])


def test_scale_x():
    c = Coordinate1D(1)
    c.scale_x(2 * units.Unit('arcsec'))
    assert c.unit == 'arcsec' and c.x == 2 * units.Unit('arcsec')


def test_invert_x():
    c = Coordinate1D(2)
    c.invert_x()
    assert c.x == -2


def test_invert():
    c = Coordinate1D(2)
    c.invert()
    assert c.x == -2


def test_add():
    c = Coordinate1D(2)
    c.add(Coordinate1D(1))
    assert c.x == 3
    c.add(1)
    assert c.x == 4


def test_subtract():
    c = Coordinate1D(3)
    c.subtract(Coordinate1D(1))
    assert c.x == 2
    c.subtract(1)
    assert c.x == 1


def test_parse_header():
    c = Coordinate1D()
    h = fits.Header()
    c.parse_header(h, 'X')
    assert c.x == 0
    c = Coordinate1D()
    c.parse_header(h, 'X', default=1 * units.Unit('arcsec'))
    assert c.x == 1 * units.Unit('arcsec')
    c = Coordinate1D()
    c.parse_header(h, 'X', default=1)
    assert c.x == 1
    h['X1'] = 5
    c.parse_header(h, 'X', default=1, alt=None)
    assert c.x == 5
    del h['X1']
    c = Coordinate1D()
    c.parse_header(h, 'X', default=Coordinate1D(2))
    assert c.x == 2


def test_edit_header(simple):
    h = fits.Header()
    simple.edit_header(h, 'X')
    assert len(h) == 0
    c = Coordinate1D(1)
    c.edit_header(h, 'X')
    assert h['X1'] == 1
    assert h.comments['X1'] == 'The reference x coordinate.'
    c = Coordinate1D(1, unit='arcsec')
    c.edit_header(h, 'X')
    assert h['X1'] == 1
    assert h.comments['X1'] == 'The reference x coordinate (arcsec).'


def test_mean(simple):
    assert simple.mean().x == 2
