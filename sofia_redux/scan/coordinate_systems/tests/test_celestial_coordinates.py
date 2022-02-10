# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import B1950


class Celestial(CelestialCoordinates):

    def get_equatorial_pole(self):
        return EquatorialCoordinates([0, 90], epoch='J2000')

    def get_zero_longitude(self):
        return 0 * units.Unit('degree')


class CelestialPole60(CelestialCoordinates):
    def get_equatorial_pole(self):
        return EquatorialCoordinates([0, 60], epoch='J2000')

    def get_zero_longitude(self):  # pragma: no cover
        return 0 * units.Unit('degree')


@pytest.fixture
def celestial():
    x, y = np.meshgrid([-2, -1, 0, 1, 2], [-60, -45, -30, 0, 30, 45, 60])
    return Celestial([x.ravel(), y.ravel()], unit='degree')


@pytest.fixture
def zero(celestial):
    return celestial.copy()[17]


def test_init(celestial):
    c = celestial.copy()
    c2 = Celestial(c)
    assert c2 == c


def test_copy(celestial):
    c = celestial.copy()
    assert isinstance(c, CelestialCoordinates)
    assert c == celestial
    c = Celestial([1, 1])
    assert np.allclose(c.coordinates.value, [1, 1])


def test_getitem(celestial):
    c = celestial[17]
    assert c.singular and c.is_null()


def test_get_pole(zero):
    reference = zero.copy()
    inclination = np.arange(-2, 3) * 30 * units.Unit('degree')
    rising_ra = np.arange(-2, 3) * units.Unit('degree')

    pole = CelestialCoordinates.get_pole(inclination, rising_ra,
                                         reference=reference)
    assert np.allclose(pole.x.value, [-92, -91, -90, -89, -88])
    assert np.allclose(pole.y.value, [150, 120, 90, 60, 30])
    # below are equatorial coordinates (negative x)
    pole = CelestialCoordinates.get_pole(inclination, rising_ra)
    assert np.allclose(pole.x.value, [92, 91, 90, 89, 88])
    assert np.allclose(pole.y.value, [150, 120, 90, 60, 30])


def test_get_zero_longitude_from(celestial, zero):
    lon = CelestialCoordinates.get_zero_longitude_from(celestial, zero)
    assert np.allclose(lon, celestial.x)


def test_get_equatorial_class():
    assert CelestialCoordinates.get_equatorial_class() == EquatorialCoordinates


def test_get_equatorial_position_angle():
    c = CelestialPole60([0, 61], unit='degree')
    angle = c.get_equatorial_position_angle().to('degree')
    assert np.isclose(angle.value, -180)


def test_get_equatorial(celestial):
    c = celestial.get_equatorial()
    assert np.allclose(c.y, celestial.y)
    assert np.allclose(c.x, 180 * units.Unit('degree') + celestial.x)


def test_to_equatorial(celestial):
    c = celestial.copy()
    e_no_epoch = c.copy()
    e_no_epoch.epoch = None
    e = c.to_equatorial(equatorial=e_no_epoch)
    assert np.allclose(e.x.value, c.x.value + 180)
    assert np.allclose(e.y, c.y)
    e2 = c.to_equatorial()
    assert np.allclose(e2.coordinates, e.coordinates)


def test_from_equatorial(celestial):
    c = celestial.copy()
    e = c.to_equatorial()
    c2 = Celestial()
    c2.from_equatorial(e)
    assert c2 == c
    c3 = Celestial()
    e.epoch = B1950
    c3.from_equatorial(e)
    assert c3 != c


def test_convert_from(celestial):
    c = celestial.copy()
    c2 = c.get_class_for('2d')([1, 1], unit='degree')
    c_conv = Celestial()
    c_conv.convert_from(c2)
    assert np.allclose(c_conv.coordinates.value, [1, 1])

    c_conv = Celestial()
    c_conv.convert_from(c)
    assert c_conv == c


def test_convert_to(celestial):
    c = celestial.copy()
    c2 = c.get_class_for('2d')([1, 1], unit='degree')
    c.convert_to(c2)
    assert np.allclose(c.coordinates, c2.coordinates)

    e = c.to_equatorial()
    c.convert_to(e)
    assert np.allclose(e.coordinates, c.coordinates)


def test_convert_from_celestial(celestial):
    c = celestial.copy()
    e = c.to_equatorial()
    c2 = c.copy()
    c2.convert_from_celestial(e)
    assert np.allclose(c2.coordinates, e.coordinates)


def test_convert_to_celestial(celestial):
    c = celestial.copy()
    e = c.to_equatorial()
    c.convert_to_celestial(e)
    assert np.allclose(e.coordinates, c.coordinates)


def test_convert(celestial):
    c = celestial.copy()
    c2 = c.get_class_for('2d')()
    c.convert(c, c2)
    assert np.allclose(c2.coordinates, c.coordinates)
