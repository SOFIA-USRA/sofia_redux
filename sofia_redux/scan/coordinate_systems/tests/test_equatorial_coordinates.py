# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates, EQUATORIAL_POLE)
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D


def test_equatorial_pole_class():
    assert EQUATORIAL_POLE.ra == 0 * units.Unit('degree')
    assert EQUATORIAL_POLE.dec == 90 * units.Unit('degree')


def test_init():
    e = EquatorialCoordinates()
    assert e.unit == 'degree'
    assert e.epoch.equinox.jyear == 2000


def test_copy():
    e = EquatorialCoordinates([1, 2])
    e2 = e.copy()
    assert e2 == e and e is not e2


def test_setup_coordinate_system():
    e = EquatorialCoordinates()
    e.setup_coordinate_system()
    assert e.default_coordinate_system.name == 'Equatorial Coordinates'
    assert e.longitude_axis.label == 'Right Ascension'
    assert e.longitude_axis.short_label == 'RA'
    assert e.longitude_axis.reverse
    assert e.latitude_axis.label == 'Declination'
    assert e.latitude_axis.short_label == 'DEC'
    assert not e.latitude_axis.reverse
    assert e.x_offset_axis.label == 'Right Ascension Offset'
    assert e.x_offset_axis.short_label == 'dRA'
    assert e.x_offset_axis.reverse
    assert e.y_offset_axis.label == 'Declination Offset'
    assert e.y_offset_axis.short_label == 'dDEC'
    assert not e.y_offset_axis.reverse


def test_fits_longitude_stem():
    e = EquatorialCoordinates()
    assert e.fits_longitude_stem == 'RA--'


def test_fits_latitude_step():
    e = EquatorialCoordinates()
    assert e.fits_latitude_stem == 'DEC-'


def test_two_letter_code():
    e = EquatorialCoordinates()
    assert e.two_letter_code == 'EQ'


def test_ra():
    e = EquatorialCoordinates()
    assert e.ra is None
    e = EquatorialCoordinates([[-1, 0, 1], [4, 5, 6]])
    assert np.allclose(e.ra.value, [359, 0, 1])
    e.ra = np.full(3, 1) * units.Unit('degree')
    assert np.allclose(e.ra.value, 1)


def test_dec():
    e = EquatorialCoordinates()
    assert e.dec is None
    e = EquatorialCoordinates([[0, 1, 2], [-1, 0, 1]])
    assert np.allclose(e.dec.value, [-1, 0, 1])
    e.dec = np.full(3, 1) * units.Unit('degree')
    assert np.allclose(e.dec.value, 1)


def test_equatorial_pole():
    pole = EquatorialCoordinates().equatorial_pole
    assert pole.ra.value == 0 and pole.dec.value == 90


def test_zero_longitude():
    assert EquatorialCoordinates().zero_longitude == 0 * units.Unit('degree')


def test_str():
    e = EquatorialCoordinates()
    assert str(e) == 'Empty coordinates (J2000.0)'
    e = EquatorialCoordinates(np.arange(4).reshape((2, 2)))
    assert str(e) == 'RA=0h00m00s->0h04m00s DEC=2d00m00s->3d00m00s (J2000.0)'
    e.nan()
    assert str(e) == 'RA=NaN->NaN DEC=NaN->NaN (J2000.0)'
    e = EquatorialCoordinates([1, 2])
    assert str(e) == 'RA=0h04m00s DEC=2d00m00s (J2000.0)'
    e.nan()
    assert str(e) == 'RA=NaN DEC=NaN (J2000.0)'


def test_getitem():
    e = EquatorialCoordinates(np.arange(10).reshape(2, 5))
    e1 = e[1]
    assert np.allclose(e1.coordinates.value, [-1, 6])


def test_get_equatorial_pole():
    e = EquatorialCoordinates()
    assert e.get_equatorial_pole() == EQUATORIAL_POLE


def test_get_zero_longitude():
    e = EquatorialCoordinates()
    assert e.get_zero_longitude() == 0 * units.Unit('degree')


def test_to_horizontal_offset():
    e = EquatorialCoordinates()
    c = Coordinate2D([np.arange(5) - 2, np.arange(5) - 2], unit='degree')
    position_angle = 30 * units.Unit('degree')
    c0 = c.copy()
    o = e.to_horizontal_offset(c0, position_angle, in_place=True)
    assert o is c0 and o != c
    assert np.allclose(o.x.value,
                       [0.73205081, 0.3660254, 0., -0.3660254, -0.73205081])
    assert np.allclose(o.y.value,
                       [-2.73205081, -1.3660254, 0., 1.3660254, 2.73205081])

    o2 = e.to_horizontal_offset(c0, position_angle, in_place=False)
    assert o2 is not c0 and o2 == c


def test_set_ra():
    e = EquatorialCoordinates()
    e.set_ra(1 * units.Unit('degree'))
    assert e.ra == 1 * units.Unit('degree')
    assert np.allclose(e.coordinates.value, [-1, 0])


def test_set_dec():
    e = EquatorialCoordinates()
    e.set_dec(1 * units.Unit('degree'))
    assert e.dec == 1 * units.Unit('degree')
    assert np.allclose(e.coordinates.value, [0, 1])


def test_get_parallactic_angle():
    e = EquatorialCoordinates([np.arange(5) - 2, np.full(5, 45)])
    lst = 0 * units.Unit('hourangle')
    site = GeodeticCoordinates([45, 45])
    pa = e.get_parallactic_angle(site, lst).to('degree')
    assert np.allclose(
        pa.value, [89.29285732, 89.64644212, 0., -89.64644212, -89.29285732])
    e = EquatorialCoordinates()
    assert e.get_parallactic_angle(site, lst) is None


def test_get_equatorial_position_angle():
    e = EquatorialCoordinates([1, 1])
    assert e.get_equatorial_position_angle() == 0 * units.Unit('radian')
    e.ra = np.arange(10)
    assert np.allclose(e.get_equatorial_position_angle(),
                       np.zeros(10) * units.Unit('radian'))


def test_to_equatorial():
    e = EquatorialCoordinates()
    e2 = e.to_equatorial()
    assert e == e2 and e is not e2

    c = e.get_class_for('2d')()
    e.ra = np.arange(5) * units.Unit('degree')
    e.dec = (np.arange(5) + 10) * units.Unit('degree')

    e2 = e.to_equatorial(c)
    assert np.allclose(e2.coordinates, c.coordinates)
    assert not isinstance(c, EquatorialCoordinates)


def test_from_equatorial():
    e = EquatorialCoordinates()
    c = e.get_class_for('2d')(np.arange(10).reshape(2, 5), unit='degree')
    e.from_equatorial(c)
    assert np.allclose(e.coordinates, c.coordinates)


def test_to_horizontal():
    e = EquatorialCoordinates(np.arange(10).reshape(2, 5))
    site = e.get_instance('geodetic')
    site.lon = 0 * units.Unit('degree')
    site.lat = 45 * units.Unit('degree')
    lst = 0 * units.Unit('hourangle')
    h = e.to_horizontal(site, lst)
    assert np.allclose(
        h.x.value,
        [-180., 178.41990101, 176.77749242, 175.07016896, 173.29531247])
    assert np.allclose(
        h.y.value,
        [50, 50.99024969, 51.96022922, 52.90873446, 53.8344934])

    # This happens all the time...
    h2 = e.to_horizontal(site, lst.value * units.Unit('hour'))
    assert h2 == h


def test_precess_to_epoch():
    e = EquatorialCoordinates([1, 1])
    epoch = e.epoch.get_epoch('B1950')
    e.precess_to_epoch(epoch)
    assert np.isclose(e.ra.value, 0.35958501900843515)
    assert np.isclose(e.dec.value, 0.7216802450171778)
    e0 = e.copy()
    e.precess_to_epoch(epoch)
    assert e == e0


def test_convert():
    e = EquatorialCoordinates([1, 1])
    c = Coordinate2D()
    e.convert(e, c)
    assert np.allclose(c.coordinates.value, [-1, 1])
    e = EquatorialCoordinates()
    e.convert(c, e)
    assert np.allclose(e.coordinates.value, [-1, 1])

    c2 = Coordinate2D()
    e.convert(c, c2)
    assert np.allclose(c2.coordinates.value, [-1, 1])


def test_edit_header():
    h = fits.Header()
    t = EquatorialCoordinates([2, 3])
    t.edit_header(h, 'FOO')
    assert h['FOO1'] == 2
    assert h['FOO2'] == 3
    assert h['WCSNAME'] == 'Equatorial Coordinates'
    h = fits.Header()
    t = EquatorialCoordinates(np.arange(10).reshape(2, 5))
    t.edit_header(h, 'FOO')
    assert len(h) == 0
