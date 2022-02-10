# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.ecliptic_coordinates import \
    EclipticCoordinates


def test_init():
    e = EclipticCoordinates()
    assert e.epoch.year == 2000
    assert e.unit == 'degree'
    assert e.coordinates is None


def test_setup_coordinate_system():
    e = EclipticCoordinates()
    e.setup_coordinate_system()
    assert e.default_coordinate_system.name == 'Ecliptic Coordinates'
    assert e.longitude_axis.label == 'Ecliptic Longitude'
    assert e.longitude_axis.short_label == 'ELON'
    assert e.longitude_axis.reverse
    assert e.latitude_axis.label == 'Ecliptic Latitude'
    assert e.latitude_axis.short_label == 'ELAT'
    assert not e.latitude_axis.reverse
    assert e.x_offset_axis.label == 'Ecliptic Longitude Offset'
    assert e.x_offset_axis.short_label == 'dELON'
    assert e.x_offset_axis.reverse
    assert e.y_offset_axis.label == 'Ecliptic Latitude Offset'
    assert e.y_offset_axis.short_label == 'dELAT'
    assert not e.y_offset_axis.reverse


def test_fits_longitude_stem():
    e = EclipticCoordinates()
    assert e.fits_longitude_stem == 'ELON'


def test_fits_latitude_stem():
    e = EclipticCoordinates()
    assert e.fits_latitude_stem == 'ELAT'


def test_two_letter_code():
    e = EclipticCoordinates()
    assert e.two_letter_code == 'EC'


def test_equatorial_pole():
    pole = EclipticCoordinates().equatorial_pole
    assert pole.ra == 270 * units.Unit('degree')
    assert np.isclose(pole.dec.to('degree').value, 66.55833333333)


def test_get_equatorial_pole():
    assert (EclipticCoordinates().get_equatorial_pole()
            == EclipticCoordinates.EQUATORIAL_POLE)


def test_get_zero_longitude():
    assert EclipticCoordinates().get_zero_longitude() == (
        90 * units.Unit('degree'))


def test_precess_to_epoch():
    e = EclipticCoordinates([30, 30])
    epoch = e.epoch.get_epoch('J3000')
    e.precess_to_epoch(epoch)
    assert np.allclose(e.coordinates.value, [-29.99999821, 29.99999816])


def test_edit_header():
    h = fits.Header()
    t = EclipticCoordinates([2, 3])
    t.edit_header(h, 'FOO')
    assert h['FOO1'] == 2
    assert h['FOO2'] == 3
    assert h['WCSNAME'] == 'Ecliptic Coordinates'
    h = fits.Header()
    t = EclipticCoordinates(np.arange(10).reshape(2, 5))
    t.edit_header(h, 'FOO')
    assert len(h) == 0
