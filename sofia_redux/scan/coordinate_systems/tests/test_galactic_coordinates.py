# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.galactic_coordinates import \
    GalacticCoordinates


def test_init():
    g = GalacticCoordinates()
    assert g.coordinates is None and g.unit == 'degree'


def test_setup_coordinate_system():
    g = GalacticCoordinates()
    g.setup_coordinate_system()
    assert g.default_coordinate_system.name == 'Galactic Coordinates'
    assert g.longitude_axis.label == 'Galactic Longitude'
    assert g.longitude_axis.short_label == 'GLON'
    assert g.longitude_axis.reverse
    assert g.latitude_axis.label == 'Galactic Latitude'
    assert g.latitude_axis.short_label == 'GLAT'
    assert not g.latitude_axis.reverse
    assert g.x_offset_axis.label == 'Galactic Longitude Offset'
    assert g.x_offset_axis.short_label == 'dGLON'
    assert g.x_offset_axis.reverse
    assert g.y_offset_axis.label == 'Galactic Latitude Offset'
    assert g.y_offset_axis.short_label == 'dGLAT'
    assert not g.y_offset_axis.reverse


def test_fits_longitude_stem():
    assert GalacticCoordinates().fits_longitude_stem == 'GLON'


def test_fits_latitude_stem():
    assert GalacticCoordinates().fits_latitude_stem == 'GLAT'


def test_two_letter_code():
    assert GalacticCoordinates().two_letter_code == 'GA'


def test_equatorial_pole():
    pole = GalacticCoordinates().equatorial_pole
    assert np.isclose(pole.ra.value, 192.85918559)
    assert np.isclose(pole.dec.value, 27.1283158766)


def test_get_equatorial_pole():
    pole = GalacticCoordinates().get_equatorial_pole()
    assert np.isclose(pole.ra.value, 192.85918559)
    assert np.isclose(pole.dec.value, 27.1283158766)


def test_get_zero_longitude():
    phi0 = GalacticCoordinates().get_zero_longitude()
    assert np.isclose(phi0, 123 * units.Unit('degree'))


def test_edit_header():
    h = fits.Header()
    t = GalacticCoordinates([2, 3])
    t.edit_header(h, 'FOO')
    assert h['FOO1'] == 2
    assert h['FOO2'] == 3
    assert h['WCSNAME'] == 'Galactic Coordinates'
    h = fits.Header()
    t = GalacticCoordinates(np.arange(10).reshape(2, 5))
    t.edit_header(h, 'FOO')
    assert len(h) == 0
