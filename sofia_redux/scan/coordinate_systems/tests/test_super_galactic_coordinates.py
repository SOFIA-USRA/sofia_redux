# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.super_galactic_coordinates import \
    SuperGalacticCoordinates


def test_init():
    g = SuperGalacticCoordinates()
    assert g.coordinates is None and g.unit == 'degree'


def test_copy():
    g = SuperGalacticCoordinates([1, 2])
    g2 = g.copy()
    assert g2 == g and g2 is not g


def test_setup_coordinate_system():
    g = SuperGalacticCoordinates()
    g.setup_coordinate_system()
    assert g.default_coordinate_system.name == 'Supergalactic Coordinates'
    assert g.longitude_axis.label == 'Supergalactic Longitude'
    assert g.longitude_axis.short_label == 'SGL'
    assert g.longitude_axis.reverse
    assert g.latitude_axis.label == 'Supergalactic Latitude'
    assert g.latitude_axis.short_label == 'SGB'
    assert not g.latitude_axis.reverse
    assert g.x_offset_axis.label == 'Galactic Longitude Offset'
    assert g.x_offset_axis.short_label == 'dSGL'
    assert g.x_offset_axis.reverse
    assert g.y_offset_axis.label == 'Galactic Latitude Offset'
    assert g.y_offset_axis.short_label == 'dSGB'
    assert not g.y_offset_axis.reverse


def test_fits_longitude_stem():
    assert SuperGalacticCoordinates().fits_longitude_stem == 'SLON'


def test_fits_latitude_stem():
    assert SuperGalacticCoordinates().fits_latitude_stem == 'SLAT'


def test_two_letter_code():
    assert SuperGalacticCoordinates().two_letter_code == 'SG'


def test_equatorial_pole():
    sg = SuperGalacticCoordinates()
    expected = sg.get_instance('equatorial')
    expected.ra = 18.91483401 * units.Unit('hourangle')
    expected.dec = 15.64835128 * units.Unit('degree')
    assert sg.equatorial_pole == expected


def test_get_equatorial_pole():
    sg = SuperGalacticCoordinates()
    assert sg.equatorial_pole == sg.get_equatorial_pole()


def test_get_zero_longitude():
    sg = SuperGalacticCoordinates()
    assert sg.get_zero_longitude() == -137.37 * units.Unit('degree')


def test_edit_header():
    h = fits.Header()
    t = SuperGalacticCoordinates([2, 3])
    t.edit_header(h, 'FOO')
    assert h['FOO1'] == 2
    assert h['FOO2'] == 3
    assert h['WCSNAME'] == 'Supergalactic Coordinates'
    h = fits.Header()
    t = SuperGalacticCoordinates(np.arange(10).reshape(2, 5))
    t.edit_header(h, 'FOO')
    assert len(h) == 0
