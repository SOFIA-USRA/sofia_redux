# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.telescope_coordinates import \
    TelescopeCoordinates


def test_init():
    t = TelescopeCoordinates()
    assert t.coordinates is None and t.unit == 'degree'


def test_copy():
    t = TelescopeCoordinates([1, 2])
    t2 = t.copy()
    assert t == t2 and t is not t2


def test_setup_coordinate_system():
    t = TelescopeCoordinates()
    t.setup_coordinate_system()
    assert t.default_coordinate_system.name == 'Telescope Coordinates'
    assert t.longitude_axis.label == 'Telescope Cross-elevation'
    assert t.longitude_axis.short_label == 'XEL'
    assert not t.longitude_axis.reverse
    assert t.latitude_axis.label == 'Telescope Elevation'
    assert t.latitude_axis.short_label == 'EL'
    assert not t.latitude_axis.reverse
    assert t.x_offset_axis.label == 'Telescope Cross-elevation Offset'
    assert t.x_offset_axis.short_label == 'dXEL'
    assert not t.x_offset_axis.reverse
    assert t.y_offset_axis.label == 'Telescope Elevation Offset'
    assert t.y_offset_axis.short_label == 'dEL'
    assert not t.y_offset_axis.reverse


def test_fits_longitude_stem():
    assert TelescopeCoordinates().fits_longitude_stem == 'TLON'


def test_fits_latitude_stem():
    assert TelescopeCoordinates().fits_latitude_stem == 'TLAT'


def test_two_letter_code():
    assert TelescopeCoordinates().two_letter_code == 'TE'


def test_xel():
    t = TelescopeCoordinates()
    assert t.xel is None
    t.xel = 1 * units.Unit('degree')
    assert t.xel == 1 * units.Unit('degree')


def test_cross_elevation():
    t = TelescopeCoordinates()
    assert t.cross_elevation is None
    t.cross_elevation = 1 * units.Unit('degree')
    assert t.cross_elevation == 1 * units.Unit('degree')


def test_el():
    t = TelescopeCoordinates()
    assert t.el is None
    t.el = 2 * units.Unit('degree')
    assert t.el == 2 * units.Unit('degree')


def test_elevation():
    t = TelescopeCoordinates()
    assert t.elevation is None
    t.elevation = 2 * units.Unit('degree')
    assert t.elevation == 2 * units.Unit('degree')


def test_getitem():
    t = TelescopeCoordinates(np.arange(10).reshape(2, 5))
    assert np.allclose(t[1].coordinates.value, [1, 6])


def test_to_equatorial_offset():
    t = TelescopeCoordinates()
    c = t.get_class_for('2d')([1, 1], unit='degree')
    vpa = 45 * units.Unit('degree')
    c1 = t.to_equatorial_offset(c, vpa, in_place=True)
    assert np.allclose(c1.coordinates.value, [0, np.sqrt(2)])
    assert c1 is c
    c2 = t.to_equatorial_offset(c1, vpa, in_place=False)
    assert np.allclose(c2.coordinates.value, [1, 1])
    assert c2 is not c1


def test_edit_header():
    h = fits.Header()
    t = TelescopeCoordinates([2, 3])
    t.edit_header(h, 'FOO')
    assert h['FOO1'] == 2
    assert h['FOO2'] == 3
    assert h['WCSNAME'] == 'Telescope Coordinates'
    h = fits.Header()
    t = TelescopeCoordinates(np.arange(10).reshape(2, 5))
    t.edit_header(h, 'FOO')
    assert len(h) == 0
