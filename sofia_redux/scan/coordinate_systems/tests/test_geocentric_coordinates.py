# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.geocentric_coordinates import \
    GeocentricCoordinates


def test_copy():
    g = GeocentricCoordinates([1, 2])
    g2 = g.copy()
    assert g2 == g and g2 is not g


def test_get_default_system():
    g = GeocentricCoordinates()
    system, local_system = g.get_default_system()
    assert system.name == 'Geocentric Coordinates'
    assert local_system.name == 'Geocentric Offsets'


def test_two_letter_code():
    assert GeocentricCoordinates().two_letter_code == 'GC'


def test_getitem():
    g = GeocentricCoordinates(np.arange(10).reshape(2, 5))
    assert np.allclose(g[1].coordinates.value, [1, 6])


def test_edit_header():
    h = fits.Header()
    g = GeocentricCoordinates([1, 6])
    g.edit_header(h, 'FOO')
    assert h['FOO1'] == 1
    assert h['FOO2'] == 6
    assert h['WCSNAME'] == 'Geocentric Coordinates'
    g = GeocentricCoordinates(np.arange(10).reshape(2, 5))
    h = fits.Header()
    g.edit_header(h, 'FOO')
    assert len(h) == 0
