# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.geocentric_coordinates import \
    GeocentricCoordinates


def test_init():
    gd = GeodeticCoordinates()
    assert gd.coordinates is None and gd.unit == 'degree'
    gc = GeocentricCoordinates([1, 2], unit='arcminute')
    gd = GeodeticCoordinates(gc)
    assert np.isclose(gd.x.value, 1 / 60)
    assert np.isclose(gd.y.value, 0.03413800694112625)


def test_copy():
    g = GeodeticCoordinates([1, 2])
    g2 = g.copy()
    assert g == g2 and g2 is not g


def test_get_default_system():
    g = GeodeticCoordinates()
    system, local_system = g.get_default_system()
    assert system.name == 'Geodetic Coordinates'
    assert local_system.name == 'Geodetic Offsets'


def test_two_letter_code():
    assert GeodeticCoordinates().two_letter_code == 'GD'


def test_getitem():
    g = GeodeticCoordinates(np.arange(10).reshape(2, 5))
    assert np.allclose(g[1].coordinates.value, [1, 6])


def test_edit_header():
    h = fits.Header()
    g = GeodeticCoordinates([3, 4])
    g.edit_header(h, 'FOO')
    assert h['FOO1'] == 3
    assert h['FOO2'] == 4
    assert h['WCSNAME'] == 'Geodetic Coordinates'
    h = fits.Header()
    g = GeodeticCoordinates([[1, 2], [3, 4]])
    g.edit_header(h, 'FOO')
    assert len(h) == 0


def test_from_geocentric():
    gc = GeocentricCoordinates(np.arange(10).reshape(2, 5))
    g = GeodeticCoordinates()
    g.from_geocentric(gc)
    assert np.allclose(g.x.value, [0, 1, 2, 3, 4])
    assert np.allclose(
        g.y.value, [5.12008921, 6.14378469, 7.167305, 8.19062147, 9.2137057])
