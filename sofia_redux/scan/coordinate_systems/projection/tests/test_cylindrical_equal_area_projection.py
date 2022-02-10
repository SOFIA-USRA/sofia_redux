# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
CP = get_projection_class('cylindrical_equal_area')


def test_init():
    p = CP()
    assert p.stretch == 1


def test_get_phi_theta():
    p = CP()
    o = Coordinate2D([2, 2], unit='degree')
    pt = p.get_phi_theta(o)
    assert np.allclose(pt.coordinates.value, [2, 2.00040638])
    p.stretch = 2.0
    pt = p.get_phi_theta(o)
    assert np.allclose(pt.coordinates.value, [2, 4.0032564])
    pt1 = pt.copy()
    pt2 = p.get_phi_theta(o, phi_theta=pt1)
    assert pt2 == pt and pt2 is pt1
    p.stretch = 1.0
    o = Coordinate2D([1, 0.5])  # No unit, assume radians
    pt = p.get_phi_theta(o)
    assert np.allclose(pt.coordinates.value, [57.29577951, 30])


def test_get_offsets():
    p = CP()
    deg = units.Unit('degree')
    o = p.get_offsets(1 * deg, 1 * deg)
    assert np.allclose(o.coordinates.value, [1, 0.99994923])
    o1 = o.copy()
    o2 = p.get_offsets(1 * deg, 1 * deg, offsets=o1)
    assert o1 is o2 and o2 == o


def test_get_fits_id():
    assert CP.get_fits_id() == 'CEA'


def test_get_full_name():
    assert CP.get_full_name() == 'Cylindrical Equal Area'


def test_parse_header():
    p = CP()
    h = fits.Header()
    h['PV2_1'] = 2.0
    p.parse_header(h)
    assert p.stretch == 2


def test_edit_header():
    h = fits.Header()
    CP().edit_header(h)
    assert h['CTYPE1'] == 'LON--CEA'
    assert h['CTYPE2'] == 'LAT--CEA'
    assert h['PV2_1'] == 1.0


def test_consistency():
    p = CP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
