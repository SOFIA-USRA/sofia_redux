# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
CP = get_projection_class('cylindrical_perspective')


def test_init():
    p = CP()
    assert p.mu == 1 and p.la == 1


def test_get_fits_id():
    assert CP.get_fits_id() == 'CYP'


def test_get_full_name():
    assert CP.get_full_name() == 'Cylindrical Perspective'


def test_get_phi_theta():
    p = CP()
    o = Coordinate2D([1, 1], unit='degree')
    c = p.get_phi_theta(o)
    assert np.allclose(c.coordinates.value, [1, 0.99997462])
    c1 = c.copy()
    c2 = p.get_phi_theta(o, phi_theta=c1)
    assert c1 is c2 and c2 == c


def test_get_offsets():
    p = CP()
    deg = units.Unit('degree')
    o = p.get_offsets(1 * deg, 1 * deg)
    assert np.allclose(o.coordinates.value, [1, 1.00002539])
    o1 = o.copy()
    o2 = p.get_offsets(1 * deg, 1 * deg, offsets=o1)
    assert o2 is o1 and o2 == o


def test_parse_header():
    p = CP()
    h = fits.Header()
    h['PV2_1'] = 2.0
    h['PV2_2'] = 3.0
    p.parse_header(h)
    assert p.mu == 2 and p.la == 3


def test_edit_header():
    p = CP()
    h = fits.Header()
    p.mu = 2.0
    p.la = 3.0
    p.edit_header(h)
    assert h['CTYPE1'] == 'LON--CYP'
    assert h['CTYPE2'] == 'LAT--CYP'
    assert h['PV2_1'] == 2
    assert h['PV2_2'] == 3


def test_consistency():
    p = CP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
