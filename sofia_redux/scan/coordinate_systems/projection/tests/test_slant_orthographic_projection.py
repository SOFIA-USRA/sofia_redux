# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
SOP = get_projection_class('slant_orthographic')


def test_init():
    p = SOP()
    assert p.native_reference.lon == 0
    assert p.native_reference.lat == 90 * units.Unit('degree')


def test_get_fits_id():
    assert SOP().get_fits_id() == 'SIN'


def test_get_full_name():
    assert SOP().get_full_name() == 'Slant Orthographic'


def test_r():
    assert np.isclose(SOP.r(90 * units.Unit('degree')), 0)
    assert np.isclose(SOP.r(0), 1 * units.Unit('radian'))


def test_theta_of_r():
    assert np.isclose(SOP.theta_of_r(0), 90 * units.Unit('degree'))
    assert np.isclose(SOP.theta_of_r(1 * units.Unit('radian')), 0)


def test_consistency():
    p = SOP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
