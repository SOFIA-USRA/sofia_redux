# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
ZEA = get_projection_class('zenithal_equal_area')


def test_init():
    p = ZEA()
    assert p.native_reference.lon == 0
    assert p.native_reference.lat == 90 * units.Unit('degree')


def test_get_fits_id():
    assert ZEA().get_fits_id() == 'ZEA'


def test_get_full_name():
    assert ZEA().get_full_name() == 'Zenithal Equal-Area'


def test_r():
    assert np.isclose(ZEA.r(90 * units.Unit('degree')), 0)
    assert np.isclose(ZEA.r(0), 81.02846845 * units.Unit('degree'))


def test_theta_of_r():
    deg = units.Unit('degree')
    assert np.isclose(ZEA.theta_of_r(0), 90 * deg)
    assert np.isclose(ZEA.theta_of_r(30 * deg), 59.64628345 * deg)


def test_consistency():
    p = ZEA()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
