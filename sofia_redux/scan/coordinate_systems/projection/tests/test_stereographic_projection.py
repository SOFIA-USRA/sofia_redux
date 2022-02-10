# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
STG = get_projection_class('stereographic')


def test_init():
    p = STG()
    assert p.native_reference.lon == 0
    assert p.native_reference.lat == 90 * units.Unit('degree')


def test_get_fits_id():
    assert STG().get_fits_id() == 'STG'


def test_get_full_name():
    assert STG().get_full_name() == 'Stereographic'


def test_r():
    assert np.isclose(STG.r(90 * units.Unit('degree')), 0)
    assert np.isclose(STG.r(0), 2 * units.Unit('radian'))


def test_theta_of_r():
    deg = units.Unit('degree')
    ud = units.dimensionless_unscaled
    assert np.isclose(STG.theta_of_r(0), 90 * deg)
    assert np.isclose(STG.theta_of_r(30 * deg), 60.65851389 * deg)
    assert np.isclose(STG.theta_of_r(np.pi / 4 * ud), 47.120219 * deg)


def test_consistency():
    p = STG()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
