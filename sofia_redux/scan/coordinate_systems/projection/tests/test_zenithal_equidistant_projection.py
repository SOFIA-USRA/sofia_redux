# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
ARC = get_projection_class('zenithal_equidistant')


def test_init():
    p = ARC()
    assert p.native_reference.lon == 0
    assert p.native_reference.lat == 90 * units.Unit('degree')


def test_get_fits_id():
    assert ARC().get_fits_id() == 'ARC'


def test_get_full_name():
    assert ARC().get_full_name() == 'Zenithal Equidistant'


def test_r():
    assert np.isclose(ARC.r(90 * units.Unit('degree')), 0)
    assert np.isclose(ARC.r(0), 90 * units.Unit('degree'))


def test_theta_of_r():
    deg = units.Unit('degree')
    assert np.isclose(ARC.theta_of_r(0), 90 * deg)
    assert np.isclose(ARC.theta_of_r(30 * deg), 60 * deg)


def test_consistency():
    p = ARC()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
