# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
# Simple class for testing
ZP = get_projection_class('zenithal_equidistant')


def test_init():
    p = ZP()
    assert p.native_reference.lon == 0
    assert p.native_reference.lat == 90 * units.Unit('degree')


def test_calculate_celestial_pole():
    p = ZP()
    p.reference = SphericalCoordinates([0, 60])
    p.calculate_celestial_pole()
    assert p.celestial_pole == p.reference


def test_get_phi_theta():
    p = ZP()
    pt = p.get_phi_theta(Coordinate2D([1, 2], unit='degree'))
    assert np.allclose(pt.coordinates.value, [153.43494882, 87.76393202])
    pt1 = pt.copy()
    pt2 = p.get_phi_theta(Coordinate2D([1, 2], unit='degree'), phi_theta=pt1)
    assert pt1 is pt2 and pt2 == pt


def test_get_offsets():
    p = ZP()
    o = p.get_offsets(1 * units.Unit('degree'), 2 * units.Unit('degree'))
    assert np.allclose(o.coordinates.value, [3.10605521, -88.9457836])
    o1 = o.copy()
    o2 = p.get_offsets(1 * units.Unit('degree'), 2 * units.Unit('degree'),
                       offsets=o1)
    assert o2 is o1 and o2 == o


def test_consistency():
    p = ZP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
