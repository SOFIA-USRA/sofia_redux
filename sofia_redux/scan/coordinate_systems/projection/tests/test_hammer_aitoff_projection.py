# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
HP = get_projection_class('hammer_aitoff')


def test_init():
    p = HP()
    assert p.reference.size == 0


def test_get_fits_id():
    assert HP.get_fits_id() == 'AIT'


def test_get_full_name():
    assert HP.get_full_name() == 'Hammer-Aitoff'


def test_z2():
    offset = Coordinate2D([0.5, 1])
    z2 = HP.z2(offset)
    assert z2.value == 0.734375

    offset = Coordinate2D([0.5, 1], unit=units.dimensionless_unscaled)
    z2 = HP.z2(offset)
    assert z2.value == 0.734375

    offset = Coordinate2D([0.5, 1], unit='radian')
    z2 = HP.z2(offset)
    assert z2.value == 0.734375


def test_gamma():
    deg = units.Unit("degree")
    assert np.isclose(HP.gamma(45 * deg, 45 * deg), 1.0998706095521211)


def test_get_phi_theta():
    pt = HP.get_phi_theta(Coordinate2D([1, 1], unit='degree'))
    assert np.allclose(pt.coordinates.value, [1.00011742, 1.00000317])
    pt = HP.get_phi_theta(
        Coordinate2D([1, 1], unit=units.dimensionless_unscaled))  # assume rads
    assert np.allclose(pt.coordinates.value, [[95.73917048, 56.01215642]])
    pt1 = pt.copy()
    pt2 = HP.get_phi_theta(Coordinate2D([1, 1]), phi_theta=pt1)
    assert pt2 is pt1 and pt2 == pt


def test_get_offsets():
    deg = units.Unit('degree')
    o = HP.get_offsets(1 * deg, 1 * deg)
    assert np.allclose(o.coordinates.value, [[0.99988259, 0.99999683]])
    o1 = o.copy()
    o2 = HP.get_offsets(1 * deg, 1 * deg, offsets=o1)
    assert o2 is o1 and o2 == o


def test_consistency():
    p = HP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
