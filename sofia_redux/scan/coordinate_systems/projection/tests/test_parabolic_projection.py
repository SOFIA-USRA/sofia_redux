# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
PP = get_projection_class('parabolic')


def test_init():
    p = PP()
    assert p.reference.size == 0


def test_get_fits_id():
    assert PP.get_fits_id() == 'PAR'


def test_get_full_name():
    assert PP.get_full_name() == 'Parabolic Projection'


def test_get_phi_theta():
    p = PP()
    pt = p.get_phi_theta(Coordinate2D([1, 2], unit='degree'))
    assert np.allclose(pt.coordinates.value, [1.00049407, 1.90989862])
    pt1 = pt.copy()
    pt2 = p.get_phi_theta(Coordinate2D([1, 2], unit='degree'), phi_theta=pt1)
    assert pt1 is pt2 and pt2 == pt


def test_get_offsets():
    p = PP()
    phi = 1 * units.Unit('degree')
    theta = 2 * units.Unit('degree')
    o = p.get_offsets(theta, phi)
    assert np.allclose(o.coordinates.value, [0.99945848, 2.09434784])
    o1 = o.copy()
    o2 = p.get_offsets(theta, phi, offsets=o1)
    assert o1 is o2 and o2 == o


def test_consistency():
    p = PP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
