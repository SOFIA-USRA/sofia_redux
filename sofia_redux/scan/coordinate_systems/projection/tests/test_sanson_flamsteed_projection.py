# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
SFP = get_projection_class('sanson_flamsteed')


def test_init():
    p = SFP()
    assert p.reference.size == 0


def test_get_fits_id():
    assert SFP.get_fits_id() == 'SFL'


def test_get_full_name():
    assert SFP.get_full_name() == 'Sanson-Flamsteed'


def test_get_phi_theta():
    p = SFP()
    pt = p.get_phi_theta(Coordinate2D([30, 60], unit='degree'))
    assert np.allclose(pt.coordinates.value, 60)
    pt1 = pt.copy()
    pt2 = p.get_phi_theta(Coordinate2D([30, 60], unit='degree'), phi_theta=pt1)
    assert pt2 is pt1 and pt2 == pt


def test_get_offsets():
    p = SFP()
    o = p.get_offsets(np.pi / 3, np.pi / 3)
    assert np.allclose(o.coordinates.value, [30, 60])
    o1 = o.copy()
    o2 = p.get_offsets(np.pi / 3, np.pi / 3, offsets=o1)
    assert o1 is o2 and o2 == o


def test_consistency():
    p = SFP()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
