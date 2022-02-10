# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.plate_carree_projection \
    import PlateCarreeProjection


def test_init():
    p = PlateCarreeProjection()
    assert p.reference.size == 0


def test_get_fits_id():
    assert PlateCarreeProjection.get_fits_id() == 'CAR'


def test_get_full_name():
    assert PlateCarreeProjection.get_full_name() == 'Plate carree'


def test_get_phi_theta():
    p = PlateCarreeProjection()
    pt = p.get_phi_theta(Coordinate2D([1, 1], unit='degree'))
    assert np.allclose(pt.coordinates.value, 1)
    pt1 = pt.copy()
    pt2 = p.get_phi_theta(Coordinate2D([1, 1], unit='degree'), phi_theta=pt1)
    assert pt2 is pt1 and pt2 == pt


def test_get_offsets():
    p = PlateCarreeProjection()
    o = p.get_offsets(1 * units.Unit('degree'), 2 * units.Unit('degree'))
    assert np.allclose(o.coordinates.value, [2, 1])
    o1 = o.copy()
    o2 = p.get_offsets(1 * units.Unit('degree'), 2 * units.Unit('degree'),
                       offsets=o1)
    assert o2 is o1 and o2 == o


def test_consistency():
    p = PlateCarreeProjection()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
