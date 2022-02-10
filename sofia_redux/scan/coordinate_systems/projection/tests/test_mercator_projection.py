# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates \
    import SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.mercator_projection \
    import MercatorProjection


def test_init():
    p = MercatorProjection()
    assert p.reference.size == 0


def test_get_fits_id():
    assert MercatorProjection.get_fits_id() == 'MER'


def test_get_full_name():
    assert MercatorProjection.get_full_name() == 'Mercator'


def test_get_phi_theta():
    p = MercatorProjection()
    o = Coordinate2D([1, 30], unit='degree')
    pt = p.get_phi_theta(o)
    assert np.allclose(pt.coordinates.value, [1, 28.71628445])

    o = Coordinate2D([1, 2])  # Assume radians
    pt = p.get_phi_theta(o)
    assert np.allclose(pt.coordinates.value, [57.29577951, 74.58537319])

    o = Coordinate2D([1, 2], unit=units.dimensionless_unscaled)
    pt = p.get_phi_theta(o)
    assert np.allclose(pt.coordinates.value, [57.29577951, 74.58537319])

    pt1 = pt.copy()
    pt2 = p.get_phi_theta(o, phi_theta=pt1)
    assert pt2 is pt1 and pt2 == pt


def test_get_offsets():
    p = MercatorProjection()
    o = p.get_offsets(1, 1)
    assert np.allclose(o.coordinates.value, [57.29577951, 70.25557897])
    ud = units.dimensionless_unscaled
    o = p.get_offsets(1 * ud, 1 * ud)
    assert np.allclose(o.coordinates.value, [57.29577951, 70.25557897])
    o = p.get_offsets(1 * units.Unit('degree'), 1 * units.Unit('degree'))
    assert np.allclose(o.coordinates.value, [1, 1.00005077])
    o1 = o.copy()
    o2 = p.get_offsets(1 * units.Unit('degree'), 1 * units.Unit('degree'),
                       offsets=o1)
    assert o1 is o2 and o1 == o


def test_consistency():
    p = MercatorProjection()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
