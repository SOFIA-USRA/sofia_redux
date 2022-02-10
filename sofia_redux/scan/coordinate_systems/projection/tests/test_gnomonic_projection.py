# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates \
    import SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.gnomonic_projection \
    import GnomonicProjection


def test_init():
    p = GnomonicProjection()
    assert p.reference.size == 0


def test_get_fits_id():
    assert GnomonicProjection.get_fits_id() == 'TAN'


def test_get_full_name():
    assert GnomonicProjection.get_full_name() == 'Gnomonic'


def test_r():
    assert GnomonicProjection.r(np.pi / 2) == 0
    assert np.isclose(GnomonicProjection.r(np.pi / 4),
                      1 * units.Unit('radian'))
    theta = np.full(5, np.pi / 2)
    theta[np.arange(3) * 2] -= np.pi / 4
    r = GnomonicProjection.r(theta)
    assert np.allclose(r.value, [1, 0, 1, 0, 1])


def test_theta_of_r():
    rad = units.Unit('radian')
    assert np.isclose(GnomonicProjection.theta_of_r(0), np.pi / 2 * rad)
    assert np.isclose(GnomonicProjection.theta_of_r(1), np.pi / 4 * rad)

    theta = GnomonicProjection.theta_of_r(np.array([0, 1]) * rad)
    assert np.allclose(theta.value, [np.pi / 2, np.pi / 4])


def test_consistency():
    p = GnomonicProjection()
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
