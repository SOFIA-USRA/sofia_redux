# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.bonnes_projection import \
    BonnesProjection


def test_init():
    p = BonnesProjection()
    assert p.theta1 == 0.0 * units.Unit('radian')
    assert p.y0 == 0.0 * units.Unit('radian')


def test_theta1():
    p = BonnesProjection()
    p.theta1 = 45 * units.Unit('degree')
    assert p.theta1 == 45 * units.Unit('degree')
    assert np.isclose(
        p.y0, (1 * units.Unit('radian')) + (45 * units.Unit('degree')))
    assert np.isclose(p.native_reference.latitude, p.theta1)


def test_y0():
    p = BonnesProjection()
    p.y0 = 45 * units.Unit('degree')
    assert p.y0 == 45 * units.Unit('degree')
    p.y0 = np.pi / 3
    assert p.y0 == 60 * units.Unit('degree')


def test_set_theta1():
    p = BonnesProjection()
    p.set_theta_1(45 * units.Unit('degree'))
    assert p.theta1 == 45 * units.Unit('degree')
    assert np.isclose(
        p.y0, (1 * units.Unit('radian')) + (45 * units.Unit('degree')))
    assert np.isclose(p.native_reference.latitude, p.theta1)
    y1 = p.y0
    p.set_theta_1(45 * units.Unit('degree'))
    assert p.y0 is y1


def test_get_phi_theta():
    p = BonnesProjection()
    p.theta1 = 45 * units.Unit('degree')
    offset = Coordinate2D([1, 1], unit='degree')
    pt = p.get_phi_theta(offset)
    assert np.allclose(pt.coordinates.value, [1.00016707, 0.99506408])
    p.theta1 = -45 * units.Unit('degree')
    pt = p.get_phi_theta(offset)
    assert np.allclose(pt.coordinates.value, [-323.57821363, 1.00484036])
    o = pt.copy()
    pt2 = p.get_phi_theta(offset, phi_theta=o)
    assert pt2 is o and pt2 == pt


def test_get_offsets():
    p = BonnesProjection()
    deg = units.Unit('degree')
    p.theta1 = 45 * deg
    o = p.get_offsets(1 * deg, 1 * deg)
    assert np.allclose(o.coordinates.value, [0.99983146, 1.0049345])
    o1 = o.copy()
    o2 = p.get_offsets(1 * deg, 1 * deg, offsets=o1)
    assert o == o2 and o1 is o2


def test_get_fits_id():
    assert BonnesProjection.get_fits_id() == 'BON'


def test_get_full_name():
    assert BonnesProjection.get_full_name() == "Bonne's Projection"


def test_parse_header():
    p = BonnesProjection()
    h = fits.Header()
    h['PV2_1'] = 45
    p.parse_header(h)
    assert p.theta1 == 45 * units.Unit('degree')
    assert np.isclose(
        p.y0, (1 * units.Unit('radian')) + (45 * units.Unit('degree')))


def test_edit_header():
    p = BonnesProjection()
    h = fits.Header()
    p.theta1 = 45 * units.Unit('degree')
    p.edit_header(h)
    assert h['CTYPE1'] == 'LON--BON'
    assert h['CTYPE2'] == 'LAT--BON'
    assert h['PV2_1'] == 45


def test_consistency():
    # Note that this does not work unless the theta1
    # (standard parallel) is set.
    p = BonnesProjection()
    p.theta1 = 30 * units.Unit('degree')
    pt = SphericalCoordinates([1, 2], unit='degree')
    o = p.get_offsets(pt.y, pt.x)
    phi_theta = p.get_phi_theta(o)
    assert phi_theta == pt
