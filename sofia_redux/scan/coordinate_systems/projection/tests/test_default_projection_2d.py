# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D


def test_init():
    p = DefaultProjection2D()
    assert p.reference == Coordinate2D([0, 0])


def test_get_coordinate_instance():
    c = DefaultProjection2D().get_coordinate_instance()
    assert c == Coordinate2D()


def test_project():
    p = DefaultProjection2D()
    c = EquatorialCoordinates([1, 2])
    projected = p.project(c)
    assert projected == Coordinate2D([-1, 2], unit='degree')  # RA is reversed
    p2 = p.project(projected, projected=projected)
    assert p2 is projected and p2 == Coordinate2D([-1, 2], unit='degree')


def test_deproject():
    p = DefaultProjection2D()
    c = EquatorialCoordinates([1, 2])
    dp = p.deproject(c)
    assert dp == Coordinate2D([-1, 2], unit='degree')  # RA is reversed
    dp2 = p.deproject(dp, coordinates=dp)
    assert dp2 is dp and dp2 == Coordinate2D([-1, 2], unit='degree')


def test_get_fits_id():
    assert DefaultProjection2D().get_fits_id() == ''


def test_get_full_name():
    assert DefaultProjection2D().get_full_name() == 'Cartesian'


def test_parse_header():
    h = fits.Header()
    p = DefaultProjection2D()
    p0 = p.copy()
    h['CDELT1'] = 3.0
    p.parse_header(h)
    assert p == p0  # no change


def test_edit_header():
    h = fits.Header()
    p = DefaultProjection2D()
    p.edit_header(h)
    assert len(h) == 0  # unchanged
