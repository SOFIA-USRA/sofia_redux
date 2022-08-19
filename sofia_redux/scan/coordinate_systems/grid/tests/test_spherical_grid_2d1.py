# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.projection.gnomonic_projection \
    import GnomonicProjection
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.grid.spherical_grid_2d1 import \
    SphericalGrid2D1


degree = units.Unit('degree')
um = units.Unit('um')


@pytest.fixture
def equatorial_reference():
    xy = EquatorialCoordinates([10, 20])
    return Coordinate2D1(xy=xy, z=30 * um)


@pytest.fixture
def g2d1(equatorial_reference):
    g = SphericalGrid2D1(reference=equatorial_reference.copy())
    g.resolution = Coordinate2D1([1, 2, 3], xy_unit='arcsec', z_unit='um')
    return g


def test_init(equatorial_reference):
    g = SphericalGrid2D1(reference=equatorial_reference)
    assert g.reference == equatorial_reference


def test_reference(g2d1):
    g = g2d1.copy()
    reference = g.reference.copy()
    reference.scale(2)
    g.reference = reference
    assert np.isclose(g.reference.x, -20 * degree)
    assert np.isclose(g.reference.y, 40 * degree)
    assert np.isclose(g.reference.z, 60 * um)


def test_fits_z_unit(g2d1):
    assert g2d1.fits_z_unit == 'um'


def test_get_default_xy_unit():
    assert SphericalGrid2D1.get_default_xy_unit() == 'degree'


def test_get_default_z_unit(g2d1):
    assert g2d1.get_default_z_unit() == 'um'


def test_get_coordinate_instance_for():
    c = SphericalGrid2D1.get_coordinate_instance_for('RA---TAN')
    assert isinstance(c, Coordinate2D1)
    assert isinstance(c.xy_coordinates, EquatorialCoordinates)


def test_set_reference(equatorial_reference):
    reference = equatorial_reference.copy()
    g = SphericalGrid2D1()
    g.set_reference(reference)
    assert g.reference == reference


def test_is_reverse_x(g2d1):
    assert g2d1.is_reverse_x()


def test_is_reverse_y(g2d1):
    assert not g2d1.is_reverse_y()


def test_is_reverse_z(g2d1):
    g = g2d1.copy()
    assert not g.is_reverse_z()
    g.z_axis.reverse = True
    assert g.is_reverse_z()


def test_parse_projection():
    g = SphericalGrid2D1()
    h = fits.Header()
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CTYPE3'] = 'WAVE'
    h['CUNIT3'] = 'um'
    g.parse_projection(h)
    assert isinstance(g.projection, GnomonicProjection)
    assert g.z_axis.unit == 'um'
