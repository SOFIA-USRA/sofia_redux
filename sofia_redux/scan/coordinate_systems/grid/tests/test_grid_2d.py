# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid


@pytest.fixture
def affine_offset():
    g = FlatGrid2D()
    m = np.asarray([[2, 0], [1, 4]])
    g.set_transform(m)
    g.reference = Coordinate2D([5, 6])
    return g


@pytest.fixture
def affine_offset_quantity():
    g = FlatGrid2D()
    m = np.asarray([[2, 0], [1, 4]])
    g.set_transform(m * units.Unit('degree'))
    g.reference = Coordinate2D([5, 6], unit='degree')
    return g


def test_init():
    g = FlatGrid2D()
    assert g.projection is not None
    assert g._reference_index.is_null()
    assert np.allclose(g.m, np.eye(2))
    assert np.allclose(g.i, g.m)


def test_copy():
    g = FlatGrid2D()
    g2 = g.copy()
    assert g2 == g and g2 is not g


def test_get_dimensions():
    g = FlatGrid2D()
    assert g.get_dimensions() == 2


def test_set_defaults():
    g = FlatGrid2D()
    g.m = g.i = None
    g.reference_index.nan()
    g.set_defaults()
    assert np.allclose(g.i, np.eye(2))
    assert np.allclose(g.m, g.i)
    assert g.reference_index.is_null()


def test_reference_index():
    g = FlatGrid2D()
    c1 = Coordinate2D([1, 1])
    g.reference_index = c1
    assert g.reference_index == c1


def test_projection():
    g = FlatGrid2D()
    p = g.projection.copy()
    p.reference = Coordinate2D([1, 1])
    g.projection = p
    assert g.projection == p


def test_transform():
    g = FlatGrid2D()
    assert np.allclose(g.transform, g.m)


def test_inverse_transform():
    g = FlatGrid2D()
    assert np.allclose(g.inverse_transform, g.i)


def test_rectilinear():
    g = FlatGrid2D()
    assert g.rectilinear
    g.m[0, 1] = 0.5
    assert not g.rectilinear


def test_x_axis():
    assert FlatGrid2D().x_axis.label == 'x'


def test_y_axis():
    assert FlatGrid2D().y_axis.label == 'y'


def test_fits_x_unit():
    assert FlatGrid2D().fits_x_unit == units.dimensionless_unscaled


def test_fits_y_unit():
    assert FlatGrid2D().fits_y_unit == units.dimensionless_unscaled


def test_eq():
    g = FlatGrid2D()
    assert g == g
    assert g is not None
    g2 = g.copy()
    g2.projection.reference = Coordinate2D([1, 1])
    assert g != g2
    g2 = g.copy()
    g2.reference_index = Coordinate2D([1, 1])
    assert g != g2
    g2 = g.copy()
    g.m[0, 1] = 0.5
    assert g != g2
    g2 = g.copy()
    g.i[0, 1] = 0.5
    assert g != g2
    g2 = g.copy()
    assert g == g2


def test_str():
    assert str(FlatGrid2D()) == ('Coordinate2D: x=0.0 y=0.0\n'
                                 'Projection: Cartesian ()\n'
                                 'Grid Spacing: 1.0 x 1.0\n'
                                 'Reference Pixel: x=0.0 y=0.0 '
                                 'C-style, 0-based')


def test_repr():
    s = repr(FlatGrid2D())
    assert str(FlatGrid2D()) in s and 'FlatGrid2D object' in s


def test_to_coordinate2d():
    c = Coordinate2D([1, 1])
    assert FlatGrid2D.to_coordinate2d(c) is c
    c1 = FlatGrid2D.to_coordinate2d(np.asarray(1))
    assert c1 == c
    c1 = FlatGrid2D.to_coordinate2d(1.0)
    assert c1 == c
    c2 = FlatGrid2D.to_coordinate2d(1 * units.Unit('degree'))
    assert c2 == Coordinate2D([1, 1], unit='degree')
    c2 = FlatGrid2D.to_coordinate2d(np.arange(2))
    assert c2 == Coordinate2D([0, 1])


def test_get_coordinate_instance_for():
    e = Grid2D.get_coordinate_instance_for('equatorial')
    assert e.__class__.__name__ == 'EquatorialCoordinates'


def test_get_default_unit():
    assert Grid2D.get_default_unit() is None


def test_from_header():
    h = fits.Header()
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CUNIT1'] = 'arcsec'
    h['CUNIT2'] = 'arcsec'
    g = Grid2D.from_header(h)
    assert g.__class__.__name__ == 'SphericalGrid'
    assert np.allclose(g.m, np.eye(2) * units.Unit('arcsec'))


def test_get_grid_2d_instance_for():
    fg = FlatGrid2D()
    sg = SphericalGrid()
    assert Grid2D.get_grid_2d_instance_for(None, None) == fg
    assert Grid2D.get_grid_2d_instance_for('a', 'b') == fg
    assert Grid2D.get_grid_2d_instance_for('RA---TAN', 'FOOOOOOO') == fg
    assert Grid2D.get_grid_2d_instance_for('RA---TAN', 'DEC--TAN') == sg
    assert Grid2D.get_grid_2d_instance_for('LON--TAN', 'LAT--TAN') == sg
    assert Grid2D.get_grid_2d_instance_for('GLON-TAN', 'ELAT-TAN') == fg
    assert Grid2D.get_grid_2d_instance_for('GLON-TAN', 'GLAT-TAN') == sg
    assert Grid2D.get_grid_2d_instance_for('XYLN-TAN', 'XZLT-TAN') == fg
    assert Grid2D.get_grid_2d_instance_for('XYLN-TAN', 'XYLT-TAN') == sg
    assert Grid2D.get_grid_2d_instance_for('XYLN-TAN', 'XYRA-TAN') == fg


def test_to_string():
    g = FlatGrid2D()
    g.reference = Coordinate2D([1, 2], unit='degree')
    g.resolution = 1 * units.Unit('degree')
    s = g.to_string()
    assert s == ('Coordinate2D: x=1.0 deg y=2.0 deg\n'
                 'Projection: Cartesian ()\n'
                 'Grid Spacing: 1.0 x 1.0 deg\n'
                 'Reference Pixel: x=0.0 y=0.0 C-style, 0-based')


def test_for_resolution():
    g = FlatGrid2D()
    g.reference = Coordinate2D([1, 1], unit='degree')
    g.reference_index = Coordinate2D([1, 1])
    g.resolution = 1 * units.Unit('degree')
    g2 = g.for_resolution(2 * units.Unit('degree'))
    assert g2.reference_index == Coordinate2D([0.5, 0.5])
    assert g2.resolution == Coordinate2D([2, 2], unit='degree')


def test_get_pixel_area():
    g = FlatGrid2D()
    assert g.get_pixel_area() == 1
    g.m[0, 1] = g.m[1, 0] = 0.5
    assert g.get_pixel_area() == 0.75
    g.m = g.m * units.Unit('degree')
    assert g.get_pixel_area() == 0.75 * units.Unit('degree2')


def test_set_resolution():
    g = FlatGrid2D()
    g.set_resolution(np.array([1.0, 2.0]))
    assert g.resolution == Coordinate2D([1, 2])
    assert np.allclose(g.m, [[1, 0], [0, 2]])
    assert np.allclose(g.i, [[1, 0], [0, 0.5]])
    g.resolution = 1.5 * units.Unit('degree')
    assert g.resolution == Coordinate2D([1.5, 1.5], unit='degree')


def test_calculate_inverse_transform():
    g = FlatGrid2D()
    g.m = np.array([[2, 0], [1, 4]])
    g.calculate_inverse_transform()
    assert np.allclose(g.i, [[0.5, 0], [-0.125, 0.25]])


def test_get_transform():
    g = FlatGrid2D()
    assert np.allclose(g.get_transform(), np.eye(2))


def test_set_transform():
    g = FlatGrid2D()
    m = np.asarray([[2, 0], [1, 4]])
    g.set_transform(m)
    assert np.allclose(g.m, m) and g.m is not m
    assert np.allclose(g.i, [[0.5, 0], [-0.125, 0.25]])
    with pytest.raises(ValueError) as err:
        g.set_transform(np.zeros((3, 3)))
    assert "should have shape (2, 2)" in str(err.value)


def test_get_inverse_transform():
    g = FlatGrid2D()
    i = g.get_inverse_transform()
    assert np.allclose(g.i, i) and i is not g.i


def test_is_horizontal():
    g = FlatGrid2D()
    assert not g.is_horizontal()
    g.reference = g.reference.get_instance('horizontal')
    assert g.is_horizontal()


def test_is_equatorial():
    g = FlatGrid2D()
    assert not g.is_equatorial()
    g.reference = g.reference.get_instance('equatorial')
    assert g.is_equatorial()


def test_is_ecliptic():
    g = FlatGrid2D()
    assert not g.is_ecliptic()
    g.reference = g.reference.get_instance('ecliptic')
    assert g.is_ecliptic()


def test_is_galactic():
    g = FlatGrid2D()
    assert not g.is_galactic()
    g.reference = g.reference.get_instance('galactic')
    assert g.is_galactic()


def test_is_super_galactic():
    g = FlatGrid2D()
    assert not g.is_super_galactic()
    g.reference = g.reference.get_instance('super_galactic')
    assert g.is_super_galactic()


def test_local_affine_transform():
    g = FlatGrid2D()
    m = np.asarray([[2, 0], [1, 4]])
    g.set_transform(m)
    g.reference = Coordinate2D([1, 1])
    grid_indices = Coordinate2D([1, 1])
    transform = g.local_affine_transform(grid_indices)
    assert transform == Coordinate2D([2, 5])
    grid_indices = Coordinate2D([[1, 2], [1, 2]])
    transform = g.local_affine_transform(grid_indices)
    assert transform == Coordinate2D([[2, 4], [5, 10]])

    g.m = g.m * units.Unit('degree')
    g.reference = Coordinate2D(g.reference, unit='degree')
    with pytest.raises(ValueError) as err:
        _ = g.local_affine_transform(grid_indices)
    assert 'Grid indices should be quantities' in str(err.value)

    grid_indices = Coordinate2D(grid_indices, unit='degree')
    transform = g.local_affine_transform(grid_indices)
    assert transform == Coordinate2D([[2, 4], [5, 10]], unit='degree')


def test_get_resolution(affine_offset):
    assert affine_offset.get_resolution() == Coordinate2D([2, 4])


def test_get_pixel_size(affine_offset):
    assert affine_offset.get_pixel_size() == Coordinate2D([2, 4])


def test_get_pixel_size_x(affine_offset):
    assert affine_offset.get_pixel_size_x() == 2


def test_get_pixel_size_y(affine_offset):
    assert affine_offset.get_pixel_size_y() == 4


def test_rotate(affine_offset):
    g = affine_offset.copy()
    g.rotate(90 * units.Unit('degree'))
    assert np.allclose(g.m, [[0, -4], [2, 0]])
    assert np.allclose(g.i, [[0, 0.5], [-0.25, 0]])


def test_is_reverse_x():
    g = FlatGrid2D()
    assert not g.is_reverse_x()
    g.x_axis.reverse = True
    assert g.is_reverse_x()


def test_is_reverse_y():
    g = FlatGrid2D()
    assert not g.is_reverse_y()
    g.y_axis.reverse = True
    assert g.is_reverse_y()


def test_parse_header():
    nd = units.dimensionless_unscaled
    deg = units.Unit('degree')
    h = fits.Header()
    g = FlatGrid2D()
    g.parse_header(h)
    assert g.x_axis.unit == nd and g.y_axis.unit == nd
    assert np.allclose(g.m, np.eye(2))
    assert g.reference == Coordinate2D([0, 0])
    assert g.reference_index == Coordinate2D([0, 0])

    h['CUNIT1'] = 'degree'
    h['CUNIT2'] = 'degree'
    g.parse_header(h)
    assert np.allclose(g.m, np.eye(2) * deg)
    assert g.reference == Coordinate2D([0, 0])

    h['CD1_1'] = 2.0
    h['CD1_2'] = 0.0
    h['CD2_1'] = 1.0
    h['CD2_2'] = 4.0
    g.parse_header(h)
    assert np.allclose(g.m, [[2, 0], [1, 4]] * deg)
    assert g.reference == Coordinate2D([0, 0])

    h['CROTA2'] = 90.0
    g.parse_header(h)
    assert np.allclose(g.m, [[-1, -4], [2, 0]] * deg)
    assert g.reference_index == Coordinate2D([0, 0])

    g.x_axis.reverse = True
    g.parse_header(h)
    assert np.allclose(g.m, [[1, -4], [-2, 0]] * deg)

    g.y_axis.reverse = True
    g.parse_header(h)
    assert np.allclose(g.m, [[-1, -4], [-2, 0]] * deg)

    h['CRVAL1'] = 1.0
    h['CRVAL2'] = 2.0
    g.parse_header(h)
    assert g.reference == Coordinate2D([1, 2])
    assert g.reference_index == Coordinate2D([0, 0])

    h['CRPIX1'] = 2.0
    h['CRPIX2'] = 3.0
    g.parse_header(h)
    assert g.reference == Coordinate2D([1, 2])
    assert g.reference_index == Coordinate2D([1, 2])


def test_edit_header():
    g = FlatGrid2D()
    h = fits.Header()
    g.edit_header(h)

    for key in ['CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2']:
        assert h[key] == 1
    assert h['CRVAL1'] == 0 and h['CRVAL2'] == 0
    assert h['CTYPE1'] == 'x' and h['CTYPE2'] == 'y'

    g.x_axis.reverse = g.y_axis.reverse = True
    g.edit_header(h)
    assert h['CDELT1'] == -1 and h['CDELT2'] == -1

    nd = units.dimensionless_unscaled
    deg = units.Unit('degree')

    g.m = g.m * nd
    g.edit_header(h)
    assert '(deg)' not in h.comments['CDELT1']
    assert '(deg)' not in h.comments['CDELT2']
    assert 'CUNIT1' not in h and 'CUNIT2' not in h

    g.m = g.m * deg
    g.edit_header(h)
    assert '(deg)' in h.comments['CDELT1'] and '(deg)' in h.comments['CDELT2']
    assert h['CUNIT1'] == 'deg'
    assert h['CUNIT2'] == 'deg'

    g.m = g.m.value
    g.x_axis.unit = deg
    g.y_axis.unit = deg
    g.edit_header(h)
    assert '(deg)' in h.comments['CDELT1'] and '(deg)' in h.comments['CDELT2']
    assert h['CUNIT1'] == 'deg'
    assert h['CUNIT2'] == 'deg'
    assert h['CDELT1'] == -1 and h['CDELT2'] == -1

    h = fits.Header()
    g.x_axis.unit = units.Unit('arcmin')
    g.y_axis.unit = units.Unit('arcmin')
    g.m[0, 1] = 0.5
    g.m = g.m * units.Unit('degree')
    g.edit_header(h)

    assert h['CRPIX1'] == 1 and h['CRPIX2'] == 1
    assert h['CRVAL1'] == 0 and h['CRVAL2'] == 0
    assert h['CD1_1'] == -60
    assert h['CD1_2'] == -30
    assert h['CD2_1'] == 0
    assert h['CD2_2'] == -60
    assert h['CUNIT1'] == 'arcmin' and h['CUNIT2'] == 'arcmin'
    assert h['CTYPE1'] == 'x' and h['CTYPE2'] == 'y'


def test_index_of():
    g = FlatGrid2D()
    c = Coordinate2D([1, 1])
    i = g.index_of(c)
    assert i == c and i is not c
    i = g.index_of(c, grid_indices=c)
    assert i is c


def test_offset_to_index(affine_offset_quantity):
    g = FlatGrid2D()
    c = Coordinate2D([3, 4])
    i = g.offset_to_index(c, in_place=False)
    assert i == c and i is not c
    i = g.offset_to_index(c, in_place=True)
    assert i is c

    g = affine_offset_quantity.copy()
    with pytest.raises(ValueError) as err:
        _ = g.offset_to_index(c)
    assert 'Offsets should be quantities' in str(err.value)

    c = Coordinate2D(c, unit='degree')
    i = g.offset_to_index(c, in_place=False)
    assert i == Coordinate2D([1.5, 0.625])

    c = Coordinate2D([[3, 10], [4, 20]], unit='degree')
    i = g.offset_to_index(c, in_place=False)
    assert i == Coordinate2D([[1.5, 5], [0.625, 3.75]])


def test_index_to_offset(affine_offset_quantity):
    g = FlatGrid2D()
    i = Coordinate2D([1.5, 0.625])
    c = g.index_to_offset(i, in_place=True)
    assert i is c
    c = g.index_to_offset(i, in_place=False)
    assert i == c and i is not c

    g = affine_offset_quantity.copy()
    c = g.index_to_offset(i)
    assert c == Coordinate2D([3, 4], unit='degree')

    i = Coordinate2D([[1.5, 5], [0.625, 3.75]])
    c = g.index_to_offset(i)
    assert c == Coordinate2D([[3, 10], [4, 20]], unit='degree')


def test_coordinates_at():
    g = FlatGrid2D()
    i = Coordinate2D([1, 2])
    c = g.coordinates_at(i)
    assert c == i and c is not i
    c = g.coordinates_at(i, coordinates=i)
    assert c is i


def test_get_reference():
    g = FlatGrid2D()
    assert g.get_reference() == Coordinate2D([0, 0])
    g._projection = None
    assert g.get_reference() is None


def test_set_reference():
    g = FlatGrid2D()
    c = Coordinate2D([1, 2])
    g.set_reference(c)
    assert g.projection.reference == c


def test_get_reference_index():
    assert FlatGrid2D().get_reference_index() == Coordinate2D([0, 0])


def test_set_reference_index():
    g = FlatGrid2D()
    c = Coordinate2D([1, 2])
    g.set_reference_index(c)
    assert g.reference_index == c


def test_get_projection():
    projection = FlatGrid2D().get_projection()
    assert projection.__class__.__name__ == 'DefaultProjection2D'


def test_set_projection():
    g = FlatGrid2D()
    p1 = g.projection.copy()
    p2 = p1.copy()
    p2.reference = Coordinate2D([2, 3])
    assert p2 != p1
    g.set_projection(p2)
    assert g.projection == p2


def test_get_coordinate_index(affine_offset):
    g = affine_offset.copy()
    c = Coordinate2D([1, 2])
    i = g.get_coordinate_index(c)
    assert i == Coordinate2D([0.5, 0.375])
    assert i is not c
    i = g.get_coordinate_index(c, indices=c)
    assert i == Coordinate2D([0.5, 0.375]) and i is c


def test_get_offset_index(affine_offset):
    g = affine_offset.copy()
    c = Coordinate2D([3, 4])
    i = g.get_offset_index(c)
    assert np.allclose([i.x, i.y], [1, 1], atol=1)
    assert i is not c
    i = g.get_offset_index(c, indices=c)
    assert np.allclose([i.x, i.y], [1, 1], atol=1)
    assert i is c


def test_get_coordinates(affine_offset):
    g = affine_offset.copy()
    i = Coordinate2D([3, 4])
    c = g.get_coordinates(i)
    assert c == Coordinate2D([6, 19]) and c is not i
    c = g.get_coordinates(i, coordinates=i)
    assert c == Coordinate2D([6, 19]) and c is i


def test_get_offset(affine_offset):
    g = affine_offset.copy()
    i = Coordinate2D([3, 4])
    o = g.get_offset(i)
    assert o == Coordinate2D([6, 19]) and o is not i
    o = g.get_offset(i, offset=i)
    assert o == Coordinate2D([6, 19]) and o is i


def test_toggle_native():
    g = FlatGrid2D()
    c = Coordinate2D([1, 1])
    c2 = g.toggle_native(c, in_place=False)
    assert c2 is not c and c2 == c
    g.x_axis.reverse = True
    c3 = g.toggle_native(c2, in_place=True)
    assert c3 is c2 and c2 == Coordinate2D([-1, 1])

    g.y_axis.reverse = True
    g.toggle_native(c)
    assert c == Coordinate2D([-1, -1])
