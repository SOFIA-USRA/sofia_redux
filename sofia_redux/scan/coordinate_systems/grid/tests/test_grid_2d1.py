# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.grid.grid_1d import Grid1D
from sofia_redux.scan.coordinate_systems.projection.gnomonic_projection \
    import GnomonicProjection
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d1 import FlatGrid2D1
from sofia_redux.scan.coordinate_systems.grid.spherical_grid_2d1 import \
    SphericalGrid2D1
from sofia_redux.scan.coordinate_systems.grid.grid_2d1 import Grid2D1
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates


class BasicGrid(Grid2D1):
    def parse_projection(self, header):
        pass


@pytest.fixture
def spherical_header():
    h = fits.Header()
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CTYPE3'] = 'WAVE'
    h['CUNIT1'] = 'degree'
    h['CUNIT2'] = 'degree'
    h['CUNIT3'] = 'um'
    h['CDELT1'] = 1.0 / 3600
    h['CDELT2'] = 2.0 / 3600
    h['CDELT3'] = 3.0
    h['CRVAL1'] = 10.0
    h['CRVAL2'] = 11.0
    h['CRVAL3'] = 12.0
    h['CRPIX1'] = 20.0
    h['CRPIX2'] = 21.0
    h['CRPIX3'] = 22.0
    return h


@pytest.fixture
def gnomonic_grid():
    g = Grid2D1.get_grid_2d1_instance_for('RA---TAN', 'DEC--TAN')
    g.resolution = Coordinate2D1(xy=[1, 2] * units.Unit('arcsec'),
                                 z=3 * units.Unit('um'))
    g.reference_index = Coordinate2D1([12, 12, 12])
    g.reference = Coordinate2D1(EquatorialCoordinates([10, 20]),
                                30 * units.Unit('um'))
    return g


@pytest.fixture
def spherical_grid_2d1(spherical_header):
    return Grid2D1.from_header(spherical_header)


def test_init():
    g = BasicGrid()
    assert isinstance(g.z, Grid1D)


def test_copy():
    g = BasicGrid()
    g2 = g.copy()
    assert g2 == g and g2 is not g


def test_copy_2d_from():
    g = BasicGrid()
    g2 = FlatGrid2D()
    g2.reference = Coordinate2D([1, 2])
    g.copy_2d_from(g2)
    assert g.reference.x == 1 and g.reference.y == 2 and g.reference.z == 0


def test_reference():
    g = BasicGrid()
    reference = g.reference.copy()
    assert isinstance(reference, Coordinate2D1)
    g.projection = DefaultProjection2D()
    g.reference = reference
    assert g.reference == reference


def test_reference_index():
    g = BasicGrid()
    ri = g.reference_index.copy()
    assert isinstance(ri, Coordinate2D1)
    g.reference_index = ri
    assert g.reference_index == ri


def test_resolution():
    g = BasicGrid()
    resolution = g.resolution.copy()
    assert isinstance(resolution, Coordinate2D1)
    g.resolution = resolution
    assert g.resolution == resolution


def test_z_axis():
    g = BasicGrid()
    assert g.z_axis.label == 'z'


def test_fits_z_unit():
    g = BasicGrid()
    assert g.fits_z_unit == units.dimensionless_unscaled


def test_eq():
    g = BasicGrid()
    g2 = g.copy()
    assert g == g2
    assert g != None
    g2.z.resolution = 2
    assert g != g2


def test_to_coordinate2d1():
    f = Grid2D1.to_coordinate2d1
    c = Coordinate2D1([1, 2, 3])
    assert f(c) is c
    assert f([1, 2]) == Coordinate2D1([1, 1, 2])
    assert f([1, 2, 3]) == Coordinate2D1([1, 2, 3])
    assert f(1) == Coordinate2D1([1, 1])
    assert f([1]) == Coordinate2D1([1, 1])
    assert f([] * units.Unit('m')).xy_unit == 'm'
    c = f([])
    assert c.x is None and c.y is None and c.z is None
    c = f([1, 2, 3, 4])
    assert c.x is None and c.y is None and c.z is None


def test_get_coordinate_instance_for():
    c = Grid2D1.get_coordinate_instance_for('spherical')
    assert isinstance(c, Coordinate2D1)
    assert isinstance(c.xy_coordinates, SphericalCoordinates)


def test_get_grid_2d1_instance_for():
    g = Grid2D1.get_grid_2d1_instance_for('RA---TAN', 'DEC--TAN')
    assert isinstance(g, SphericalGrid2D1)
    assert isinstance(g.projection, GnomonicProjection)
    g = Grid2D1.get_grid_2d1_instance_for('X', 'Y')
    assert isinstance(g, FlatGrid2D1)


def test_from_header(spherical_header):
    h = spherical_header.copy()
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CTYPE3'] = 'WAVE'
    h['CUNIT1'] = 'degree'
    h['CUNIT2'] = 'degree'
    h['CUNIT3'] = 'um'
    h['CDELT1'] = 1.0 / 3600
    h['CDELT2'] = 2.0 / 3600
    h['CDELT3'] = 3.0
    h['CRVAL1'] = 10.0
    h['CRVAL2'] = 11.0
    h['CRVAL3'] = 12.0
    h['CRPIX1'] = 20.0
    h['CRPIX2'] = 21.0
    h['CRPIX3'] = 22.0
    g = Grid2D1.from_header(h)
    assert g.reference_index == Coordinate2D1([19, 20, 21])
    assert g.reference.x == -10 * units.Unit('degree')
    assert g.reference.y == 11 * units.Unit('degree')
    assert g.reference.z == 12 * units.Unit('um')
    assert g.resolution == Coordinate2D1([-1, 2, 3], xy_unit='arcsec',
                                         z_unit='um')
    g = Grid2D1.from_header(fits.Header(), alt='C')
    assert isinstance(g, FlatGrid2D1)
    assert g.z.variant == 2


def test_get_default_coordinate_instance():
    g = BasicGrid()
    assert isinstance(g.get_default_coordinate_instance(), Coordinate2D1)


def test_to_string():
    g = Grid2D1.get_grid_2d1_instance_for('RA---TAN', 'DEC--TAN')
    g.resolution = Coordinate2D1(xy=[1, 2] * units.Unit('arcsec'),
                                 z=3 * units.Unit('um'))
    assert str(g) == ('Spherical: x=Empty, y=Empty, z=0.0\n'
                      'Projection: Gnomonic (TAN)\n'
                      'Grid Spacing: (1.000 x 2.000 arcsec) x 3.00000 um\n'
                      'Reference Pixel: x=0.0 y=0.0, z=0.0 C-style, 0-based')


def test_for_resolution(gnomonic_grid):
    g = gnomonic_grid.copy()
    resolution = Coordinate2D1([1, 1, 1], xy_unit='arcsec', z_unit='um')
    g2 = g.for_resolution(resolution)
    assert g2.resolution == resolution
    assert g2.reference_index == Coordinate2D1([12, 24, 36])


def test_pixel_volume(gnomonic_grid):
    v = gnomonic_grid.get_pixel_volume()
    assert v == 6 * units.Unit('arcsec2 um')


def test_get_resolution(gnomonic_grid):
    resolution = gnomonic_grid.get_resolution()
    assert resolution == Coordinate2D1([1, 2, 3], xy_unit='arcsec',
                                       z_unit='um')


def test_set_resolution(gnomonic_grid):
    g = gnomonic_grid.copy()
    resolution = Coordinate2D1([2, 3, 4], xy_unit='arcsec', z_unit='um')
    g.set_resolution(resolution)
    assert g.resolution == resolution


def test_get_pixel_size(gnomonic_grid):
    assert gnomonic_grid.get_pixel_size() == Coordinate2D1(
        [1, 2, 3], xy_unit='arcsec', z_unit='um')


def test_get_pixel_size_z(gnomonic_grid):
    assert gnomonic_grid.get_pixel_size_z().x == 3 * units.Unit('um')


def test_is_reverse_z():
    assert not BasicGrid().is_reverse_z()


def test_parse_header(gnomonic_grid):
    g = gnomonic_grid.copy()
    h = fits.Header()
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CTYPE3'] = 'WAVE'
    h['CUNIT1'] = 'degree'
    h['CUNIT2'] = 'degree'
    h['CUNIT3'] = 'um'
    h['CDELT1'] = 1.0 / 3600
    h['CDELT2'] = 2.0 / 3600
    h['CDELT3'] = 3.0
    h['CRVAL1'] = 10.0
    h['CRVAL2'] = 11.0
    h['CRVAL3'] = 12.0
    h['CRPIX1'] = 20.0
    h['CRPIX2'] = 21.0
    h['CRPIX3'] = 22.0
    g.parse_header(h)
    arcsec = units.Unit('arcsec')
    degree = units.Unit('degree')
    um = units.Unit('um')
    assert g.reference.x == -10 * degree
    assert g.reference.y == 11 * degree
    assert g.reference.z == 12 * um
    assert g.reference_index.x == 19
    assert g.reference_index.y == 20
    assert g.reference_index.z == 21
    assert g.resolution.x == -1 * arcsec
    assert g.resolution.y == 2 * arcsec
    assert g.resolution.z == 3 * um


def test_edit_header(gnomonic_grid):
    g = gnomonic_grid.copy()
    h = fits.Header()
    g.reference_index = Coordinate2D1([10, 11, 12])
    g.edit_header(h)
    assert h['CTYPE1'] == 'RA---TAN'
    assert h['CTYPE2'] == 'DEC--TAN'
    assert h['CTYPE3'] == 'z'
    assert h['CRPIX1'] == 11
    assert h['CRPIX2'] == 12
    assert h['CRPIX3'] == 13
    assert h['CRVAL1'] == 10
    assert h['CRVAL2'] == 20
    assert h['CRVAL3'] == 30
    assert h['CUNIT1'] == 'deg'
    assert h['CUNIT2'] == 'deg'
    assert h['CUNIT3'] == 'um'


def test_coordinates_at(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = g.reference_index.copy()
    c = g.coordinates_at(index)
    assert np.isclose(c.x, g.reference.x)
    assert np.isclose(c.y, g.reference.y)
    assert np.isclose(c.z, g.reference.z)
    coordinates = c.copy()
    c = g.coordinates_at(index, coordinates=coordinates)
    assert c is coordinates
    assert np.isclose(c.x, g.reference.x)
    assert np.isclose(c.y, g.reference.y)
    assert np.isclose(c.z, g.reference.z)


def test_index_of(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    c = g.reference.copy()
    index = g.index_of(c)
    assert index == g.reference_index
    grid_indices = Coordinate2D1()
    index2 = g.index_of(c, grid_indices=grid_indices)
    assert index2 is grid_indices and index2 == index


def test_offset_to_index(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    c = Coordinate2D([0, 0], unit='arcsec')
    index = g.offset_to_index(c)
    assert isinstance(index, Coordinate2D)
    assert np.isclose(index.x, 19) and np.isclose(index.y, 20)
    c = Coordinate2D1([0, 0], 0, xy_unit='arcsec', z_unit='um')
    index = g.offset_to_index(c, in_place=True)
    assert index is c and index == g.reference_index
    c = Coordinate2D1([0, 0], 0, xy_unit='arcsec', z_unit='um')
    index = g.offset_to_index(c, in_place=False)
    assert index is not c and index == g.reference_index


def test_index_to_offset(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = Coordinate2D([0, 0])
    c2 = g.index_to_offset(index)
    assert isinstance(c2, Coordinate2D)
    assert np.isclose(c2.x, 19 * units.Unit('arcsec'))
    assert np.isclose(c2.y, -40 * units.Unit('arcsec'))

    index = g.reference_index.copy()
    c = g.index_to_offset(index, in_place=True)
    assert c is index and np.all(c.is_null())
    index = g.reference_index.copy()
    c = g.index_to_offset(index, in_place=False)
    assert c is not index and np.all(c.is_null())


def test_get_reference(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    reference = g.get_reference()
    assert isinstance(reference, Coordinate2D1)
    assert reference.x == -10 * units.Unit('degree')
    assert reference.y == 11 * units.Unit('degree')
    assert reference.z == 12 * units.Unit('um')


def test_set_reference(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    reference = g.reference.copy()
    reference.scale(2)
    g.set_reference(reference)
    assert g.reference.x == -20 * units.Unit('degree')
    assert g.reference.y == 22 * units.Unit('degree')
    assert g.reference.z == 24 * units.Unit('um')


def test_get_reference_index(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = g.get_reference_index()
    assert isinstance(index, Coordinate2D1)
    assert index.x == 19 and index.y == 20 and index.z == 21


def test_set_reference_index(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = g.reference_index.copy()
    index.scale(2)
    g.set_reference_index(index)
    assert g.reference_index.x == 38
    assert g.reference_index.y == 40
    assert g.reference_index.z == 42


def test_get_coordinate_index(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = g.get_coordinate_index(g.reference)
    assert index is not g.reference_index and index == g.reference_index
    i2 = g.get_coordinate_index(g.reference, indices=index)
    assert i2 is index and i2 == g.reference_index


def test_get_offset_index(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    offset = g.reference.copy()
    offset.zero()
    index = g.get_offset_index(offset)
    assert index == g.reference_index
    i2 = g.get_offset_index(offset, indices=index)
    assert i2 is index and i2 == g.reference_index


def test_get_coordinates(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = g.reference_index.copy()
    c = g.get_coordinates(index)
    assert np.isclose(c.x, g.reference.x)
    assert np.isclose(c.y, g.reference.y)
    assert np.isclose(c.z, g.reference.z)
    c2 = g.get_coordinates(index, coordinates=c)
    assert c2 is c
    assert np.isclose(c2.x, g.reference.x)
    assert np.isclose(c2.y, g.reference.y)
    assert np.isclose(c2.z, g.reference.z)


def test_get_offset(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    index = g.reference_index.copy()
    offset = g.get_offset(index)
    assert isinstance(offset, Coordinate2D1)
    assert np.all(offset.is_null())
    o = offset
    offset = g.get_offset(index, offset=o)
    assert offset is o and np.all(o.is_null())
    index_3d = Coordinate3D([index.x, index.y, index.z])

    index = index.xy_coordinates.copy()
    offset = g.get_offset(index)
    assert isinstance(offset, Coordinate2D) and offset.is_null()
    offset = g.get_offset(index_3d)
    assert isinstance(offset, Coordinate2D1) and np.all(offset.is_null())


def test_toggle_native(spherical_grid_2d1):
    g = spherical_grid_2d1.copy()
    offset = Coordinate2D1([1, 2, 3])
    g.z_axis.reverse = True
    o2 = g.toggle_native(offset, in_place=False)
    assert o2 is not offset and o2.x == -1 and o2.y == 2 and o2.z == -3
    o2 = g.toggle_native(offset, in_place=True)
    assert o2 is offset and o2.x == -1 and o2.y == 2 and o2.z == -3
