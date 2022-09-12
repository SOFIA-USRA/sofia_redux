# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.cartesian_grid import \
    CartesianGrid


def test_init():
    g = CartesianGrid()
    assert g.ndim == 1 and g.first_axis == 1
    g = CartesianGrid(first_axis=2, dimensions=3)
    assert g.first_axis == 2 and g.ndim == 3


def test_eq():
    g = CartesianGrid()
    assert g == g
    assert g != None
    g2 = g.copy()
    assert g2 == g
    g2.axis_resolution.x = 2.0
    assert g != g2
    g2 = g.copy()
    g2.reference_value.x = 2.0
    assert g != g2
    g2 = g.copy()
    g2.reference_index.x = 2.0
    assert g != g2
    g2 = g.copy()
    g.first_axis = 2
    assert g != g2


def test_variant_id():
    g = CartesianGrid()
    assert g.variant_id == ''
    g.variant = 1
    assert g.variant_id == 'B'


def test_unit():
    g = CartesianGrid()
    assert g.unit is None
    g.resolution.change_unit('arcsec')
    assert g.unit == 'arcsec'


def test_get_dimensions():
    g = CartesianGrid()
    assert g.get_dimensions() == 1
    g = CartesianGrid(dimensions=3)
    assert g.get_dimensions() == 3
    g._coordinate_system = None
    assert g.get_dimensions() is None


def test_set_first_axis():
    g = CartesianGrid()
    g.set_first_axis_index(3)
    assert g.first_axis == 3


def test_set_resolution():
    g = CartesianGrid()
    resolution = Coordinate1D(1.5)
    g.set_resolution(resolution)
    assert isinstance(g.resolution, Coordinate1D)
    assert g.resolution.x == 1.5
    g.set_resolution(2.0)
    assert g.resolution.x == 2
    g.set_resolution([2.5])
    assert g.resolution.x == 2.5


def test_get_resolution():
    g = CartesianGrid()
    assert g.get_resolution() == Coordinate1D(1.0)


def test_set_reference():
    g = CartesianGrid()
    g.set_reference(Coordinate1D(1.5))
    assert g.reference == Coordinate1D(1.5)
    g.set_reference(2.0)
    assert g.reference.x == 2
    g.set_reference([2.5])
    assert g.reference.x == 2.5


def test_get_reference():
    g = CartesianGrid()
    assert g.get_reference() == Coordinate1D(0.0)


def test_set_reference_index():
    g = CartesianGrid()
    g.set_reference_index(Coordinate1D(1.5))
    assert g.reference_index.x == 1.5
    g.set_reference_index(2.0)
    assert g.reference_index.x == 2.0
    g.set_reference_index([2.5])
    assert g.reference_index.x == 2.5
    g.set_reference_index(6 * units.Unit('m'))
    assert g.reference_index.x == 6


def test_get_reference_index():
    assert CartesianGrid().get_reference_index() == Coordinate1D(0.0)


def test_coordinates_at():
    g = CartesianGrid()
    assert g.coordinates_at(Coordinate1D(1.0)) == Coordinate1D(1.0)
    g.reference_index = Coordinate1D(2.0)
    c = g.coordinates_at(Coordinate1D(1.0))
    assert c == Coordinate1D(-1.0)
    c2 = g.coordinates_at(Coordinate1D(2.0), coordinates=c)
    assert c2 is c and c2.x == 0


def test_index_of():
    g = CartesianGrid()
    assert g.index_of(Coordinate1D(1.0)) == Coordinate1D(1.0)
    g.reference_index = Coordinate1D(2.0)
    c = g.index_of(Coordinate1D(1.0))
    assert c == Coordinate1D(3.0)
    c2 = g.index_of(Coordinate1D(2.0), grid_indices=c)
    assert c2 is c and c2.x == 4


def test_index_to_offset():
    g = CartesianGrid()
    c = Coordinate1D(2.0)
    g.resolution = Coordinate1D(0.5)
    g.reference_index = Coordinate1D(10.0)
    offset = g.index_to_offset(c, in_place=False)
    assert offset is not c and offset == Coordinate1D(-4.0)
    offset = g.index_to_offset(c, in_place=True)
    assert offset is c and offset == Coordinate1D(-4.0)


def test_offset_to_index():
    g = CartesianGrid()
    m = units.Unit('m')
    g.set_resolution(0.5 * m)
    g.set_reference_index(1.0)
    offset = Coordinate1D(2 * m)
    index = g.offset_to_index(offset, in_place=False)
    assert index is not offset and index == Coordinate1D(5)
    index = g.offset_to_index(offset, in_place=True)
    assert index is offset and index == Coordinate1D(5)
    g = CartesianGrid(dimensions=2)
    g.set_resolution([0.5, 1.0])
    g.set_reference_index(1.0)
    offset = Coordinate2D([10, 10])
    index = g.offset_to_index(offset)
    assert index == Coordinate2D([21, 11])


def test_parse_header():
    g = CartesianGrid(dimensions=2)
    h = fits.Header()
    h['CTYPE1'] = 'U'
    h['CTYPE2'] = 'V'
    h['CUNIT1'] = 'cm'
    h['CUNIT2'] = 'm'
    h['CRPIX1'] = 10.0
    h['CRPIX2'] = 11.0
    h['CRVAL1'] = 20.0
    h['CRVAL2'] = 21.0
    h['CDELT1'] = 30.0
    h['CDELT2'] = 31.0
    g.parse_header(h)
    assert g.resolution == Coordinate2D([30, 3100], unit='cm')
    assert g.reference_index == Coordinate2D([9, 10])
    assert g.reference_value == Coordinate2D([20, 2100], unit='cm')
    assert g.coordinate_system.axes[0].short_label == 'U'
    assert g.coordinate_system.axes[1].short_label == 'V'
    g = CartesianGrid(dimensions=2)
    del h['CUNIT1']
    del h['CUNIT2']
    g.parse_header(h)
    assert g.resolution == Coordinate2D([30, 31])


def test_edit_header():
    g = CartesianGrid(dimensions=2)
    g.coordinate_system.axes[0].short_label = 'U'
    g.coordinate_system.axes[1].short_label = 'V'
    g.resolution = Coordinate2D([1, 2])
    g.reference_index = Coordinate2D([10, 11])
    g.reference_value = Coordinate2D([20, 21])
    h = fits.Header()
    g.edit_header(h)
    assert h['CTYPE1'] == 'U'
    assert h['CTYPE2'] == 'V'
    assert h['CRPIX1'] == 11
    assert h['CRPIX2'] == 12
    assert h['CDELT1'] == 1
    assert h['CDELT2'] == 2
    assert h['CRVAL1'] == 20
    assert h['CRVAL2'] == 21
    assert 'CUNIT1' not in h and 'CUNIT2' not in h
    g.resolution.change_unit('m')
    g.reference_value.change_unit('m')
    g.coordinate_system.axes[0].unit = units.Unit('m')
    g.coordinate_system.axes[1].unit = units.Unit('m')
    g.edit_header(h)
    assert h['CUNIT1'] == 'm' and h['CUNIT2'] == 'm'
