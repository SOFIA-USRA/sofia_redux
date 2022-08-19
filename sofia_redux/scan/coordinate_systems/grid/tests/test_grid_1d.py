# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.grid.grid_1d import Grid1D
from sofia_redux.scan.coordinate_systems.coordinate_axis import CoordinateAxis


@pytest.fixture
def g1d():
    g = Grid1D()
    g.resolution = 0.5
    g.reference_index = 3
    g.reference = 100
    return g


def test_init():
    g = Grid1D()
    assert g.ndim == 1


def test_copy():
    g = Grid1D()
    g2 = g.copy()
    assert g == g2 and g is not g2


def test_axis():
    g = Grid1D()
    assert isinstance(g.axis, CoordinateAxis)


def test_resolution():
    g = Grid1D()
    assert isinstance(g.resolution, Coordinate1D) and g.resolution.x == 1
    resolution = g.resolution.copy()
    resolution.scale(2)
    g.resolution = resolution
    assert g.resolution == resolution


def test_reference():
    g = Grid1D()
    reference = g.reference.copy()
    assert isinstance(reference, Coordinate1D) and reference.x == 0
    reference.x = 2
    g.reference = reference
    assert g.reference.x == 2


def test_get_default_unit():
    g = Grid1D()
    assert g.get_default_unit() == 'pixel'
    g.resolution = Coordinate1D(1, unit='arcsec')
    assert g.get_default_unit() == 'arcsec'
    g = Grid1D()
    g.resolution = 1 * units.Unit('um')
    assert g.get_default_unit() == 'um'


def test_get_default_coordinate_instance():
    assert isinstance(Grid1D().get_default_coordinate_instance(), Coordinate1D)


def test_set_resolution():
    g = Grid1D()
    resolution = 1 * units.Unit('um')
    g.set_resolution(resolution)
    assert g.resolution.x == 1 * units.Unit('um')
    resolution = Coordinate1D(2, unit='um')
    g.set_resolution(resolution)
    assert g.resolution.x == 2 * units.Unit('um')


def test_get_resolution():
    g = Grid1D()
    assert g.get_resolution() == Coordinate1D(1)


def test_set_reference():
    g = Grid1D()
    g.set_reference(2)
    assert g.reference.x == 2
    g.set_reference(Coordinate1D(3))
    assert g.reference.x == 3


def test_set_reference_index():
    g = Grid1D()
    g.set_reference_index(5)
    assert g.reference_index.x == 5
    g.set_reference_index(Coordinate1D(1.5))
    assert g.reference_index.x == 1.5


def test_coordinates_at(g1d):
    g = g1d.copy()
    c = g.coordinates_at(Coordinate1D(3))
    assert c.x == 100
    c = g.coordinates_at(Coordinate1D(6))
    assert c.x == 101.5


def test_index_of(g1d):
    g = g1d.copy()
    c = g.index_of(Coordinate1D(102))
    assert c.x == 7


def test_index_to_offset(g1d):
    assert g1d.index_to_offset(Coordinate1D(7)).x == 2


def test_offset_to_index(g1d):
    c = g1d.offset_to_index(Coordinate1D(2))
    assert c.x == 7


def test_parse_header():
    g = Grid1D()
    h = fits.Header()
    h['CTYPE1'] = 'x'
    h['CUNIT1'] = 'arcsec'
    h['CRPIX1'] = 50
    h['CRVAL1'] = 100
    h['CDELT1'] = 1.5
    g.parse_header(h)
    assert g.resolution.x == 1.5 * units.Unit('arcsec')
    assert g.reference_index.x == 49
    assert g.reference_value.x == 100 * units.Unit('arcsec')
    del h['CUNIT1']
    g = Grid1D()
    g.parse_header(h)
    assert g.resolution.x == 1.5
    assert g.reference_index.x == 49
    assert g.reference_value.x == 100


def test_edit_header():
    h = fits.Header()
    g = Grid1D()
    arcsec = units.Unit('arcsec')
    g.resolution = 1.5 * arcsec
    g.reference = 100 * arcsec
    g.reference_index = Coordinate1D(50.0)
    g.edit_header(h)
    assert h['CTYPE1'] == 'x'
    assert h['CUNIT1'] == 'arcsec'
    assert h['CRPIX1'] == 51
    assert h['CRVAL1'] == 100
    assert h['CDELT1'] == 1.5
