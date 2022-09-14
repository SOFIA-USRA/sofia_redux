# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d1 import FlatGrid2D1
from sofia_redux.scan.coordinate_systems.grid.spherical_grid_2d1 import \
    SphericalGrid2D1
from sofia_redux.scan.source_models.beams.gaussian_2d1 import Gaussian2D1
from sofia_redux.scan.source_models.maps.image_2d1 import Image2D1
from sofia_redux.scan.source_models.maps.map_2d1 import Map2D1

arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
um = units.Unit('um')


@pytest.fixture
def spike():
    data = np.zeros((10, 11, 12))
    data[5, 6, 7] = 1.0
    return data


@pytest.fixture
def spike_image(spike):
    return Image2D1(data=spike.copy(), unit='Jy')


@pytest.fixture
def spike_map(spike_image):
    return Map2D1(data=spike_image)


@pytest.fixture
def initialized_spike_map(spike_map):
    m = spike_map.copy()
    m.set_resolution(Coordinate2D1(xy=[1, 1] * arcsec, z=1 * um))
    m.smoothing_beam = m.get_pixel_smoothing()
    m.set_underlying_beam(Coordinate2D1(xy=[2, 2] * arcsec, z=2 * um))
    m.set_filtering(Coordinate2D1(xy=[1, 1] * arcsec, z=1 * um))
    m.set_filter_blanking(10.0)
    m.set_display_grid_unit('arcsec')
    m.set_z_display_grid_unit('um')
    m.grid.reference = Coordinate2D1([0, 0] * arcsec, 0 * um)
    return m


def test_init():
    m = Map2D1()
    assert isinstance(m.grid, FlatGrid2D1)
    assert isinstance(m.reuse_index, Coordinate2D1)
    assert isinstance(m.filter_fwhm, Coordinate2D1)
    assert isinstance(m.correcting_fwhm, Coordinate2D1)
    assert m.z_display_grid_unit is None


def test_copy():
    m = Map2D1()
    m2 = m.copy()
    assert m == m2 and m is not m2


def test_size_x(spike_map):
    assert spike_map.size_x() == 12


def test_size_y(spike_map):
    assert spike_map.size_y() == 11


def test_size_z(spike_map):
    assert spike_map.size_z() == 10


def test_default_beam():
    assert Map2D1.default_beam() == Gaussian2D1


def test_numpy_to_fits():
    x = np.arange(3)
    c = Map2D1.numpy_to_fits(x)
    assert c.x == 2 and c.y == 1 and c.z == 0
    c = Map2D1.numpy_to_fits([1, 2, 3])
    assert c.x == 3 and c.y == 2 and c.z == 1


def test_fits_to_numpy():
    x = Coordinate2D1([1, 2, 3])
    c = Map2D1.fits_to_numpy(x)
    assert c == [3.0, 2.0, 1.0]


def test_pixel_volume(initialized_spike_map):
    assert initialized_spike_map.pixel_volume == 1 * units.Unit('arcsec2 um')


def test_reference(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.reference == Coordinate2D1([0, 0] * arcsec, 0 * um)
    reference = Coordinate2D1([1, 2] * arcsec, 3 * um)
    m.reference = reference
    assert m.reference == reference


def test_reference_index(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.reference_index == Coordinate2D1([0, 0, 0])
    c = Coordinate2D1([1, 2, 3])
    m.reference_index = c
    assert m.reference_index == c


def test_reset_filtering(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.filter_fwhm = Coordinate2D1([1, 1] * arcsec, 1 * um)
    m.correcting_fwhm = Coordinate2D1([1, 1] * arcsec, 1 * um)
    m.reset_filtering()
    assert np.all(m.filter_fwhm.is_nan())
    assert np.all(m.correcting_fwhm.is_nan())


def test_is_filtered(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.is_filtered()
    m.reset_filtering()
    assert not m.is_filtered()


def test_is_corrected(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert not m.is_corrected()
    m.correcting_fwhm = Coordinate2D1([1, 1] * arcsec, 1 * um)
    assert m.is_corrected()


def test_set_correcting_fwhm(spike_map):
    m = spike_map.copy()
    m.set_correcting_fwhm(Coordinate2D1([1, 1] * arcsec, 1 * um))
    assert m.correcting_fwhm == Coordinate2D1([1, 1] * arcsec, 1 * um)


def test_set_filtering(spike_map):
    m = spike_map.copy()
    m.set_filtering(Coordinate2D1([1, 1] * arcsec, 1 * um))
    assert m.filter_fwhm == Coordinate2D1([1, 1] * arcsec, 1 * um)


def test_set_grid(initialized_spike_map):
    m = initialized_spike_map.copy()
    grid = m.grid.copy()
    smoothing_beam = m.smoothing_beam.copy()
    grid.resolution = Coordinate2D1([2, 2] * arcsec, 2 * um)
    m.set_grid(grid)
    assert m.grid is grid
    assert m.smoothing_beam != smoothing_beam
    m.smoothing_beam = None
    m.set_grid(grid)
    assert m.smoothing_beam == m.get_pixel_smoothing()
    grid = m.grid.copy()
    grid.resolution = Coordinate2D1([1, 1] * arcsec, 1 * um)
    m.smoothing_beam = Gaussian2D1(x_fwhm=10 * arcsec, y_fwhm=10 * arcsec,
                                   z_fwhm=10 * um)
    m.set_grid(grid)
    assert np.isclose(m.smoothing_beam.x_fwhm, 9.8219056 * arcsec, atol=1e-5)
    assert np.isclose(m.smoothing_beam.y_fwhm, 9.8219056 * arcsec, atol=1e-5)
    assert np.isclose(m.smoothing_beam.z_fwhm, 9.8219056 * um, atol=1e-5)


def test_set_resolution(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.set_resolution(Coordinate2D1([2, 2] * arcsec, 2 * um), redo=True)
    assert np.isclose(m.smoothing_beam.x_fwhm, 1.878875 * arcsec, atol=1e-5)
    assert np.isclose(m.smoothing_beam.y_fwhm, 1.878875 * arcsec, atol=1e-5)
    assert np.isclose(m.smoothing_beam.z_fwhm, 1.878875 * um, atol=1e-5)
    m.smoothing_beam = None
    m.set_resolution(Coordinate2D1([2, 2] * arcsec, 2 * um))
    assert isinstance(m.smoothing_beam, Gaussian2D1)
    assert np.isclose(m.smoothing_beam.x_fwhm, 1.878875 * arcsec, atol=1e-5)
    assert np.isclose(m.smoothing_beam.y_fwhm, 1.878875 * arcsec, atol=1e-5)
    assert np.isclose(m.smoothing_beam.z_fwhm, 1.878875 * um, atol=1e-5)


def test_set_underlying_beam(spike_map):
    m = spike_map.copy()
    g = Gaussian2D1(x_fwhm=1 * arcsec, y_fwhm=2 * arcsec, z_fwhm=3 * um)
    m.set_underlying_beam(g)
    assert m.underlying_beam == g
    m.underlying_beam = None
    m.set_underlying_beam(g.extent())
    assert isinstance(m.underlying_beam, Gaussian2D1)
    assert m.underlying_beam == g


def test_set_smoothing(spike_map):
    m = spike_map.copy()
    g = Gaussian2D1(x_fwhm=1 * arcsec, y_fwhm=2 * arcsec, z_fwhm=3 * um)
    m.set_smoothing(g)
    assert m.smoothing_beam == g
    m.smoothing_beam = None
    m.set_smoothing(g.extent())
    assert isinstance(m.smoothing_beam, Gaussian2D1)
    assert m.smoothing_beam == g


def test_set_image(spike_map):
    m = spike_map.copy()
    data = np.zeros((4, 5, 6))
    m.set_image(data)
    assert m.shape == (4, 5, 6)


def test_set_z_display_grid_unit(spike_map):
    m = spike_map.copy()
    m.set_z_display_grid_unit('K')
    assert m.z_display_grid_unit == 1 * units.Unit('K')
    m.set_z_display_grid_unit(arcsec)
    assert m.z_display_grid_unit == 1 * arcsec
    m.set_z_display_grid_unit(2 * um)
    assert m.z_display_grid_unit == 2 * um
    m.set_z_display_grid_unit(None)
    assert m.z_display_grid_unit == 2 * um
    with pytest.raises(ValueError) as err:
        m.set_z_display_grid_unit(1)
    assert 'Unit must be' in str(err.value)


def test_get_z_display_grid_unit(spike_map):
    m = spike_map.copy()
    assert m.get_z_display_grid_unit() == 'pixel'
    m.set_z_display_grid_unit('arcsec')
    assert m.get_z_display_grid_unit() == arcsec


def test_get_z_default_grid_unit(spike_map, initialized_spike_map):
    m = spike_map.copy()
    m.grid = None
    assert m.get_z_default_grid_unit() == 'pixel'
    assert initialized_spike_map.get_z_default_grid_unit() == 'um'


def test_get_volume(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.get_volume() == 1320 * units.Unit('arcsec2 um')


def test_get_image_beam_volume(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert np.isclose(m.get_image_beam_volume(),
                      10.64909578 * units.Unit('arcsec2 um'), atol=1e-5)
    m.smoothing_beam = None
    assert np.isclose(m.get_image_beam_volume(),
                      9.64909578 * units.Unit('arcsec2 um'), atol=1e-5)
    m.underlying_beam = None
    assert m.get_image_beam_volume() == 0


def test_get_filter_area(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert np.isclose(m.get_filter_area(), 1.13309004 * arcsec * arcsec)
    m.filter_fwhm = None
    assert m.get_filter_area() == 0 * units.Unit('degree2')


def test_get_filter_volume(initialized_spike_map):
    assert np.isclose(initialized_spike_map.get_filter_volume(),
                      1.20613697 * units.Unit('arcsec2 um'), atol=1e-5)


def test_get_filter_correction_factor(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert np.isclose(m.get_filter_correction_factor(), 4.9204709, atol=1e-5)
    m.underlying_beam = None
    assert np.isclose(m.get_filter_correction_factor(), 0.5467190, atol=1e-5)
    m.smoothing_beam = None
    assert m.get_filter_correction_factor() == 1
    m.filter_fwhm.zero()
    assert m.get_filter_correction_factor() == 1
    m.filter_fwhm = Coordinate2D1([1, 1] * arcsec, 1 * um)
    assert np.isclose(m.get_filter_correction_factor(
        underlying_fwhm=Coordinate2D1([2, 2] * arcsec, 2 * um)), 9)
    m.filter_fwhm.nan()
    assert m.get_filter_correction_factor() == 1


def test_get_pixel_smoothing(initialized_spike_map):
    g = initialized_spike_map.get_pixel_smoothing()
    assert isinstance(g, Gaussian2D1)
    assert np.isclose(g.x_fwhm, 0.939437 * arcsec, atol=1e-5)
    assert np.isclose(g.y_fwhm, 0.939437 * arcsec, atol=1e-5)
    assert np.isclose(g.z_fwhm, 0.939437 * um, atol=1e-5)


def test_get_resolution(initialized_spike_map):
    assert initialized_spike_map.get_resolution() == Coordinate2D1(
        [1, 1] * arcsec, 1 * um)


def test_get_anti_aliasing_beam_image_for(initialized_spike_map):
    b = initialized_spike_map.get_anti_aliasing_beam_image_for(
        initialized_spike_map)
    assert b is None


def test_get_anti_aliasing_beam_for(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.set_resolution(Coordinate2D1([5, 5] * arcsec, 5 * um))
    m.smoothing_beam = Gaussian2D1(x_fwhm=1 * arcsec, y_fwhm=1 * arcsec,
                                   z_fwhm=1 * um)
    b = m.get_anti_aliasing_beam_for(m)
    assert isinstance(b, Gaussian2D1)
    assert np.isclose(b.x_fwhm, 4.589505 * arcsec, atol=1e-5)
    assert np.isclose(b.y_fwhm, 4.589505 * arcsec, atol=1e-5)
    assert np.isclose(b.z_fwhm, 4.589505 * um, atol=1e-5)


def test_get_index_transform_to(initialized_spike_map):
    m = initialized_spike_map.copy()
    m2 = m.copy()
    m.set_resolution(Coordinate2D1([2, 2] * arcsec, 2 * um))
    inds = m.get_index_transform_to(m2)
    assert inds.span == Coordinate2D1([22, 20, 18])
    assert inds.min == Coordinate2D1([0, 0, 0])


def test_get_info(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.get_info() == [
        'Map information:',
        'Image Size: 12x11x10 pixels (1.000 x 1.000 arcsec) x 1.00000 um.',
        'Coordinate2D: x=0.0 arcsec y=0.0 arcsec, z=0.0 um\n'
        'Projection: Cartesian ()\n'
        'Grid Spacing: (1.000 x 1.000 arcsec) x 1.00000 um\n'
        'Reference Pixel: x=0.0 y=0.0, z=0.0 C-style, 0-based',
        'Instrument PSF: 2.00000 arcsec, 2.00000 um (includes pixelization)',
        'Image resolution: 2.20965 arcsec, 2.00000 um (includes smoothing)']
    m2 = Map2D1()
    assert m2.get_info() == [
        'Map information:',
        'Image Size: 0 pixels (1.000 x 1.000) x 1.00000.',
        'Coordinate2D: x=0.0 y=0.0, z=0.0\n'
        'Projection: Cartesian ()\n'
        'Grid Spacing: (1.000 x 1.000) x 1.00000\n'
        'Reference Pixel: x=0.0 y=0.0, z=0.0 C-style, 0-based',
        'Instrument PSF: 0.00000, 0.00000 (includes pixelization)',
        'Image resolution: 0.00000, 0.00000 (includes smoothing)']


def test_get_points_per_smoothing_beam(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smoothing_beam = Gaussian2D1(x_fwhm=5 * arcsec, y_fwhm=5 * arcsec,
                                   z_fwhm=5 * um)
    assert np.isclose(m.get_points_per_smoothing_beam(), 150.767122, atol=1e-5)
    m.smoothing_beam = None
    assert m.get_points_per_smoothing_beam() == 1


def test_copy_properties_from(initialized_spike_map):
    m = Map2D1()
    m.copy_properties_from(initialized_spike_map)
    assert m.display_grid_unit == 1 * arcsec


def test_merge_properties_from(initialized_spike_map):
    m = initialized_spike_map.copy()
    m2 = initialized_spike_map.copy()
    m.smoothing_beam = None
    m2.filter_fwhm.scale(0.5)
    m.merge_properties_from(m2)
    assert m.smoothing_beam == m2.smoothing_beam
    assert m.filter_fwhm == m2.filter_fwhm
    m2.smoothing_beam.scale(2)
    m.merge_properties_from(m2)
    assert m.smoothing_beam == m2.smoothing_beam


def test_add_smoothing(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smoothing_beam = None
    g = Gaussian2D1(x_fwhm=1 * arcsec, y_fwhm=1 * arcsec, z_fwhm=1 * um)
    m.add_smoothing(g)
    assert isinstance(m.smoothing_beam, Gaussian2D1) and m.smoothing_beam == g
    m.add_smoothing(g)
    assert np.isclose(m.smoothing_beam.x_fwhm, np.sqrt(2) * arcsec)
    assert np.isclose(m.smoothing_beam.y_fwhm, np.sqrt(2) * arcsec)
    assert np.isclose(m.smoothing_beam.z_fwhm, np.sqrt(2) * um)
    m.smoothing_beam = None
    m.add_smoothing(Coordinate2D1([1, 1] * arcsec, 1 * um))
    assert isinstance(m.smoothing_beam, Gaussian2D1) and m.smoothing_beam == g


def test_filter_beam_correct(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.underlying_beam = None
    m.filter_beam_correct()
    assert np.all(m.correcting_fwhm.is_null())
    m.underlying_beam = Gaussian2D1(
        x_fwhm=1 * arcsec, y_fwhm=1 * arcsec, z_fwhm=1 * um)
    m.filter_beam_correct()
    assert m.correcting_fwhm.x == 1 * arcsec
    assert m.correcting_fwhm.y == 1 * arcsec
    assert m.correcting_fwhm.z == 1 * um


def test_undo_filter_correct(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.undo_filter_correct()
    assert not m.is_corrected()
    m.underlying_beam = Gaussian2D1(
        x_fwhm=1 * arcsec, y_fwhm=1 * arcsec, z_fwhm=1 * um)
    m.filter_beam_correct()
    assert m.is_corrected()
    assert np.isclose(m.data.max(), 1.093438, atol=1e-5)
    m.undo_filter_correct(reference=np.zeros(m.shape))
    assert not m.is_corrected()
    assert np.isclose(m.data.max(), 1)
    m.filter_beam_correct()
    m.undo_filter_correct(reference=Image2D1(data=np.zeros(m.shape)))
    assert np.isclose(m.data.max(), 1)
    m.filter_beam_correct()
    m.undo_filter_correct()
    assert np.isclose(m.data.max(), 1)


def test_update_filtering(initialized_spike_map):
    m = initialized_spike_map.copy()
    fwhm = Coordinate2D1([1, 2] * arcsec, 3 * um)
    m.filter_fwhm.nan()
    m.update_filtering(fwhm)
    assert m.filter_fwhm == fwhm
    fwhm = Coordinate2D1([0.5, 2.5] * arcsec, 1.5 * um)
    m.update_filtering(fwhm)
    assert m.filter_fwhm.x == 0.5 * arcsec
    assert m.filter_fwhm.y == 2 * arcsec
    assert m.filter_fwhm.z == 1.5 * um


def test_parse_coordinate_info():
    m = Map2D1()
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
    m.parse_coordinate_info(h)
    assert isinstance(m.grid, SphericalGrid2D1)


def test_parse_corrected_beam(initialized_spike_map):
    m = initialized_spike_map.copy()
    h = fits.Header()
    m.parse_corrected_beam(h)
    assert np.all(m.correcting_fwhm.is_nan())
    h['CBMAJ'] = 4, 'Beam major axis (arcsec)'
    h['CBMIN'] = 5, 'Beam minor axis (arcsec)'
    h['CBPA'] = 25, 'Beam position angle (deg)'
    h['CB1D'] = (1, '(um)')
    m.parse_corrected_beam(h)
    assert m.correcting_fwhm.x == 5 * arcsec
    assert m.correcting_fwhm.y == 4 * arcsec
    assert m.correcting_fwhm.z == 1 * um


def test_parse_smoothing_beam(initialized_spike_map):
    pix = units.Unit('pixel')
    m = initialized_spike_map.copy()
    h = fits.Header()
    s = m.smoothing_beam.copy()
    m.parse_smoothing_beam(h)
    assert m.smoothing_beam == s

    h['SBMAJ'] = 4, 'Beam major axis (arcsec)'
    h['SBMIN'] = 5, 'Beam minor axis (arcsec)'
    h['SBPA'] = 25, 'Beam position angle (deg)'
    h['SB1D'] = (1, '(um)')
    m.parse_smoothing_beam(h)
    assert m.smoothing_beam.x_fwhm == 5 * arcsec
    assert m.smoothing_beam.y_fwhm == 4 * arcsec
    assert m.smoothing_beam.z_fwhm == 1 * um

    m = Map2D1()
    m.parse_smoothing_beam(fits.Header())
    assert np.isclose(m.smoothing_beam.x_fwhm, 0.939437 * pix)
    assert np.isclose(m.smoothing_beam.y_fwhm, 0.939437 * pix)
    assert np.isclose(m.smoothing_beam.z_fwhm, 0.939437 * pix)


def test_filter_beam(initialized_spike_map):
    m = initialized_spike_map.copy()
    h = fits.Header()
    m.parse_filter_beam(h)
    assert np.all(m.filter_fwhm.is_nan())

    h['XBMAJ'] = 4, 'Beam major axis (arcsec)'
    h['XBMIN'] = 5, 'Beam minor axis (arcsec)'
    h['XBPA'] = 25, 'Beam position angle (deg)'
    h['XB1D'] = (1, '(um)')
    m.parse_filter_beam(h)
    assert m.filter_fwhm.x == 5 * arcsec
    assert m.filter_fwhm.y == 4 * arcsec
    assert m.filter_fwhm.z == 1 * um

    m = Map2D1()
    m.parse_filter_beam(fits.Header())
    assert np.all(m.filter_fwhm.is_nan())


def test_parse_underlying_beam(spike_map):
    m = spike_map.copy()
    m.grid = FlatGrid2D1()
    h = fits.Header()
    m.underlying_beam = None
    m.parse_underlying_beam(h)
    assert isinstance(m.underlying_beam, Gaussian2D1)
    assert m.underlying_beam.x_fwhm == 0
    assert m.underlying_beam.z_fwhm == 0
    assert m.underlying_beam.x_fwhm.unit == 'pixel'
    assert m.underlying_beam.z_fwhm.unit == 'pixel'
    m.grid.resolution = Coordinate2D1([1, 1] * arcsec, 1 * um)
    m.display_grid_unit = 1 * units.Unit('degree')
    m.z_display_grid_unit = 1 * units.Unit('m')
    m.parse_underlying_beam(h)
    assert m.underlying_beam.x_fwhm == 0
    assert m.underlying_beam.z_fwhm == 0
    assert m.underlying_beam.x_fwhm.unit == 'degree'
    assert m.underlying_beam.z_fwhm.unit == 'm'

    m.underlying_beam = None
    m.smoothing_beam = None
    h['RESOLUTN'] = 3, '(arcsec)'
    h['RESOLUTZ'] = 2, '(um)'
    m.parse_underlying_beam(h)
    assert isinstance(m.underlying_beam, Gaussian2D1)
    assert np.isclose(m.underlying_beam.x_fwhm.value, 3 / 3600)
    assert np.isclose(m.underlying_beam.y_fwhm.value, 3 / 3600)
    assert np.isclose(m.underlying_beam.z_fwhm.value, 2e-6)
    m.smoothing_beam = Gaussian2D1(x_fwhm=1 * arcsec, y_fwhm=1 * arcsec,
                                   z_fwhm=1 * um)
    m.parse_underlying_beam(h)
    assert np.isclose(m.underlying_beam.x_fwhm.value, 0.00078567)
    assert np.isclose(m.underlying_beam.y_fwhm.value, 0.00078567)
    assert np.isclose(m.underlying_beam.z_fwhm.value, 1e-6)
    m.smoothing_beam.set_xyz_fwhm(20 * arcsec, 20 * arcsec, 20 * um)
    m.parse_underlying_beam(h)
    assert m.underlying_beam.x_fwhm == 0
    assert m.underlying_beam.z_fwhm == 0
    h['BMAJ'] = 4
    h['BMIN'] = 3
    h['B1D'] = 5
    m.smoothing_beam = None
    m.parse_underlying_beam(h)
    assert m.underlying_beam.x_fwhm == 4 * arcsec
    assert m.underlying_beam.y_fwhm == 3 * arcsec
    assert m.underlying_beam.z_fwhm == 5 * um

    h['BEAM'] = 5, '(arcmin)'
    h['BEAMZ'] = 6, '(um)'
    m.parse_underlying_beam(h)
    assert m.underlying_beam.x_fwhm.value == 5 / 60
    assert m.underlying_beam.z_fwhm.value == 6e-6

    h['IBMAJ'] = 8
    h['IBMIN'] = 7
    h['IB1D'] = 6
    m.parse_underlying_beam(h)
    assert m.underlying_beam.x_fwhm == 8 * arcsec
    assert m.underlying_beam.y_fwhm == 7 * arcsec
    assert m.underlying_beam.z_fwhm == 6 * um


def test_edit_header(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.filter_fwhm = Coordinate2D1([2, 1] * arcsec, 3 * um)
    m.correcting_fwhm = Coordinate2D1([5, 4] * arcsec, 6 * um)
    h = fits.Header()
    m.edit_header(h)
    expected = {'CRPIX1': 1.0,
                'CRPIX2': 1.0,
                'CRVAL1': 0.0,
                'CRVAL2': 0.0,
                'CDELT1': 1.0,
                'CDELT2': 1.0,
                'CUNIT1': 'arcsec',
                'CUNIT2': 'arcsec',
                'CTYPE3': 'z',
                'CUNIT3': 'um',
                'CRPIX3': 1.0,
                'CRVAL3': 0.0,
                'CDELT3': 1.0,
                'BNAM': 'image',
                'BMAJ': 2.209647573847605,
                'BMIN': 2.209647573847605,
                'BPA': 0.0,
                'B1D': 2.209647573847605,
                'RESOLUTN': 2.209647573847605,
                'RESOLUTZ': 2.209647573847605,
                'IBNAM': 'instrument',
                'IBMAJ': 2.0,
                'IBMIN': 2.0,
                'IBPA': 0.0,
                'IB1D': 2.0,
                'SBNAM': 'smoothing',
                'SBMAJ': 0.9394372786996513,
                'SBMIN': 0.9394372786996513,
                'SBPA': 0.0,
                'SB1D': 0.9394372786996515,
                'SMOOTH': 0.9394372786996513,
                'SMOOTHZ': 0.9394372786996515,
                'XBNAM': 'Extended Structure Filter',
                'XBMAJ': 2.0,
                'XBMIN': 1.0,
                'XBPA': 0.0,
                'XB1D': 3.0,
                'CBNAM': 'Peak Corrected',
                'CBMAJ': 5.0,
                'CBMIN': 4.0,
                'CBPA': 0.0,
                'CB1D': 6.0,
                'SMTHRMS': True,
                'DATAMIN': 0.0,
                'DATAMAX': 1.0,
                'BZERO': 0.0,
                'BSCALE': 1.0,
                'BUNIT': 'Jy'}
    for key, value in expected.items():
        compare = h[key]
        if not isinstance(compare, str):
            assert np.isclose(compare, value, rtol=0.1)
        else:
            assert compare == value


def test_count_beams(initialized_spike_map):
    assert np.isclose(initialized_spike_map.count_beams(), 123.95418606259405,
                      atol=1e-5)


def test_count_independent_points(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.count_independent_points(100 * units.Unit('arcsec2 um')) == 2
    m.filter_fwhm.nan()
    assert m.count_independent_points(100 * units.Unit('arcsec2 um')) == 11
    m.smoothing_beam = None
    assert m.count_independent_points(100 * units.Unit('arcsec2 um')) == 0


def test_nearest_to_offset(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.nearest_to_offset([1.5, 1.4, 2.5]) == (2, 1, 3)


def test_convert_range_value_to_index(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.verbose = True
    i = m.convert_range_value_to_index([1.5, 2.4, 2.5])
    assert isinstance(i, np.ndarray) and i.dtype in [int, np.int64]
    assert np.allclose(i, [2, 2, 3])


def test_crop(initialized_spike_map):
    m = initialized_spike_map.copy()
    ranges = [[0, 2], [1, 4], [2, 7]]
    m.crop(ranges)
    assert m.shape == (6, 4, 3)
    ranges = Coordinate2D1(ranges)
    m = initialized_spike_map.copy()
    m.crop(ranges)
    assert m.shape == (6, 4, 3)
    m.basis._data = None
    m.crop(ranges)
    assert m.size == 0


def test_smooth_to(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smooth_to(Coordinate2D1([5, 5] * arcsec, 5 * um))
    max_value = m.data.max()
    assert np.isclose(max_value, 0.00735, atol=1e-4)
    assert m.data[5, 6, 7] == max_value
    d0 = m.data.copy()
    m.smoothing_beam.scale(1000)
    m.smooth_to(Coordinate2D1([5, 5] * arcsec, 5 * um))
    assert np.allclose(m.data, d0)


def test_smooth_with_psf(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smooth_with_psf(Coordinate2D1([2, 2] * arcsec, 2 * um))
    assert np.isclose(m.data.max(), 0.103636, atol=1e-4)


def test_smooth(initialized_spike_map):
    m = initialized_spike_map.copy()
    ref = Coordinate2D1([1, 1, 1])
    beam_map = np.ones((3, 3, 3))
    m.smooth(beam_map, reference_index=ref)
    assert np.isclose(m.data.max(), 1 / 27, atol=1e-3)
    m = initialized_spike_map.copy()
    ref = Coordinate3D([1, 1, 1])
    m.smooth(beam_map, reference_index=ref)
    assert np.isclose(m.data.max(), 1 / 27, atol=1e-3)


def test_fast_smooth(initialized_spike_map):
    m = initialized_spike_map.copy()
    beam_map = np.ones((5, 5, 5))
    ref = Coordinate2D1([2, 2, 2])
    steps = Coordinate2D1([2, 2, 2])
    m.fast_smooth(beam_map, steps, reference_index=ref)
    assert np.isclose(m.data.max(), 0.0057, atol=1e-3)
    ref = Coordinate3D([2, 2, 2])
    steps = Coordinate3D([2, 2, 2])
    m = initialized_spike_map.copy()
    m.fast_smooth(beam_map, steps, reference_index=ref)
    assert np.isclose(m.data.max(), 0.0057, atol=1e-3)


def test_fft_filter_above(initialized_spike_map):
    m = initialized_spike_map.copy()
    weights = Image2D1(data=m.data * 0)
    m.fft_filter_above(Coordinate2D1([1, 1] * arcsec, 1 * um), weight=weights,
                       valid=np.full(m.shape, True))
    assert m.data.max() == 1
    m.fft_filter_above(Coordinate2D1([1, 1] * arcsec, 1 * um))
    assert np.isclose(m.data.max(), 0.297725, atol=1e-3)


def test_resample(initialized_spike_map):
    m = initialized_spike_map.copy()
    resolution = [0.5 * arcsec, 0.5 * arcsec, 0.5 * um]
    m.resample(resolution)
    assert m.data.shape == (20, 22, 24)
