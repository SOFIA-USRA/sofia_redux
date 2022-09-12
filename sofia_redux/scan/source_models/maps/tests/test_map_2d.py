# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.fits_properties.fits_properties import (
    FitsProperties)
from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.source_models.maps.map_2d import Map2D


arcsec = units.Unit('arcsec')
asec2 = arcsec * arcsec
ud = units.dimensionless_unscaled
pix = units.Unit('pixel')


@pytest.fixture
def spike():
    data = np.zeros((10, 11))
    data[5, 6] = 1.0
    return data


@pytest.fixture
def spike_image(spike):
    image = Image2D(data=spike.copy(), unit='Jy')
    return image


@pytest.fixture
def spike_map(spike_image):
    map2d = Map2D(data=spike_image)
    return map2d


@pytest.fixture
def initialized_spike_map(spike_map):
    map2d = spike_map.copy()
    map2d.set_resolution(1 * arcsec)
    map2d.smoothing_beam = map2d.get_pixel_smoothing()
    map2d.set_underlying_beam(2 * units.Unit('arcsec'))
    map2d.set_filtering(1 * units.Unit('arcsec'))
    map2d.set_filter_blanking(10.0)
    map2d.set_display_grid_unit('arcsec')
    return map2d


def test_class():
    assert Map2D.UNDERLYING_BEAM_FITS_ID == 'I'
    assert Map2D.SMOOTHING_BEAM_FITS_ID == 'S'
    assert Map2D.CORRECTED_BEAM_FITS_ID == 'C'
    assert Map2D.FILTER_BEAM_FITS_ID == 'X'


def test_init(spike_image):
    data = spike_image.copy()
    map2d = Map2D()
    assert map2d.data is None
    map2d = Map2D(data=data)
    assert map2d.unit == 1 * units.Unit('Jy')
    assert map2d.basis is data
    assert isinstance(map2d.grid, FlatGrid2D)
    assert isinstance(map2d.fits_properties, FitsProperties)
    assert map2d.display_grid_unit is None
    assert map2d.underlying_beam is None
    assert map2d.smoothing_beam is None
    assert np.isnan(map2d.filter_fwhm) and map2d.filter_fwhm.unit == 'degree'
    assert np.isnan(map2d.correcting_fwhm)
    assert map2d.correcting_fwhm.unit == 'degree'
    assert map2d.filter_blanking == np.inf
    assert map2d.reuse_index.size == 0


def test_copy(spike_map):
    m1 = spike_map
    m2 = m1.copy()
    assert m1 == m2
    m3 = m1.copy(with_contents=False)
    assert m1 != m3


def test_numpy_to_fits():
    c = Map2D.numpy_to_fits(np.asarray([1, 2]))
    assert isinstance(c, Coordinate2D)
    assert c.x == 2 and c.y == 1


def test_eq(spike_map):
    m = spike_map.copy()
    assert m == m
    assert m != 1
    m2 = m.copy()
    assert m2 == m
    m2.fits_properties = 1
    assert m2 != m
    m2 = m.copy()
    m2.correcting_fwhm = 1 * units.Unit('degree')
    assert m2 != m
    m2 = m.copy()
    m2.filter_fwhm = 1 * units.Unit('degree')
    assert m2 != m
    m2 = m.copy()
    m2.grid = None
    assert m2 != m
    m2 = m.copy()
    m2.display_grid_unit = units.Unit('m')
    assert m2 != m
    m2 = m.copy()
    m2.underlying_beam = 1
    assert m2 != m
    m2 = m.copy()
    m2.smoothing_beam = 1
    assert m2 != m
    m2 = m.copy()
    m.correcting_fwhm = Coordinate2D([1, 1], unit='arcsec')
    m2.correcting_fwhm = Coordinate2D([2, 2], unit='arcsec')
    assert m != m2
    m2 = m.copy()
    m.filter_fwhm = Coordinate2D([1, 1], unit='arcsec')
    m2.filter_fwhm = Coordinate2D([2, 2], unit='arcsec')
    assert m != m2


def test_pixel_area(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.pixel_area == 1 * asec2
    m.set_resolution(np.asarray([2, 3]) * arcsec)
    assert m.pixel_area == 6 * asec2


def test_reference(initialized_spike_map):
    m = initialized_spike_map.copy()
    c = Coordinate2D([1, 2])
    c0 = c.copy()
    c0.zero()
    assert m.reference == c0
    m.reference = c
    assert m.reference == c


def test_reference_index(initialized_spike_map):
    m = initialized_spike_map.copy()
    c = Coordinate2D([1, 2])
    c0 = c.copy()
    c0.zero()
    assert m.reference_index == c0
    m.reference_index = c
    assert m.reference_index == c


def test_projection(initialized_spike_map):
    m = initialized_spike_map.copy()
    projection = m.projection.copy()
    projection.reference = Coordinate2D([1, 1])
    assert m.projection != projection
    m.projection = projection
    assert m.projection == projection


def test_reset_processing(initialized_spike_map):
    m = initialized_spike_map.copy()
    fwhm1 = m.smoothing_beam.fwhm.copy()
    m.smoothing_beam.fwhm *= 2
    m.reset_processing()
    assert m.smoothing_beam.fwhm == fwhm1
    assert np.isnan(m.filter_fwhm)
    assert np.isnan(m.correcting_fwhm)
    assert np.isnan(m.filter_blanking)


def test_reset_smoothing(initialized_spike_map):
    m = initialized_spike_map.copy()
    fwhm1 = m.smoothing_beam.fwhm.copy()
    m.smoothing_beam.fwhm *= 3
    m.reset_smoothing()
    assert m.smoothing_beam.fwhm == fwhm1


def test_reset_filtering(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.reset_filtering()
    assert np.isnan(m.filter_fwhm)
    assert np.isnan(m.correcting_fwhm)
    assert np.isnan(m.filter_blanking)


def test_renew(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.renew()
    assert np.isnan(m.filter_fwhm)
    assert np.allclose(m.data, 0)


def test_add_proprietary_units(spike_map):
    m = spike_map.copy()
    m.local_units = {}
    m.alternate_unit_names = {}
    m.add_proprietary_units()
    assert 'beam' in m.local_units and 'pix' in m.local_units
    assert 'BEAM' in m.alternate_unit_names
    assert 'Pixels' in m.alternate_unit_names


def test_no_data(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.no_data()
    assert np.all(np.isnan(m.data))


def test_is_filtered():
    m = Map2D()
    assert not m.is_filtered()
    m.filter_fwhm = 1 * arcsec
    assert m.is_filtered()


def test_is_corrected():
    m = Map2D()
    assert not m.is_corrected()
    m.correcting_fwhm = 1 * arcsec
    assert m.is_corrected()


def test_is_filter_blanked():
    m = Map2D()
    assert not m.is_filter_blanked()
    m.filter_blanking = 1.0
    assert m.is_filter_blanked()


def test_set_correcting_fwhm():
    m = Map2D()
    m.set_correcting_fwhm(1 * arcsec)
    assert m.correcting_fwhm == 1 * arcsec


def test_set_default_unit(spike_map):
    m = spike_map.copy()
    m.local_units = {}
    m.set_default_unit()
    assert 'beam' in m.local_units


def test_set_unit(spike_map):
    m = spike_map.copy()
    m.set_unit('m')
    assert m.unit == 1 * units.Unit('m')
    assert m.get_image().unit == 1 * units.Unit('m')


def test_set_grid(initialized_spike_map):
    m = initialized_spike_map.copy()
    fwhm1 = m.smoothing_beam.fwhm
    new_grid = m.grid.copy()
    new_grid.set_resolution(2 * arcsec)
    m.set_grid(new_grid)
    assert np.isclose(m.smoothing_beam.fwhm, 2 * fwhm1)
    assert m.grid is new_grid
    m.smoothing_beam = None
    m.set_grid(new_grid)
    assert isinstance(m.smoothing_beam, Gaussian2D)
    assert np.isclose(m.smoothing_beam.fwhm, 2 * fwhm1)

    # Deconvolution and encompassing occurs
    new_grid.set_resolution(1 * arcsec)
    m.set_grid(new_grid)
    assert np.allclose(m.smoothing_beam.x_fwhm, 2 * fwhm1)
    assert np.allclose(m.smoothing_beam.y_fwhm, 1.627153 * arcsec, atol=1e-6)


def test_set_resolution(initialized_spike_map):
    m = initialized_spike_map.copy()
    c = Coordinate2D([1, 2], unit=arcsec)
    f1 = m.smoothing_beam.fwhm
    m.set_resolution(c)

    assert m.get_resolution() == c
    assert np.isclose(m.smoothing_beam.x_fwhm, f1 * 2)
    assert np.isclose(m.smoothing_beam.y_fwhm, f1)
    # Rotation occurred because y-fwhm > x-fwhm
    assert m.smoothing_beam.theta == 90 * units.Unit('degree')

    m.set_resolution(1 * arcsec)
    assert np.allclose(m.get_resolution().coordinates, 1 * arcsec)
    # Smoothing beam is not encompassed
    assert np.isclose(m.smoothing_beam.x_fwhm, f1 * 2)
    assert np.isclose(m.smoothing_beam.y_fwhm, f1)

    m.set_resolution(1 * arcsec, redo=True)
    # Test deconvolution occurred
    assert np.isclose(m.smoothing_beam.y_fwhm, f1)
    assert np.isclose(m.smoothing_beam.x_fwhm, 1.627153 * arcsec, atol=1e-6)

    m.smoothing_beam = None
    m.set_resolution(1 * arcsec)
    assert isinstance(m.smoothing_beam, Gaussian2D)
    assert np.isclose(m.smoothing_beam.x_fwhm, f1)
    assert np.isclose(m.smoothing_beam.y_fwhm, f1)


def test_set_underlying_beam():
    m = Map2D()
    psf = Gaussian2D(x_fwhm=2.0 * arcsec, y_fwhm=1.0 * arcsec)
    m.set_underlying_beam(psf)
    assert m.underlying_beam == psf and m.underlying_beam is not psf
    m.set_underlying_beam(2.0 * arcsec)
    assert m.underlying_beam == Gaussian2D(
        x_fwhm=2.0 * arcsec, y_fwhm=2.0 * arcsec)


def test_set_pixel_smoothing(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smoothing_beam = None
    m.set_pixel_smoothing()
    assert isinstance(m.smoothing_beam, Gaussian2D)
    factor = Gaussian2D.FWHM_TO_SIZE
    assert np.isclose(m.smoothing_beam.x_fwhm, 1 * arcsec / factor)
    assert np.isclose(m.smoothing_beam.y_fwhm, 1 * arcsec / factor)
    assert m.smoothing_beam.theta == 0


def test_set_smoothing():
    m = Map2D()
    psf = Gaussian2D(x_fwhm=2.0 * arcsec, y_fwhm=1.0 * arcsec)
    m.set_smoothing(psf)
    assert m.smoothing_beam == psf and m.smoothing_beam is not psf
    m.set_smoothing(2.0 * arcsec)
    assert m.smoothing_beam == Gaussian2D(
        x_fwhm=2.0 * arcsec, y_fwhm=2.0 * arcsec)


def test_set_filtering():
    m = Map2D()
    m.set_filtering(1 * arcsec)
    assert m.filter_fwhm == 1 * arcsec


def test_set_filter_blanking():
    m = Map2D()
    m.set_filter_blanking(None)
    assert np.isnan(m.filter_blanking)
    m.set_filter_blanking(1)
    assert m.filter_blanking == 1


def test_set_image():
    m = Map2D()
    image = np.ones((10, 10), dtype=float)
    m.set_image(image)
    assert np.allclose(m.data, image)
    assert np.allclose(m.basis.data, image)


def test_set_display_grid_unit():
    m = Map2D()
    for value in ['degree', units.Unit('degree'), 1 * units.Unit('degree')]:
        m.set_display_grid_unit(value)
        assert m.display_grid_unit == 1 * units.Unit('degree')
    m = Map2D()
    m.set_display_grid_unit(None)
    assert m.display_grid_unit is None
    with pytest.raises(ValueError) as err:
        m.set_display_grid_unit(1)
    assert "Unit must be" in str(err.value)


def test_get_display_grid_unit():
    m = Map2D()
    m.grid = None
    assert m.display_grid_unit is None
    assert m.get_display_grid_unit() == 1 * units.Unit('pixel')
    m.set_display_grid_unit('degree')
    assert m.get_display_grid_unit() == 1 * units.Unit('degree')


def test_get_area(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert m.get_area() == 110 * asec2


def test_get_default_grid_unit():
    m = Map2D()
    m.grid = None
    assert m.get_display_grid_unit() == 1 * units.Unit('pixel')
    m = Map2D()
    assert m.get_display_grid_unit() == 1 * units.Unit('pixel')


def test_get_image(initialized_spike_map, capsys):
    m = initialized_spike_map.copy()
    assert m.get_image() is m.basis
    m.get_image(dtype=int)
    assert "Cannot change base image type" in capsys.readouterr().err


def test_get_image_beam():
    m = Map2D()
    assert m.get_image_beam() is None
    psf1 = Gaussian2D(x_fwhm=1 * ud, y_fwhm=1 * ud)
    psf2 = Gaussian2D(x_fwhm=2 * ud, y_fwhm=2 * ud)
    m.smoothing_beam = psf1.copy()
    assert m.get_image_beam() == psf1
    m.smoothing_beam = None
    m.underlying_beam = psf2.copy()
    assert m.get_image_beam() == psf2
    m.smoothing_beam = psf1
    m.underlying_beam = psf2
    psf = m.get_image_beam()
    assert np.isclose(psf.fwhm, 2.236068, atol=1e-6)


def test_get_image_beam_area():
    m = Map2D()
    m.smoothing_beam = Gaussian2D(x_fwhm=1 * ud, y_fwhm=1 * ud)
    m.underlying_beam = Gaussian2D(x_fwhm=2 * ud, y_fwhm=2 * ud)
    assert np.isclose(m.get_image_beam_area(), m.get_image_beam().area)
    m.smoothing_beam = None
    assert np.isclose(m.get_image_beam_area(), m.underlying_beam.area)
    m.underlying_beam = None
    assert m.get_image_beam_area() == 0


def test_get_filter_area():
    m = Map2D()
    assert np.isnan(m.get_filter_area())
    m.filter_fwhm = None
    assert m.get_filter_area() == 0
    m.filter_fwhm = 1 * arcsec
    assert np.isclose(m.get_filter_area(), 1.133090 * asec2, atol=1e-6)


def test_get_filter_correction_factor():
    m = Map2D()
    m.filter_fwhm = np.nan
    assert m.get_filter_correction_factor() == 1
    m.filter_fwhm = 0.0
    assert m.get_filter_correction_factor() == 1
    m.filter_fwhm = 1.0
    assert m.get_filter_correction_factor(underlying_fwhm=1.0) == 2
    m.filter_fwhm = 1 * arcsec
    assert m.get_filter_correction_factor() == 1.0

    m.underlying_beam = Gaussian2D(x_fwhm=2 * arcsec, y_fwhm=2 * arcsec)
    m.smoothing_beam = Gaussian2D(x_fwhm=1 * arcsec, y_fwhm=1 * arcsec)
    assert m.get_filter_correction_factor() == 2.5


def test_get_pixel_smoothing(initialized_spike_map):
    m = initialized_spike_map.copy()
    psf = m.get_pixel_smoothing()
    factor = Gaussian2D.FWHM_TO_SIZE
    assert np.isclose(psf.x_fwhm, 1 * arcsec / factor)
    assert np.isclose(psf.y_fwhm, 1 * arcsec / factor)
    assert psf.theta == 0


def test_get_resolution(initialized_spike_map):
    c = Coordinate2D([1, 1], unit=arcsec)
    assert initialized_spike_map.get_resolution() == c


def test_get_anti_aliasing_beam_for(initialized_spike_map):
    m1 = initialized_spike_map.copy()
    m2 = m1.copy()
    m2.smoothing_beam.fwhm = 2 * arcsec
    aa_beam = m1.get_anti_aliasing_beam_for(m2)
    assert aa_beam is None  # because smoothing beam is encompassing pixels
    m2.smoothing_beam.fwhm = 0.5 * arcsec
    aa_beam = m1.get_anti_aliasing_beam_for(m2)
    assert np.isclose(aa_beam.fwhm, 0.795325 * arcsec, atol=1e-6)
    m2.smoothing_beam = None
    aa_beam = m1.get_anti_aliasing_beam_for(m2)
    assert aa_beam == m1.get_pixel_smoothing()


def test_get_anti_aliasing_beam_image_for(initialized_spike_map):
    m1 = initialized_spike_map.copy()
    m2 = m1.copy()
    m2.smoothing_beam.fwhm = 2 * arcsec
    assert m1.get_anti_aliasing_beam_image_for(m2) is None
    m2.smoothing_beam.fwhm = 0.5 * arcsec
    beam_image = m1.get_anti_aliasing_beam_image_for(m2)
    assert beam_image.shape == (7, 7)
    assert np.allclose(beam_image[2:5, 2:5],
                       [[0, 0.0125, 0],
                        [0.0125, 1, 0.0125],
                        [0, 0.0125, 0]], atol=1e-3)
    m2.smoothing_beam.fwhm = 4 * arcsec
    beam_image = m1.get_anti_aliasing_beam_image_for(m2)
    assert beam_image is None


def test_get_index_transform_to(initialized_spike_map):
    m1 = initialized_spike_map.copy()
    m2 = m1.copy()
    m2.set_resolution(2 * arcsec)
    indices = m1.get_index_transform_to(m2)
    expected = np.stack([x.ravel() for x in np.indices(
        m1.shape)])[::-1] / 2
    assert np.allclose(expected, indices.coordinates)


def test_get_table_entry(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert np.isclose(m.get_table_entry('beams'), 19.883015, atol=1e-6)
    assert m.get_table_entry('min') == 0
    assert m.get_table_entry('max') == 1
    assert m.get_table_entry('unit') == 'Jy'
    assert np.isclose(m.get_table_entry('mean'), 1 / 110)
    assert m.get_table_entry('median') == 0
    assert m.get_table_entry('rms') == 0
    assert m.get_table_entry('foo') is None


def test_get_info(initialized_spike_map):
    m = initialized_spike_map.copy()
    result = m.get_info()
    assert result == [
        'Map information:',
        'Image Size: 11x10 pixels (1.0 x 1.0 arcsec).',
        ('Coordinate2D: x=0.0 y=0.0\nProjection: Cartesian ()\n'
         'Grid Spacing: 1.0 x 1.0 arcsec\nReference Pixel: x=0.0 y=0.0 '
         'C-style, 0-based'),
        'Instrument PSF: 2.00000 arcsec (includes pixelization)',
        'Image resolution: 2.20965 arcsec (includes smoothing)'
    ]
    expected = result.copy()
    m.display_grid_unit = None
    assert m.get_info() == expected


def test_get_points_per_smoothing_beam(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smoothing_beam.fwhm = 2 * arcsec
    assert np.isclose(m.get_points_per_smoothing_beam(), 4.532360, atol=1e-6)
    m.smoothing_beam = None
    assert m.get_points_per_smoothing_beam() == 1


def test_copy_processing_from(spike_map):
    m1 = spike_map.copy()
    m2 = spike_map.copy()
    m2.underlying_beam = 1
    m2.smoothing_beam = 2
    m2.filter_fwhm = 3 * units.Unit('degree')
    m2.filter_blanking = 4
    m2.correcting_fwhm = 5 * units.Unit('degree')
    assert m1 != m2
    m1.copy_processing_from(m2)
    assert m1 == m2


def test_copy_properties_from(spike_map):
    m1 = Map2D()
    m2 = spike_map.copy()
    m1.copy_properties_from(m2)
    assert m1.fits_properties == m2.fits_properties
    assert np.isclose(m1.filter_fwhm, m2.filter_fwhm, equal_nan=True)
    assert np.isclose(m1.correcting_fwhm, m2.correcting_fwhm, equal_nan=True)
    assert np.isclose(m1.filter_blanking, m2.filter_blanking, equal_nan=True)
    assert m1.grid == m2.grid
    assert m1.display_grid_unit == m2.display_grid_unit
    assert m1.underlying_beam == m2.underlying_beam
    assert m1.smoothing_beam == m2.smoothing_beam
    m2.fits_properties = None
    m1.copy_properties_from(m2)
    assert m1.fits_properties is None


def test_merge_properties_from(initialized_spike_map):
    m1 = initialized_spike_map.copy()
    m2 = m1.copy()
    m2.filter_fwhm = 0.5 * units.Unit('arcsec')
    m2.smoothing_beam.fwhm *= 2
    m1.merge_properties_from(m2)
    assert m1.smoothing_beam == m2.smoothing_beam  # encompassed
    assert m1.filter_fwhm == m2.filter_fwhm
    m1.smoothing_beam = None
    m1.merge_properties_from(m2)
    assert m1.smoothing_beam == m2.smoothing_beam  # copied
    assert m1.filter_fwhm == m2.filter_fwhm


def test_add_smoothing(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.add_smoothing(1.0 * arcsec)
    assert np.isclose(m.smoothing_beam.fwhm, 1.372058 * arcsec, atol=1e-6)
    m.smoothing_beam = None
    m.add_smoothing(1.0 * arcsec)
    assert isinstance(m.smoothing_beam, Gaussian2D)
    assert m.smoothing_beam.fwhm == 1 * arcsec


def test_filter_beam_correct(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.data.fill(1.0)
    assert not m.is_corrected()

    m.filter_beam_correct()
    assert np.allclose(m.data, 2.655983, atol=1e-6)
    assert m.is_corrected()
    assert m.correcting_fwhm == 2 * arcsec

    # Try again - should be no change since equal FWHM
    m.filter_beam_correct()
    assert np.allclose(m.data, 2.655983, atol=1e-6)
    assert m.is_corrected()
    assert m.correcting_fwhm == 2 * arcsec

    m = initialized_spike_map.copy()
    m.fill(1)
    m.underlying_beam = None
    m.filter_beam_correct()
    assert np.allclose(m.data, 0.531197, atol=1e-6)
    assert m.is_corrected()
    assert m.correcting_fwhm == 0 * arcsec


def test_filter_correct(initialized_spike_map):
    m = initialized_spike_map.copy()
    data = m.data.copy()
    data.fill(1)
    data[1, 1] = 10
    m.data = data
    m.filter_blanking = 9.0
    m0 = m.copy()

    m.filter_correct(1 * arcsec)
    expected = np.full(m.shape, 1.062393)
    expected[1, 1] = 10  # no change for invalid element
    assert np.allclose(m.data, expected, atol=1e-6)

    m.filter_correct(1 * arcsec)  # should be no change
    assert np.allclose(m.data, expected, atol=1e-6)

    # Apply a different correction factor
    m.filter_correct(2 * arcsec)  # Previously corrected
    m0.filter_correct(2 * arcsec)  # Not previously corrected
    # Results should be equal
    assert np.allclose(m.data, m0.data)

    m = initialized_spike_map.copy()
    m.data = data / 2
    m.filter_blanking = 9

    # The point (1, 1) should not be blanked, but will be if we pass in the
    # original data as a reference
    m.filter_correct(1 * arcsec, reference=data)
    assert np.allclose(m.data, expected / 2)


def test_undo_filter_correct(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.filter_blanking = 9
    data = np.ones(m.shape, dtype=float)
    data[1, 1] = 10  # one invalid point (> filter blanking)

    m.data = data.copy()
    assert not m.is_corrected()
    m.undo_filter_correct()
    assert np.allclose(m.data, data)  # No change since not corrected
    m.filter_correct(1 * arcsec)
    assert not np.allclose(m.data, data)
    assert m.is_corrected()
    m.undo_filter_correct()
    assert np.allclose(m.data, data)
    assert not m.is_corrected()

    m.data = np.ones(m.shape, dtype=float)
    m.set_correcting_fwhm(1 * arcsec)  # as if it were corrected
    m.undo_filter_correct(reference=data)  # A single point should be invalid
    mask = m.data == 1
    assert mask[1, 1]
    assert not np.any(m.data[~mask] == 1)


def test_update_filtering():
    m = Map2D()
    assert np.isnan(m.filter_fwhm)
    m.update_filtering(10 * arcsec)
    assert m.filter_fwhm == 10 * arcsec
    m.update_filtering(5 * arcsec)
    assert m.filter_fwhm == 5 * arcsec
    m.update_filtering(10 * arcsec)
    assert m.filter_fwhm == 5 * arcsec


def test_parse_coordinate_info():
    h = fits.Header()
    h['CDELT1'] = 1.0
    h['CDELT2'] = 2.0
    m = Map2D()
    m.parse_coordinate_info(h)
    assert m.grid.resolution.x == 1 * pix and m.grid.resolution.y == 2 * pix


def test_parse_corrected_beam():
    m = Map2D()
    h = fits.Header()
    h['CBMAJ'] = 2.0
    h['CBMIN'] = 2.0
    m.parse_corrected_beam(h)
    assert m.correcting_fwhm == 2 * pix
    h = fits.Header()
    h['CORRECTN'] = 2.5
    m.parse_corrected_beam(h)
    assert m.correcting_fwhm == 2.5 * pix
    m.set_display_grid_unit('arcsec')
    m.parse_corrected_beam(h)
    assert m.correcting_fwhm == 2.5 * arcsec
    m.parse_corrected_beam(fits.Header())
    assert np.isclose(m.correcting_fwhm, np.nan * arcsec, equal_nan=True)


def test_parse_smoothing_beam():
    m = Map2D()
    m.parse_smoothing_beam(fits.Header())
    assert np.isclose(m.smoothing_beam.fwhm, 0.939437 * pix, atol=1e-6)
    h = fits.Header()
    h['SBMAJ'] = 2.0
    h['SBMIN'] = 2.0
    m.parse_smoothing_beam(h)
    assert m.smoothing_beam.fwhm == 2 * pix
    m.set_display_grid_unit(arcsec)
    h = fits.Header()
    h['SMOOTH'] = 2.5
    m.parse_smoothing_beam(h)
    assert m.smoothing_beam.fwhm == 2.5 * arcsec


def test_parse_filter_beam():
    m = Map2D()
    h = fits.Header()
    h['XBMAJ'] = 2.0
    h['XBMIN'] = 2.0
    m.parse_filter_beam(h)
    assert m.filter_fwhm == 2 * pix
    h = fits.Header()
    h['EXTFLTR'] = 2.5
    m.parse_filter_beam(h)
    assert m.filter_fwhm == 2.5 * pix
    m.set_display_grid_unit('arcsec')
    m.parse_filter_beam(h)
    assert m.filter_fwhm == 2.5 * arcsec
    m.parse_filter_beam(fits.Header())
    assert np.isclose(m.filter_fwhm, np.nan * arcsec, equal_nan=True)


def test_parse_underlying_beam():
    m = Map2D()
    m.parse_underlying_beam(fits.Header())
    assert m.underlying_beam.fwhm == 0 * pix
    m.display_grid_unit = 1 * arcsec
    h = fits.Header()
    h['IBMAJ'] = 2.0
    h['IBMIN'] = 2.0
    m.parse_underlying_beam(h)
    assert m.underlying_beam.fwhm == 2 * pix
    h = fits.Header()
    h['BEAM'] = 2.0
    m.parse_underlying_beam(h)
    assert m.underlying_beam.fwhm == 2 * arcsec
    h = fits.Header()
    h['BMAJ'] = 3.0
    h['BMIN'] = 3.0
    m.parse_underlying_beam(h)
    assert m.underlying_beam.fwhm == 3 * pix
    h = fits.Header()
    h['RESOLUTN'] = 2.5
    m.parse_underlying_beam(h)
    assert m.underlying_beam.fwhm == 2.5 * arcsec
    m.smoothing_beam = Gaussian2D(x_fwhm=1 * arcsec, y_fwhm=1 * arcsec)
    m.parse_underlying_beam(h)
    assert np.isclose(m.underlying_beam.fwhm, 2.291288 * arcsec, atol=1e-6)
    h['RESOLUTN'] = 0.5
    m.parse_underlying_beam(h)
    assert m.underlying_beam.fwhm == 0 * arcsec


def test_parse_header():
    m = Map2D()
    h = fits.Header()
    h['CDELT1'] = 1.0
    h['CDELT2'] = 2.0
    h['CORRECTN'] = 1.5
    h['SMOOTH'] = 2.5
    h['EXTFLTR'] = 3.5
    h['BEAM'] = 4.5
    m.parse_header(h)
    assert m.grid.resolution.x == 1 * pix and m.grid.resolution.y == 2 * pix
    assert m.correcting_fwhm == 1.5 * pix
    assert m.smoothing_beam.fwhm == 2.5 * pix
    assert m.filter_fwhm == 3.5 * pix
    assert m.underlying_beam.fwhm == 4.5 * pix


def test_edit_coordinate_info(initialized_spike_map):
    m = initialized_spike_map.copy()
    h = fits.Header()
    m.edit_coordinate_info(h)
    assert h['CRPIX1'] == 1 and h['CRPIX2'] == 1
    assert h['CRVAL1'] == 0 and h['CRVAL2'] == 0
    assert h['CDELT1'] == 1 and h['CDELT2'] == 1
    assert h['CUNIT1'] == 'arcsec' and h['CUNIT2'] == 'arcsec'
    assert h['CTYPE1'] == 'x' and h['CTYPE2'] == 'y'


def test_edit_header(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.correcting_fwhm = 2.5 * arcsec
    h = fits.Header()
    m.edit_header(h)
    d = dict(h)
    d['HISTORY'] = list(d['HISTORY'])
    assert d == {'CRPIX1': 1.0,
                 'CRPIX2': 1.0,
                 'CRVAL1': 0.0,
                 'CRVAL2': 0.0,
                 'CDELT1': 1.0,
                 'CDELT2': 1.0,
                 'CUNIT1': 'arcsec',
                 'CUNIT2': 'arcsec',
                 'CTYPE1': 'x',
                 'CTYPE2': 'y',
                 'BNAM': 'image',
                 'BMAJ': 2.209647573847605,
                 'BMIN': 2.209647573847605,
                 'BPA': 0.0,
                 'RESOLUTN': 2.209647573847605,
                 'IBNAM': 'instrument',
                 'IBMAJ': 2.0,
                 'IBMIN': 2.0,
                 'IBPA': 0.0,
                 'SBNAM': 'smoothing',
                 'SBMAJ': 0.9394372786996513,
                 'SBMIN': 0.9394372786996513,
                 'SBPA': 0.0,
                 'SMOOTH': 0.9394372786996513,
                 'XBNAM': 'Extended Structure Filter',
                 'XBMAJ': 1.0,
                 'XBMIN': 1.0,
                 'XBPA': 0.0,
                 'CBNAM': 'Peak Corrected',
                 'CBMAJ': 2.5,
                 'CBMIN': 2.5,
                 'CBPA': 0.0,
                 'SMTHRMS': True,
                 'DATAMIN': 0.0,
                 'DATAMAX': 1.0,
                 'BZERO': 0.0,
                 'BSCALE': 1.0,
                 'BUNIT': 'Jy',
                 'HISTORY': [
                     'pasted new content: 11x10', 'pasted new content: 11x10']}

    h = fits.Header()
    m.skip_model_edit_header = True
    m.edit_header(h)
    assert 'SMTHRMS' not in h


def test_claim_image(initialized_spike_map):
    m = initialized_spike_map.copy()
    o = Map2D()
    o.parallelism = 1
    o.executor = 'foo'
    m.claim_image(o)
    assert o.unit == 1 * units.Unit('Jy')
    assert o.parallelism == 0
    assert o.executor is None


def test_count_beams(initialized_spike_map):
    m = initialized_spike_map.copy()
    assert np.isclose(m.count_beams(), 19.883015, atol=1e-6)


def test_count_independents_points(initialized_spike_map):
    m = initialized_spike_map.copy()
    area = 121 * asec2
    assert m.count_independent_points(area) == 3
    m.filter_fwhm = np.nan * arcsec
    assert m.count_independent_points(area) == 23
    m.smoothing_beam = None
    assert m.count_independent_points(area) == 0


def test_nearest_to_offset():
    m = Map2D()
    ix, iy = m.nearest_to_offset([1.5, 2.5])
    assert ix == 2 and iy == 3


def test_convert_range_value_to_index(initialized_spike_map, capsys):
    m = initialized_spike_map.copy()
    m.set_display_grid_unit(arcsec)
    m.verbose = True
    ranges = np.asarray([[4, 7], [5, 9]]) * arcsec
    index_ranges = m.convert_range_value_to_index(ranges)
    assert np.allclose(index_ranges, [[4, 7], [5, 9]])
    assert index_ranges.dtype in [int, np.int64]


def test_crop(initialized_spike_map):
    m = initialized_spike_map.copy()
    ranges = np.asarray([[4, 7], [5, 9]]) * arcsec
    m.crop(ranges)
    assert m.shape == (5, 4)
    assert m.grid.reference_index == Coordinate2D([-4, -5])

    ranges = np.asarray([[2, 4], [3, 5]])
    m = initialized_spike_map.copy()
    m.crop(ranges)
    assert m.shape == (3, 3)
    assert m.grid.reference_index == Coordinate2D([-2, -3])

    m = Map2D()
    m.crop(ranges)
    assert m.data is None


def test_auto_crop(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.verbose = True
    data = m.data.copy()
    m.auto_crop()
    assert m.shape == data.shape  # No change

    data[0] = np.nan
    data[:, 0] = np.nan
    m.data = data
    ranges = m.auto_crop()
    assert ranges.dtype == int and np.allclose(ranges, [[1, 11], [1, 10]])
    assert m.shape == (9, 10)

    data.fill(np.nan)
    m.data = data
    ranges = m.auto_crop()
    assert m.size == 0
    assert np.allclose(ranges, -1)

    m = Map2D()
    m.auto_crop()
    assert m.data is None


def test_smooth_to(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smooth_to(5)  # arc seconds (deconvolved by smoothing beam)
    assert np.allclose(m.data[4:7, 5:8],
                       [[0.02985814, 0.03375953, 0.03102982],
                        [0.03349578, 0.03787248, 0.03481021],
                        [0.03071402, 0.03472725, 0.0319193]], atol=1e-6)

    m = initialized_spike_map.copy()
    m.smooth_to(0.1 * arcsec)  # Should not do anything
    assert np.allclose(m.data, initialized_spike_map.data)


def test_smooth_with_psf(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.smooth_with_psf(5)  # arcseconds
    assert np.allclose(m.data[4:7, 5:8],
                       [[0.02913677, 0.03282827, 0.03034042],
                        [0.03255413, 0.0366786, 0.03389896],
                        [0.03000677, 0.0338085, 0.03124636]], atol=1e-6)


def test_smooth(initialized_spike_map):
    m = initialized_spike_map.copy()
    beam_map = np.ones((3, 3)) / 9
    m.smooth(beam_map)
    assert np.allclose(m.data[4:7, 5:8], beam_map)
    assert np.isclose(m.smoothing_beam.fwhm, 1.328565 * arcsec, atol=1e-6)


def test_fast_smooth(initialized_spike_map):
    m = initialized_spike_map.copy()
    beam_map = np.ones((3, 3)) / 9
    steps = np.full(2, 1)
    m.fast_smooth(beam_map, steps)
    assert np.allclose(m.data[4:7, 5:8], beam_map)
    assert np.isclose(m.smoothing_beam.fwhm, 1.328565 * arcsec, atol=1e-6)


def test_filter_above(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.filter_fwhm = np.inf * arcsec
    m.filter_above(5 * arcsec)

    assert np.allclose(m.data[4:7, 5:8],
                       [[-0.02985814, -0.03375953, -0.03102982],
                        [-0.03349578, 0.96212752, -0.03481021],
                        [-0.03071402, -0.03472725, -0.0319193]],
                       atol=1e-6)
    assert m.filter_fwhm == 5 * arcsec
    assert np.all(m.valid)

    m = initialized_spike_map.copy()
    valid = m.valid
    valid[5, 5] = False
    data = m.data
    data[5, 5] = np.nan
    m.data = data
    m.filter_above(5 * arcsec, valid=valid)
    assert np.allclose(m.data[4:7, 5:8],
                       [[-0.02985814, -0.03375953, -0.03102982],
                        [np.nan, 0.96212752, -0.03481021],
                        [-0.03071402, -0.03472725, -0.0319193]],
                       atol=1e-6, equal_nan=True)


def test_fft_filter_above(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.filter_fwhm = np.inf * arcsec
    m.fft_filter_above(5 * arcsec)
    assert np.allclose(m.data[4:7, 5:8],
                       [[-0.02827913, -0.0315959, -0.02827913],
                        [-0.0315959, 0.9646983, -0.0315959],
                        [-0.02827913, -0.0315959, -0.02827913]],
                       atol=1e-6)
    assert m.filter_fwhm == 5 * arcsec

    m = initialized_spike_map.copy()
    data = m.data
    data[5, 5] = np.nan
    m.data = data
    valid = m.valid
    weight = np.zeros_like(data)
    m.fft_filter_above(5 * arcsec, valid=valid, weight=weight)
    assert np.allclose(m.data, data, equal_nan=True)  # no difference

    weight = m.basis.copy()
    weight.data.fill(1.0)
    m.fft_filter_above(5 * arcsec, weight=weight)
    assert np.allclose(m.data[4:7, 5:8],
                       [[-0.02827913, -0.0315959, -0.02827913],
                        [np.nan, 0.9646983, -0.0315959],
                        [-0.02827913, -0.0315959, -0.02827913]],
                       atol=1e-6, equal_nan=True)


def test_resample_from_map(initialized_spike_map):
    map2d = initialized_spike_map.copy()
    m = Map2D()
    m.set_resolution(0.5 * arcsec)
    m.data = np.zeros((20, 22), dtype=float)
    m.resample_from_map(map2d)
    assert np.allclose(m.data[9:12, 11:14],
                       [[0.14621069, 0.19355833, 0.19191715],
                        [0.19746058, 0.26140456, 0.25918811],
                        [0.19742679, 0.26135982, 0.25914375]],
                       atol=1e-6)


def test_resample(initialized_spike_map):
    m = initialized_spike_map.copy()
    m.resample(0.5 * arcsec)
    assert np.allclose(m.data[9:12, 11:14],
                       [[0.14621069, 0.19355833, 0.19191715],
                        [0.19746058, 0.26140456, 0.25918811],
                        [0.19742679, 0.26135982, 0.25914375]],
                       atol=1e-6)
    assert m.shape == (20, 22)
