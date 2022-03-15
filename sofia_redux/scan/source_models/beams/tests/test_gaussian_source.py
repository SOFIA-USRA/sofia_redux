# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.beams.gaussian_source import GaussianSource
from sofia_redux.scan.source_models.maps.map_2d import Map2D
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D
from sofia_redux.scan.utilities.range import Range


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')


@pytest.fixture
def gaussian2d():
    g = Gaussian2D(x_fwhm=10, y_fwhm=5, theta=30, peak=2.0, peak_unit='Jy',
                   position_unit=arcsec)
    return g


@pytest.fixture
def gaussian_source(gaussian2d):
    return GaussianSource(gaussian_model=gaussian2d)


@pytest.fixture
def obs2d(gaussian2d):
    g = gaussian2d.copy()
    x, y = np.meshgrid(np.linspace(-30, 30, 61), np.linspace(-30, 30, 61)
                       ) * arcsec
    data = g.model(x, y)
    obs2d = Observation2D(data=data)
    weight = np.full_like(data, 2.0)
    exposure = np.ones_like(data)
    obs2d.set_weight_image(weight)
    obs2d.set_exposure_image(exposure)
    obs2d.grid.set_resolution(1 * arcsec)
    obs2d.reference = Coordinate2D([0, 0], unit='degree')
    return obs2d


@pytest.fixture
def map2d(obs2d):
    map2d = Map2D(data=obs2d)
    map2d.grid = obs2d.grid
    return map2d


@pytest.fixture
def filtered_map2d(map2d, gaussian_source):
    image = map2d.copy()
    image.filter_fwhm = 2 * arcsec
    image.smoothing_beam = gaussian_source.copy()
    image.underlying_beam = gaussian_source.copy()
    image.smoothing_beam.fwhm = 1 * arcsec
    image.underlying_beam.fwhm = 3 * arcsec
    image.filter_fwhm = 2 * arcsec
    image.filter_blanking = 0.5
    return image


@pytest.fixture
def grid_gaussian_source(gaussian_source, obs2d):
    g = gaussian_source.copy()
    g.grid = obs2d.grid.copy()
    g.set_center_index(Coordinate2D([30, 30]))
    return g


def test_init(gaussian2d):
    model = gaussian2d.copy()
    g = GaussianSource(peak=2.0, x_mean=3.0, y_mean=4.0,
                       x_fwhm=6.0, y_fwhm=5.0, theta=7,
                       peak_unit='Jy', position_unit=arcsec)
    assert g.positioning_method == 'position'
    assert g.coordinates is None
    assert g.source_mask is None
    assert g.source_sum == 0
    assert g.grid is None
    assert g.center_index is None
    assert g.peak_weight == 1
    assert g.fwhm_weight == 1
    assert not g.is_corrected
    assert g.x_fwhm == 6 * arcsec
    assert g.y_fwhm == 5 * arcsec
    assert g.peak == 2
    assert g.unit == 'Jy'
    assert g.x_mean == 3 * arcsec
    assert g.y_mean == 4 * arcsec
    assert g.theta == 7 * degree

    g = GaussianSource(gaussian_model=model)
    assert g.x_fwhm == 10 * arcsec
    assert g.y_fwhm == 5 * arcsec
    assert g.theta == 30 * degree
    assert g.peak == 2
    assert g.unit == 'Jy'
    assert g.x_mean == 0 * arcsec
    assert g.y_mean == 0 * arcsec


def test_copy(gaussian_source):
    g = gaussian_source
    g2 = g.copy()
    assert g2 == g and g2 is not g
    assert g2.grid is g.grid  # Referenced attribute


def test_referenced_attributes():
    g = GaussianSource()
    assert 'grid' in g.referenced_attributes


def test_position():
    g = GaussianSource()
    c = g.position
    assert isinstance(c, Coordinate2D) and c.is_null()
    c = Coordinate2D([1 * arcsec, 2 * arcsec])
    g.position = c
    assert g.position == c


def test_peak_significance():
    g = GaussianSource()
    g.peak = -3.0
    g.peak_weight = 4.0
    assert g.peak_significance == 6


def test_peak_rms():
    g = GaussianSource()
    g.peak_weight = 4.0
    assert g.peak_rms == 0.5
    g.peak_weight = -4.0
    assert g.peak_rms == 0


def test_fwhm_significance():
    g = GaussianSource()
    g.fwhm = 5.0 * arcsec
    g.fwhm_weight = 4.0
    assert g.fwhm_significance == 10


def test_fwhm_rms():
    g = GaussianSource()
    g.fwhm_weight = 4.0
    g.fwhm = 5 * arcsec
    assert g.fwhm_rms == 0.5 * arcsec


def test_eq(gaussian_source):
    g = gaussian_source.copy()
    assert g == g
    g2 = g.copy()
    assert g == g2
    g2.is_corrected = True
    assert g != g2
    g2 = g.copy()
    g2.fwhm = 900 * arcsec
    assert g != g2


def test_set_positioning_method():
    g = GaussianSource()
    g.set_positioning_method('position')
    assert g.positioning_method == 'position'
    g.set_positioning_method('peak')
    assert g.positioning_method == 'position'
    g.set_positioning_method('centroid')
    assert g.positioning_method == 'centroid'
    with pytest.raises(ValueError) as err:
        g.set_positioning_method('foo')
    assert 'Available positioning methods are' in str(err.value)


def test_set_peak_position():
    g = GaussianSource()
    c = Coordinate2D([3, 4], unit=arcsec)
    g.set_peak_position(c)
    assert g.coordinates == c
    assert g.x_mean == c.x
    assert g.y_mean == c.y

    ud = units.dimensionless_unscaled
    c = np.array([5 * ud, 6 * ud], dtype=object)
    g.set_peak_position(c)
    assert g.x_mean == 5 and g.y_mean == 6


def test_fit_map_least_squares(gaussian_source, map2d, obs2d):
    g = gaussian_source.copy()
    m = map2d.copy()
    o = obs2d.copy()
    assert g.grid is None
    integrated_value = g.fit_map_least_squares(m)
    assert np.isclose(integrated_value, 113.309004, atol=1e-6)
    assert g.grid is m.grid
    assert np.isclose(g.center_index.x, 30)
    assert np.isclose(g.center_index.y, 30)
    assert np.isclose(g.fwhm, 7.071068 * arcsec, atol=1e-6)
    assert g.fwhm_weight == 1  # Does not get calculated without significance

    g = gaussian_source.copy()
    integrated_value = g.fit_map_least_squares(o)  # Fits on significance
    assert np.isclose(integrated_value, 113.357305, atol=1e-6)
    assert np.isclose(g.peak, 2.000853, atol=1e-6)
    assert np.isclose(g.peak_weight, 2, atol=1e-6)
    assert np.isclose(g.fwhm_weight, 0.080068 / arcsec ** 2, atol=1e-6)


def test_get_lsq_fit_parameters(gaussian_source, obs2d):
    g = gaussian_source.copy()
    o = obs2d.copy()
    g.grid = o.grid
    func, p0, bounds = g.get_lsq_fit_parameters(o)
    assert func == g.gaussian_2d_fit
    assert p0[0] == 2.0  # peak
    assert p0[1] == 30.0  # x-center
    assert p0[2] == 30.0  # y-center
    assert np.isclose(p0[3], 3.002806, atol=1e-6)  # gaussian sigma (pixels)
    assert bounds[0][:3] == (-np.inf, -np.inf, -np.inf)
    assert np.isclose(bounds[0][3], 3.002806, atol=1e-6)
    assert bounds[1] == (np.inf, np.inf, np.inf, np.inf)


def test_gaussian_2d_fit():
    coordinates = np.linspace(-5, 5, 11), np.linspace(-2.5, 2.5, 11)
    amplitude = 2.0
    x0 = 0.0
    y0 = 0.0
    sigma = 3.0
    fit = GaussianSource.gaussian_2d_fit(coordinates, amplitude, x0, y0, sigma)
    assert np.allclose(
        fit,
        [0.35240862, 0.65838598, 1.07052286, 1.51493026, 1.86582392, 2,
         1.86582392, 1.51493026, 1.07052286, 0.65838598, 0.35240862])


def test_fit_map(gaussian_source, obs2d, map2d):
    g = gaussian_source.copy()
    o = obs2d.copy()
    m = map2d.copy()

    assert g.center_index is None
    assert g.grid is None
    assert g.source_mask is None
    assert g.source_radius is None
    assert g.source_sum == 0

    g.fit_map(m)
    assert np.isclose(g.source_sum, 104.864403, atol=1e-6)
    assert isinstance(g.source_mask, np.ndarray)
    assert np.sum(g.source_mask) == 253
    assert isinstance(g.source_radius, units.Quantity)
    assert np.isclose(g.source_radius, 9 * arcsec)
    assert isinstance(g.center_index, Coordinate2D)
    assert g.center_index.x == 30 and g.center_index.y == 30
    assert g.fwhm_weight == 1 / arcsec ** 2
    assert np.isclose(g.fwhm, 6.802473 * arcsec, atol=1e-6)

    g = gaussian_source.copy()  # Uses significance map instead
    g.fit_map(o)
    assert np.isclose(g.fwhm_weight, 0.0864423 / arcsec ** 2, atol=1e-6)
    assert g.center_index.x == 30 and g.center_index.y == 30
    assert np.sum(g.source_mask) == 253
    assert np.isclose(g.source_sum, 148.300661, atol=1e-6)
    assert np.isclose(g.source_radius, 9 * arcsec)
    assert np.isclose(g.fwhm, 6.802473 * arcsec, atol=1e-6)
    o.data.fill(0.0)
    g.fit_map(o)
    assert not np.isfinite(g.fwhm_weight) and g.fwhm_weight > 0


def test_set_center_index(grid_gaussian_source):
    g = grid_gaussian_source.copy()
    center_index = Coordinate2D([4, 5])
    g.set_center_index(center_index)
    assert np.isclose(g.position.x, 4 * arcsec)
    assert np.isclose(g.position.y, 5 * arcsec)
    assert g.center_index == center_index


def test_get_grid_coordinates(grid_gaussian_source):
    g = grid_gaussian_source.copy()
    index = Coordinate2D([5, 6])
    coordinates = g.get_grid_coordinates(index)
    assert np.isclose(coordinates.x, 5 * arcsec)
    assert np.isclose(coordinates.y, 6 * arcsec)


def test_find_source_extent(grid_gaussian_source, obs2d):
    g = grid_gaussian_source.copy()
    assert g.source_radius is None
    assert g.source_mask is None
    assert g.source_sum == 0
    g.find_source_extent(obs2d, max_iterations=40, radius_increment=1.1,
                         tolerance=0.05)
    assert np.isclose(g.source_sum, 104.864403, atol=1e-6)
    assert isinstance(g.source_mask, np.ndarray)
    assert g.source_mask.sum() == 253
    assert g.source_radius == 9 * arcsec

    o = obs2d.copy()
    o.data *= -1
    g.find_source_extent(o, max_iterations=40, radius_increment=1.1,
                         tolerance=0.05)

    assert np.isclose(g.source_sum, -104.864403, atol=1e-6)
    assert g.source_mask.sum() == 253
    assert g.source_radius == 9 * arcsec

    o.data = np.zeros_like(o.data)

    g.find_source_extent(o, max_iterations=40, radius_increment=1.1,
                         tolerance=0.05)
    assert g.source_sum == 0
    assert g.source_mask.sum() == 3697
    assert g.source_radius == 41 * arcsec

    g.find_source_extent(o, max_iterations=40, radius_increment=-1.1,
                         tolerance=0.05)
    assert g.source_sum == 0
    assert g.source_mask.sum() == 1
    assert isinstance(g.source_radius, units.Quantity)
    assert np.isclose(g.source_radius, 0)

    g.grid.set_resolution(-1 * arcsec)
    g.find_source_extent(o, max_iterations=40, radius_increment=1e-6,
                         tolerance=0.05)
    assert g.source_sum == 0
    assert g.source_mask.sum() == 0
    assert np.isclose(g.source_radius, -41 * arcsec)


def test_get_center_offset(grid_gaussian_source):
    g = grid_gaussian_source.copy()
    offset = Coordinate2D()
    c = g.get_center_offset()
    expected = c.copy()
    assert np.isclose(c.x, 30 * arcsec) and np.isclose(c.y, 30 * arcsec)
    c = g.get_center_offset(offset=offset)
    assert c is offset and c == expected


def test_find_peak(grid_gaussian_source, obs2d):
    g = grid_gaussian_source.copy()
    g.positioning_method = 'position'
    p1 = g.find_peak(obs2d)
    g.positioning_method = 'centroid'
    p2 = g.find_peak(obs2d)
    assert p1 == p2  # close
    assert p1.x != p2.x  # but not exact
    assert p1.x == 30 and p1.y == 30  # These are pixel coordinates

    g.positioning_method = 'position'
    p3 = g.find_peak(obs2d, grid=g.grid)

    assert p3 == p1  # FlatGrid2D projection

    g.positioning_method = 'foo'
    with pytest.raises(ValueError) as err:
        _ = g.find_peak(obs2d)
    assert "Unknown positioning method" in str(err.value)


def test_find_local_peak(obs2d):
    image = obs2d.copy()
    g = GaussianSource()
    data = image.data.copy()
    data[20, 20] = -1
    image.data = data
    pos = Coordinate2D([30, 30.0])
    neg = Coordinate2D([19.99969224639764, 19.998476676389178])
    assert g.find_local_peak(image, sign=0) == pos
    assert g.find_local_peak(image, sign=1) == pos
    assert g.find_local_peak(image, sign=-1) == neg


def test_find_local_centroid(obs2d):
    centroid = GaussianSource.find_local_centroid(obs2d)
    assert np.allclose(centroid.coordinates, [30, 30], atol=1e-8)


def test_set_peak_from(obs2d, grid_gaussian_source, map2d):
    g = grid_gaussian_source.copy()
    assert g.unit == 'Jy'
    g.set_peak_from(obs2d)
    assert g.unit == units.dimensionless_unscaled
    assert g.peak == 2
    assert g.peak_weight == 2
    g.set_peak_from(map2d)  # No weight
    assert g.peak_weight == np.inf
    assert g.peak == 2
    assert g.unit == units.dimensionless_unscaled


def test_set_unit():
    ud = units.dimensionless_unscaled
    g = GaussianSource()
    g.set_unit(None)
    assert g.peak == 1 and g.unit == ud
    g.set_unit(2 * units.Unit('K'))
    assert g.peak == 2 and g.unit == 'K'
    g.set_unit(units.Unit('Jy'))
    assert g.peak == 2 and g.unit == 'Jy'
    g.set_unit(3)
    assert g.peak == 6 and g.unit == ud
    g.set_unit('degree')
    assert g.peak == 6 and g.unit == 'degree'


def test_scale_peak():
    g = GaussianSource()
    g.peak = 1.0
    g.peak_weight = 1.0
    g.scale_peak(1)
    assert g.peak == 1 and g.peak_weight == 1
    g.scale_peak(2)
    assert g.peak == 2 and g.peak_weight == 0.25


def test_scale_fwhm():
    g = GaussianSource()
    g.fwhm = 3 * arcsec
    g.fwhm_weight = 1.0
    g.scale_fwhm(1)
    assert g.fwhm == 3 * arcsec and g.fwhm_weight == 1
    g.scale_fwhm(2)
    assert g.fwhm == 6 * arcsec and g.fwhm_weight == 0.25


def test_set_exact():
    g = GaussianSource()
    g.peak_weight = 1.0
    g.set_exact()
    assert g.peak_weight == np.inf


def test_get_correction_factor(map2d, gaussian_source):
    image = map2d.copy()
    g = GaussianSource()
    g.peak = 1.0
    g.peak_weight = 4.0
    assert g.peak_significance == 2.0
    assert g.get_correction_factor(image) == 1
    image.filter_fwhm = 2 * arcsec
    image.smoothing_beam = gaussian_source.copy()
    image.underlying_beam = gaussian_source.copy()
    image.smoothing_beam.fwhm = 1 * arcsec
    image.underlying_beam.fwhm = 3 * arcsec
    assert np.isclose(g.get_correction_factor(image), 2.6)

    image.filter_blanking = 0.5
    assert np.isclose(g.get_correction_factor(image), 13 / 11)


def test_correct(filtered_map2d):
    image = filtered_map2d.copy()
    g = GaussianSource()
    g.peak = 1.0
    g.peak_weight = 4.0
    g.correct(image)
    assert g.is_corrected
    assert np.isclose(g.peak, 13 / 11)
    g.correct(image)
    assert np.isclose(g.peak, 13 / 11)  # No change


def test_uncorrect(filtered_map2d):
    image = filtered_map2d.copy()
    g = GaussianSource()
    g.peak = 1.0
    g.peak_weight = 4.0
    g.uncorrect(image)
    assert g.peak == 1 and g.peak_weight == 4
    g.correct(image)
    assert np.isclose(g.peak, 13 / 11)
    assert np.isclose(g.peak_weight, 2.8639053, atol=1e-6)
    assert g.is_corrected
    g.uncorrect(image)
    assert np.isclose(g.peak, 1)
    assert np.isclose(g.peak_weight, 4)
    assert not g.is_corrected


def test_get_gaussian_2d(gaussian_source):
    g = gaussian_source.copy()
    g2 = g.get_gaussian_2d()
    assert isinstance(g2, Gaussian2D) and not isinstance(g2, GaussianSource)
    assert g.peak == g2.peak
    assert g.x_fwhm == g2.x_fwhm
    assert g.y_fwhm == g2.y_fwhm
    assert g.x_mean == g2.x_mean
    assert g.y_mean == g2.y_mean
    assert g.theta == g2.theta


def test_deconvolve_with(gaussian_source):
    g = gaussian_source.copy()
    g.set_xy_fwhm(16 * arcsec, 4 * arcsec)
    assert g.fwhm == 8 * arcsec

    g2 = g.copy()
    psf = g.get_gaussian_2d()
    psf.fwhm = 2 * arcsec
    g2.deconvolve_with(psf)
    assert np.isclose(g2.fwhm, 7.4155855 * arcsec, atol=1e-6)
    assert np.isclose(g2.fwhm_weight, 0.85923294 / arcsec ** 2, atol=1e-6)
    assert g2.x_fwhm == g2.y_fwhm

    psf.fwhm = 20 * arcsec
    g2.deconvolve_with(psf)
    assert g2.fwhm == 0 and g2.fwhm_weight == 0


def test_convolve_with(gaussian_source):
    g = gaussian_source.copy()
    g.set_xy_fwhm(16 * arcsec, 4 * arcsec)
    psf = g.get_gaussian_2d()
    psf.fwhm = 2 * arcsec
    g.convolve_with(psf)
    assert np.isclose(g.x_fwhm, 16.1245155 * arcsec, atol=1e-6)
    assert np.isclose(g.y_fwhm, 4.47213595 * arcsec, atol=1e-6)
    assert np.isclose(g.fwhm_weight, 1.12673477 / arcsec ** 2, atol=1e-6)

    g = gaussian_source.copy()
    g.fwhm = 0 * arcsec
    psf.fwhm = 0 * arcsec
    g.convolve_with(psf)
    assert g.fwhm == 0 and g.fwhm_weight == 0


def test_edit_header(gaussian_source):
    g = gaussian_source.copy()
    g.fwhm = 10 * arcsec
    g.fwhm_weight = 1 / arcsec ** 2
    g.peak = 2.0
    g.peak_weight = 4.0
    h = fits.Header()
    g.edit_header(h)
    assert h['SRCPEAK'] == 2
    assert h['SRCPKERR'] == 0.5
    assert h['SRCFWHM'] == 10
    assert h['SRCWERR'] == 1
    assert h.comments['SRCPEAK'] == '(Jy) source peak flux.'
    assert h.comments['SRCPKERR'] == '(Jy) peak flux error.'
    assert h.comments['SRCFWHM'] == '(arcsec) source FWHM.'
    assert h.comments['SRCWERR'] == '(arcsec) FWHM error.'

    g.set_unit(None)
    g.edit_header(h, size_unit='arcmin')
    assert h['SRCPEAK'] == 2
    assert h['SRCPKERR'] == 0.5
    assert h['SRCFWHM'] == 10 / 60
    assert h['SRCWERR'] == 1 / 60
    assert h.comments['SRCPEAK'] == 'source peak flux.'
    assert h.comments['SRCPKERR'] == 'peak flux error.'
    assert h.comments['SRCFWHM'] == '(arcmin) source FWHM.'
    assert h.comments['SRCWERR'] == '(arcmin) FWHM error.'

    ud = units.dimensionless_unscaled
    g.fwhm = 10 * ud
    g.fwhm_weight = 1 * ud
    g.edit_header(h, size_unit=ud)
    assert h['SRCPEAK'] == 2
    assert h['SRCPKERR'] == 0.5
    assert h['SRCFWHM'] == 10
    assert h['SRCWERR'] == 1
    assert h.comments['SRCPEAK'] == 'source peak flux.'
    assert h.comments['SRCPKERR'] == 'peak flux error.'
    assert h.comments['SRCFWHM'] == 'source FWHM.'
    assert h.comments['SRCWERR'] == 'FWHM error.'


def test_get_integral(gaussian_source):
    g = gaussian_source.copy()
    g.peak = 10.0
    g.peak_weight = 1.0
    psf_area = g.area / 2
    integral, weight = g.get_integral(psf_area)
    assert integral == 20 and weight == 0.25


def test_pointing_info(filtered_map2d, gaussian_source):
    image = filtered_map2d.copy()
    g = gaussian_source.copy()
    info = g.pointing_info(image)

    assert info == ['Peak: 2.00000 Jy (S/N ~ 2.00000)',
                    'Integral: 11.1111 +- 5.5556 Jy',
                    'FWHM: 7.0711 +- 1.0000 (arcsec)']

    g.fwhm = 0 * arcsec
    info = g.pointing_info(image)
    assert info == ['Peak: 2.00000 Jy (S/N ~ 2.00000)',
                    'Integral: 0.0000 Jy',
                    'FWHM: 0.0000 +- 1.0000 (arcsec)']

    g.fwhm = 3 * arcsec
    image.underlying_beam.fwhm = 0 * arcsec
    info = g.pointing_info(image)
    assert info == ['Peak: 2.00000 Jy (S/N ~ 2.00000)',
                    'Integral: inf Jy',
                    'FWHM: 3.0000 +- 1.0000 (arcsec)']

    g = gaussian_source.copy()
    g.fwhm = 3 * units.dimensionless_unscaled
    g.fwhm_weight = np.inf * units.dimensionless_unscaled
    info = g.pointing_info(image)

    assert info == ['Peak: 2.00000 Jy (S/N ~ 2.00000)',
                    'Integral: inf Jy',
                    'FWHM: 3.0000']

    g.peak = np.inf
    info = g.pointing_info(image)
    assert info == ['Peak: inf Jy (S/N ~ inf)',
                    'Integral: inf Jy',
                    'FWHM: 3.0000']


def test_get_asymmetry_2d(grid_gaussian_source, filtered_map2d):
    image = filtered_map2d.copy()
    g = grid_gaussian_source.copy()
    radial_range = Range(1 * arcsec, 10 * arcsec)
    angle = 0 * degree
    g.center_index.subtract(Coordinate2D([1, 1]))
    asymmetry = g.get_asymmetry_2d(image, angle, radial_range)
    assert np.isclose(asymmetry.x, 0.142215, atol=1e-6)
    assert np.isclose(asymmetry.x_weight, 76.938829, atol=1e-6)
    assert np.isclose(asymmetry.y, 0.181397, atol=1e-6)
    assert np.isclose(asymmetry.y_weight, 76.938829, atol=1e-6)


def test_get_representation(grid_gaussian_source):
    g = grid_gaussian_source.copy()
    assert np.isclose(g.center_index.x, 30)
    assert np.isclose(g.center_index.y, 30)
    grid = g.grid.copy()
    grid.resolution = 2 * arcsec
    g2 = g.get_representation(grid)
    assert np.isclose(g2.center_index.x, 15)
    assert np.isclose(g2.center_index.y, 15)


def test_get_data(filtered_map2d, grid_gaussian_source):
    image = filtered_map2d.copy()
    g = grid_gaussian_source.copy()
    data = g.get_data(image)
    jy = units.Unit('Jy')
    assert data['peak'] == 2 * jy
    assert data['dpeak'] == 1 * jy
    assert data['peakS2N'] == 2
    assert np.isclose(data['int'], 100 / 9 * jy)
    assert np.isclose(data['dint'], 50 / 9 * jy)
    assert np.isclose(data['intS2N'], 2)
    assert np.isclose(data['FWHM'], 7.071068 * arcsec, atol=1e-6)
    assert np.isclose(data['dFWHM'], 1 * arcsec)
    d0 = data.copy()

    data = g.get_data(image, size_unit='degree')
    for key, value in data.items():
        assert np.isclose(data[key], d0[key])
    assert data['FWHM'].unit == degree
    assert data['dFWHM'].unit == degree

    image.underlying_beam = None
    g.fwhm = 3 * units.dimensionless_unscaled
    data = g.get_data(image, size_unit='arcsec')
    assert data['peak'] == 2 * jy
    assert data['dpeak'] == 1 * jy
    assert data['peakS2N'] == 2
    assert np.isclose(data['int'], 0 * jy)
    assert np.isclose(data['dint'], 0 * jy)
    assert np.isclose(data['intS2N'], 0)
    assert np.isclose(data['FWHM'], 3 * arcsec)
    assert np.isclose(data['dFWHM'], 1 * arcsec)




