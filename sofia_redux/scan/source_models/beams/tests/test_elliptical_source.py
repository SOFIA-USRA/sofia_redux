# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.beams.gaussian_source import GaussianSource
from sofia_redux.scan.source_models.beams.elliptical_source import \
    EllipticalSource
from sofia_redux.scan.source_models.maps.map_2d import Map2D
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


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
def elliptical_source(gaussian2d):
    return EllipticalSource(gaussian_model=gaussian2d)


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
def grid_elliptical_source(elliptical_source, obs2d):
    e = elliptical_source.copy()
    e.grid = obs2d.grid.copy()
    e.set_center_index(Coordinate2D([30, 30]))
    return e


def test_init(gaussian_source):
    g = EllipticalSource()
    assert g.elongation == 0
    assert g.angle_weight is None

    g = EllipticalSource(gaussian_model=gaussian_source)
    assert np.isclose(g.elongation, 1/3)
    assert np.isclose(g.fwhm, 7.07106781 * arcsec, atol=1e-6)
    assert g.fwhm == g.x_fwhm
    assert g.y_fwhm == g.y_fwhm


def test_major_fwhm(elliptical_source):
    e = elliptical_source.copy()
    assert np.isclose(e.major_fwhm, 10 * arcsec)


def test_minor_fwhm(elliptical_source):
    e = elliptical_source.copy()
    assert np.isclose(e.minor_fwhm, 5 * arcsec)


def test_major_fwhm_weight(elliptical_source):
    e = elliptical_source.copy()
    assert e.major_fwhm_weight == np.inf
    e.elongation_weight = 0.25
    e.fwhm_weight = 1.0
    assert np.isclose(e.major_fwhm_weight, 0.00441989 / arcsec ** 2, atol=1e-6)
    e.fwhm_weight = 0.0
    assert np.isclose(e.major_fwhm_weight, 0 / arcsec ** 2, atol=1e-6)


def test_minor_fwhm_weight(elliptical_source):
    e = elliptical_source.copy()
    assert e.minor_fwhm_weight == np.inf
    e.elongation_weight = 0.25
    e.fwhm_weight = 0.5
    assert np.isclose(e.minor_fwhm_weight, 0.0043956 / arcsec ** 2, atol=1e-6)
    e.fwhm_weight = 0.0
    assert np.isclose(e.minor_fwhm_weight, 0 / arcsec ** 2, atol=1e-6)


def test_major_fwhm_rms(elliptical_source):
    e = elliptical_source.copy()
    e.elongation_weight = 10.0
    assert np.isclose(e.major_fwhm_rms, 2.622022 * arcsec, atol=1e-6)


def test_minor_fwhm_rms(elliptical_source):
    e = elliptical_source.copy()
    e.elongation_weight = 20.0
    assert np.isclose(e.minor_fwhm_rms, 2.015564 * arcsec, atol=1e-6)


def test_angle(elliptical_source):
    e = elliptical_source.copy()
    assert e.angle == 30 * degree
    e.angle = 45 * degree
    assert e.angle == 45 * degree


def test_angle_rms(elliptical_source):
    e = elliptical_source.copy()
    assert e.angle_rms == 0 * degree
    e.angle_weight = 0.25 / (degree ** 2)
    assert e.angle_rms == 2 * degree


def test_elongation_rms(elliptical_source):
    e = elliptical_source.copy()
    assert e.elongation_rms == 0
    e.elongation_weight = 0.25
    assert e.elongation_rms == 2


def test_set_elongation(elliptical_source):
    e = elliptical_source.copy()
    assert np.isclose(e.elongation, 1/3)
    e.set_elongation()
    assert np.isclose(e.elongation, 1/3)
    assert e.elongation_weight == np.inf

    e.set_xy_fwhm(0 * arcsec, 0 * arcsec)
    e.set_elongation(weight=None)
    assert e.elongation == 0
    assert e.elongation_weight == np.inf

    e.set_elongation(major=1 * arcsec, minor=3 * arcsec, weight=4.0,
                     angle=45 * degree)
    assert np.isclose(e.elongation, 0.5)
    assert np.isclose(e.major_fwhm, 3 * arcsec)
    assert np.isclose(e.minor_fwhm, 1 * arcsec)
    assert np.isclose(e.position_angle, 135 * degree)
    assert np.isclose(e.elongation_weight, 4.0)


def test_edit_header(elliptical_source):
    e = elliptical_source.copy()
    header = fits.Header()
    e.edit_header(header)

    expected_values = {'SRCPEAK': 2.0,
                       'SRCPKERR': 1.0,
                       'SRCFWHM': 7.0710678118654755,
                       'SRCWERR': 1.0,
                       'SRCMAJ': 10.0,
                       'SRCMAJER': 0.0,
                       'SRCMIN': 5.0,
                       'SRCMINER': 0.0,
                       'SRCPA': 30.0,
                       'SRCPAERR': 0.0}

    expected_comments = {'SRCPEAK': '(Jy) source peak flux.',
                         'SRCPKERR': '(Jy) peak flux error.',
                         'SRCFWHM': '(arcsec) source FWHM.',
                         'SRCWERR': '(arcsec) FWHM error.',
                         'SRCMAJ': '(arcsec) source major axis.',
                         'SRCMAJER': '(arcsec) major axis error.',
                         'SRCMIN': '(arcsec) source minor axis.',
                         'SRCMINER': '(arcsec) minor axis error.',
                         'SRCPA': '(deg) source position angle.',
                         'SRCPAERR': '(deg) source angle error.'}

    for key, value in expected_values.items():
        assert np.isclose(header[key], value)
        assert header.comments[key] == expected_comments[key]

    header = fits.Header()
    e.fwhm = 1 * units.dimensionless_unscaled
    e.fwhm_weight = 4.0
    e.edit_header(header)

    expected_values['SRCFWHM'] = 1.0
    expected_values['SRCWERR'] = 0.5
    expected_values['SRCMAJ'] = np.sqrt(2)
    expected_values['SRCMIN'] = 1 / np.sqrt(2)
    expected_values['SRCMINER'] = 0.0
    expected_values['SRCMAJER'] = 0.0

    for key in ['SRCFWHM', 'SRCWERR', 'SRCMAJ', 'SRCMIN',
                'SRCMINER', 'SRCMAJER']:
        expected_comments[key] = expected_comments[key][
            len('(arcsec) '):]

    for key, value in expected_values.items():
        assert np.isclose(header[key], value)
        assert header.comments[key] == expected_comments[key]

    header = fits.Header()
    e.edit_header(header, size_unit=2 * units.dimensionless_unscaled)

    assert np.isclose(header['SRCMAJ'], 2 * expected_values['SRCMAJ'])
    assert np.isclose(header['SRCMIN'], 2 * expected_values['SRCMIN'])


def test_pointing_info(filtered_map2d, elliptical_source):
    image = filtered_map2d.copy()
    e = elliptical_source.copy()
    info = e.pointing_info(image)

    assert info == ['Peak: 2.00000 Jy (S/N ~ 2.00000)',
                    'Integral: 11.1111 +- 5.5556 Jy',
                    'FWHM: 7.0711 +- 1.0000 (arcsec)',
                    '(a=10.000000+-0.000000 arcsec, '
                    'b=5.000000+-0.000000 arcsec, angle=30.000000 deg)']

    e.fwhm = 10 * units.dimensionless_unscaled
    e.model.theta = np.nan * degree
    image.display_grid_unit = None
    info = e.pointing_info(image)
    assert info == [
        'Peak: 2.00000 Jy (S/N ~ 2.00000)',
        'Integral: 945448228803.3822 +- 472724114401.6911 Jy',
        'FWHM: 10.0000 +- 1.0000',
        '(a=14.142136+-0.000000, b=7.071068+-0.000000, angle=nan deg)']


def test_find_source_extent(filtered_map2d, grid_elliptical_source):
    image = filtered_map2d.copy()
    e = grid_elliptical_source.copy()
    assert np.isclose(e.elongation, 1/3)
    e.find_source_extent(image)
    assert np.isclose(e.elongation, 0.666680, atol=1e-6)
    assert np.isclose(e.elongation_weight, 111.5182710, atol=1e-6)


def test_fit_map_least_squares(filtered_map2d, grid_elliptical_source):
    image = filtered_map2d.copy()
    e = grid_elliptical_source.copy()
    integral = e.fit_map_least_squares(image)
    assert np.isclose(integral, 113.309004, atol=1e-6)
    assert np.isclose(e.elongation, 0.6281, atol=1e-3)
    assert np.isclose(e.elongation_weight, 111.518, atol=1e-3)


def test_measure_shape(filtered_map2d, grid_elliptical_source):
    image = filtered_map2d.copy()
    e = grid_elliptical_source.copy()
    e.measure_shape(image)
    assert np.isclose(e.elongation, 0.666680, atol=1e-6)
    assert np.isclose(e.elongation_weight, 111.518271, atol=1e-6)
    assert np.isclose(e.angle, 28.602487 * degree, atol=1e-6)
    assert np.isclose(e.angle_weight, 0.01509857 / degree ** 2, atol=1e-6)

    image.data = np.zeros_like(image.data)
    e.measure_shape(image)
    assert e.elongation == 0
    assert e.elongation_weight == 0
    assert e.angle == 0 * degree
    assert e.angle_weight == 0 / degree ** 2


def test_get_data(filtered_map2d, grid_elliptical_source):
    image = filtered_map2d.copy()
    e = grid_elliptical_source.copy()
    data = e.get_data(image)
    assert data['a'] == 10 * arcsec
    assert data['b'] == 5 * arcsec
    assert data['da'] == 0 * arcsec
    assert data['db'] == 0 * arcsec
    assert data['angle'] == 30 * degree
    assert data['dangle'] == 0 * degree
