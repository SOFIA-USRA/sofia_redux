# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.io import fits
from astropy.modeling import functional_models
from astropy.stats import gaussian_sigma_to_fwhm
import numpy as np
import pytest

from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D


arcsec = units.Unit('arcsec')


@pytest.fixture
def gaussian2d():
    g = Gaussian2D(
        peak=10 * units.Unit('Jy'),
        x_mean=1 * arcsec,
        y_mean=2 * arcsec,
        x_fwhm=3 * arcsec,  # Should get swapped with y
        y_fwhm=4 * arcsec,  # Should get swapped with x
        theta=30 * units.Unit('degree'),  # Should get set to 120 degrees
    )
    return g


def test_class():
    assert np.isclose(Gaussian2D.AREA_FACTOR, 1.133090035, atol=1e-8)
    assert np.isclose(Gaussian2D.FWHM_TO_SIZE, 1.064467019, atol=1e-8)
    assert np.isclose(Gaussian2D.QUARTER, 90 * units.Unit('degree'))


def test_init():
    g = Gaussian2D()
    du = units.dimensionless_unscaled
    for attribute in ['x_mean', 'y_mean', 'x_fwhm', 'y_fwhm']:
        value = getattr(g, attribute)
        assert value == 0 and value.unit == du
    assert g.peak == 1 * du
    assert g.theta == 0 * units.Unit('degree')
    assert g.unit == du

    g = Gaussian2D(peak=1.5 * units.Unit('K'),
                   x_mean=1 * arcsec,
                   y_mean=2 * units.Unit('arcmin'),
                   x_fwhm=3 * arcsec,
                   y_fwhm=4 * arcsec,
                   theta=3600 * arcsec)

    assert g.peak == 1.5
    assert g.unit == units.Unit('K')
    assert g.x_mean == 1 * arcsec
    assert g.y_mean == 2 * units.Unit('arcmin')
    assert g.x_fwhm == 4 * arcsec
    assert g.y_fwhm == 3 * arcsec
    assert g.theta == 91 * units.Unit('degree')

    g = Gaussian2D(peak=1, peak_unit='Jy',
                   x_mean=1, y_mean=2, x_fwhm=4, y_fwhm=3, theta=45.0,
                   position_unit='degree')

    assert g.peak == 1
    assert g.unit == units.Unit('Jy')
    assert g.x_mean == 1 * units.Unit('degree')
    assert g.y_mean == 2 * units.Unit('degree')
    assert g.x_fwhm == 4 * units.Unit('degree')
    assert g.y_fwhm == 3 * units.Unit('degree')
    assert g.theta == 45 * units.Unit('degree')


def test_copy(gaussian2d):
    g = gaussian2d
    g2 = g.copy()
    assert g is not g2 and g == g2


def test_referenced_attributes(gaussian2d):
    refs = gaussian2d.referenced_attributes
    assert isinstance(refs, set) and len(refs) == 0


def test_x_fwhm(gaussian2d):
    g = gaussian2d.copy()
    assert g.x_fwhm == 4 * arcsec
    g.x_fwhm = 5 * arcsec
    assert g.x_fwhm == 5 * arcsec
    assert g.y_fwhm == 3 * arcsec
    assert g.theta == 120 * units.Unit('degree')
    g.x_fwhm = 1 * arcsec
    assert g.x_fwhm == 3 * arcsec
    assert g.y_fwhm == 1 * arcsec
    assert g.theta == 30 * units.Unit('degree')


def test_y_fwhm(gaussian2d):
    g = gaussian2d.copy()
    assert g.y_fwhm == 3 * arcsec
    assert g.x_fwhm == 4 * arcsec
    g.y_fwhm = 1 * arcsec
    assert g.y_fwhm == 1 * arcsec
    assert g.x_fwhm == 4 * arcsec
    assert g.theta == 120 * units.Unit('degree')
    g.y_fwhm = 5 * arcsec
    assert g.x_fwhm == 5 * arcsec
    assert g.y_fwhm == 4 * arcsec
    assert g.theta == 30 * units.Unit('degree')


def test_x_stddev():
    x = 5 * arcsec * gaussian_sigma_to_fwhm
    y = 5 * arcsec * gaussian_sigma_to_fwhm
    g = Gaussian2D(x_fwhm=x, y_fwhm=y)
    assert g.x_stddev == 5 * arcsec and g.y_stddev == 5 * arcsec
    g.x_stddev = 6 * arcsec
    assert g.x_stddev == 6 * arcsec and g.y_stddev == 5 * arcsec
    assert g.theta == 0
    g.x_stddev = 1 * arcsec
    assert g.x_stddev == 5 * arcsec and g.y_stddev == 1 * arcsec
    assert g.theta == 90 * units.Unit('degree')


def test_y_stddev():
    x = 4 * arcsec * gaussian_sigma_to_fwhm
    y = 4 * arcsec * gaussian_sigma_to_fwhm
    g = Gaussian2D(x_fwhm=x, y_fwhm=y)
    assert g.x_stddev == 4 * arcsec and g.y_stddev == 4 * arcsec
    g.y_stddev = 3 * arcsec
    assert g.x_stddev == 4 * arcsec and g.y_stddev == 3 * arcsec
    assert g.theta == 0
    g.y_stddev = 5 * arcsec
    assert g.x_stddev == 5 * arcsec and g.y_stddev == 4 * arcsec
    assert g.theta == 90 * units.Unit('degree')


def test_position_angle(gaussian2d):
    g = gaussian2d.copy()
    g.position_angle = 270 * units.Unit('degree')
    assert g.position_angle == 90 * units.Unit('degree')


def test_theta(gaussian2d):
    g = gaussian2d.copy()
    g.theta = -270 * units.Unit('degree')
    assert g.theta == 90 * units.Unit('degree')


def test_major_fwhm(gaussian2d):
    g = gaussian2d
    assert g.major_fwhm == g.x_fwhm


def test_minor_fwhm(gaussian2d):
    g = gaussian2d
    assert g.minor_fwhm == g.y_fwhm


def test_fwhm(gaussian2d):
    g = gaussian2d.copy()
    assert g.fwhm == np.sqrt(12) * arcsec
    g.fwhm = 5 * arcsec
    assert g.fwhm == 5 * arcsec
    assert g.x_fwhm == g.y_fwhm and g.x_fwhm == g.fwhm


def test_area(gaussian2d):
    g = gaussian2d.copy()
    assert np.isclose(g.area, 13.597080425481582 * units.Unit('arcsec2'))
    g.area = 4 * units.Unit('arcsec2')
    assert g.area == 4 * units.Unit('arcsec2')
    assert np.isclose(g.fwhm, 1.8788745573993026 * arcsec)
    assert g.fwhm == g.x_fwhm and g.fwhm == g.y_fwhm


def test_x_mean():
    g = Gaussian2D()
    model = functional_models.Gaussian2D()
    g.model = model
    assert isinstance(g.x_mean, float) and g.x_mean == 0

    g = Gaussian2D()
    g.x_mean = 1 * arcsec
    assert g.x_mean == 1 * arcsec


def test_y_mean():
    g = Gaussian2D()
    model = functional_models.Gaussian2D()
    g.model = model
    assert isinstance(g.y_mean, float) and g.y_mean == 0

    g = Gaussian2D()
    g.y_mean = 1 * arcsec
    assert g.y_mean == 1 * arcsec


def test_peak(gaussian2d):
    g = gaussian2d.copy()
    assert g.peak == 10 and isinstance(g.peak, float)
    g.peak = 11
    assert g.peak == 11 and isinstance(g.peak, float)


def test_str(gaussian2d):
    s = str(gaussian2d)
    assert s == ('x_fwhm=4.0 arcsec, y_fwhm=3.0 arcsec, x_mean=1.0 arcsec, '
                 'y_mean=2.0 arcsec, theta=120.0 deg, peak=10.0 Jy')


def test_repr(gaussian2d):
    s = repr(gaussian2d)
    assert s.endswith('x_fwhm=4.0 arcsec, y_fwhm=3.0 arcsec, x_mean=1.0 '
                      'arcsec, y_mean=2.0 arcsec, theta=120.0 deg, '
                      'peak=10.0 Jy')
    assert 'Gaussian2D object' in s


def test_eq(gaussian2d):
    g = gaussian2d.copy()
    assert g == g
    g2 = g.copy()
    assert g == g2
    x = -9999 * arcsec
    g2.y_mean = x
    assert g != g2
    g2.x_mean = x
    assert g != g2
    g2.y_fwhm = x
    assert g != g2
    g2.x_fwhm = x
    assert g != g2
    g2.theta = x
    assert g != g2
    assert g != 1


def test_set_xy_fwhm():
    g = Gaussian2D()
    g.set_xy_fwhm(1 * arcsec, 2 * arcsec)
    assert g.x_fwhm == 2 * arcsec and g.y_fwhm == 1 * arcsec
    assert g.theta == 90 * units.Unit('degree')
    g = Gaussian2D()
    g.set_xy_fwhm(2 * arcsec, 1 * arcsec)
    assert g.x_fwhm == 2 * arcsec and g.y_fwhm == 1 * arcsec
    assert g.theta == 0 * units.Unit('degree')


def test_validate():
    g = Gaussian2D()
    g.model.x_stddev = 1 * arcsec
    g.model.y_stddev = 2 * arcsec
    g.validate()
    assert g.model.x_stddev == 2 * arcsec and g.model.y_stddev == 1 * arcsec
    assert g.position_angle == 90 * units.Unit('degree')
    g = Gaussian2D()
    g.model.x_stddev = 2 * arcsec
    g.model.y_stddev = 1 * arcsec
    assert g.model.x_stddev == 2 * arcsec and g.model.y_stddev == 1 * arcsec
    assert g.position_angle == 0 * units.Unit('degree')


def test_get_circular_equivalent_fwhm(gaussian2d):
    g = gaussian2d
    assert np.isclose(g.get_circular_equivalent_fwhm(),
                      3.46410162 * arcsec, atol=1e-6)


def test_combine_with(gaussian2d):
    g = gaussian2d.copy()
    g2 = g.copy()
    g1 = g.copy()
    g1.combine_with(g2)
    assert np.isclose(g1.x_fwhm, 5.6568542 * arcsec, atol=1e-6)
    assert np.isclose(g1.y_fwhm, 4.2426407 * arcsec, atol=1e-6)
    assert np.isclose(g1.theta, 120 * units.Unit('degree'))

    g1 = g.copy()
    g2 = g.copy()
    g2.set_xy_fwhm(10 * arcsec, 10 * arcsec)
    g2.theta = 0 * units.Unit('degree')
    g1.combine_with(g2, deconvolve=True)
    assert g1.x_fwhm == 0 and g1.y_fwhm == 0

    g1 = g2.copy()
    g2.set_xy_fwhm(9.9 * arcsec, 9.9 * arcsec)
    g1.combine_with(g2, deconvolve=True)
    assert np.isclose(g1.x_fwhm, 1.4106736 * arcsec, atol=1e-6)
    assert np.isclose(g1.y_fwhm, 1.4106736 * arcsec, atol=1e-6)
    assert g1.theta == 0

    g1.set_xy_fwhm(2 * arcsec, 1 * arcsec)
    g1.theta = 0 * units.Unit('degree')
    g2.set_xy_fwhm(2 * arcsec, 1 * arcsec)
    g2.theta = 30 * units.Unit('degree')

    g1.combine_with(g2)
    assert np.isclose(g1.theta, 15 * units.Unit('degree'))
    assert np.isclose(g1.x_fwhm, 2.7564608 * arcsec, atol=1e-6)
    assert np.isclose(g1.y_fwhm, 1.5498141 * arcsec, atol=1e-6)

    g0 = g1.copy()
    g1.combine_with(None)
    assert g1 == g0


def test_convolve_with(gaussian2d):
    g = gaussian2d.copy()
    g2 = g.copy()
    g.convolve_with(g2)
    assert np.isclose(g.x_fwhm, 5.6568542 * arcsec, atol=1e-6)
    assert np.isclose(g.y_fwhm, 4.2426407 * arcsec, atol=1e-6)
    assert np.isclose(g.theta, 120 * units.Unit('degree'))


def test_deconvolve_with(gaussian2d):
    g = gaussian2d.copy()
    g2 = g.copy()
    g.deconvolve_with(g2)
    assert np.isclose(g.fwhm, 0 * arcsec)


def test_encompass(gaussian2d):
    g = gaussian2d.copy()
    g1 = g.copy()
    g1.encompass(1 * arcsec)
    assert g1 == g
    g1.encompass(10 * arcsec)
    assert g1.x_fwhm == 10 * arcsec
    assert g1.x_fwhm == 10 * arcsec
    g1 = g.copy()
    g2 = g.copy()
    g2.x_fwhm = 5 * arcsec
    g2.y_fwhm = 6 * arcsec
    g1.encompass(g2)
    assert g1.x_fwhm == 6 * arcsec
    assert g1.y_fwhm == 5 * arcsec
    assert g1.theta == 30 * units.Unit('degree')


def test_rotate(gaussian2d):
    g = gaussian2d.copy()
    g.rotate(130 * units.Unit('degree'))
    # 120 + 130 = 250; 250 % 180 == 70
    assert g.theta == 70 * units.Unit('degree')


def test_scale(gaussian2d):
    g = gaussian2d.copy()
    g.scale(2)
    assert g.x_fwhm == 8 * arcsec and g.y_fwhm == 6 * arcsec


def test_parse_header():
    h = fits.Header()
    h['BMAJ'] = 4, 'Beam major axis (arcmin)'
    h['BMIN'] = 5, 'Beam minor axis (arcsec)'
    h['BPA'] = 25, 'Beam position angle (deg)'
    g = Gaussian2D()
    g.parse_header(h)

    assert g.x_fwhm == 4 * units.Unit('arcmin') and g.y_fwhm == 5 * arcsec
    assert g.theta == 25 * units.Unit('degree')

    g.parse_header(h, size_unit='arcsecond')
    assert g.x_fwhm == 5 * arcsec and g.y_fwhm == 4 * arcsec
    # Rotated by 90 because minor FWHM > major FWHM
    assert g.theta == 115 * units.Unit('degree')

    del h['BMIN']
    g.parse_header(h)
    assert g.x_fwhm == 4 * units.Unit('arcmin') and g.x_fwhm == g.y_fwhm
    assert g.theta == 25 * units.Unit('degree')

    g = Gaussian2D()
    g0 = g.copy()
    with log.log_to_list() as log_list:
        g.parse_header(h, fits_id='A')
    assert log_list[0].msg == (
        "FITS header contains no beam description for type 'A'.")
    assert g == g0  # No change


def test_edit_header(gaussian2d):
    g = gaussian2d.copy()
    header = fits.Header()
    g.edit_header(header, beam_name='foo')
    assert header['BNAM'] == 'foo'
    assert header.comments['BNAM'] == 'Beam name.'
    assert header['BMAJ'] == 4.0
    assert header.comments['BMAJ'] == 'Beam major axis (arcsec).'
    assert header['BMIN'] == 3.0
    assert header.comments['BMIN'] == 'Beam minor axis (arcsec).'
    assert header['BPA'] == 120.0
    assert header.comments['BPA'] == 'Beam position angle (deg).'

    g.edit_header(header, size_unit='arcmin')
    assert header['BMAJ'] == 4 / 60
    assert header.comments['BMAJ'] == 'Beam major axis (arcmin).'
    assert header['BMIN'] == 3 / 60
    assert header.comments['BMIN'] == 'Beam minor axis (arcmin).'

    g = Gaussian2D(x_fwhm=2.0, y_fwhm=1.0, theta=45.0)
    header = fits.Header()
    g.edit_header(header, fits_id='A')
    assert header['ABMAJ'] == 2.0
    assert header.comments['ABMAJ'] == 'Beam major axis.'
    assert header['ABMIN'] == 1.0
    assert header.comments['ABMIN'] == 'Beam minor axis.'
    assert header['ABPA'] == 45.0
    assert header.comments['ABPA'] == 'Beam position angle (deg).'


def test_is_circular():
    g = Gaussian2D(x_fwhm=1, y_fwhm=1)
    assert g.is_circular()
    g = Gaussian2D(x_fwhm=2, y_fwhm=1)
    assert not g.is_circular()


def test_is_encompassing(gaussian2d):
    g1 = gaussian2d.copy()
    g2 = gaussian2d.copy()
    assert g1.is_encompassing(g2)
    g2.rotate(30 * units.Unit('degree'))
    assert not g1.is_encompassing(g2)
    g2.x_fwhm = 10 * units.Unit('arcsec')
    assert not g1.is_encompassing(g2)


def test_extent(gaussian2d):
    extent = gaussian2d.extent()
    assert np.isclose(extent.x, 3.278719 * arcsec, atol=1e-6)
    assert np.isclose(extent.y, 3.774917 * arcsec, atol=1e-6)


def test_get_beam_map(gaussian2d):
    grid = FlatGrid2D()
    grid.set_resolution(1 * arcsec)
    g = gaussian2d.copy()
    g.theta = 0 * units.Unit('degree')
    beam_map = g.get_beam_map(grid, sigmas=1.0)
    assert beam_map.shape == (7, 9) and beam_map.max() == 1
    assert beam_map[3, 4] == 1
    assert np.isclose(beam_map[5, 5], 0.24523252, atol=1e-6)
    # Check for symmetry
    idx = np.nonzero(beam_map == beam_map[5, 5])
    assert np.allclose(idx[0], [1, 1, 5, 5])
    assert np.allclose(idx[1], [3, 5, 3, 5])

    g.theta = 30 * units.Unit('degree')
    beam_map = g.get_beam_map(grid, sigmas=2.0)
    assert beam_map.shape == (15, 17)
    assert beam_map.max() == 1 and beam_map[7, 8] == 1
    assert np.isclose(beam_map[9, 10], 0.23257977, atol=1e-6)
    idx = np.nonzero(beam_map == beam_map[9, 10])
    assert np.allclose(idx[0], [5, 9])
    assert np.allclose(idx[1], [6, 10])


def test_get_equivalent(gaussian2d):
    g = gaussian2d.copy()
    grid = FlatGrid2D()
    grid.set_resolution(0.5 * arcsec)
    beam_map = g.get_beam_map(grid, sigmas=3)
    g2 = g.get_equivalent(beam_map, 0.5 * arcsec)
    assert np.isclose(g2.fwhm, g.fwhm)


def test_set_equivalent(gaussian2d):
    g = Gaussian2D()
    grid = FlatGrid2D()
    resolution = 0.5 * arcsec
    grid.set_resolution(resolution)
    beam_map = gaussian2d.get_beam_map(grid, sigmas=3)
    fwhm = gaussian2d.fwhm
    g.set_equivalent(beam_map, resolution)
    assert np.isclose(g.fwhm, fwhm)

    c = Coordinate2D([resolution, resolution])
    g.set_equivalent(beam_map, c)
    assert np.isclose(g.fwhm, fwhm)
    q = c.coordinates
    g.set_equivalent(beam_map, q)
    assert np.isclose(g.fwhm, fwhm)

    g = Gaussian2D()
    g.set_equivalent(beam_map, 0.5)
    assert np.isclose(g.fwhm.value, fwhm.value)


def test_set_area():
    g = Gaussian2D()
    g.set_area(4)
    assert g.area == 4
    g.set_area(4 * units.Unit('degree'))
    assert g.area == 4 * units.Unit('degree')
