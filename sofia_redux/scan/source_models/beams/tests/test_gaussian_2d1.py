# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d1 import FlatGrid2D1
from sofia_redux.scan.source_models.beams.gaussian_1d import Gaussian1D
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.beams.gaussian_2d1 import Gaussian2D1

arcsec = units.Unit('arcsec')
um = units.Unit('um')
ud = units.dimensionless_unscaled


def test_class():
    assert np.isclose(Gaussian2D1.fwhm_to_size, 1.064467, atol=1e-5)


def test_init():
    g = Gaussian2D1()
    assert isinstance(g.z, Gaussian1D)


def test_copy():
    g = Gaussian2D1()
    g2 = g.copy()
    assert g == g2 and g is not g2


def test_referenced_attributes():
    assert Gaussian2D1().referenced_attributes == set([])


def test_z_fwhm():
    g = Gaussian2D1()
    assert g.z_fwhm == 0
    g.z_fwhm = 1
    assert g.z_fwhm == 1


def test_z_stddev():
    g = Gaussian2D1()
    assert g.z_stddev == 0
    g.z_stddev = 1
    assert g.z_stddev == 1


def test_volume():
    g = Gaussian2D1(x_fwhm=1, y_fwhm=2, z_fwhm=3)
    assert np.isclose(g.volume, 7.23682184, atol=1e-5)


def test_z_integral():
    g = Gaussian2D1(z_fwhm=1)
    assert g.z_integral == Gaussian2D1.fwhm_to_size
    g.z_integral = Gaussian2D1.fwhm_to_size * 2
    assert g.z_fwhm == 2


def test_z_mean():
    g = Gaussian2D1(z_mean=1 * um)
    assert g.z_mean == 1 * um
    g.z_mean = 2 * um
    assert g.z_mean == 2 * um


def test_str():
    g = Gaussian2D1()
    s = str(g)
    assert s == ('x_fwhm=0.0, y_fwhm=0.0, z_fwhm=0.0, x_mean=0.0, y_mean=0.0, '
                 'z_mean=0.0, theta=0.0 deg, peak=1.0')
    g = Gaussian2D1(peak_unit='Jy')
    assert str(g) == ('x_fwhm=0.0, y_fwhm=0.0, z_fwhm=0.0, x_mean=0.0, '
                      'y_mean=0.0, z_mean=0.0, theta=0.0 deg, peak=1.0 Jy')


def test_eq():
    g = Gaussian2D1()
    assert g == g
    g2 = g.copy()
    assert g2 == g
    g2.x_fwhm += 2
    assert g2 != g
    g2 = g.copy()
    g2.z_fwhm += 2
    assert g != g2


def test_set_xyz_fwhm():
    g = Gaussian2D1()
    g.set_xyz_fwhm(1, 2, 3)
    assert g.x_fwhm == 2 and g.y_fwhm == 1 and g.z_fwhm == 3
    assert g.theta == 90 * units.Unit('degree')


def test_validate():
    g = Gaussian2D1()
    g.model.x_stddev = 1 * ud
    g.model.y_stddev = 2 * ud
    assert g.theta == 0 * units.Unit('degree')
    g.validate()
    assert g.model.x_stddev == 2 * ud
    assert g.model.y_stddev == 1 * ud
    assert g.theta == 90 * units.Unit('degree')


def test_combine_with():
    g0 = Gaussian2D1(x_fwhm=2, y_fwhm=1, z_fwhm=3)
    g2 = Gaussian2D1(x_fwhm=1, y_fwhm=0.5, z_fwhm=2)
    g = g0.copy()
    g.combine_with(None)
    assert g == g0
    g.combine_with(g2, deconvolve=False)
    assert np.isclose(g.x_fwhm, 2.23606798, atol=1e-5)
    assert np.isclose(g.y_fwhm, 1.11803399, atol=1e-5)
    assert np.isclose(g.z_fwhm, 3.60555128, atol=1e-5)
    g.combine_with(g2, deconvolve=True)
    assert g == g0


def test_convolve_with():
    g = Gaussian2D1(x_fwhm=2, y_fwhm=1, z_fwhm=3)
    g2 = Gaussian2D1(x_fwhm=1, y_fwhm=0.5, z_fwhm=2)
    g.convolve_with(g2)
    assert np.isclose(g.x_fwhm, 2.23606798, atol=1e-5)
    assert np.isclose(g.y_fwhm, 1.11803399, atol=1e-5)
    assert np.isclose(g.z_fwhm, 3.60555128, atol=1e-5)


def test_deconvolve_with():
    g = Gaussian2D1(x_fwhm=2, y_fwhm=1, z_fwhm=3)
    g2 = Gaussian2D1(x_fwhm=1, y_fwhm=0.5, z_fwhm=2)
    g.deconvolve_with(g2)
    assert np.isclose(g.x_fwhm, 1.73205081, atol=1e-5)
    assert np.isclose(g.y_fwhm, 0.8660254, atol=1e-5)
    assert np.isclose(g.z_fwhm, 2.23606798, atol=1e-5)


def test_encompass():
    g = Gaussian2D1(x_fwhm=2, y_fwhm=2, z_fwhm=2)
    g.encompass(Gaussian2D1(x_fwhm=3, y_fwhm=3, z_fwhm=3))
    assert g.x_fwhm == 3 and g.y_fwhm == 3 and g.z_fwhm == 3
    g.encompass(Gaussian2D1(x_fwhm=3, y_fwhm=3, z_fwhm=3),
                z_psf=Gaussian1D(fwhm=4))
    assert g.x_fwhm == 3 and g.y_fwhm == 3 and g.z_fwhm == 4


def test_scale_z():
    g = Gaussian2D1(z_fwhm=1)
    g.scale_z(2)
    assert g.z_fwhm == 2


def test_parse_header():
    h = fits.Header()
    h['BMAJ'] = 4, 'Beam major axis (arcsec)'
    h['BMIN'] = 5, 'Beam minor axis (arcsec)'
    h['BPA'] = 25, 'Beam position angle (deg)'
    h['B1D'] = (1, '(um)')
    g = Gaussian2D1()
    g.parse_header(h)
    assert g.x_fwhm == 5 * arcsec
    assert g.y_fwhm == 4 * arcsec
    assert g.z_fwhm == 1 * um
    assert g.theta == 115 * units.Unit('degree')


def test_edit_header():
    h = fits.Header()
    g = Gaussian2D1(x_fwhm=2, y_fwhm=1, z_fwhm=3)
    g.edit_header(h)
    assert h['BMAJ'] == 2
    assert h['BMIN'] == 1
    assert h['B1D'] == 3
    assert h['BPA'] == 0


def test_is_encompassing():
    g = Gaussian2D1(x_fwhm=2, y_fwhm=2, z_fwhm=2)
    g2 = Gaussian2D(x_fwhm=1, y_fwhm=2)
    assert g.is_encompassing(g2)
    g2.x_fwhm = 3 * ud
    assert not g.is_encompassing(g2)
    g2 = g.copy()
    assert g.is_encompassing(g2)
    g2.z_fwhm = 3 * ud
    assert not g.is_encompassing(g2)


def test_extent():
    g = Gaussian2D1(x_fwhm=2, y_fwhm=1, z_fwhm=3)
    extent = g.extent()
    assert isinstance(extent, Coordinate2D1)
    assert extent.x == 2 and extent.y == 1 and extent.z == 3


def test_get_beam_map():
    grid = FlatGrid2D1()
    g = Gaussian2D1(x_fwhm=5, y_fwhm=5, z_fwhm=5)
    beam_map = g.get_beam_map(grid, sigmas=2)
    assert beam_map.shape == (21, 21, 21)
    assert beam_map[10, 10, 10] == 1
    assert np.isclose(beam_map[10, 10, 8], 0.6417129, atol=1e-5)
    beam_map = g.get_beam_map(grid, sigmas=Coordinate2D1([1, 2, 3]))
    assert beam_map.shape == (31, 21, 11)
    beam_map = g.get_beam_map(grid, sigmas=[1])
    assert beam_map.shape == (11, 11, 11)
    beam_map = g.get_beam_map(grid, sigmas=np.asarray(1))
    assert beam_map.shape == (11, 11, 11)
    beam_map = g.get_beam_map(grid, sigmas=[1, 2])
    assert beam_map.shape == (21, 11, 11)
    beam_map = g.get_beam_map(grid, sigmas=[1, 2, 3])
    assert beam_map.shape == (31, 21, 11)

    g = Gaussian2D1(x_fwhm=5 * arcsec,
                    y_fwhm=5 * arcsec,
                    z_fwhm=5 * um,
                    theta=45 * units.Unit('degree'))
    grid = FlatGrid2D1()
    grid.resolution = Coordinate2D1(xy=[1, 1] * arcsec, z=1 * um)
    reference = grid.resolution.copy()
    reference.zero()
    grid.reference = reference
    beam_map = g.get_beam_map(grid)
    for i in range(beam_map.ndim):
        assert 31 <= beam_map.shape[i] <= 33


def test_get_equivalent():
    grid = FlatGrid2D1()
    g = Gaussian2D1(x_fwhm=5, y_fwhm=5, z_fwhm=5)
    beam_map = g.get_beam_map(grid)
    g2 = Gaussian2D1.get_equivalent(beam_map, Coordinate2D1([1, 1, 1]))
    assert g2 == g


def test_set_equivalent():
    grid = FlatGrid2D1()
    g = Gaussian2D1(x_fwhm=5, y_fwhm=5, z_fwhm=5)
    beam_map = g.get_beam_map(grid)
    g2 = Gaussian2D1()
    g2.set_equivalent(beam_map, Coordinate2D1([1, 1, 1]))
    assert g2 == g
