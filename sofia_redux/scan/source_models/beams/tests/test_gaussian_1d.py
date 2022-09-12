# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.source_models.beams.gaussian_1d import Gaussian1D
from sofia_redux.scan.coordinate_systems.grid.grid_1d import Grid1D

ud = units.dimensionless_unscaled
arcsec = units.Unit('arcsec')
jy = units.Unit('Jy')


def test_unit():
    g = Gaussian1D()
    assert g.unit == ud
    assert g.fwhm.unit == ud
    g = Gaussian1D(peak=1 * units.Unit('K'), position_unit='arcsec')
    assert g.fwhm.unit == arcsec
    assert g.unit == 'K'
    g = Gaussian1D(peak_unit='Jy')
    assert g.unit == 'Jy'


def test_copy():
    g = Gaussian1D()
    g2 = g.copy()
    assert g == g2 and g is not g2


def test_referenced_attributes():
    assert Gaussian1D().referenced_attributes == set([])


def test_fwhm():
    g = Gaussian1D()
    g.fwhm = 1
    assert g.fwhm == 1 * ud


def test_stddev():
    g = Gaussian1D()
    g.stddev = 1
    assert g.stddev == 1 * ud


def test_mean():
    g = Gaussian1D()
    g.mean = 1
    assert g.mean == 1 * ud
    g.mean = 1 * arcsec
    assert g.mean == 1 * arcsec


def test_peak():
    g = Gaussian1D()
    assert g.peak == 1
    g.peak = 2
    assert g.peak == 2
    g = Gaussian1D(peak_unit='K')
    assert g.peak == 1
    g.peak = 3000 * units.Unit('mK')
    assert g.peak == 3


def test_position_unit():
    g = Gaussian1D()
    assert g.position_unit == ud
    g = Gaussian1D(position_unit=arcsec)
    assert g.position_unit == arcsec


def test_str():
    assert str(Gaussian1D(peak_unit='Jy')) == 'fwhm=0.0, mean=0.0, peak=1.0 Jy'
    assert str(Gaussian1D()) == 'fwhm=0.0, mean=0.0, peak=1.0'


def test_repr():
    s = repr(Gaussian1D())
    assert s.endswith('fwhm=0.0, mean=0.0, peak=1.0') and 'object' in s


def test_eq():
    g = Gaussian1D(peak=1 * jy, fwhm=2 * arcsec, mean=3 * arcsec)
    g2 = g.copy()
    assert g == g
    assert g == g2
    assert g != None
    g2.fwhm *= 1.5
    assert g != g2
    g2 = g.copy()
    g2.mean *= 1.5
    assert g != g2


def test_validate():
    Gaussian1D().validate()  # does nothing


def test_combine_with():
    g = Gaussian1D(fwhm=1)
    g0 = g.copy()
    g.combine_with(None)
    assert g == g0
    g.combine_with(g, deconvolve=False)
    assert np.isclose(g.fwhm, np.sqrt(2))
    g.combine_with(Gaussian1D(fwhm=2), deconvolve=True)
    assert np.isclose(g.fwhm, 0)


def test_convolve_with():
    g = Gaussian1D(fwhm=1)
    g.convolve_with(g)
    assert g.fwhm == np.sqrt(2)


def test_deconvolve_with():
    g = Gaussian1D(fwhm=1)
    g.deconvolve_with(g)
    assert g.fwhm == 0


def test_encompass():
    g = Gaussian1D(fwhm=1)
    g.encompass(2 * ud)
    assert g.fwhm == 2
    g.encompass(Gaussian1D(fwhm=3))
    assert g.fwhm == 3


def test_is_encompassing():
    g = Gaussian1D(fwhm=1)
    assert g.is_encompassing(0.5 * ud)
    assert not g.is_encompassing(1.5 * ud)
    assert g.is_encompassing(Gaussian1D(fwhm=0.5))
    assert not g.is_encompassing(Gaussian1D(fwhm=1.5))


def test_scale():
    g = Gaussian1D(fwhm=2)
    g.scale(1.5)
    assert np.isclose(g.fwhm, 3)


def test_extent():
    g = Gaussian1D(fwhm=1)
    assert g.extent() == Coordinate1D(1)


def test_get_integral():
    g = Gaussian1D(fwhm=1)
    assert np.isclose(g.get_integral(), 1.06446702, atol=1e-5)


def test_set_integral():
    g = Gaussian1D(fwhm=3)
    g.set_integral(1.06446702)
    assert np.isclose(g.fwhm, 1)


def test_value_at():
    g = Gaussian1D(fwhm=1, peak=1, mean=0)
    assert np.isclose(g.value_at(0), 1)
    assert np.isclose(g.value_at(1), 0.0625)


def test_get_beam_unit():
    g = Gaussian1D(fwhm=1)
    assert np.isclose(g.get_beam_unit(), 1.06446702 * units.Unit('1/beam'),
                      atol=1e-5)


def test_get_beam_map():
    g = Gaussian1D(fwhm=1)
    grid = Grid1D()
    grid.resolution = 0.5 * ud
    beam = g.get_beam_map(grid)
    assert beam.size == 13
    assert np.allclose(beam[4:9], [0.0625, 0.5, 1, 0.5, 0.0625])

    g = Gaussian1D(fwhm=1 * arcsec, mean=0 * arcsec)
    grid = Grid1D()
    grid.resolution = 0.5 * arcsec
    beam = g.get_beam_map(grid)
    assert beam.size == 13
    assert np.allclose(beam[4:9], [0.0625, 0.5, 1, 0.5, 0.0625])


def test_get_equivalent():
    g = Gaussian1D.get_equivalent(
        np.array([0.0625, 0.5, 1, 0.5, 0.0625]), 0.5)
    assert np.isclose(g.fwhm, 1, atol=0.1)


def test_set_equivalent():
    g = Gaussian1D()
    beam_map = np.array([0.0625, 0.5, 1, 0.5, 0.0625])
    g.set_equivalent(beam_map, 0.5)
    assert np.isclose(g.fwhm, 1.0, atol=0.1)

    g.set_equivalent(beam_map, Coordinate1D(0.25))
    assert np.isclose(g.fwhm, 0.5, atol=0.05)


def test_parse_header():
    h = fits.Header()
    g = Gaussian1D()
    g.parse_header(h)
    assert g.fwhm == 0
    h['B1D'] = (1, '(degree)')
    g.parse_header(h)
    assert g.fwhm == 1 * units.Unit('degree')
    g.parse_header(h, size_unit='arcsec')
    assert g.fwhm == 1 * units.Unit('arcsec')


def test_edit_header():
    h = fits.Header()
    g = Gaussian1D(fwhm=1 * arcsec)
    g.edit_header(h, beam_name='FOO')
    assert h['BNAM'] == 'FOO'
    assert h['B1D'] == 1 and h.comments['B1D'] == 'Beam 1D axis (arcsec).'
    g.edit_header(h, beam_name='FOO', size_unit='arcmin')
    assert np.isclose(h['B1D'], 1/60)
    assert h.comments['B1D'] == 'Beam 1D axis (arcmin).'

    g = Gaussian1D()
    g.edit_header(h)
    assert h['B1D'] == 0 and h.comments['B1D'] == 'Beam 1D axis.'
