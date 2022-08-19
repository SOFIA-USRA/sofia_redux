# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.modeling.functional_models import Gaussian1D
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.simulation.source_models.single_gaussian_2d1 import \
    SingleGaussian2d1


arcsec = units.Unit('arcsec')
um = units.Unit('um')


@pytest.fixture
def g_source():
    return SingleGaussian2d1(fwhm=5 * arcsec, z_fwhm=2 * um, z_mean=100 * um)


def test_init():
    g = SingleGaussian2d1(fwhm=5 * arcsec, z_fwhm=2 * um, z_mean=100 * um)
    assert isinstance(g.z_model, Gaussian1D)
    assert g.name == 'single_gaussian_2d1'


def test_copy(g_source):
    g = g_source.copy()
    assert isinstance(g_source, SingleGaussian2d1) and g_source is not g
    assert g.model.x_fwhm == 5 * arcsec


def test_initialize_model(g_source):
    g = g_source.copy()
    options = {'fwhm': 10 * arcsec, 'z_fwhm': 4 * um, 'z_mean': 120 * um}
    g.initialize_model(**options)
    assert g.model.x_fwhm == 10 * arcsec
    assert g.model.y_fwhm == 10 * arcsec
    assert g.z_model.fwhm == 4 * um
    assert g.z_model.mean == 120 * um
    assert g.z_model.amplitude == 1
    assert g.z_model.stddev == 4 * gaussian_fwhm_to_sigma * um
    del options['z_fwhm']
    options['z_stddev'] = 3 * gaussian_fwhm_to_sigma * um
    g.initialize_model(**options)
    assert g.z_model.fwhm == 3 * um
    del options['z_stddev']
    with pytest.raises(ValueError) as err:
        g.initialize_model(**options)
    assert 'Spectral width not specified' in str(err.value)
    del options['z_mean']
    with pytest.raises(ValueError) as err:
        g.initialize_model(**options)
    assert 'Spectral center not specified' in str(err.value)


def test_apply_to_offsets(g_source):
    c = Coordinate2D1([0 * arcsec, 0 * arcsec, 100 * um])
    data = g_source.apply_to_offsets(c)
    assert data == 1
    x = np.linspace(-2, 2, 5) * arcsec
    y = np.linspace(-2, 2, 5) * arcsec
    z = np.linspace(98, 102, 5) * um
    c = Coordinate2D1([x, y, z])
    data = g_source.apply_to_offsets(c)
    assert np.allclose(
        data, [0.02573722, 0.40053494, 1, 0.40053494, 0.02573722], atol=1e-4)


def test_apply_to_horizontal(g_source):
    with pytest.raises(NotImplementedError) as err:
        g_source.apply_to_horizontal(None)
    assert 'Cannot determine model from horizontal' in str(err.value)
