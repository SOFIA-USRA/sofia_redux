# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.simulation.source_models.single_gaussian import \
    SingleGaussian

arcsec = units.Unit('arcsec')


def test_init():
    source = SingleGaussian(fwhm=10 * arcsec)
    assert source.name == 'single_gaussian'
    assert source.model.x_fwhm == 10 * arcsec


def test_copy():
    source = SingleGaussian(fwhm=10 * arcsec)
    source2 = source.copy()
    assert isinstance(source2, SingleGaussian) and source2 is not source
    assert source2.model.x_fwhm == 10 * arcsec


def test_initialize_model():
    source = SingleGaussian(fwhm=1 * arcsec)
    assert source.model.amplitude == 1
    assert source.model.x_mean == 0 * arcsec
    assert source.model.y_mean == 0 * arcsec
    assert source.model.x_fwhm == 1 * arcsec
    assert source.model.y_fwhm == 1 * arcsec

    with pytest.raises(ValueError) as err:
        source.initialize_model()
    assert 'Gaussian width parameter has not been supplied' in str(err.value)

    source.initialize_model(stddev=1 * arcsec)
    assert source.model.x_stddev == 1 * arcsec
    assert source.model.y_stddev == 1 * arcsec

    source.initialize_model(amplitude=2 * units.Unit('Jy'),
                            x_fwhm=1 * arcsec, y_fwhm=2 * arcsec)
    assert source.model.amplitude == 2 * units.Unit('Jy')
    assert source.model.x_fwhm == 1 * arcsec
    assert source.model.y_fwhm == 2 * arcsec

    info = ExampleInfo()
    source.initialize_model(info=info)
    assert source.model.x_fwhm == 10 * arcsec
    assert source.model.y_fwhm == 10 * arcsec


def test_apply_to_offsets():
    source = SingleGaussian(fwhm=4 * arcsec)
    line = np.linspace(-2, 2, 5)
    offsets = Coordinate2D([line, line], unit='arcsec')
    data = source(offsets)
    isr2 = 1 / np.sqrt(2)
    assert np.allclose(data, [1 / 4, isr2, 1, isr2, 1 / 4])


def test_apply_to_horizontal():
    source = SingleGaussian(fwhm=2 * arcsec)
    horizontal = HorizontalCoordinates()
    with pytest.raises(NotImplementedError) as err:
        _ = source(horizontal)
    assert "Cannot determine model from horizontal" in str(err.value)
