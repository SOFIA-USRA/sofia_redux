# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy.testing as npt
import numpy as np
from sofia_redux.calibration.standard_model import thermast


@pytest.fixture
def wavelength_array():
    return np.linspace(5, 30, 5)


@pytest.fixture
def omega_array():
    return np.linspace(0, np.pi / 2, 5)


def test_bbflux(omega_array):
    flux = thermast.bbflux(omega=omega_array, tss=300, w=20)
    assert len(flux) == len(omega_array)
    assert all(flux >= 0)


def test_planck_function(wavelength_array):
    planck = thermast.planck_function(wavelength_array, 200)
    assert len(planck) == len(wavelength_array)
    assert all(np.log10(planck) < 1)


def test_thermast():
    nw = 2500
    wave, flux = thermast.thermast(Gmag=0.15, pV=0.9, phang=45,
                                   dsun=2.5, dist=2.7, nw=nw,
                                   rsize=300)
    assert len(wave) == nw
    assert len(flux) == nw
    assert all(flux) > 0
    # Rough test to make sure in Janskys
    npt.assert_array_less(np.log10(flux), 3)


def test_flux_at_w(wavelength_array):
    flux1 = thermast.flux_at_w(wavelength_array, 300,
                               0, np.pi / 2.)
    flux2 = thermast.flux_at_w(wavelength_array, 300)
    flux3 = thermast.flux_at_w(wavelength_array, 300,
                               -np.pi / 2, np.pi)
    flux = np.array([flux2, flux3]) / flux1
    npt.assert_allclose(flux, np.ones_like(flux))


@pytest.mark.parametrize('angle,factor', [(0., 1),
                                          (30., 0.75858),
                                          (45., 0.660693),
                                          (90., 0.436516)])
def test_correction_factor(angle, factor):
    correction = thermast.correction_factor(angle)
    npt.assert_allclose(correction, factor, atol=1e-5)
