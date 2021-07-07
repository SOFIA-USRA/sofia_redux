# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy.testing as npt
import numpy as np
from sofia_redux.calibration.standard_model import background as bg


def strictly_decreasing(arr):
    return all(x > y for x, y in zip(arr, arr[1:]))


def strictly_increasing(arr):
    return all(x < y for x, y in zip(arr, arr[1:]))


@pytest.fixture(scope='module')
def warr():
    return np.linspace(50, 200, 5)


@pytest.fixture(scope='module')
def sources():
    return ['Ceres', 'Neptune']


@pytest.fixture(scope='module')
def temperatures(sources):
    Ttel = 240.  # Telescope temperature
    Tsky = 240.  # Sky temperature
    Twin = 278.  # Window temperature
    Tfo = 293.  # Foreoptics temperature
    Tinst = 10.  # Internal optics temperature

    temps = {'atmosphere': Tsky, 'telescope': Ttel,
             'window': Twin, 'foreoptics': Tfo,
             'instrument': Tinst}
    return temps


@pytest.fixture(scope='module')
def plancks(warr, temperatures):
    return bg.derive_background_photon_flux(warr, temperatures)


@pytest.fixture(scope='module')
def emissivity():
    eta_wabs = np.array([0.63, 0.67, 0.79, 0.86, 0.87])
    emiss = {'window': 1 - eta_wabs,
             'foreoptics': 0.04,
             'telescope': 0.15,
             'atmosphere': 1,
             'instrument': 1}
    return emiss


@pytest.fixture(scope='module')
def etas():
    eta_win = 0.92 * np.array([0.63, 0.67, 0.79, 0.86, 0.87])
    eta_inst = np.array([0.146, 0.190, 0.213, 0.286, 0.247])
    e = {'telescope': 0.85, 'foreoptics': 0.96,
         'window': eta_win, 'instrument': eta_inst}
    return e


@pytest.fixture(scope='module')
def total_throughput(warr):
    return 0.008 * np.ones_like(warr)


@pytest.fixture(scope='module')
def field_order():
    return ['atmosphere', 'telescope', 'window', 'foreoptics', 'instrument']


@pytest.fixture(scope='module')
def numerators(field_order):
    num = dict()
    for field in field_order:
        num[field] = np.arange(1, 10, 3)
    return num


def tfi(warr):
    return 0.9 * np.ones_like(warr)


def test_derive_background_photon_flux(warr, temperatures):
    planks = bg.derive_background_photon_flux(warr, temperatures)
    assert set(planks.keys()) == set(temperatures.keys())
    for key, value in planks.items():
        assert len(value) == len(warr)


def test_background_integrand_1(plancks, temperatures, warr, total_throughput,
                                emissivity, etas):
    bg_int_0 = bg.background_integrand_1(plancks, emissivity, etas,
                                         total_throughput, 0, 0)
    bg_int_1 = bg.background_integrand_1(plancks, emissivity, etas,
                                         total_throughput, 1, 2)
    assert bg_int_0.shape == warr.shape
    assert strictly_decreasing(bg_int_0)

    assert bg_int_1.shape == warr.shape
    assert strictly_decreasing(bg_int_1)
    assert np.alltrue(bg_int_0 < bg_int_1)
    assert np.alltrue(bg_int_1 < 1)


@pytest.mark.parametrize('filter_number', range(5))
def test_background_integrand_2(plancks, temperatures, warr, total_throughput,
                                emissivity, etas, filter_number):
    bg_int = bg.background_integrand_2(plancks, temperatures, warr,
                                       total_throughput, emissivity, etas,
                                       transmissions=tfi(warr),
                                       filter_number=filter_number)
    assert len(bg_int) == len(warr)
    assert strictly_decreasing(bg_int)


def test_integrand2(numerators, warr, temperatures, field_order):
    factors = bg.integrand2(numerators, warr, temperatures, field_order)
    assert len(factors) == len(field_order)
    assert set(factors.keys()) == set(field_order)
    for field, value in factors.items():
        assert len(value) == len(warr)
        assert np.alltrue(np.isfinite(value))


def test_background_integrand_coeff(warr, temperatures, field_order):
    numerator = np.arange(1, 10, 3)
    for field in field_order:
        factor = bg.background_integrand_coeff(numerator, warr,
                                               temperatures[field])
        assert len(factor) == len(warr)
        assert np.alltrue(np.isfinite(factor))
        assert strictly_increasing(factor)


def test_background_power(warr):
    num_pix = 6 * np.ones_like(warr)
    power = bg.background_power(telescope_area=5, omega_pix=1e-10,
                                bg_integrand1=5, num_pix=num_pix,
                                wavelengths=warr)
    npt.assert_allclose(power, 2.25e-6)


def test_setup_integrands(total_throughput, warr):
    model_flux = np.log10(warr)
    telescope_area = 5
    npix = 6
    fwhm = 5
    integrands = bg.setup_integrands(
        total_throughput=total_throughput,
        atmosphere_transmission=np.ones_like(warr),
        telescope_area=telescope_area,
        warr=warr,
        model_flux=model_flux,
        num_pix=npix * np.ones_like(warr),
        fwhm=fwhm * np.ones_like(warr))

    assert len(integrands) == 13
    for key, val in integrands.items():
        assert len(val) == len(warr)
    npt.assert_allclose(integrands['1'] / integrands['0'], telescope_area)
    npt.assert_allclose(integrands['2'] / integrands['1'], warr)
    npt.assert_allclose(integrands['12'] * warr**3, integrands['1'])


def test_integrate_integrands(total_throughput, warr):
    model_flux = np.log10(warr)
    telescope_area = 5
    npix = 6
    fwhm = 5
    integrands = bg.setup_integrands(
        total_throughput=total_throughput,
        atmosphere_transmission=np.ones_like(warr),
        telescope_area=telescope_area,
        warr=warr,
        model_flux=model_flux,
        num_pix=npix * np.ones_like(warr),
        fwhm=fwhm * np.ones_like(warr))
    integrals = bg.integrate_integrands(integrands, warr)
    assert set(integrals.keys()) == set(integrands.keys())
    for key, val in integrals.items():
        assert isinstance(val, np.float)
