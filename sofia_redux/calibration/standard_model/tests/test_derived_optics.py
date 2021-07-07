# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy.testing as npt

import numpy as np

from sofia_redux.calibration.standard_model import derived_optics as dopt
from sofia_redux.calibration.standard_model import calibration_io as cio


def strictly_increasing(arr):
    return all(x < y for x, y in zip(arr, arr[1:]))


@pytest.fixture(scope='module')
def integrals():
    ints = dict()
    ints['0'] = 2
    ints['1'] = 4
    ints['2'] = 921
    ints['3'] = 32
    ints['4'] = 8
    ints['5'] = 918111
    ints['6'] = 55
    ints['7'] = 1e-4
    ints['8'] = 0.4
    ints['9'] = 1e3
    ints['10'] = 100
    ints['11'] = 3
    return ints


@pytest.fixture(scope='function')
def results():
    res = dict()
    res['lambda_pivot'] = 50
    res['lambda_mean'] = 40
    res['lambda_1'] = 60
    res['nep'] = 5e-14
    res['flux_mean'] = 15
    res['flux_nu_mean'] = 10
    return res


@pytest.fixture(scope='module')
def wavelengths():
    return np.linspace(5, 300, 30)


@pytest.fixture(scope='module')
def model_flux_in_filter(wavelengths):
    flux = 0.5 * wavelengths + 3
    return flux


def test_mean_fluxes(integrals, results):
    res = dopt.mean_fluxes(results, integrals)
    npt.assert_allclose(res['flux_mean'], 2)
    npt.assert_allclose(res['flux_nu_mean'], 1667820475990760)


def test_mean_pixels_in_beam(wavelengths):
    num_pix = 5 * np.ones_like(wavelengths)
    total_throughput = 0.01 * np.ones_like(wavelengths)
    npix = dopt.mean_pixels_in_beam(num_pix, total_throughput, wavelengths)
    npt.assert_allclose(num_pix, 5)
    assert isinstance(npix, float)


def test_noise_equivalent_power(wavelengths):
    integrand1 = 5
    integrand2 = 2.5 * np.ones_like(wavelengths)
    num_pix = 5 * np.ones_like(wavelengths)
    omega_pix = 2 * np.ones_like(wavelengths)
    telescope_area = 10
    correct_nep = 3.56241888e-8

    nep = dopt.noise_equivalent_power(bg_integrand1=integrand1,
                                      bg_integrand2=integrand2,
                                      num_pix=num_pix,
                                      wavelengths=wavelengths,
                                      omega_pix=omega_pix,
                                      telescope_area=telescope_area)
    assert len(nep) == len(wavelengths)
    npt.assert_allclose(nep, correct_nep)


def test_limiting_flux(results, integrals):
    snr_ref = 4.
    correct_nefd = 0.005559
    correct_mdcf = 0.524145
    res = dopt.limiting_flux(results, integrals, snr_ref)

    npt.assert_allclose(res['nefd'], correct_nefd, rtol=1e-4)
    npt.assert_allclose(res['mdcf'], correct_mdcf, rtol=1e-4)


def test_response(results, integrals):
    resp, respnu = dopt.response(results, integrals)

    npt.assert_allclose(resp, 4)
    npt.assert_allclose(respnu, 4.796679e-18)


def test_source_descriptions(results, integrals):
    ffrac = 0.715
    correct_size = 0.05
    correct_fwhm = 125
    correct_rate = 5.72
    res = dopt.source_descriptions(results, integrals, ffrac)
    npt.assert_allclose(res['source_size'], correct_size)
    npt.assert_allclose(res['source_fwhm'], correct_fwhm)
    npt.assert_allclose(res['source_rate'], correct_rate)


def test_color_terms(results, model_flux_in_filter, wavelengths):
    correct_k0 = 0.6521739130434783
    correct_k1 = 0.30303030303030304
    res = dopt.color_terms(results, fref=None, pl=False, bb=False,
                           alpha=-2, wref=None, temp=300,
                           model_flux_in_filter=model_flux_in_filter,
                           wavelengths=wavelengths)

    assert res['color_term_k0'] == correct_k0
    assert res['color_term_k1'] == correct_k1


def test_flux_reference_wavelength(results, model_flux_in_filter, wavelengths):
    correct = [23, 33]
    lam = dopt.flux_reference_wavelength(
        results, model_flux_in_filter=model_flux_in_filter,
        wavelengths=wavelengths)
    npt.assert_allclose(lam, correct, rtol=1e-4)


def test_flux_reference_wavelength_powerlaw(results):
    fref = 3
    wref = 9
    alpha = -1.5
    correct = [5.26682e-14, 4.300338e-14]
    lam = dopt.flux_reference_wavelength(results, pl=True, alpha=alpha,
                                         fref=fref, wref=wref)
    npt.assert_allclose(lam, correct, rtol=1e-4)


def test_flux_reference_wavelength_blackbody(results):
    correct = [1.150442, 0.302007]
    lam = dopt.flux_reference_wavelength(results, bb=True, temp=250)
    npt.assert_allclose(lam, correct, rtol=1e-4)


@pytest.mark.parametrize('iq,sig', [(5.08, 2), (0, 0)])
def test_pointing_optics_sigma(iq, sig):
    opt = dopt.pointing_optics_sigma(iq)
    npt.assert_allclose(opt, sig)


def test_source_size(wavelengths):
    iq = 0.5
    theta_pix = 0.2 * np.ones_like(wavelengths)
    fwhm, num_pix, ffrac = dopt.source_size(wavelengths, iq, theta_pix)
    assert len(fwhm) == len(wavelengths)
    assert len(num_pix) == len(wavelengths)
    assert strictly_increasing(fwhm)
    npt.assert_allclose(ffrac, 0.715)


@pytest.mark.parametrize('filt,wmin,wmax,natm', [('A', 44.50, 65.75, 10),
                                                 ('B', 53.85, 75.45, 20),
                                                 ('C', 73.70, 113.10, 30),
                                                 ('D', 118.80, 210.32, 40),
                                                 ('E', 172.08, 292.46, 50)])
def test_apply_filter(wavelengths, filt, wmin, wmax, natm):
    caldata = cio.calibration_data_path()
    filter_name = f'HAWC_band{filt}.txt'
    awave = np.linspace(5, 300, natm)
    atran = np.ones_like(awave)
    transmission = np.ones_like(wavelengths)
    flux = np.log10(wavelengths)
    wf, tf, taf, fsi, warr, fname = \
        dopt.apply_filter(caldata, filter_name, atmosphere_wave=awave,
                          atmosphere_transmission=atran,
                          model_wave=wavelengths, model_flux=flux)
    assert filter_name in fname
    assert np.min(warr) >= wmin
    assert np.max(warr) <= wmax
    assert np.min(taf) >= np.min(transmission)
    assert np.max(taf) <= np.max(transmission)
    assert np.min(fsi) >= np.min(flux)
    assert np.max(fsi) <= np.max(flux)
