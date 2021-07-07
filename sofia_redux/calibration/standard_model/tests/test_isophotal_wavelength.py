# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy.testing as npt
import numpy as np
import pandas as pd

from sofia_redux.calibration.standard_model import isophotal_wavelength as iso


@pytest.fixture(scope='module')
def result():
    s = dict()
    s['lambda_c'] = 50
    s['lambda_mean'] = 50
    s['lambda_1'] = 50
    s['lambda_pivot'] = 50
    s['lambda_eff'] = 50
    s['lambda_eff_jv'] = 50
    s['isophotal_wt'] = 50
    s['width'] = 50
    s['response'] = 50
    s['flux_mean'] = 10000
    s['flux_nu_mean'] = 50
    s['color_term_k0'] = 50
    s['color_term_k1'] = 50
    s['source_rate'] = 50
    s['source_size'] = 50
    s['source_fwhm'] = 50
    s['background_power'] = 50
    s['nep'] = 50
    s['nefd'] = 50
    s['mdcf'] = 50
    s['npix_mean'] = 50
    s['lambda_prime'] = 50
    s['lamcorr'] = 50
    s = pd.Series(s)
    return s


@pytest.fixture(scope='module')
def integrals():
    i = dict()
    i['1'] = 4
    i['2'] = 2
    i['3'] = 10
    i['4'] = 38
    i['5'] = 8
    i['6'] = 5
    i['7'] = 1
    i['8'] = 20
    i['9'] = 3
    i['10'] = 50
    i['11'] = 15
    i['12'] = 30
    return i


@pytest.fixture(scope='module')
def wavelengths():
    return np.linspace(5, 300, 30)


@pytest.fixture(scope='module')
def model_flux_in_filter(wavelengths):
    flux = 0.5 * wavelengths + 3
    return flux


@pytest.fixture(scope='module')
def curved_model_flux_in_filter(wavelengths):
    flux = wavelengths + 200 * np.sin(wavelengths / 40)
    return flux


@pytest.fixture(scope='module')
def atmosphere_transmission_in_filter(wavelengths):
    trans = np.ones_like(wavelengths)
    return trans


def test_calc_isophotal(result, wavelengths, curved_model_flux_in_filter,
                        atmosphere_transmission_in_filter):
    result['flux_mean'] = np.mean(curved_model_flux_in_filter)
    r = iso.calc_isophotal(result, wavelengths, False, None,
                           curved_model_flux_in_filter,
                           atmosphere_transmission_in_filter,
                           total_throughput=8 * np.ones_like(wavelengths))
    assert r['isophotal'] < wavelengths.max()
    assert r['isophotal'] > wavelengths.min()
    assert r['isophotal_wt'] < wavelengths.max()
    assert r['isophotal_wt'] > wavelengths.min()


def test_calc_isophotal_pl(result, wavelengths, curved_model_flux_in_filter,
                           atmosphere_transmission_in_filter):
    r = iso.calc_isophotal(result, wavelengths, True, None,
                           curved_model_flux_in_filter,
                           atmosphere_transmission_in_filter,
                           total_throughput=8)
    assert r['isophotal'] < wavelengths.max()
    assert r['isophotal'] > wavelengths.min()
    assert r['isophotal_wt'] < wavelengths.max()
    assert r['isophotal_wt'] > wavelengths.min()


def test_isophotal_from_powerlaw_alpha2(result):
    isopho = iso.isophotal_from_powerlaw(result, alpha=-2)
    npt.assert_allclose(isopho, result['lambda_mean'])
    assert len(isopho) == 2


def test_isophotal_from_powerlaw_single_zero(result, wavelengths,
                                             model_flux_in_filter):
    taf = np.ones_like(wavelengths)
    fsi = np.ones_like(wavelengths)
    result['flux_mean'] = np.mean(model_flux_in_filter)
    fzero = model_flux_in_filter - np.mean(model_flux_in_filter)
    isopho = iso.isophotal_from_powerlaw(result, alpha=-1,
                                         warr=wavelengths,
                                         flux=fzero,
                                         total_throughput=4,
                                         taf=taf, fsi=fsi)
    npt.assert_allclose(isopho, np.mean(wavelengths))
    assert len(isopho) == 2


def test_isophotal_from_powerlaw_multiple_zeros(result, wavelengths,
                                                curved_model_flux_in_filter):
    taf = np.ones_like(wavelengths)
    fsi = np.ones_like(wavelengths)
    result['flux_mean'] = np.mean(curved_model_flux_in_filter)
    fzero = curved_model_flux_in_filter - np.mean(curved_model_flux_in_filter)
    isopho = iso.isophotal_from_powerlaw(result, alpha=-1,
                                         warr=wavelengths,
                                         flux=fzero,
                                         total_throughput=4,
                                         taf=taf, fsi=fsi)
    npt.assert_allclose(isopho, 127.7031207)
    assert len(isopho) == 2


def test_isophotal_from_zeros_single_zero(wavelengths):
    wave_zeros = 150
    total_throughput = 4
    fsi = np.ones_like(wavelengths)
    taf = np.ones_like(wavelengths)

    isopho = iso.isophotal_from_zeros(wave_zeros=wave_zeros, taf=taf,
                                      total_throughput=total_throughput,
                                      warr=wavelengths, fsi=fsi)
    npt.assert_allclose(isopho, wave_zeros)
    npt.assert_almost_equal(isopho[0], isopho[1])
    assert len(isopho) == 2


def test_isophotal_from_zeros_multiple_zeros(wavelengths):
    wave_zeros = [75, 150]
    correct_isopho = [112.5, 125]
    total_throughput = 4
    fsi = 3 * wavelengths
    taf = np.ones_like(wavelengths)

    isopho = iso.isophotal_from_zeros(wave_zeros=wave_zeros, taf=taf,
                                      total_throughput=total_throughput,
                                      warr=wavelengths, fsi=fsi)
    npt.assert_array_almost_equal(isopho, correct_isopho)
    assert len(isopho) == 2


@pytest.mark.parametrize('x0,y0', [(50, 238.396),
                                   (-30, -174.07473)])
def test_interpol(wavelengths, curved_model_flux_in_filter, x0, y0):
    y = iso.interpol(curved_model_flux_in_filter, wavelengths, x0)
    npt.assert_allclose(y, y0, rtol=5e-4)


def test_calculated_lambdas(result, integrals):
    correct = dict()
    correct['lambda_1'] = 25
    correct['nuref'] = 0.666666666
    correct['irac'] = 0.4
    correct['sdss'] = 1.1051709180756477
    correct['eff_ph'] = 0.625
    correct['lambda_mean'] = 0.5
    correct['lambda_eff'] = 0.21052631578947367
    # correct['lambda_pivot'] = 0.26666666666666666
    correct['lambda_pivot'] = 0.516398
    correct['lambda_eff_jv'] = 0.5
    correct['rms'] = 2.179449471770337
    correct['lambda_prime'] = 0.4
    correct['lamcorr'] = 0.75
    result = iso.calculated_lambdas(result, integrals)
    for key, value in correct.items():
        npt.assert_allclose(result[key], value, rtol=1e-5)
