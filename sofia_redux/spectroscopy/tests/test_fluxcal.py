# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
import pytest

from sofia_redux.spectroscopy.fluxcal import fluxcal


def spectral_data(nw=10):
    spectra = {}
    # three orders
    for order in range(3):
        spectra[order] = []
        # two spectra each
        for spec_i in range(2):
            # make a flat spectrum with a few peak values
            flux = np.full((nw, nw), 1.0)
            flux[:, 4] = 1.5
            flux[:, 5] = 2.0
            flux[:, 6] = 1.5
            error = np.full((nw, nw), 1.0)
            error[:, 4] = 1.5
            error[:, 5] = 2.0
            error[:, 6] = 1.5
            wave = np.arange(nw)
            spectral_flux = np.full((nw,), 1.0)
            spectral_flux[4] = 1.5
            spectral_flux[5] = 2.0
            spectral_flux[6] = 1.5
            spectral_error = np.full((nw,), 1.0)
            spectral_error[4] = 1.5
            spectral_error[5] = 2.0
            spectral_error[6] = 1.5
            spec = {'flux': flux, 'error': error, 'wave': wave,
                    'spectral_flux': spectral_flux,
                    'spectral_error': spectral_error}
            spectra[order].append(spec)

    atran = np.array([np.arange(nw * 2), np.full((nw * 2,), 0.5)])

    return spectra, atran


def test_default():
    spectra, atran = spectral_data()
    nw = spectra[0][0]['wave'].size

    result = fluxcal(spectra, atran)
    keys_2d = ['flux', 'error']
    keys_1d = ['wave', 'spectral_flux', 'spectral_error',
               'transmission', 'response', 'response_error']
    keys_0d = ['wave_shift', 'atran_index']
    assert len(result) == 3
    for order in result:
        for speci, spec in enumerate(result[order]):
            for key in keys_2d:
                assert key in spec
                assert spec[key].ndim == 2
                assert spec[key].shape[1] == nw
                # with atran=0.5 and no response, values are 2x input
                assert np.allclose(spec[key],
                                   2 * spectra[order][speci][key])
            for key in keys_1d:
                assert key in spec
                assert spec[key].ndim == 1
                assert spec[key].size == nw
                # with atran=0.5 and no response, values are 2x input
                if 'spectral' in key:
                    assert np.allclose(spec[key],
                                       2 * spectra[order][speci][key])
            for key in keys_0d:
                assert key in spec
                assert isinstance(spec[key], int) \
                    or isinstance(spec[key], float)
                # by default, wave_shift is zero, and without opt,
                # atran_index is also zero
                assert np.allclose(spec[key], 0)


def test_fail(capsys):
    spectra, atran = spectral_data()

    # make data bad in one spectrum
    spectra[1][1]['spectral_flux'] *= np.nan

    # result is None
    result = fluxcal(spectra, atran)
    assert result is None
    assert 'No good flux in order 1, spectrum 1' in capsys.readouterr().err


def test_manual_waveshift():
    spectra, atran = spectral_data()

    # add a 2 pixel waveshift for one spectrum
    spectra[1][1]['wave_shift'] = 2.0
    wave = spectra[0][0]['wave'].copy()

    # data in that spectrum is shifted, others stay the same
    result = fluxcal(spectra, atran)
    for order in result:
        for speci, spec in enumerate(result[order]):
            # wavelengths stay the same
            assert np.allclose(spec['wave'], wave)
            if order == 1 and speci == 1:
                # peak values are shifted
                assert np.allclose(spec['flux'][:, 7], 4.0)
                assert np.allclose(spec['error'][:, 7], 4.0)
                assert np.allclose(spec['spectral_flux'][7], 4.0)
                assert np.allclose(spec['spectral_error'][7], 4.0)
                assert np.allclose(spec['wave_shift'], 2.0)

                # first two values are nan
                assert np.all(np.isnan(spec['flux'][:, 0:2]))
                assert np.all(np.isnan(spec['error'][:, 0:2]))
                assert np.all(np.isnan(spec['spectral_flux'][0:2]))
                assert np.all(np.isnan(spec['spectral_error'][0:2]))
            else:
                # values are not shifted
                assert np.allclose(spec['flux'][:, 7], 2.0)
                assert np.allclose(spec['error'][:, 7], 2.0)
                assert np.allclose(spec['spectral_flux'][7], 2.0)
                assert np.allclose(spec['spectral_error'][7], 2.0)
                assert np.allclose(spec['wave_shift'], 0.0)

                # no nans
                assert not np.any(np.isnan(spec['flux']))
                assert not np.any(np.isnan(spec['error']))
                assert not np.any(np.isnan(spec['spectral_flux']))
                assert not np.any(np.isnan(spec['spectral_error']))


@pytest.mark.parametrize('peak_index,subsample',
                         [(126, 1), (128, 1), (130, 1),
                          (130, 10)])
def test_auto_waveshift(peak_index, subsample):
    nw = 256
    spectra, atran = spectral_data(nw)

    # make some more feature-rich spectral data
    spectra = {0: [spectra[0][0]]}
    spectra[0][0]['wave'] = np.arange(nw, dtype=float)
    spectra[0][0]['spectral_flux'] = np.full(nw, 1.0)
    spectra[0][0]['spectral_error'] = np.full(nw, 1.0)

    from astropy.modeling.models import Gaussian1D
    center = 128
    offset = 30
    g1 = Gaussian1D(amplitude=2.0, mean=center, stddev=3)
    g2 = Gaussian1D(amplitude=2.0, mean=(center + offset), stddev=3)
    spectra[0][0]['spectral_flux'] += g1(spectra[0][0]['wave'])
    spectra[0][0]['spectral_flux'] += g2(spectra[0][0]['wave'])

    # add almost the same peaks to atran to correlate with, but shifted
    g3 = Gaussian1D(amplitude=2.0, mean=peak_index, stddev=3)
    g4 = Gaussian1D(amplitude=2.0, mean=(peak_index + offset), stddev=3)
    atran[1, :] = 1.0
    atran[1] += g3(atran[0])
    atran[1] += g4(atran[0])

    result = fluxcal(spectra, atran, auto_shift=True,
                     shift_limit=3, model_order=0,
                     shift_subsample=subsample)
    for order in result:
        for spec in result[order]:
            # the direction of the shift should be negative if
            # atran peak is lower, positive if higher, zero if same
            assert np.sign(spec['wave_shift']) == np.sign(peak_index - center)
            assert np.allclose(spec['wave_shift'], peak_index - center)


def test_waveshift_limits(mocker, capsys):
    # set log level to debug
    olog = log.level
    log.setLevel('DEBUG')

    spectra, atran = spectral_data()

    # mock shift function to return specific shift values

    # too small value: set to zero
    mocker.patch('sofia_redux.spectroscopy.fluxcal.get_wave_shift',
                 return_value=0.05)
    result = fluxcal(spectra, atran, auto_shift=True)
    for order in result:
        for spec in result[order]:
            assert spec['wave_shift'] == 0
    assert 'very small. Setting to zero' in capsys.readouterr().out

    # too big value: set to zero, warn
    mocker.patch('sofia_redux.spectroscopy.fluxcal.get_wave_shift',
                 return_value=6.0)
    result = fluxcal(spectra, atran, auto_shift=True, shift_limit=5)
    for order in result:
        for spec in result[order]:
            assert spec['wave_shift'] == 0
    assert 'too large. Not applying' in capsys.readouterr().err

    # reset log level
    log.setLevel(olog)


def test_bad_waveshift(mocker, capsys):
    spectra, atran = spectral_data()

    # set correction data to Nan -- waveshift will return NaN
    atran[1, :] = np.nan

    result = fluxcal(spectra, atran, auto_shift=True)
    for order in result:
        for spec in result[order]:
            assert spec['wave_shift'] == 0
    assert 'Could not calculate wave shift; ' \
           'setting to 0.' in capsys.readouterr().err


def test_optimize_atran():
    spectra, atran = spectral_data()
    atran_list = []
    for i in range(5):
        # add a peak to correct out, at a different position
        # in each atran
        atran_copy = atran.copy()
        atran_copy[1, 7 - i] *= 2.0
        atran_list.append(atran_copy)

    # run fluxcal with wave shift off
    result = fluxcal(spectra, atran_list, auto_shift=False,
                     model_order=3)

    # the middle one should match best
    for order in result:
        for spec in result[order]:
            assert spec['atran_index'] == 2

            # check that additional diagnostic info is added
            assert 'fit_chisq' in spec
            assert 'fit_rchisq' in spec
            assert 'all_corrected' in spec
            assert len(spec['all_corrected']) == 5


def test_response():
    spectra, atran = spectral_data()

    response = {}
    for order in spectra:
        # set response to spectra / atran -- should come out 1.0
        # (also expand the wavelength range to test interpolation)
        rtest = 1 / atran[1]
        rtest[:10] *= spectra[order][0]['spectral_flux']
        rtest = np.hstack([[1.0, 1.0, 1.0], rtest])
        response[order] = {'wave': np.hstack([[-3, -2, -1], atran[0]]),
                           'response': rtest,
                           'error': np.full_like(rtest, 1.0)}

    result = fluxcal(spectra, atran, response=response, auto_shift=False)

    for order in result:
        for speci, spec in enumerate(result[order]):
            # all fluxes come out 1
            for key in ['flux', 'spectral_flux']:
                assert np.allclose(spec[key], 1.0)
            # error is a little higher than 1
            for key in ['error', 'spectral_error']:
                assert np.all(spec[key] > 1.0)
                assert np.all(spec[key] < 2.0)
