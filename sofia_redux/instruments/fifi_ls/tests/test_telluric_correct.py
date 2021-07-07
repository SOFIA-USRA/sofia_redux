# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.fifi_ls.telluric_correct \
    import telluric_correct, wrap_telluric_correct, apply_atran_correction
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_scm_files


class TestTelluricCorrect(FIFITestCase):

    def test_success(self):
        files = get_scm_files()
        result = telluric_correct(files[0], write=False, outdir=None)
        assert isinstance(result, fits.HDUList)
        assert result[0].header['PRODTYPE'] == 'telluric_corrected'
        bunit = ['adu/(Hz s)', 'adu/(Hz s)',
                 'adu/(Hz s)', 'adu/(Hz s)', 'um',
                 'arcsec', 'arcsec', '', '']
        exts = ['FLUX', 'STDDEV',
                'UNCORRECTED_FLUX', 'UNCORRECTED_STDDEV',
                'LAMBDA', 'XS', 'YS', 'ATRAN', 'UNSMOOTHED_ATRAN']
        for i, extname in enumerate(exts):
            assert extname in result
            assert result[extname].header['BUNIT'] == bunit[i]

    def test_write(self, tmpdir):
        files = get_scm_files()
        result = telluric_correct(files[0], write=True, outdir=str(tmpdir))
        assert os.path.isfile(result)
        result = telluric_correct(files[0], write=False, outdir=str(tmpdir))
        assert isinstance(result, fits.HDUList)

    def test_cutoff(self):
        files = get_scm_files()
        result1 = telluric_correct(
            files[0], write=False, outdir=None, cutoff=0)
        result2 = telluric_correct(
            files[0], write=False, outdir=None, cutoff=0.999)
        n1 = np.isnan(result1['FLUX'].data).sum()
        n2 = np.isnan(result2['FLUX'].data).sum()
        assert n2 > n1

    def test_no_correction(self):
        filename = get_scm_files()[0]
        default = telluric_correct(filename, skip_corr=False, cutoff=0.8)
        result = telluric_correct(filename, skip_corr=True, cutoff=0.8)

        assert 'Telluric corrected' in str(default[0].header['HISTORY'])
        assert 'Telluric corrected' not in str(result[0].header['HISTORY'])
        assert 'Telluric spectrum attached' in str(result[0].header['HISTORY'])

        # output does not have uncorrected extensions; should
        # have everything else
        for extname in ['FLUX', 'STDDEV', 'LAMBDA', 'XS', 'YS',
                        'ATRAN', 'UNSMOOTHED_ATRAN']:
            assert extname in result
        for extname in ['UNCORRECTED_FLUX', 'UNCORRECTED_STDDEV']:
            assert extname not in result

        # output data should have fewer nans
        assert np.sum(np.isnan(result['FLUX'].data)) < \
            np.sum(np.isnan(default['FLUX'].data))

        # data is same as default uncorrected data
        assert np.allclose(result['FLUX'].data,
                           default['UNCORRECTED_FLUX'].data,
                           equal_nan=True)
        assert np.allclose(result['STDDEV'].data,
                           default['UNCORRECTED_STDDEV'].data,
                           equal_nan=True)

    def test_hdul_input(self):
        filename = get_scm_files()[0]
        hdul = fits.open(filename)
        result = telluric_correct(hdul)
        assert isinstance(result, fits.HDUList)

    def test_bad_parameters(self, capsys):
        files = get_scm_files()

        # bad output directory
        result = telluric_correct(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = telluric_correct('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_cal_failure(self, mocker, capsys):
        filename = get_scm_files()[0]

        # mock failures in subroutines

        mocker.patch(
            'sofia_redux.instruments.fifi_ls.telluric_correct.apply_atran',
            return_value=None)
        result = telluric_correct(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Unable to apply ATRAN' in capt.err

        mocker.patch(
            'sofia_redux.instruments.fifi_ls.telluric_correct.get_atran',
            return_value=None)
        result = telluric_correct(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Unable to get ATRAN' in capt.err

        mocker.patch(
            'sofia_redux.instruments.fifi_ls.telluric_correct.get_resolution',
            return_value=None)
        result = telluric_correct(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Unable to determine spectral resolution' in capt.err

    def test_wrap(self):
        files = get_scm_files()

        # serial
        result = wrap_telluric_correct(files, write=False)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

        # multithreading
        result = wrap_telluric_correct(files, write=False, jobs=-1)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.telluric_correct.multitask',
            return_value=['test', None])

        # bad files
        result = wrap_telluric_correct(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_scm_files()
        wrap_telluric_correct(files[0], write=False,
                              allow_errors=False)

        # allow errors
        result = wrap_telluric_correct(files, write=False,
                                       allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_telluric_correct(files, write=False,
                                       allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_otf_reshape(self):
        files = get_scm_files()
        hdul = fits.open(files[0])

        # mock OTF data: should have 3D flux, stddev that get modified;
        # rest of extensions are copied as is
        nramp = 15
        nw = hdul['FLUX'].data.shape[0]
        flux = np.repeat(hdul['FLUX'].data.reshape(1, nw, 25),
                         nramp, axis=0)
        stddev = np.repeat(hdul['STDDEV'].data.reshape(1, nw, 25),
                           nramp, axis=0)
        hdul['FLUX'].data = flux
        hdul['STDDEV'].data = stddev

        result = telluric_correct(hdul, write=False, outdir=None)
        assert result['FLUX'].data.shape == (nramp, nw, 25)
        assert result['STDDEV'].data.shape == (nramp, nw, 25)

        for ext in result[1:]:
            name = ext.header['EXTNAME']
            if name == 'FLUX' or name == 'STDDEV':
                assert not np.allclose(ext.data, hdul[name].data)
            elif name == 'UNCORRECTED_FLUX' or name == 'UNCORRECTED_STDDEV':
                assert np.allclose(ext.data,
                                   hdul[name.replace('UNCORRECTED_', '')].data,
                                   equal_nan=True)
            elif 'ATRAN' in name:
                continue
            else:
                assert np.allclose(ext.data, hdul[name].data)

        # flux and stddev should have same factors
        assert np.allclose(result['FLUX'].data / hdul['FLUX'].data,
                           result['STDDEV'].data / hdul['STDDEV'].data,
                           equal_nan=True)
        # same correction should be applied to all ramps
        for ramp in range(nramp):
            data = result['FLUX'].data[ramp]
            good = ~np.isnan(data)
            assert np.allclose(hdul['FLUX'].data[ramp][good] / data[good],
                               result['ATRAN'].data[good])

    def test_apply_atran(self):
        wave = np.repeat((np.arange(16, dtype=float) + 2).reshape(16, 1), 25,
                         axis=1)
        data = np.ones((10, 16, 25))
        var = np.ones((10, 16, 25))
        awave = np.arange(100, dtype=float)
        atran = np.arange(100, dtype=float) / 100.

        expected_atran = np.ones((10, 16, 25))
        expected_atran *= atran[None, 2:18, None]

        result = apply_atran_correction(wave, data, var,
                                        np.array([awave, atran]), 0.0)
        tel_corr, var_corr, atran_store = result

        # returned atran should be close to expected
        assert np.allclose(atran_store, expected_atran)
        # corrected data should be divided by it
        assert np.allclose(tel_corr, 1 / expected_atran)
        # variance should be divided by it, squared
        assert np.allclose(var_corr, 1 / expected_atran ** 2)

    def test_unsmoothed_atran_trim(self):
        files = get_scm_files()
        hdul = fits.open(files[0])

        # for blue data, unsmoothed atran should have wavelengths < 130
        hdul[0].header['CHANNEL'] = 'BLUE'
        blue = telluric_correct(hdul, write=False)
        assert np.all(blue['UNSMOOTHED_ATRAN'].data[0] < 130.)

        # for red data, unsmoothed atran should have wavelengths > 90
        hdul[0].header['CHANNEL'] = 'RED'
        red = telluric_correct(hdul, write=False)
        assert np.all(red['UNSMOOTHED_ATRAN'].data[0] > 90.)
