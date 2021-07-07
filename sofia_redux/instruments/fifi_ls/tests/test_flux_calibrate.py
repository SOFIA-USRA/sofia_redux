# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.fifi_ls.get_response import get_response
from sofia_redux.instruments.fifi_ls.flux_calibrate \
    import flux_calibrate, wrap_flux_calibrate, apply_response
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_tel_files, get_scm_files


class TestFluxCalibrate(FIFITestCase):

    def test_success(self):
        files = get_tel_files()
        result = flux_calibrate(files[0], write=False, outdir=None)
        assert isinstance(result, fits.HDUList)
        assert result[0].header['PRODTYPE'] == 'flux_calibrated'
        bunit = ['Jy/pixel', 'Jy/pixel', 'um',
                 'arcsec', 'arcsec', '', 'adu/(Hz s Jy)', '']
        exts = ['FLUX', 'STDDEV', 'LAMBDA', 'XS', 'YS', 'ATRAN',
                'RESPONSE', 'UNSMOOTHED_ATRAN']
        for i, extname in enumerate(exts):
            assert extname in result
            assert result[extname].header['BUNIT'] == bunit[i]

    def test_no_tell(self):
        # check that a non-telluric-corrected file still calibrates okay
        filename = get_scm_files()[0]
        result = flux_calibrate(filename)
        assert isinstance(result, fits.HDUList)
        for extname in ['FLUX', 'STDDEV', 'LAMBDA', 'XS', 'YS',
                        'RESPONSE']:
            assert extname in result
        assert 'ATRAN' not in result
        assert 'UNSMOOTHED_ATRAN' not in result

    def test_invalid_response(self, capsys):
        filename = get_tel_files()[0]
        hdul = fits.open(filename)
        response = get_response(hdul[0].header)
        invalid_response = response.copy()
        invalid_response[1, :] = np.nan
        result = apply_response(hdul, invalid_response)
        assert result is None
        capt = capsys.readouterr()
        assert 'No valid response data' in capt.err

    def test_write(self, tmpdir):
        files = get_tel_files()
        result = flux_calibrate(files[0], write=True, outdir=str(tmpdir))
        assert os.path.isfile(result)
        result = flux_calibrate(files[0], write=False, outdir=str(tmpdir))
        assert isinstance(result, fits.HDUList)

    def test_hdul_input(self):
        filename = get_tel_files()[0]
        hdul = fits.open(filename)
        result = flux_calibrate(hdul)
        assert isinstance(result, fits.HDUList)

    def test_bad_parameters(self, capsys):
        files = get_tel_files()

        # bad output directory
        result = flux_calibrate(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = flux_calibrate('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_cal_failure(self, mocker, capsys):
        filename = get_tel_files()[0]

        # mock failure in apply response
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.flux_calibrate.apply_response',
            return_value=None)
        result = flux_calibrate(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Failed to apply response' in capt.err

        # mock failure in get response
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.flux_calibrate.get_response',
            return_value=None)
        result = flux_calibrate(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Failed to get response' in capt.err

    def test_wrap(self):
        files = get_tel_files()

        # serial
        result = wrap_flux_calibrate(files, write=False)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

        # multithreading
        result = wrap_flux_calibrate(files, write=False, jobs=-1)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.flux_calibrate.multitask',
            return_value=['test', None])

        # bad files
        result = wrap_flux_calibrate(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_tel_files()
        wrap_flux_calibrate(files[0], write=False,
                            allow_errors=False)

        # allow errors
        result = wrap_flux_calibrate(files, write=False,
                                     allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_flux_calibrate(files, write=False,
                                     allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err
