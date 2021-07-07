# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.io import fits

from sofia_redux.instruments.fifi_ls.correct_wave_shift \
    import correct_wave_shift, wrap_correct_wave_shift, correct_lambda
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_cal_files, get_scm_files


class TestCorrectWaveShift(FIFITestCase):

    def test_success(self):
        files = get_cal_files()
        result = correct_wave_shift(files[0], write=False, outdir=None)
        assert isinstance(result, fits.HDUList)
        assert result[0].header['PRODTYPE'] == 'wavelength_shifted'
        bunit = ['Jy/pixel', 'Jy/pixel', 'um',
                 'arcsec', 'arcsec', '', 'adu/(Hz s Jy)', 'um']
        exts = ['FLUX', 'STDDEV', 'LAMBDA', 'XS', 'YS', 'ATRAN',
                'RESPONSE', 'UNCORRECTED_LAMBDA']
        for i, extname in enumerate(exts):
            assert extname in result
            assert result[extname].header['BUNIT'] == bunit[i]

        ulam = result['UNCORRECTED_LAMBDA'].data
        dw = result['LAMBDA'].data - ulam
        baryvel = result[0].header.get('BARYSHFT', 0)
        lsrvel = result[0].header.get('LSRSHFT', 0)
        assert np.allclose(baryvel, 9.86e-5)
        assert np.allclose(lsrvel, 3.438e-5)
        assert np.allclose(baryvel * ulam, dw)

    def test_write(self, tmpdir):
        files = get_cal_files()
        for write in [False, True]:
            result = correct_wave_shift(files[0], write=write,
                                        outdir=str(tmpdir))
            if write:
                assert isinstance(result, str)
            else:
                assert isinstance(result, fits.HDUList)

    def test_no_telluric(self, capsys):
        # get an uncorrected file
        scm = get_scm_files()[0]

        # test correct_lambda directly
        hdul = fits.open(scm)
        result = correct_lambda(hdul)
        assert result is None
        capt = capsys.readouterr()
        assert 'No telluric correction performed, ' \
               'not shifting wavelengths' in capt.err

        # test that wrappers don't crash in this case
        result = correct_wave_shift(hdul)
        assert result is None
        result = wrap_correct_wave_shift(hdul)
        assert len(result) == 0

    def test_bad_parameters(self, capsys):
        files = get_cal_files()

        # bad output directory
        result = correct_wave_shift(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = correct_wave_shift('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_wrap(self):
        files = get_cal_files()

        # serial
        result = wrap_correct_wave_shift(files, write=False)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

        # multithreading
        result = wrap_correct_wave_shift(files, write=False, jobs=-1)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.correct_wave_shift.multitask',
            return_value=['test', None])

        # bad files
        result = wrap_correct_wave_shift(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_cal_files()
        wrap_correct_wave_shift(files[0], write=False,
                                allow_errors=False)

        # allow errors
        result = wrap_correct_wave_shift(files, write=False,
                                         allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_correct_wave_shift(files, write=False,
                                         allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err
