# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.fifi_ls.combine_grating_scans \
    import combine_grating_scans, wrap_combine_grating_scans
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_flf_files


class TestCombineGratingScans(FIFITestCase):

    def test_success(self):
        files = get_flf_files()
        result = combine_grating_scans(
            files[0], write=False, outdir=None, correct_bias=False)
        assert isinstance(result, fits.HDUList)
        assert result[0].header['PRODTYPE'] == 'scan_combined'

        # test on an HDUList
        hdul = fits.open(files[0])
        result2 = combine_grating_scans(
            hdul, write=False, outdir=None)
        assert isinstance(result2, fits.HDUList)
        assert result2[0].header['PRODTYPE'] == 'scan_combined'

        # test bunit in extensions
        # check bunit
        assert result2['FLUX'].header['BUNIT'] == 'adu/(Hz s)'
        assert result2['STDDEV'].header['BUNIT'] == 'adu/(Hz s)'
        assert result2['LAMBDA'].header['BUNIT'] == 'um'
        assert result2['XS'].header['BUNIT'] == 'arcsec'
        assert result2['YS'].header['BUNIT'] == 'arcsec'

    def test_write(self, tmpdir):
        files = get_flf_files()
        result = combine_grating_scans(
            files[0], write=True, outdir=str(tmpdir), correct_bias=False)
        failure = False
        if not isinstance(result, str) or not os.path.isfile(result):
            print("error: not a file (%s)" % repr(result))
            failure = True
        assert not failure

    def test_correct_bias(self):
        files = get_flf_files()
        r1 = combine_grating_scans(
            files[0], write=False, outdir=None, correct_bias=False)
        r2 = combine_grating_scans(
            files[0], write=False, outdir=None, correct_bias=True)
        d1 = r1[1].data
        d2 = r2[1].data
        v1 = np.nansum(np.abs(d1))
        v2 = np.nansum(np.abs(d2))
        assert v2 <= v1

    def test_no_bias(self, capsys, mocker):
        # mock no overlap
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.combine_grating_scans.'
            'get_lambda_overlap',
            return_value=(10, 10))
        files = get_flf_files()
        r1 = combine_grating_scans(
            files[0], write=False, outdir=None, correct_bias=False)
        capsys.readouterr()
        r2 = combine_grating_scans(
            files[0], write=False, outdir=None, correct_bias=True)
        d1 = r1[1].data
        d2 = r2[1].data
        capt = capsys.readouterr()
        assert 'No overlapping wavelengths' in capt.out
        assert np.allclose(d1, d2, equal_nan=True)

        # mock a small overlap, so mean can't be used - same effect
        # as not correcting bias
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.combine_grating_scans.'
            'get_lambda_overlap',
            return_value=(118.17, 118.2))
        r3 = combine_grating_scans(
            files[0], write=False, outdir=None, correct_bias=True)
        d3 = r3[1].data
        assert np.allclose(d2, d3, equal_nan=True)

    def test_wrap_combine_grating_scans(self):
        files = get_flf_files()

        # serial
        result = wrap_combine_grating_scans(files, write=False)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

        # multithreading
        result = wrap_combine_grating_scans(files, write=False, jobs=-1)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

    def test_bad_parameters(self, tmpdir, capsys, mocker):
        files = get_flf_files()

        # bad output directory
        result = combine_grating_scans(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = combine_grating_scans('badfile.fits', outdir=str(tmpdir))
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

        # combination failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.combine_grating_scans.'
            'combine_extensions',
            return_value=None)
        result = combine_grating_scans(files[0], outdir=str(tmpdir))
        assert result is None
        capt = capsys.readouterr()
        assert 'Combination failed' in capt.err

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.combine_grating_scans.'
            'multitask',
            return_value=['test', None])

        # bad files
        result = wrap_combine_grating_scans(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_flf_files()
        wrap_combine_grating_scans(files[0], write=False,
                                   allow_errors=False)

        # allow errors
        result = wrap_combine_grating_scans(files, write=False,
                                            allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_combine_grating_scans(files, write=False,
                                            allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_otf_combine(self, capsys):
        files = get_flf_files()
        hdul = fits.open(files[0])

        # mock OTF data with multiple grating scans
        nramp = 15
        for i in range(hdul[0].header['NGRATING']):
            # lambda stays the same shape, all others need to be 3D
            flux = np.repeat(hdul[f'FLUX_G{i}'].data.reshape(1, 16, 25),
                             nramp, axis=0)
            stddev = np.repeat(hdul[f'STDDEV_G{i}'].data.reshape(1, 16, 25),
                               nramp, axis=0)
            xs = np.repeat(hdul[f'XS_G{i}'].data.reshape(1, 1, 25),
                           nramp, axis=0)
            ys = np.repeat(hdul[f'YS_G{i}'].data.reshape(1, 1, 25),
                           nramp, axis=0)
            hdul[f'FLUX_G{i}'].data = flux
            hdul[f'STDDEV_G{i}'].data = stddev
            hdul[f'XS_G{i}'].data = xs
            hdul[f'YS_G{i}'].data = ys

        # raises error for multiple scans with 3D data
        result = combine_grating_scans(hdul, write=False, outdir=None)
        assert result is None
        assert "Grating scans are not supported " \
               "in OTF mode" in capsys.readouterr().err

        # set to 1 grating -- extras will be dropped
        hdul[0].header['NGRATING'] = 1
        result = combine_grating_scans(hdul, write=False, outdir=None)
        assert isinstance(result, fits.HDUList)
        assert len(result) == 6

        # check shapes of output extensions
        assert result['FLUX'].shape == (nramp, 16, 25)
        assert result['STDDEV'].shape == (nramp, 16, 25)
        assert result['LAMBDA'].shape == (16, 25)
        # XS and YS are expanded to match flux shape
        assert result['XS'].shape == (nramp, 16, 25)
        assert result['YS'].shape == (nramp, 16, 25)
