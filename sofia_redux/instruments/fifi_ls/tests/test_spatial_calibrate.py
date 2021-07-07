# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
from astropy.table import Table
import numpy as np

from sofia_redux.instruments.fifi_ls.spatial_calibrate \
    import (spatial_calibrate, wrap_spatial_calibrate, clear_spatial_cache,
            get_spatial_from_cache, store_spatial_in_cache, offset_xy,
            get_deltavec_coeffs, calculate_offsets)
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_wav_files


class TestSpatialCalibrate(FIFITestCase):

    def test_success(self):
        files = get_wav_files()
        result = spatial_calibrate(
            files[0], write=False, outdir=None, obsdate=None, rotate=False,
            flipsign=False)
        assert isinstance(result, fits.HDUList)
        assert result['XS_G0'].data.shape == (25,)
        assert result['YS_G0'].data.shape == (25,)

    def test_write(self, tmpdir):
        files = get_wav_files()
        result = spatial_calibrate(
            files[0], write=True, outdir=str(tmpdir), obsdate=None)
        assert os.path.isfile(result)
        result = spatial_calibrate(
            files[0], write=False, outdir=str(tmpdir), obsdate=None)
        assert isinstance(result, fits.HDUList)

    def test_parallel(self, tmpdir):
        files = get_wav_files()
        result = wrap_spatial_calibrate(
            files, outdir=str(tmpdir), obsdate=None,
            jobs=-1, allow_errors=False, write=True)
        for f in result:
            assert os.path.isfile(f)

    def test_serial(self, tmpdir):
        files = get_wav_files()
        result = wrap_spatial_calibrate(
            files, outdir=str(tmpdir), obsdate=None, jobs=None,
            allow_errors=False, write=True)
        for f in result:
            assert os.path.isfile(f)

    def test_flipsign(self):
        files = get_wav_files()
        result1 = spatial_calibrate(
            files[0], write=False, outdir=None, obsdate=None, rotate=False,
            flipsign=False)
        result2 = spatial_calibrate(
            files[0], write=False, outdir=None, obsdate=None, rotate=False,
            flipsign=True)
        x1 = result1['XS_G0'].data
        x2 = result2['XS_G0'].data
        x1x2 = x1 * x2
        assert (x1x2 < 0).all()

    def test_rotate(self):
        files = get_wav_files()
        result1 = spatial_calibrate(
            files[0], write=False, outdir=None, obsdate=None, rotate=False,
            flipsign=False)
        result2 = spatial_calibrate(
            files[0], write=False, outdir=None, obsdate=None, rotate=True,
            flipsign=False)
        x1 = result1['XS_G0'].data
        x2 = result2['XS_G0'].data
        assert not np.allclose(x1, x2)
        assert result1[0].header['SKY_ANGL'] != 0
        assert result2[0].header['SKY_ANGL'] == 0

    def test_offset_xy_param(self, capsys, tmpdir, mocker):
        date = [2018, 12, 1]
        calfile, offsets = offset_xy(date)
        assert os.path.isfile(calfile)
        assert isinstance(offsets, np.ndarray)

        # bad date
        assert offset_xy([2018, 12]) is None
        assert 'DATE must be a 3-element array' in capsys.readouterr().err
        assert offset_xy('2018-12-01T00:00:00') is None
        assert 'DATE must be a 3-element array' in capsys.readouterr().err

        # mock the calibration directory
        mock_file = tmpdir.join('test_file')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.__file__', str(mock_file))
        assert offset_xy(date) is None
        assert 'not a file' in capsys.readouterr().err

        # mock a file with no blue channel
        os.makedirs(tmpdir.join('data', 'spatial_cal'))
        cal_file = tmpdir.join('data', 'spatial_cal', 'poscal_default.txt')
        cal_file.write('99999999  r   spatial_cal/poscal_20081212_r.txt\n')
        assert offset_xy(date, blue=True) is None
        assert 'No spatial calibration' in capsys.readouterr().err

        # read red channel, but file is missing
        assert offset_xy(date, blue=False) is None
        assert 'not a file' in capsys.readouterr().err

    def test_deltavec(self, capsys, tmpdir, mocker):
        header = fits.Header({'PRIMARAY': 'RED',
                              'CHANNEL': 'BLUE',
                              'DICHROIC': 105})
        obsdate = [2018, 12, 1]

        # check keys in default result
        c1, c2 = get_deltavec_coeffs(header, obsdate)
        assert isinstance(c1, dict)
        assert isinstance(c2, dict)
        keys = ['ax', 'bx', 'rx', 'ay', 'by', 'ry']
        for key in keys:
            assert key in c1
            assert key in c2
            assert isinstance(c1[key], float)
            assert isinstance(c2[key], float)
            assert c1[key] != c2[key]
        assert c2['required']

        # test empty header
        clear_spatial_cache()
        assert get_deltavec_coeffs(fits.Header(), obsdate) is None
        assert 'No boresight offsets' in capsys.readouterr().err

        # missing cal file
        mock_file = tmpdir.join('test_file')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.__file__', str(mock_file))
        assert get_deltavec_coeffs(fits.Header(), obsdate) is None
        assert 'not a file' in capsys.readouterr().err

    def test_calculate_offsets(self, capsys, mocker):
        filename = get_wav_files()[0]
        hdul = fits.open(filename)

        result = calculate_offsets(hdul)
        assert isinstance(result, fits.HDUList)

        # bad obsdate
        hdul[0].header['DATE-OBS'] = 'BAD-VAL'
        assert calculate_offsets(hdul) is None
        assert 'Bad DATE-OBS' in capsys.readouterr().err

        del hdul[0].header['DATE-OBS']
        assert calculate_offsets(hdul) is None
        assert 'DATE-OBS not found' in capsys.readouterr().err

        hdul[0].header['DATE-OBS'] = '2019-12'
        assert calculate_offsets(hdul) is None
        assert 'DATE must be a 3-element array' in capsys.readouterr().err

        # mock deltavec to return a bad value
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.spatial_calibrate.'
            'get_deltavec_coeffs',
            return_value=None)
        assert calculate_offsets(hdul, obsdate=[2018, 12, 1]) is None
        assert 'Problem in deltavec coefficients' in capsys.readouterr().err

        # mock offset_xy to return a bad value
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.spatial_calibrate.offset_xy',
            return_value=('badfile.fits', [1, 2, 3]))
        assert calculate_offsets(hdul) is None
        assert 'Invalid number of XY offsets' in capsys.readouterr().err

    def test_telsim(self):
        # check that telsim data produces something reasonable
        filename = get_wav_files()[0]
        hdul = fits.open(filename)
        hdul[0].header['OBJECT'] = 'telsim'
        del hdul[0].header['DLAM_MAP']
        del hdul[0].header['DBET_MAP']
        result = spatial_calibrate(hdul)
        assert isinstance(result, fits.HDUList)
        assert np.all(~np.isnan(result['XS_G0'].data))
        assert np.all(~np.isnan(result['YS_G0'].data))
        assert result['XS_G0'].data.size == 25
        assert result['YS_G0'].data.size == 25

    def test_hdul_input(self):
        filename = get_wav_files()[0]
        hdul = fits.open(filename)
        result = spatial_calibrate(hdul)
        assert isinstance(result, fits.HDUList)

        # check bunit
        for ext in result[1:]:
            if 'LAMBDA' in ext.header['EXTNAME']:
                assert ext.header['BUNIT'] == 'um'
            elif ('XS' in ext.header['EXTNAME']
                  or 'YS' in ext.header['EXTNAME']):
                assert ext.header['BUNIT'] == 'arcsec'
            else:
                assert ext.header['BUNIT'] == 'adu/(Hz s)'

    def test_bad_parameters(self, capsys):
        files = get_wav_files()

        # bad output directory
        result = spatial_calibrate(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = spatial_calibrate('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_cal_failure(self, mocker, capsys):
        filename = get_wav_files()[0]

        # mock failure in calculate offsets
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.spatial_calibrate.'
            'calculate_offsets',
            return_value=None)
        result = spatial_calibrate(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Offset calculation failed' in capt.err

    def test_spatial_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_spatial'))
        spatialfile = os.path.join(tempdir, 'test01')
        obsdate = 20200201

        with open(spatialfile, 'w') as f:
            print('this is the spatial file', file=f)

        spatial = np.arange(10)
        coeffs = {'c1': 1, 'c2': 2.0}
        store = spatial, coeffs

        clear_spatial_cache()
        assert get_spatial_from_cache(spatialfile, obsdate) is None
        store_spatial_in_cache(spatialfile, obsdate, *store)

        # It should be in there now
        s, c = get_spatial_from_cache(spatialfile, obsdate)
        assert np.allclose(s, spatial)
        assert isinstance(c, dict)

        # Check it's still in there
        assert get_spatial_from_cache(spatialfile, obsdate) is not None

        # Check that a different date does not retrieve this file
        assert get_spatial_from_cache(spatialfile, obsdate + 1) is None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(spatialfile, 'w') as f:
            print('a modification', file=f)

        assert get_spatial_from_cache(spatialfile, obsdate) is None

        # Store the data again
        store_spatial_in_cache(spatialfile, obsdate, *store)

        # Make sure it's there
        assert get_spatial_from_cache(spatialfile, obsdate) is not None

        # Check clear works
        clear_spatial_cache()
        assert get_spatial_from_cache(spatialfile, obsdate) is None

        # Store then delete the file -- check that bad file
        # can't be retrieved
        store_spatial_in_cache(spatialfile, obsdate, *store)
        assert get_spatial_from_cache(spatialfile, obsdate) is not None
        os.remove(spatialfile)
        assert get_spatial_from_cache(spatialfile, obsdate) is None

    def test_wrap(self):
        files = get_wav_files()

        # serial
        result = wrap_spatial_calibrate(files, write=False)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

        # multithreading
        result = wrap_spatial_calibrate(files, write=False, jobs=-1)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.spatial_calibrate.multitask',
            return_value=['test', None])

        # bad files
        result = wrap_spatial_calibrate(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_wav_files()
        wrap_spatial_calibrate(files[0], write=False,
                               allow_errors=False)

        # allow errors
        result = wrap_spatial_calibrate(files, write=False,
                                        allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_spatial_calibrate(files, write=False,
                                        allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_scanpos(self):
        filename = get_wav_files()[0]
        hdul = fits.open(filename)
        # attach a scanpos extension, as for OTF data
        header = hdul[0].header
        header['NGRATING'] = 1
        nramp = hdul[1].data.shape[0]
        tab = Table()
        tab['DLAM_MAP'] = header['DLAM_MAP'] + np.arange(nramp, dtype=float)
        tab['DBET_MAP'] = header['DBET_MAP'] + np.arange(nramp, dtype=float)
        tab['FLAG'] = np.full(nramp, True)
        hdul.append(fits.BinTableHDU(tab, name='SCANPOS_G0'))

        result = spatial_calibrate(
            hdul, write=False, outdir=None, obsdate=None, rotate=False,
            flipsign=False)
        assert isinstance(result, fits.HDUList)

        # 25 pixel array for each ramp; central dimension is for spexels
        assert result['XS_G0'].data.shape == (nramp, 1, 25)
        assert result['YS_G0'].data.shape == (nramp, 1, 25)

        # scanpos table is dropped from output
        assert 'SCANPOS_G0' not in result

        # xs and ys should grow linearly by ramp, along the sky angle
        dx = - np.cos(np.radians(header['DET_ANGL'])) \
            + np.sin(np.radians(header['DET_ANGL']))
        dy = np.cos(np.radians(header['DET_ANGL'])) \
            + np.sin(np.radians(header['DET_ANGL']))
        assert np.allclose(result['XS_G0'].data[1:, 0, :]
                           - result['XS_G0'].data[:-1, 0, :],
                           dx)
        assert np.allclose(result['YS_G0'].data[1:, 0, :]
                           - result['YS_G0'].data[:-1, 0, :],
                           dy)

        # run with flipsign
        flip = spatial_calibrate(
            hdul, write=False, outdir=None, obsdate=None, rotate=False,
            flipsign=True)

        # diff is same, starting point is opposite
        assert np.allclose(flip['XS_G0'].data[1:, 0, :]
                           - flip['XS_G0'].data[:-1, 0, :],
                           dx)
        assert np.allclose(flip['YS_G0'].data[1:, 0, :]
                           - flip['YS_G0'].data[:-1, 0, :],
                           dy)
        assert np.allclose(flip['XS_G0'].data[0, 0, 0],
                           -1 * result['XS_G0'].data[0, 0, 0], atol=0.1)
        assert np.allclose(flip['YS_G0'].data[0, 0, 0],
                           result['YS_G0'].data[0, 0, -1], atol=0.1)
