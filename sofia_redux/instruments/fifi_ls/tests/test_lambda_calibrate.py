# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.fifi_ls.lambda_calibrate \
    import (column_level1_mapper, read_wavecal, wave,
            lambda_calibrate, wrap_lambda_calibrate, clear_wavecal_cache,
            get_wavecal_from_cache, store_wavecal_in_cache)
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_ncm_files


class TestLambdaCalibrate(FIFITestCase):

    def test_column_level1_mapper(self):

        assert column_level1_mapper('Red D105') == 'r_105'
        assert column_level1_mapper('Red D130') == 'r_130'
        assert column_level1_mapper('Blue 1st Order') == 'b1'
        assert column_level1_mapper('Blue 2nd Order') == 'b2'
        assert column_level1_mapper('Blue 2nd Order D105') == 'b2'
        with pytest.raises(ValueError) as err:
            column_level1_mapper('')
        assert "bad column name in dataframe" in str(err.value).lower()

    def test_read_wavecal(self, tmpdir):

        with pytest.raises(ValueError) as err:
            read_wavecal(calfile='__does_not_exist__')
        assert "cannot read wavelength " \
               "calibration file" in str(err.value).lower()

        badfile = str(tmpdir.mkdir('test_lambda_calibrate').join('test_01'))
        with open(badfile, 'w') as f:
            print('foo', file=f)
        with pytest.raises(ValueError) as err:
            read_wavecal(calfile=badfile)
        assert "cannot parse" in str(err.value).lower()

        df = read_wavecal()
        col0 = list(df.columns.get_level_values(0))
        for c in col0:
            assert isinstance(c, int)

        col1 = list(df.columns.get_level_values(1))
        required_col1 = ['r_105', 'r_130', 'b1', 'b2']
        assert set(col1) == set(required_col1)

        row0 = list(df.index.get_level_values(0))
        required_row0 = ['g0', 'np', 'a', 'isf', 'gamma',
                         'ps', 'qoff', 'qs', 'isoff']
        assert set(row0) == set(required_row0)

        for index in df.index:
            count = 0
            if index[0] != 'isoff':
                assert index[1] == -1
            else:
                assert index[1] > count
                count += 1

    def test_wave(self, capsys):
        # okay result
        result = wave(10000, [2018, 12, 1], 105, blue='B1')
        assert result is not None
        assert isinstance(result['wavelength'], np.ndarray)
        assert not np.all(np.isnan(result['wavelength']))

        # bad indpos
        result = wave(None, None, None)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid IND' in capt.err

        # negative indpos
        result = wave(-9999, None, None)
        assert result is None
        capt = capsys.readouterr()
        assert 'IND must be positive' in capt.err

        # bad date
        result = wave(10000, [2018, 12], None)
        assert result is None
        capt = capsys.readouterr()
        assert 'DATE must be an array of length 3' in capt.err

        # date with no data
        result = wave(10000, [2001, 12, 1], 105, blue='B1')
        assert result is None
        capt = capsys.readouterr()
        assert 'No calibration data found for date' in capt.err

        # bad blue order
        result = wave(10000, [2018, 12, 1], 105, blue=1)
        assert result is None
        capt = capsys.readouterr()
        assert 'BLUE must be B1 or B2' in capt.err

        # bad dichroic value
        result = wave(10000, [2018, 12, 1], 200)
        assert result is None
        capt = capsys.readouterr()
        assert 'column not in wavecal data' in capt.err

    def test_success(self):
        # red 105 file
        filename = get_ncm_files()[0]
        hdul = fits.open(filename)
        hdul[0].header['CHANNEL'] = 'RED'
        hdul[0].header['DICHROIC'] = 105
        red_105 = lambda_calibrate(filename, write=False,
                                   outdir=None, obsdate=None)
        assert isinstance(red_105, fits.HDUList)

        # red 130 file
        hdul[0].header['DICHROIC'] = 130
        red_130 = lambda_calibrate(filename, write=False,
                                   outdir=None, obsdate=None)
        assert isinstance(red_130, fits.HDUList)

        # blue 1 105 file
        hdul[0].header['CHANNEL'] = 'BLUE'
        hdul[0].header['G_ORD_B'] = 1
        hdul[0].header['DICHROIC'] = 105
        blue1_105 = lambda_calibrate(hdul, write=False,
                                     outdir=None, obsdate=None)
        assert isinstance(blue1_105, fits.HDUList)

        # blue 1 130 file
        hdul[0].header['DICHROIC'] = 130
        blue1_130 = lambda_calibrate(hdul, write=False,
                                     outdir=None, obsdate=None)
        assert isinstance(blue1_130, fits.HDUList)

        # blue 2 105 file
        hdul[0].header['G_ORD_B'] = 2
        hdul[0].header['DICHROIC'] = 105
        blue2_105 = lambda_calibrate(hdul, write=False,
                                     outdir=None, obsdate=None)
        assert isinstance(blue2_105, fits.HDUList)

        # blue 2 130 file
        hdul[0].header['G_ORD_B'] = 2
        hdul[0].header['DICHROIC'] = 130
        blue2_130 = lambda_calibrate(hdul, write=False,
                                     outdir=None, obsdate=None)
        assert isinstance(blue2_130, fits.HDUList)

        # wave red > blue 1 > blue 2
        # blue dichroics have same wavecal,
        # red dichroics have minor differences
        rwave1 = red_105['LAMBDA_G0'].data
        rwave2 = red_130['LAMBDA_G0'].data
        bwave1 = blue1_105['LAMBDA_G0'].data
        bwave2 = blue1_130['LAMBDA_G0'].data
        bwave3 = blue2_105['LAMBDA_G0'].data
        bwave4 = blue2_130['LAMBDA_G0'].data

        assert np.allclose(rwave1, rwave2, atol=.001)
        assert np.mean(rwave1) > np.mean(bwave1)
        assert np.allclose(bwave1, bwave2)
        assert np.mean(bwave1) > np.mean(bwave3)
        assert np.allclose(bwave3, bwave4)

    def test_error(self, capsys):
        filename = get_ncm_files()[0]

        # invalid blue order
        hdul = fits.open(filename)
        hdul[0].header['CHANNEL'] = 'BLUE'
        hdul[0].header['G_ORD_B'] = 3

        result = lambda_calibrate(hdul)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid Blue grating order' in capt.err

        # bad date
        hdul = fits.open(filename)
        hdul[0].header['CHANNEL'] = 'RED'
        hdul[0].header['DATE-OBS'] = 'BADVAL'
        result = lambda_calibrate(hdul)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid DATE-OBS' in capt.err

    def test_write(self, tmpdir):
        files = get_ncm_files()
        result = lambda_calibrate(
            files[0], write=True, outdir=str(tmpdir), obsdate=None)
        assert os.path.isfile(result)

    def test_hdul_input(self):
        filename = get_ncm_files()[0]
        hdul = fits.open(filename)
        result = lambda_calibrate(hdul)
        assert isinstance(result, fits.HDUList)

        # check bunit
        for ext in result[1:]:
            if 'LAMBDA' in ext.header['EXTNAME']:
                assert ext.header['BUNIT'] == 'um'
            else:
                assert ext.header['BUNIT'] == 'adu/(Hz s)'

    def test_bad_parameters(self, capsys):
        files = get_ncm_files()

        # bad output directory
        result = lambda_calibrate(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = lambda_calibrate('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_cal_failure(self, mocker, capsys):
        filename = get_ncm_files()[0]

        # mock failure in wave
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.lambda_calibrate.wave',
            return_value=None)
        result = lambda_calibrate(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Wavelength calibration failed' in capt.err

        # mock failure in read_wavecal
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.lambda_calibrate.read_wavecal',
            return_value=None)
        result = lambda_calibrate(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Unable to read wave calibration' in capt.err

    def test_wavecal_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_wavecal'))
        wavecalfile = os.path.join(tempdir, 'test01')

        with open(wavecalfile, 'w') as f:
            print('this is the wavecal file', file=f)

        wavecal = np.arange(10)
        store = wavecalfile, wavecal

        clear_wavecal_cache()
        assert get_wavecal_from_cache(wavecalfile) is None
        store_wavecal_in_cache(*store)

        # It should be in there now
        result = get_wavecal_from_cache(wavecalfile)
        assert np.allclose(result, wavecal)

        # Check it's still in there
        assert get_wavecal_from_cache(wavecalfile) is not None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(wavecalfile, 'w') as f:
            print('a modification', file=f)

        assert get_wavecal_from_cache(wavecalfile) is None

        # Store the data again
        store_wavecal_in_cache(*store)

        # Make sure it's there
        assert get_wavecal_from_cache(wavecalfile) is not None

        # Check clear works
        clear_wavecal_cache()
        assert get_wavecal_from_cache(wavecalfile) is None

        # Store then delete the wavecal file -- check that bad file
        # can't be retrieved
        store_wavecal_in_cache(*store)
        assert get_wavecal_from_cache(wavecalfile) is not None
        os.remove(wavecalfile)
        assert get_wavecal_from_cache(wavecalfile) is None

    def test_wrap(self, tmpdir):
        files = get_ncm_files()

        # serial
        result = wrap_lambda_calibrate(files, outdir=str(tmpdir), write=False)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

        # multithreading
        result = wrap_lambda_calibrate(files, outdir=str(tmpdir), write=False,
                                       jobs=-1)
        assert len(result) > 0
        assert isinstance(result[0], fits.HDUList)

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.lambda_calibrate.multitask',
            return_value=['test', None])

        # bad files
        result = wrap_lambda_calibrate(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_ncm_files()
        wrap_lambda_calibrate(files[0], write=False,
                              allow_errors=False)

        # allow errors
        result = wrap_lambda_calibrate(files, write=False,
                                       allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_lambda_calibrate(files, write=False,
                                       allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_scanpos(self):
        # if no scanpos data present, no errors, nothing propagated
        filename = get_ncm_files()[0]
        result = lambda_calibrate(filename, write=False)
        assert isinstance(result, fits.HDUList)
        for i in range(result[0].header['NGRATING']):
            assert f'SCANPOS_G{i}' not in result

        # if scanpos data present, it should be passed forward
        # unmodified
        filename = get_ncm_files()[0]
        hdul = fits.open(filename)
        for i in range(hdul[0].header['NGRATING']):
            hdul.append(fits.ImageHDU(np.arange(i + 10), name=f'SCANPOS_G{i}'))

        result = lambda_calibrate(hdul, write=False)
        assert isinstance(result, fits.HDUList)
        for i in range(result[0].header['NGRATING']):
            assert f'SCANPOS_G{i}' in result
            assert np.allclose(result[f'SCANPOS_G{i}'].data,
                               hdul[f'SCANPOS_G{i}'].data)
