# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy import log
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.forcast.getatran \
    import (clear_atran_cache, get_atran_from_cache,
            store_atran_in_cache, get_atran)
from sofia_redux.instruments.forcast.tests.resources import raw_specdata


class TestGetAtran(object):
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def test_atran_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_atran'))
        atranfile = os.path.join(tempdir, 'test01')
        res = 100.0

        with open(atranfile, 'w') as f:
            print('this is the atran file', file=f)

        wave = unsmoothed = smoothed = np.arange(10)

        filename = 'ATRNFILE_header_value'
        store = atranfile, res, filename, wave, unsmoothed, smoothed

        clear_atran_cache()
        assert get_atran_from_cache(atranfile, res) is None
        store_atran_in_cache(*store)

        # It should be in there now
        result = get_atran_from_cache(atranfile, res)
        assert result[0] == filename

        for r in result[1:]:
            assert np.allclose(r, wave)

        # Check it's still in there
        assert get_atran_from_cache(atranfile, res) is not None

        # Check that a different resolution does not retrieve this file
        assert get_atran_from_cache(atranfile, res * 2) is None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(atranfile, 'w') as f:
            print('a modification', file=f)

        assert get_atran_from_cache(atranfile, res) is None

        # Store the data again
        store_atran_in_cache(*store)

        # Make sure it's there
        assert get_atran_from_cache(atranfile, res) is not None

        # Check clear works
        clear_atran_cache()
        assert get_atran_from_cache(atranfile, res) is None

        # Store then delete the atran file -- check that bad file
        # can't be retrieved
        store_atran_in_cache(*store)
        assert get_atran_from_cache(atranfile, res) is not None
        os.remove(atranfile)
        assert get_atran_from_cache(atranfile, res) is None

    def test_filename(self, tmpdir, capsys):
        hdul = raw_specdata()
        header = hdul[0].header
        resolution = 123
        atranfile = tmpdir.join('test_file.fits')

        # default: gets alt/za/resolution from header, no unsmoothed data
        default = get_atran(header, resolution)
        assert default is not None
        assert isinstance(default, np.ndarray)

        # same, but get unsmoothed data
        result = get_atran(header, resolution, get_unsmoothed=True)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert np.allclose(result[0], default)
        assert result[0].size == result[1].size

        # provide a bad filename -- warns and gets default
        result = get_atran(header, resolution, filename=str(atranfile))
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'not found; retrieving default' in capt.err

        # provide a good filename, bad file
        atranfile.write('Test data\n')
        result = get_atran(header, resolution, filename=str(atranfile))
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid data' in capt.err

        # good filename, minimum required data
        data = np.arange(100).reshape(2, 50).astype(float)
        hdul = fits.HDUList(fits.PrimaryHDU(data=data))
        hdul.writeto(str(atranfile), overwrite=True)
        result = get_atran(header, resolution, filename=str(atranfile))
        # wavelengths will match
        assert np.allclose(result[0], data[0])
        # data will be all nan
        assert np.all(np.isnan(result[1]))

    def test_header(self, capsys):
        hdul = raw_specdata()
        header = hdul[0].header
        resolution = 123

        # default
        default = get_atran(header, resolution)
        assert default is not None
        assert isinstance(default, np.ndarray)

        # bad header
        result = get_atran(['test'], resolution)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid header' in capt.err

        # bad alti_sta -- should get end
        hdr = header.copy()
        hdr['ALTI_STA'] = -9999
        result = get_atran(hdr, resolution)
        assert np.allclose(result, default)

        # bad alti_end -- should get start
        hdr = header.copy()
        hdr['ALTI_END'] = -9999
        result = get_atran(hdr, resolution)
        assert np.allclose(result, default)

        # bad za_start -- should get end
        hdr = header.copy()
        hdr['ZA_START'] = -9999
        result = get_atran(hdr, resolution)
        assert np.allclose(result, default)

        # bad za_end -- should get start
        hdr = header.copy()
        hdr['ZA_END'] = -9999
        result = get_atran(hdr, resolution)
        assert np.allclose(result, default)

        # use_wv option: without wv keywords, should warn
        # and produce same result
        hdr = header.copy()
        try:
            del hdr['WVZ_STA']
            del hdr['WVZ_END']
        except KeyError:
            pass
        result = get_atran(hdr, resolution, use_wv=True)
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'Bad WV value' in capt.out

        # add a WV value (start and/or end), but no ATRAN repo with WV files
        hdr = header.copy()
        hdr['WVZ_STA'] = 6.0
        result = get_atran(hdr, resolution, use_wv=True)
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'Using nearest Alt/ZA' in capt.out
        assert 'Using nearest Alt/ZA/WV' not in capt.out

        hdr = header.copy()
        hdr['WVZ_END'] = 6.0
        result = get_atran(hdr, resolution, use_wv=True)
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'Using nearest Alt/ZA' in capt.out
        assert 'Using nearest Alt/ZA/WV' not in capt.out

        hdr = header.copy()
        hdr['WVZ_STA'] = 6.0
        hdr['WVZ_END'] = 8.0
        result = get_atran(hdr, resolution, use_wv=True)
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert "Alt, ZA, WV: 40.00 44.25 7.00" in capt.out
        assert 'Using nearest Alt/ZA' in capt.out
        assert 'Using nearest Alt/ZA/WV' not in capt.out

        # now add wvz_obs -- should be used in place of sta/end
        hdr['WVZ_OBS'] = 5.0
        get_atran(hdr, resolution, use_wv=True)
        capt = capsys.readouterr()
        assert "Alt, ZA, WV: 40.00 44.25 5.00" in capt.out

        # but not if it has a bad value
        hdr['WVZ_OBS'] = -9999.
        get_atran(hdr, resolution, use_wv=True)
        capt = capsys.readouterr()
        assert "Alt, ZA, WV: 40.00 44.25 7.00" in capt.out

    def test_atran_dir(self, tmpdir, capsys):
        hdul = raw_specdata()
        header = hdul[0].header
        resolution = 123

        default = get_atran(header, resolution)

        # specify bad directory: should get default
        result = get_atran(header, resolution, atran_dir='badval')
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'Cannot find ATRAN directory' in capt.err

        # specify empty directory -- returns None
        result = get_atran(header, resolution, atran_dir=str(tmpdir))
        assert result is None
        capt = capsys.readouterr()
        assert 'No ATRAN file found' in capt.out

        # make a file that should match alt/za and
        # one that matches alt/za/wv
        atran1 = 'atran_40K_45deg_4-50mum.fits'
        atran2 = 'atran_40K_45deg_6pwv_4-50mum.fits'
        afile1 = str(tmpdir.join(atran1))
        afile2 = str(tmpdir.join(atran2))

        data = np.arange(100).reshape(2, 50).astype(float)
        hdul = fits.HDUList(fits.PrimaryHDU(data=data))
        hdul.writeto(afile1, overwrite=True)
        hdul.writeto(afile2, overwrite=True)

        # without use_wv, should get atran1
        result1 = get_atran(header, resolution, atran_dir=str(tmpdir))
        capt = capsys.readouterr()
        assert atran1 in capt.out
        assert result1 is not None

        # with use_wv, will get the other one (but the data is the same)
        header['WVZ_STA'] = 6.
        header['WVZ_END'] = 6.
        result2 = get_atran(header, resolution, atran_dir=str(tmpdir),
                            use_wv=True)
        capt = capsys.readouterr()
        assert 'Using nearest Alt/ZA/WV' in capt.out
        assert atran2 in capt.out
        assert np.allclose(result1, result2, equal_nan=True)
