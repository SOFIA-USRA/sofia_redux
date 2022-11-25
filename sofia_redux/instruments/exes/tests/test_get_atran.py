# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import time

from astropy import log
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments import exes
from sofia_redux.instruments.exes.get_atran \
    import (clear_atran_cache, get_atran_from_cache,
            store_atran_in_cache, get_atran)
from sofia_redux.instruments.exes.tests.resources import low_header


def check_data_dir():
    atran_dir = os.path.join(os.path.dirname(exes.__file__),
                             'data', 'transmission')
    fits_files = glob.glob(os.path.join(atran_dir, 'psg*fits'))
    if len(fits_files) >= 1:
        return True
    else:
        return False


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

    @pytest.mark.skipif(not check_data_dir(), reason='Models not available')
    def test_filename(self, tmpdir, capsys):
        header = low_header()
        resolution = 1234
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

    @pytest.mark.skipif(not check_data_dir(), reason='Models not available')
    def test_header(self, capsys):
        header = low_header()
        resolution = 1234

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

    @pytest.mark.skipif(not check_data_dir(), reason='Models not available')
    def test_zeroval(self, capsys, tmpdir):
        header = fits.Header()
        resolution = 1234

        # make a file that should match alt/za
        atran1 = 'psg_40K_45deg_5-28um.fits'
        afile1 = str(tmpdir.join(atran1))

        data = np.arange(100).reshape(2, 50).astype(float)
        hdul = fits.HDUList(fits.PrimaryHDU(data=data))
        hdul.writeto(afile1, overwrite=True)

        get_atran(header, resolution, atran_dir=str(tmpdir))
        capt = capsys.readouterr()
        assert "Alt, ZA: 0.00 0.00" in capt.out
        assert 'Using nearest Alt/ZA' in capt.out

    @pytest.mark.skipif(not check_data_dir(), reason='Models not available')
    def test_atran_dir(self, tmpdir, capsys):
        header = low_header()
        resolution = 1234

        default = get_atran(header, resolution)

        # specify bad directory: should get default
        result = get_atran(header, resolution, atran_dir='badval')
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'Cannot find transmission directory' in capt.err

        # specify empty directory -- returns None
        result = get_atran(header, resolution, atran_dir=str(tmpdir))
        assert result is None
        capt = capsys.readouterr()
        assert 'No PSG file found' in capt.out

        # make a file that should match alt/za
        atran1 = 'psg_40K_45deg_5-28um.fits'
        afile1 = str(tmpdir.join(atran1))

        data = np.arange(100).reshape(2, 50).astype(float)
        hdul = fits.HDUList(fits.PrimaryHDU(data=data))
        hdul.writeto(afile1, overwrite=True)

        # should get atran1
        result1 = get_atran(header, resolution, atran_dir=str(tmpdir))
        capt = capsys.readouterr()
        assert atran1 in capt.out
        assert result1 is not None
