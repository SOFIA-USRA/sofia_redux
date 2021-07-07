# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.fifi_ls.get_resolution \
    import (get_resolution, clear_resolution_cache,
            get_resolution_from_cache, store_resolution_in_cache)
from sofia_redux.instruments.fifi_ls.tests.resources import FIFITestCase


class TestGetResolution(FIFITestCase):

    def get_header(self):
        header = fits.Header()
        header['CHANNEL'] = 'RED'
        header['G_WAVE_R'] = 100.0
        header['G_WAVE_B'] = 100.0
        header['G_ORD_B'] = 1
        return header

    def test_success(self):
        header = self.get_header()
        resw = get_resolution(header, wmean=None, spatial=False)
        assert isinstance(resw, float)
        resf = get_resolution(header, wmean=None, spatial=True)
        assert isinstance(resf, float)
        assert resw != resf
        header['CHANNEL'] = 'BLUE'
        resb = get_resolution(header, wmean=None, spatial=True)
        assert isinstance(resb, float)
        assert resf != resb

    def test_error(self, capsys):
        # bad header
        with pytest.raises(ValueError):
            get_resolution(None)
        capt = capsys.readouterr()
        assert 'Invalid header' in capt.err

        # now start with empty header
        header = fits.Header()

        # add missing keys one at a time
        header['DATE-OBS'] = 'BADVAL'
        for key in ['CHANNEL', 'G_ORD_B', 'G_WAVE_R', 'G_WAVE_B']:
            with pytest.raises(ValueError):
                get_resolution(header)
            capt = capsys.readouterr()
            assert 'Header missing {}'.format(key) in capt.err
            header[key] = 'TESTVAL'

        # now it has all others, will complain about bad blue order
        header['CHANNEL'] = 'BLUE'
        with pytest.raises(ValueError):
            get_resolution(header)
        capt = capsys.readouterr()
        assert 'Invalid blue grating order' in capt.err

        # test that unknown channel returns expected values
        # with warning
        header['CHANNEL'] = 'BADVAL'
        resw = get_resolution(header)
        assert np.isclose(resw, 1000)
        ress = get_resolution(header, spatial=True)
        assert np.isclose(ress, 5)
        capt = capsys.readouterr()
        assert 'Channel is unknown' in capt.err

        # test bad wavelength mean
        header['CHANNEL'] = 'RED'
        header['G_WAVE_R'] = 'test'
        with pytest.raises(ValueError):
            get_resolution(header)
        capt = capsys.readouterr()
        assert 'Invalid wavelength' in capt.err

    def test_default_table(self, tmpdir, mocker, capsys):
        # test for missing/bad defaults file
        os.makedirs(tmpdir.join('resolution'))
        default = tmpdir.join('resolution', 'spectral_resolution.txt')
        default.write('test\n')

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.fifi_ls.__file__',
                     str(mock_file))

        with pytest.raises(ValueError):
            get_resolution(self.get_header())
        capt = capsys.readouterr()
        assert 'Cannot read resolution file' in capt.err

    def test_resolution_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_resolution'))
        resolutionfile = os.path.join(tempdir, 'test01')

        with open(resolutionfile, 'w') as f:
            print('this is the resolution file', file=f)

        resolution = np.arange(10)
        store = resolutionfile, resolution

        clear_resolution_cache()
        assert get_resolution_from_cache(resolutionfile) is None
        store_resolution_in_cache(*store)

        # It should be in there now
        result = get_resolution_from_cache(resolutionfile)
        assert np.allclose(result, resolution)

        # Check it's still in there
        assert get_resolution_from_cache(resolutionfile) is not None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(resolutionfile, 'w') as f:
            print('a modification', file=f)

        assert get_resolution_from_cache(resolutionfile) is None

        # Store the data again
        store_resolution_in_cache(*store)

        # Make sure it's there
        assert get_resolution_from_cache(resolutionfile) is not None

        # Check clear works
        clear_resolution_cache()
        assert get_resolution_from_cache(resolutionfile) is None

        # Store then delete the resolution file -- check that bad file
        # can't be retrieved
        store_resolution_in_cache(*store)
        assert get_resolution_from_cache(resolutionfile) is not None
        os.remove(resolutionfile)
        assert get_resolution_from_cache(resolutionfile) is None
