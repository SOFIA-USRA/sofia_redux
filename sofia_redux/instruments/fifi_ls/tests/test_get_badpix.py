# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments import fifi_ls
from sofia_redux.instruments.fifi_ls.get_badpix \
    import (get_badpix, clear_badpix_cache, get_badpix_from_cache,
            store_badpix_in_cache, read_defaults_table)
from sofia_redux.instruments.fifi_ls.tests.resources import FIFITestCase


class TestGetBadpix(FIFITestCase):

    def test_errors(self, capsys, tmpdir):
        # bad header
        assert get_badpix(None) is None
        capt = capsys.readouterr()
        assert 'Invalid header' in capt.err

        # bad date-obs - uses latest
        header = fits.Header()
        header['CHANNEL'] = 'RED'
        header['DATE-OBS'] = 'badval'
        result = get_badpix(header)
        assert result is not None
        capt = capsys.readouterr()
        assert 'Could not determine DATE-OBS' in capt.err

        # bad provided file
        badfile = tmpdir.join('badfile')
        badfile.write('bad value')
        result = get_badpix(header, filename=str(badfile))
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid badpix file' in capt.err

    def test_success(self):
        header = fits.Header()
        header['CHANNEL'] = 'RED'
        header['DATE-OBS'] = '2010-01-01T00:00:00'
        values = get_badpix(header)
        fname = header.get('BDPXFILE')
        assert isinstance(values, np.ndarray)
        assert values.ndim == 2
        assert isinstance(fname, str)
        dpath = os.path.dirname(fifi_ls.__file__)
        fpath = os.path.join(dpath, 'data', 'badpix_files', fname)
        values = get_badpix(fits.Header(), filename=fpath)
        assert isinstance(values, np.ndarray)

    def test_badpix_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_badpix'))
        badpixfile = os.path.join(tempdir, 'test01')

        with open(badpixfile, 'w') as f:
            print('this is the badpix file', file=f)

        badpix = np.arange(10)
        store = badpixfile, badpix

        clear_badpix_cache()
        assert get_badpix_from_cache(badpixfile) is None
        store_badpix_in_cache(*store)

        # It should be in there now
        result = get_badpix_from_cache(badpixfile)
        assert np.allclose(result, badpix)

        # Check it's still in there
        assert get_badpix_from_cache(badpixfile) is not None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(badpixfile, 'w') as f:
            print('a modification', file=f)

        assert get_badpix_from_cache(badpixfile) is None

        # Store the data again
        store_badpix_in_cache(*store)

        # Make sure it's there
        assert get_badpix_from_cache(badpixfile) is not None

        # Check clear works
        clear_badpix_cache()
        assert get_badpix_from_cache(badpixfile) is None

        # Store then delete the badpix file -- check that bad file
        # can't be retrieved
        store_badpix_in_cache(*store)
        assert get_badpix_from_cache(badpixfile) is not None
        os.remove(badpixfile)
        assert get_badpix_from_cache(badpixfile) is None

    def test_default_table(self, tmpdir, mocker, capsys):
        # test for missing/bad defaults file
        os.makedirs(tmpdir.join('badpix_files'))
        default = tmpdir.join('badpix_files', 'badpix_default.txt')
        default.write('test\n')

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.__file__', str(mock_file))

        with pytest.raises(ValueError):
            read_defaults_table()
        capt = capsys.readouterr()
        assert 'Could not read badpix default file' in capt.err
