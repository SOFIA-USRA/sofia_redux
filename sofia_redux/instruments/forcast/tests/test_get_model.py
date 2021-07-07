# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy import log
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.forcast.getmodel \
    import (clear_model_cache, get_model_from_cache,
            store_model_in_cache, get_model)
from sofia_redux.instruments.forcast.tests.resources import raw_specdata


class TestGetModel(object):
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def test_model_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_model'))
        modelfile = os.path.join(tempdir, 'test01')
        res = 100.0

        with open(modelfile, 'w') as f:
            print('this is the model file', file=f)

        wave = unsmoothed = smoothed = np.arange(10)

        filename = 'MODLFILE_header_value'
        store = modelfile, res, filename, wave, unsmoothed, smoothed

        clear_model_cache()
        assert get_model_from_cache(modelfile, res) is None
        store_model_in_cache(*store)

        # It should be in there now
        result = get_model_from_cache(modelfile, res)
        assert result[0] == filename
        for r in result[1:]:
            assert np.allclose(r, wave)

        # Check it's still in there
        assert get_model_from_cache(modelfile, res) is not None

        # Check that a different resolution does not retrieve this file
        assert get_model_from_cache(modelfile, res * 2) is None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(modelfile, 'w') as f:
            print('a modification', file=f)

        assert get_model_from_cache(modelfile, res) is None

        # Store the data again
        store_model_in_cache(*store)

        # Make sure it's there
        assert get_model_from_cache(modelfile, res) is not None

        # Check clear works
        clear_model_cache()
        assert get_model_from_cache(modelfile, res) is None

        # Store then delete the model file -- check that bad file
        # can't be retrieved
        store_model_in_cache(*store)
        assert get_model_from_cache(modelfile, res) is not None
        os.remove(modelfile)
        assert get_model_from_cache(modelfile, res) is None

    def test_filename(self, tmpdir, capsys):
        hdul = raw_specdata()
        header = hdul[0].header
        resolution = 123
        modelfile = tmpdir.join('test_file.fits')

        # default: gets object, date from header, no unsmoothed data
        default = get_model(header, resolution)
        assert default is not None
        assert isinstance(default, np.ndarray)

        # same, but get unsmoothed data
        result = get_model(header, resolution, get_unsmoothed=True)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert np.allclose(result[0], default)
        assert result[0].size == result[1].size

        # provide a bad filename -- warns and gets default
        result = get_model(header, resolution, filename=str(modelfile))
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'not found; retrieving default' in capt.err

        # provide a good filename, bad file
        modelfile.write('Test data\n')
        result = get_model(header, resolution, filename=str(modelfile))
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid data' in capt.err

        # good filename, minimum required data
        data = np.arange(100).reshape(2, 50).astype(float)
        hdul = fits.HDUList(fits.PrimaryHDU(data=data))
        hdul.writeto(str(modelfile), overwrite=True)
        result = get_model(header, resolution, filename=str(modelfile))
        # wavelengths will match
        assert np.allclose(result[0], data[0])
        # data will be all nan
        assert np.all(np.isnan(result[1]))

    def test_header(self, capsys):
        hdul = raw_specdata()
        header = hdul[0].header
        resolution = 123

        # default
        default = get_model(header, resolution)
        assert default is not None
        assert isinstance(default, np.ndarray)

        # bad header
        result = get_model(['test'], resolution)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid header' in capt.err

        # short date in header - okay
        header['DATE-OBS'] = '2018-12-31'
        result = get_model(header, resolution)
        assert np.allclose(result, default)
        assert '2018123100' in capsys.readouterr().out

        # bad short date in header - gets default
        header['DATE-OBS'] = '2018-AA-31'
        result = get_model(header, resolution)
        assert np.allclose(result, default)
        assert '9999999999' in capsys.readouterr().out

        # bad long date in header - gets default
        header['DATE-OBS'] = '2018-AA-31T00:00:00'
        result = get_model(header, resolution)
        assert np.allclose(result, default)
        assert '9999999999' in capsys.readouterr().out

    def test_model_dir(self, tmpdir, capsys):
        hdul = raw_specdata()
        header = hdul[0].header
        resolution = 123

        default = get_model(header, resolution)

        # specify bad directory: should get default
        result = get_model(header, resolution, model_dir='badval')
        assert np.allclose(result, default)
        capt = capsys.readouterr()
        assert 'Cannot find model directory' in capt.err

        # specify empty directory -- returns None
        result = get_model(header, resolution, model_dir=str(tmpdir))
        assert result is None
        capt = capsys.readouterr()
        assert 'No model file found' in capt.err

        # make a file that should match object and
        # one that nearly matches object/date
        model1 = 'alphaboo_model.fits'
        model2 = 'alphaboo_2018123100_model.fits'
        model3 = 'alphaboo_20190201_model.fits'
        afile1 = str(tmpdir.join(model1))
        afile2 = str(tmpdir.join(model2))
        afile3 = str(tmpdir.join(model3))

        data = np.arange(100).reshape(2, 50).astype(float)
        hdul = fits.HDUList(fits.PrimaryHDU(data=data))
        hdul.writeto(afile1, overwrite=True)
        hdul.writeto(afile2, overwrite=True)
        hdul.writeto(afile3, overwrite=True)

        # with all present, should get model1
        result1 = get_model(header, resolution, model_dir=str(tmpdir))
        capt = capsys.readouterr()
        assert model1 in capt.out
        assert result1 is not None

        # with only the date-specific ones, should get model2 (closer in time)
        tmpdir2 = tmpdir.join('dates')
        os.makedirs(str(tmpdir2))
        afile2 = str(tmpdir2.join(model2))
        afile3 = str(tmpdir2.join(model3))
        hdul.writeto(afile2, overwrite=True)
        hdul.writeto(afile3, overwrite=True)
        result2 = get_model(header, resolution, model_dir=str(tmpdir2))
        capt = capsys.readouterr()
        assert model2 in capt.out
        assert np.allclose(result1, result2, equal_nan=True)
