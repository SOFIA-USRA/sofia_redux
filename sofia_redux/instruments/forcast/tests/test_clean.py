# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.clean import addhist, clean
import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.tests.resources \
    import random_mask, add_jailbars

dripconfig.load()


def fake_data(shape=(256, 256), value=2.0, badfrac=0.1,
              jblevel=1.0, setnan=True):
    data = np.full(shape, value)
    add_jailbars(data, level=jblevel)
    badmask = random_mask(shape[-2:], frac=badfrac)
    if setnan:
        if len(data.shape) == 2:
            data[badmask] = np.nan
        elif len(data.shape) == 3:
            for frame in data:
                frame[badmask] = np.nan
    return data, badmask


class TestClean(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Clean: test history message'

    def test_2d_clean(self):
        data, badmap = fake_data()
        header = fits.header.Header()
        variance = np.full_like(data, 2.0)
        variance[np.isnan(data)] = np.nan
        dripconfig.load()
        dripconfig.configuration['jbclean'] = 'FFT'
        nans = np.isnan(data)

        result, var = clean(
            data, badmap=badmap, header=header, variance=variance)

        # NaNs should have been removed
        assert np.isnan(result).sum() < nans.sum()
        assert np.isnan(var).sum() < nans.sum()

        # jailbars should have gone, but may be a few bad pixels
        nbad = np.sum(np.abs(result[~nans] - result[~nans][0]) > 1e-5)
        assert nbad < .05 * data.size
        assert header['JBCLEAN'] == 'FFT'

        # If no badmap, just clean jailbars
        result, var = clean(data)
        assert np.allclose(result[~np.isnan(result)],
                           result[~np.isnan(result)][0])
        assert np.isnan(result).sum()
        assert var is None

        # Do not jbclean if JBCLEAN is not 'FFT'
        dripconfig.configuration['jbclean'] = 'MEDIAN'
        result, var = clean(data)
        assert not np.allclose(result[~badmap], result[~badmap][0])

    def test_3d_clean(self):
        frames = 2
        data, badmap = fake_data((frames, 256, 256))
        header = fits.header.Header()
        variance = np.full_like(data, 2.0)
        dripconfig.load()
        dripconfig.configuration['jbclean'] = 'FFT'

        result, var = clean(
            data, badmap=badmap, header=header, variance=variance)
        nans = np.isnan(data)
        variance[nans] = np.nan

        # NaNs should have been removed
        assert np.isnan(result).sum() < nans.sum()

        # variance should be unchanged
        assert np.isnan(var).sum() < nans.sum()

        # jailbars should have gone, but may be a few bad pixels
        nbad = np.sum(np.abs(result[~nans] - result[~nans][0]) > 1e-5)
        assert nbad < .05 * data.size
        assert header['JBCLEAN'] == 'FFT'

        # If no badmap, just clean jailbars
        result, var = clean(data)
        for f in result:
            assert np.allclose(f[~np.isnan(f)], f[~np.isnan(f)][0])
        assert var is None

        # Do not jbclean if JBCLEAN is not 'FFT'
        dripconfig.configuration['jbclean'] = 'MEDIAN'
        result, var = clean(data)
        for f in result:
            assert not np.allclose(f[~badmap], f[~badmap][0])

    def test_baddata(self, capsys):
        # bad data
        data = 1
        header = fits.header.Header()
        result = clean(data, header=header)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a valid array' in capt.err
        assert 'Did not clean' in repr(header['HISTORY'])

        # bad variance: continues, returns None for var
        frames = 2
        data, badmap = fake_data((frames, 256, 256))
        var = np.zeros((256, 256))
        result, var = clean(data, header=header, variance=var)
        assert result is not None
        assert var is None
        capt = capsys.readouterr()
        assert 'Variance must match data' in capt.err

    def test_jbclean_error(self, capsys, mocker):
        # jbclean failure
        def return_none(*args, **kwargs):
            return None
        mocker.patch('sofia_redux.instruments.forcast.clean.jbclean',
                     return_none)

        frames = 2
        data, badmap = fake_data((frames, 256, 256))
        header = fits.header.Header()

        dripconfig.configuration['jbclean'] = 'FFT'
        assert dripconfig.configuration['jbclean'] == 'FFT'

        result, var = clean(data, badmap=badmap, header=header)
        assert result is not None

        capt = capsys.readouterr()
        assert 'Jailbar cleaning failed' in capt.err

        history = repr(header['HISTORY'])
        for i in range(frames):
            assert 'failed on frame {}'.format(i + 1) in history

    def test_propagate_nan(self):
        data, badmap = fake_data(setnan=False)
        header = fits.header.Header()
        variance = np.full_like(data, 2.0)
        variance[np.isnan(data)] = np.nan
        nans = np.isnan(data)

        # propagate instead of interpolate
        result, var = clean(data, badmap=badmap, header=header,
                            variance=variance, propagate_nan=True)

        # NaNs should have been added ad badmap locations
        assert np.isnan(result).sum() > nans.sum()
        assert np.all(np.isnan(result[badmap]))
        assert np.all(np.isnan(var) == np.isnan(result))
