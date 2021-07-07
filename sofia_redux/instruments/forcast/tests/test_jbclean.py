# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.jbclean import jbfft, jbmedian, jbclean
from sofia_redux.instruments.forcast.tests.resources \
    import random_mask, add_jailbars


def fake_data(shape=(256, 256), value=2.0, badfrac=0.1, jblevel=1.0):
    data = np.full(shape, value)
    add_jailbars(data, level=jblevel)
    data[random_mask(shape, frac=badfrac)] = np.nan
    # add a fully masked column
    data[:, 100] = np.nan
    return data


class TestJbclean(object):

    def test_jbfft(self):
        data = fake_data()
        result = jbfft(data)
        assert np.isnan(data).any()

        # check NaNs were not interpolated over
        assert np.allclose(np.isnan(data), np.isnan(result))
        mask = ~np.isnan(data)
        std_original = np.std(data[mask])
        std_corrected = np.std(result[mask])
        assert std_original != 0
        assert std_corrected == 0
        assert np.mean(result[mask]) != 0
        result = jbfft(np.full_like(data, np.nan))
        assert np.isnan(result).all()
        result = jbfft(None)
        assert result is None

        # Check funky bar spacing
        result = jbfft(data, bar_spacing=13)
        assert not np.allclose(result[mask], data[mask])
        assert not np.allclose(result[mask], result[mask][0])
        assert result is not data

    def test_jbmedian(self):
        data = fake_data()
        result = jbmedian(data)
        assert np.isnan(data).any()

        # check NaNs were not interpolated over
        assert np.allclose(np.isnan(data), np.isnan(result))
        mask = ~np.isnan(data)
        std_original = np.std(data[mask])
        std_corrected = np.std(result[mask])
        assert not np.allclose(std_original, 0, atol=0.05)
        assert np.allclose(std_corrected, 0, atol=0.05)

        assert np.mean(result[mask]) != 0
        result = jbmedian(np.full_like(data, np.nan))
        assert np.isnan(result).all()
        result = jbmedian(None)
        assert result is None

        # Check funky bar spacing
        result = jbmedian(data, bar_spacing=13)
        assert np.allclose(result[mask], data[mask])
        assert result is not data

        # if bar_spacing + width <= 1 we should hit an error
        # no error:
        jbmedian(data, bar_spacing=16, width=-14)
        # error:
        with pytest.raises(ValueError):
            jbmedian(data, bar_spacing=16, width=-15)

    def test_jbclean(self):
        data = fake_data()
        mask = ~np.isnan(data)
        variance = np.full_like(data, 2.0)
        dripconfig.load()
        for method in ['MEDIAN', 'FFT']:
            dripconfig.configuration['jbclean'] = method
            header = fits.header.Header()
            result, var = jbclean(data, header=header, variance=variance)
            assert header['JBCLEAN'] == method
            assert np.allclose(var, variance)
            assert np.allclose(result[mask], result[mask][0])

        dripconfig.configuration['jbclean'] = 'FOO'
        jbclean(data, header=header, variance=variance)
        dripconfig.load()
        assert 'invalid method' in str(header)

        result, var = jbclean(data)
        assert np.allclose(result[mask], result[mask][0])
        assert var is None

        # check invalid data and var
        result = jbclean(1)
        assert result is None
        result = jbclean(data, variance=np.zeros(10))
        assert result is None
