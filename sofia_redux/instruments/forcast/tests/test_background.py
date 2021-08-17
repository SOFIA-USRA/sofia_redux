# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.forcast.background import background, mode
from sofia_redux.instruments.forcast.tests.resources import nmc_testdata


def noisy_data(sigma=10):
    rand = np.random.RandomState(42)
    data = nmc_testdata()['data']
    level = np.nanmax(abs(data)) / sigma
    data += rand.normal(0, level, data.shape)
    section = [data.shape[1] // 2, data.shape[0] // 2,
               data.shape[1] // 4, data.shape[0] // 4]
    data = np.stack([data, data * 2])
    return data, section


class TestBackground(object):

    def test_background(self):
        data, section = noisy_data()
        median_stat = background(data, section)
        assert isinstance(median_stat, np.ndarray)
        mode_stat = background(data, section, stat='mode')
        assert isinstance(mode_stat, np.ndarray)
        assert not np.allclose(median_stat, mode_stat)
        header = fits.header.Header()
        single_mode = background(data[0], section, header=header)
        assert len(single_mode) == 1
        assert single_mode[0] == median_stat[0]
        assert 'NLINSLEV' in header

    def test_mask(self):
        data, s = noisy_data()
        mask = np.full((data.shape[1], data.shape[2]), False)
        mask[int(s[1] - s[3] / 4): int(s[1] + s[3] / 4),
             int(s[0] - s[2] / 4): int(s[0] + s[2] / 4)] = True
        mask_med = background(data, s, mask=mask)
        mask_mod = background(data, s, mask=mask, stat='mode')
        assert isinstance(mask_med, np.ndarray)
        assert isinstance(mask_mod, np.ndarray)

    def test_errors(self):
        data = np.zeros(10)

        # test bad section
        section = [1, 2, 3]
        with pytest.raises(ValueError):
            background(data, section)

        # test bad data
        section = [1, 2, 3, 4]
        result = background(data, section)
        assert result is None

    def test_mode(self):
        # test array, not sorted
        a = np.arange(10, -1, -1)

        # all unique: return minimum
        assert mode(a) == 0

        # one most common value: return it
        a = np.append(a, 1)
        assert mode(a) == 1

        # two most common values: return the smallest
        a = np.append(a, 2)
        assert mode(a) == 1

        # non-array okay
        assert mode([1, 2, 3, 3]) == 3

        # empty
        with pytest.raises(ValueError):
            mode(np.array([]))
