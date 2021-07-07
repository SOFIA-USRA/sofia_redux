# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.darksum import darksum
import sofia_redux.instruments.forcast.configuration as dripconfig
dripconfig.load()


def fake_data():
    data = np.zeros((5, 256, 256))
    for i in range(data.shape[0]):
        data[i, :, :] = i + 1.
        for j in range(16):
            data[i, :, j * 16] += 10
    return data


class TestDarksum(object):

    def test_errors(self):
        assert darksum(1, fits.header.Header()) is None
        data = np.full((10, 10), np.nan)
        assert darksum(data, fits.header.Header()) is None
        data = np.full((10, 10), 0.0)
        assert darksum(data, fits.header.Header()) is None

    def test_success(self):
        header = fits.header.Header()
        data = fake_data()
        variance = np.full_like(data, 2.0)
        extra = {}
        result = darksum(data, header, extra=extra)
        assert len(extra) == 0
        assert np.allclose(result[0], data[2, :, :])
        assert result[1] is None
        result = darksum(data, header, darkvar=variance)
        nframes2 = data.shape[0] ** 2
        assert np.allclose(result[1],
                           np.sum(variance, axis=0) / nframes2)
        assert np.allclose(result[0], data[2])

    def test_badmap(self, mocker, capsys):
        data = fake_data()
        header = fits.header.Header()
        badmap = np.zeros_like(data[0], dtype=bool)

        # add sprinkling of bad data
        rand = np.random.RandomState(42)
        randomx = rand.choice(data.shape[2], 50)
        randomy = rand.choice(data.shape[1], 50)
        for x, y in zip(randomx, randomy):
            badmap[y, x] = True
        for frame in range(data.shape[0]):
            badframe = data[frame]
            badframe[badmap] = -9999
        extra = {}

        darksum(data, header, badmap=badmap, extra=extra)
        assert -9999 in data
        assert not np.isnan(data).any()
        assert not np.isnan(extra['cleaned']).any()

        # add large bad section
        badmap[25:50, 25:50] = True
        darksum(data, header, badmap=badmap, extra=extra)
        assert -9999 in data
        assert not np.isnan(data).any()
        assert np.isnan(extra['cleaned']).any()
        mask = ~np.isnan(extra['cleaned'])
        assert -9999 not in (extra['cleaned'][mask])

        # test clean failure
        mocker.patch('sofia_redux.instruments.forcast.darksum.clean',
                     return_value=None)
        capsys.readouterr()
        extra = {}
        darksum(data, header, badmap=badmap, extra=extra)
        capt = capsys.readouterr()
        assert 'cleaning failed' in capt.err
        assert 'cleaned' not in extra

    def test_jailbar(self):
        data = fake_data()
        header = fits.header.Header()
        variance = np.full_like(data, 2.0)
        result = darksum(data, header, jailbar=True, darkvar=variance)
        assert np.allclose(result[0], 3)
        assert np.allclose(
            result[1], np.sum(variance, axis=0) / (data.shape[0] ** 2))

    def test_single_dark(self):
        # uncleaned, single plane -- should just be returned as is
        header = fits.header.Header()
        data = fake_data()[0, :, :]
        variance = np.full_like(data, 2.0)
        result = darksum(data, header, darkvar=variance)
        assert np.allclose(result[0], data)
        assert np.allclose(result[1], variance)
