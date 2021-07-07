# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.merge_centroid \
    import addhist, merge_centroid
from sofia_redux.instruments.forcast.tests.resources \
    import nmc_testdata, npc_testdata


class TestMergeCentroid(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Merge: test history message'

    def test_nmc_merge_centroid(self):
        test = nmc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        varval = 2.0
        variance = np.full_like(data, varval)
        normmap = np.full_like(data, np.nan)
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        merged, var = merge_centroid(data, header, variance=variance,
                                     normmap=normmap)
        dripconfig.load()
        nm = np.nanmax(normmap)
        assert nm == 4
        assert np.allclose(np.nanmin(var), varval * (nm - 1) / (nm ** 2))
        assert np.allclose(np.nanmax(data),
                           np.nanmax(merged * nm / (nm - 1)),
                           atol=0.1)
        npeaks = 3
        for key in ['MRGX', 'MRGY', 'MRGDX', 'MRGDY']:
            for i in range(npeaks):
                assert '%s%i' % (key, i) in header
        assert 'MRGX3' not in header

    def test_npc_merge_centroid(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        varval = 2.0
        variance = np.full_like(data, varval)
        normmap = np.full_like(data, np.nan)
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        merged, var = merge_centroid(
            data, header, variance=variance, normmap=normmap)
        dripconfig.load()
        nm = np.nanmax(normmap)
        assert nm == 4
        assert np.allclose(np.nanmin(var), varval * nm / (nm ** 2))
        assert np.allclose(np.nanmax(data), np.nanmax(merged), atol=0.1)
        npeaks = 4
        for key in ['MRGX', 'MRGY', 'MRGDX', 'MRGDY']:
            for i in range(npeaks):
                assert '%s%i' % (key, i) in header

    def test_errors(self, capsys, mocker):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        header['INSTMODE'] = 'FOO'
        # Check unrecognized mode
        assert merge_centroid(data, header) is None

        # Check wrong number of peaks found
        header = test['header'].copy()
        data *= 0
        assert merge_centroid(data, header) is None

        # check wrong number of peaks found for NPC
        data = abs(test['data'].copy())
        assert merge_centroid(data, header) is None

        # check wrong number of peaks found for NMC
        header = nmc_testdata()['header']
        data = npc_testdata()['data']
        assert merge_centroid(data, header) is None

        # check bad header
        assert merge_centroid(data, None) is None

        # check bad data
        assert merge_centroid(None, header) is None
        assert merge_centroid(np.full_like(data, np.nan), header) is None

        # check bad variance; data still okay
        data = test['data'].copy()
        header = test['header'].copy()
        d, v = merge_centroid(data, header, variance=10)
        assert d is not None
        assert v is None

        # check bad normmap -- doesn't care, will resize to data
        nmp = np.array([10])
        d, v = merge_centroid(data, header,
                              variance=np.full_like(data, 2.0),
                              normmap=nmp)
        assert d is not None
        assert v is not None
        assert nmp.shape == d.shape

        # check bad fwhm
        dripconfig.load()
        dripconfig.configuration['mfwhm'] = -4
        result = merge_centroid(data, header)
        capt = capsys.readouterr()
        assert 'using default FWHM' in capt.out
        assert result is not None

        # check bad x/y returned (beyond edge of data)
        def peakfind(*args, **kwargs):
            return [(600, 600), (1000, 1000),
                    (800, 800), (900, 900)]
        mocker.patch(
            'sofia_redux.instruments.forcast.merge_centroid.peakfind',
            peakfind)
        result = merge_centroid(data, header)
        capt = capsys.readouterr()
        assert 'wrong peaks found' in capt.err
        assert result is None

    def test_resize(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        varval = 2.0
        variance = np.full_like(data, varval)
        normmap = np.full_like(data, np.nan)
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        msmall, vsmall = merge_centroid(
            data, header, variance=variance, normmap=normmap,
            resize=False)
        mlarge, vlarge = merge_centroid(
            data, header, variance=variance, normmap=normmap,
            resize=True)
        for s, l in zip(msmall.shape, mlarge.shape):
            assert s < l
        for s, l in zip(vsmall.shape, vlarge.shape):
            assert s < l
