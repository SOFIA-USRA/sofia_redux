# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.merge_correlation \
    import addhist, merge_correlation
from sofia_redux.instruments.forcast.tests.resources \
    import nmc_testdata, npc_testdata


class TestMergeCorrelation(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Merge: test history message'

    def test_nmc_merge_correlation(self):
        test = nmc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        varval = 2.0
        variance = np.full_like(data, varval)
        normmap = np.full_like(data, np.nan)
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        merged, var = merge_correlation(
            data, header, variance=variance, normmap=normmap)
        dripconfig.load()
        nm = np.nanmax(normmap)
        assert nm == 4
        assert np.nanmin(var) == varval * (nm - 1) / (nm ** 2)
        assert np.allclose(np.nanmax(data),
                           np.nanmax(merged * nm / (nm - 1)),
                           atol=0.1)
        for key in ['MRGDX', 'MRGDY']:
            for i in range(2):
                assert '%s%i' % (key, i) in header
        assert 'MRGX2' not in header

    def test_npc_merge_correlation(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        varval = 2.0
        variance = np.full_like(data, varval)
        normmap = np.full_like(data, np.nan)
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        merged, var = merge_correlation(
            data, header, variance=variance, normmap=normmap)
        dripconfig.load()
        nm = np.nanmax(normmap)
        assert nm == 4
        assert np.nanmin(var) == varval * nm / (nm ** 2)
        assert np.allclose(np.nanmax(data), np.nanmax(merged), atol=0.1)
        for key in ['MRGDX', 'MRGDY']:
            for i in range(2):
                assert '%s%i' % (key, i) in header

    def test_errors(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        assert merge_correlation(data, 'a') is None
        assert merge_correlation(np.array(10), header) is None
        dripconfig.load()
        dripconfig.configuration['border'] = data.shape[0]
        merged = merge_correlation(data, header)
        assert merged is None
        dripconfig.load()

        # check bad variance
        merged = merge_correlation(data, header, variance=10)
        assert merged[0] is not None
        assert merged[1] is None

    def test_upsample(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        dripconfig.load()
        dripconfig.configuration['border'] = 0

        merge_correlation(data, header, upsample=100)
        dx = header['MRGDX0']
        assert not np.allclose(dx, int(dx), atol=0.01)

        merge_correlation(data, header, upsample=1)
        dx = header['MRGDX0']
        assert dx == int(dx)
        dripconfig.load()

    def test_maxregister(self):
        test = npc_testdata()
        data = test['data'].copy()
        dmax = np.nanmax(data)
        header = test['header'].copy()

        # These settings should result in 0 chop nod so shift
        # algorithm will esentially subtract out all source
        header['CHPAMP1'] = 0
        header['NODAMP'] = 0
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        merged, _ = merge_correlation(data, header, maxshift=0)
        mmax = np.nanmax(merged)

        # maximum should be close to zero
        assert np.allclose(mmax, 0, atol=0.01)

        # Now allow a search over the whole image
        # Note that this solution may be incorrect as we cannot
        # guarantee which negative source correlates with which
        # positive source... That's why we need the shift from
        # the header as an initial guess.
        merged, _ = merge_correlation(data, header, maxregister=None)
        dripconfig.load()

        # should be closer to data than 0
        mmax = np.nanmax(merged)
        assert dmax - mmax < mmax

    def test_resize(self, capsys):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        varval = 2.0
        variance = np.full_like(data, varval)
        normmap = np.full_like(data, np.nan)
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        msmall, vsmall = merge_correlation(
            data, header, variance=variance, normmap=normmap,
            resize=False)
        mlarge, vlarge = merge_correlation(
            data, header, variance=variance, normmap=normmap,
            resize=True)
        for s, l in zip(msmall.shape, mlarge.shape):
            assert s < l
        for s, l in zip(vsmall.shape, vlarge.shape):
            assert s < l

        # test border
        dripconfig.configuration['border'] = 10
        mborder, vborder = merge_correlation(
            data, header, variance=variance, normmap=normmap,
            resize=False)
        assert mborder.shape[0] == msmall.shape[0]
        assert mborder.shape[1] == msmall.shape[1]
        capt = capsys.readouterr()
        assert 'Removing 10 pixel border from consideration' in capt.out
        assert np.allclose(msmall, mborder, equal_nan=True)
