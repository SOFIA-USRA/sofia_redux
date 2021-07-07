# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.merge_shift import addhist, merge_shift
from sofia_redux.instruments.forcast.tests.resources \
    import nmc_testdata, npc_testdata


class TestMergeShift(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Merge: test history message'

    def test_merge_shift_nmc(self):
        test = nmc_testdata()
        data = test['data']
        header = test['header']
        sh = imgshift_header(header)
        chopnod = [sh['chopx'], sh['chopy'], sh['nodx'], sh['nody']]
        variance = np.full_like(data, 2)
        normmap = np.zeros_like(data)
        merged, var = merge_shift(data, chopnod, header=header,
                                  variance=variance, nmc=True,
                                  normmap=normmap)
        assert np.nanmax(normmap) == 4
        assert np.nanmin(var) == 2 * 3 / (4 ** 2)
        # NMC has one extra exposure at the source location
        assert np.allclose(np.nanmax(data),
                           np.nanmax(merged * 4 / 3),
                           atol=0.1)
        for key in ['SHIFTORD', 'MRGDX0', 'MRGDX1', 'MRGDY0', 'MRGDY1']:
            assert key in header

    def test_merge_shift_npc(self):
        test = npc_testdata()
        data = test['data']
        header = test['header']
        sh = imgshift_header(header)
        chopnod = [sh['chopx'], sh['chopy'], sh['nodx'], sh['nody']]
        variance = np.full_like(data, 2)
        normmap = np.zeros_like(data)
        merged, var = merge_shift(data, chopnod, header=header,
                                  variance=variance, nmc=False,
                                  normmap=normmap)
        assert np.nanmax(normmap) == 4
        assert np.nanmin(var) == 2 * 4 / (4 ** 2)
        assert np.allclose(np.nanmax(data), np.nanmax(merged), atol=0.1)
        for key in ['SHIFTORD', 'MRGDX0', 'MRGDX1', 'MRGDY0', 'MRGDY1']:
            assert key in header

    def test_chop_maxshift(self):
        test = npc_testdata()
        data = test['data']
        header = test['header']
        sh = imgshift_header(header)
        chopnod = [sh['chopx'], sh['chopy'], sh['nodx'], sh['nody']]
        variance = np.full_like(data, 2)
        normmap = np.zeros_like(data)
        merged, var = merge_shift(data, chopnod, header=header,
                                  variance=variance, nmc=False,
                                  normmap=normmap, maxshift=0)
        assert np.all(merged == data)
        assert np.all(variance == var)
        assert np.all(normmap == 1)
        assert 'chop positions was not applied' in str(header)
        assert 'MRGDX0' not in header

        test = nmc_testdata()
        data = test['data']
        header = test['header']
        sh = imgshift_header(header)
        chopnod = [sh['chopx'], sh['chopy'], sh['nodx'], sh['nody']]
        variance = np.full_like(data, 2)
        normmap = np.zeros_like(data)
        merged, var = merge_shift(data, chopnod, header=header,
                                  variance=variance, nmc=True,
                                  normmap=normmap, maxshift=0)
        # check nmc normalization
        assert np.all(merged == data / 2.)
        assert np.all(var == variance / 4.)
        assert np.all(normmap == 2)
        assert 'chop positions was not applied' in str(header)
        assert 'MRGDX0' not in header

    def test_nod_maxshift(self):
        test = npc_testdata()
        data = test['data']
        header = test['header']
        chopnod = [1.0, 1.0, 3.0, 3.0]
        variance = np.full_like(data, 2)
        normmap = np.zeros_like(data)
        _, var = merge_shift(data, chopnod, header=header,
                             variance=variance, nmc=False,
                             normmap=normmap, maxshift=2)
        assert np.nanmin(var) == 2 * 2 / (2 ** 2)
        assert np.nanmin(normmap) < 2
        assert np.nanmax(normmap) == 2
        assert 'nod positions was not applied' in str(header)
        assert 'MRGDX0' in header
        assert 'MRGDX1' not in header

    def test_errors(self):
        test = nmc_testdata()
        data = test['data']
        header = test['header']
        sh = imgshift_header(header)
        chopnod = [sh['chopx'], sh['chopy'], sh['nodx'], sh['nody']]
        assert merge_shift(None, chopnod) is None
        assert merge_shift(data, None) is None
        assert merge_shift(data * np.nan, chopnod) is None

        # bad variance
        _, var = merge_shift(data, chopnod, variance=np.zeros(10))
        assert var is None
