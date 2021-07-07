# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.merge import addhist, merge
from sofia_redux.instruments.forcast.setpar import setpar
from sofia_redux.instruments.forcast.tests.resources \
    import npc_testdata, nmc_testdata


class TestMerge(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Merge: test history message'

    def test_merge_errors(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        assert merge(np.zeros(10), header) is None

        nomerge, _ = merge(data, fits.header.Header())
        assert np.allclose(nomerge, data)

        setpar('cormerge', 'XCOR')
        header['CORMERGE'] = 'XCOR'
        header['SLIT'] = 'NONE'
        assert merge(np.full_like(data, np.nan), header) is None

        # bad header -- okay
        assert merge(data, None) is not None

        # bad variance -- data is okay
        m, v = merge(data, header, variance=10)
        assert m is not None
        assert v is None

    def test_merge_algorithms(self):
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        test = npc_testdata()
        data = test['data']
        for k, v in [('HEADER', 'algorithm uses header'),
                     ('XCOR', 'algorithm uses cross-correlation'),
                     ('CENTROID', 'algorithm uses centroid')]:
            header = test['header'].copy()
            setpar('cormerge', k)
            header['CORMERGE'] = k
            header['SLIT'] = 'NONE'
            merge(data, header)
            assert v in repr(header)

        header = test['header'].copy()
        setpar('cormerge', 'NONE')
        header['CORMERGE'] = 'NONE'
        header['SLIT'] = 'FOOBAR'
        merge(data, header)
        assert 'Shift algorithm not applied for NPC mode' in repr(header)

        header = nmc_testdata()['header']
        header['CORMERGE'] = 'NONE'
        header['SLIT'] = 'FOOBAR'
        merge(data, header, variance=np.full_like(data, 2.0))
        assert 'Shift algorithm not applied for NMC mode' in repr(header)

        dripconfig.load()

    def test_merge_full(self):
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        setpar('cormerge', 'CENTROID')
        header['CORMERGE'] = 'CENTROID'
        header['SLIT'] = 'NONE'
        header['SKY_ANGL'] = 45.0
        variance = np.full_like(data, 2.0)
        normmap = np.array([])
        d, v = merge(data, header, variance=variance, normmap=normmap,
                     rotation_order=1)

        assert np.isnan(d).any()
        assert np.nanmax(d) > 0
        assert d.shape == normmap.shape
        assert np.allclose(np.nanmax(normmap), 4, atol=0.1)
        assert np.allclose(np.nanmin(v), 0.5)
        assert 'New CROTA2 after rotation is 0.0 degrees' in repr(header)
        assert header['CROTA2'] == 0
        assert header['PRODTYPE'] == 'MERGED'

    def test_merge_wcs(self):
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        header['CORMERGE'] = 'CENTROID'
        header['SLIT'] = 'NONE'
        header['SKY_ANGL'] = 45.0
        y0, x0 = data.shape[0] / 2, data.shape[1] / 2
        x0 += x0 / 4
        y0 += y0 / 4
        sx0 = x0 + 3
        sy0 = y0 + 3
        header['CRPIX1'] = x0
        header['CRPIX2'] = y0
        header['SRCPOSX'] = sx0
        header['SRCPOSY'] = sy0
        variance = np.full_like(data, 2.0)
        normmap = np.array([])
        d, v = merge(data, header, variance=variance, normmap=normmap,
                     rotation_order=1, strip_border=True)
        assert len(d.shape) == 2
        assert data.shape != d.shape
        assert normmap.shape == d.shape
        assert v.shape == d.shape
        assert (header['CRPIX1'], header['CRPIX2']) != (x0, y0)
        assert (header['SRCPOSX'], header['SRCPOSY']) != (sx0, sy0)

    def test_centroid_fallback(self, mocker, capsys):
        # make centroid fail
        mocker.patch('sofia_redux.instruments.forcast.merge.merge_centroid',
                     return_value=None)

        dripconfig.load()
        dripconfig.configuration['border'] = 0
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()

        # run with centroid -- falls back to header
        setpar('cormerge', 'CENTROID')
        dfall, _ = merge(data, header, rotation_order=1)
        capt = capsys.readouterr()
        assert 'centroid failed; falling back to ' \
               'header shifts' in capt.err

        # run with header -- should be same
        setpar('cormerge', 'HEADER')
        dhdr, _ = merge(data, header, rotation_order=1)
        assert np.allclose(dfall, dhdr, equal_nan=True)
