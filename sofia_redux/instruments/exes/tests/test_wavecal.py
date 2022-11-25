# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging
import pytest
import numpy as np

from sofia_redux.instruments.exes import wavecal as ew
from sofia_redux.instruments.exes.tests import resources


class TestWavecal(object):

    @pytest.fixture
    def header(self):
        hdr = resources.cross_dispersed_flat_header()
        return hdr

    @pytest.fixture
    def single_order_header(self):
        hdr = resources.single_order_flat_header()
        return hdr

    def test_wavecal(self, header):
        # 2D wavemap, spatmap, order mask
        wavemap = ew.wavecal(header)
        assert wavemap.shape == (3, header['NSPEC'], header['NSPAT'])

        # 1st order is marked in order mask, has 960 spectral elements,
        # 36 spatial
        order1 = wavemap[0][wavemap[2] == 1]
        assert order1.size == 960 * 36
        order1 = order1.reshape(960, 36).T
        assert np.allclose(order1, order1[0])

        # 1D wavecal only
        wavemap = ew.wavecal(header, order=1)
        assert wavemap.shape == (960,)
        assert np.allclose(wavemap, order1[0])

    def test_wavecal_single_order(self, single_order_header):
        header = single_order_header

        # 2D wavemap, spatmap, order mask
        wavemap = ew.wavecal(header)
        assert wavemap.shape == (3, header['NSPEC'], header['NSPAT'])

        # 1st order is marked in order mask, has 1020 spectral elements,
        # 350 spatial
        order1 = wavemap[0][wavemap[2] == 1]
        assert order1.size == 1020 * 350
        order1 = order1.reshape(350, 1020)
        assert np.allclose(order1, order1[0])

        # 1D wavecal only
        wavemap = ew.wavecal(header, order=1)
        assert wavemap.shape == (1020,)
        assert np.allclose(wavemap, order1[0])

    def test_parse_inputs_default_order(self, header):
        params = ew._parse_inputs(header, None)

        n_expected = 22
        assert isinstance(params, dict)
        assert params['order'] is None
        assert params['norders'] == n_expected
        assert len(params['ob']) == n_expected
        assert len(params['ot']) == n_expected
        assert len(params['os']) == n_expected
        assert len(params['oe']) == n_expected
        assert params['crossdisp'] is True

    def test_parse_inputs_defined_order(self, header):
        params = ew._parse_inputs(header, 15)

        assert isinstance(params, dict)
        assert params['order'] == 15

        n_expected = 22
        assert params['norders'] == n_expected
        assert len(params['ob']) == n_expected
        assert len(params['ot']) == n_expected
        assert len(params['os']) == n_expected
        assert len(params['oe']) == n_expected
        assert params['crossdisp'] is True

    def test_parse_inputs_bad_order(self, header):
        params = ew._parse_inputs(header, 'last')
        assert params['order'] == 'last'

    def test_parse_inputs_malformed(self, header):
        header['NORDERS'] = 7
        with pytest.raises(ValueError) as msg:
            ew._parse_inputs(header, None)

        assert "Can't determine edges" in str(msg)

    def test_parse_inputs_wno(self, header):
        header['WAVENO0'] = 1210.0
        header['WNO0'] = -9999
        params = ew._parse_inputs(header, None)
        assert params['wnoc'] == header['WAVENO0']

        header['WNO0'] = 840.0
        params = ew._parse_inputs(header, None)
        assert params['wnoc'] == header['WNO0']

        header['WNO0'] = 'a'
        params = ew._parse_inputs(header, None)
        assert params['wnoc'] == header['WAVENO0']

    def test_wavecal_malformed(self, header, caplog):
        caplog.set_level(logging.ERROR)
        header['NORDERS'] = 7
        wavemap = ew.wavecal(header)
        assert isinstance(wavemap, np.ndarray)
        assert len(wavemap) == 0
        assert "Can't determine edges" in caplog.text

    def test_setup_wavemap_default_order(self, header):
        params = ew._parse_inputs(header, None)
        keys = ['wavemap', 'wavecal', 'spatcal', 'order_mask',
                'order_idx', 'w', 's']
        for key in keys:
            assert key not in params

        ew._setup_wavemap(params)

        shape = (1024, 1024)
        assert params['order_idx'] == -1
        assert params['wavemap'].shape == (3, 1024, 1024)
        assert params['wavecal'].shape == shape
        assert params['spatcal'].shape == shape
        assert params['order_mask'].shape == shape
        assert params['w'].shape == shape
        assert params['s'].shape == shape

    def test_setup_wavemap_defined_order(self, header):
        params = ew._parse_inputs(header, 15)
        keys = ['wavemap', 'wavecal', 'spatcal', 'order_mask',
                'order_idx', 'w', 's']
        for key in keys:
            assert key not in params

        ew._setup_wavemap(params)

        norders = 22
        assert params['order_idx'] == norders - 15 + 1
        assert params['wavemap'].shape == (1004,)
        assert len(params['wavecal']) == 0
        assert len(params['spatcal']) == 0
        assert len(params['order_mask']) == 0
        assert len(params['w']) == 1024
        assert len(params['s']) == 0

    def test_check_order(self):
        order = 'UNKNOWN'
        result = ew._check_order(order)
        assert result == [0]

        result = ew._check_order(order, default=5)
        assert result == [5]

        order = '4,9,1,39,80'
        result = ew._check_order(order)
        assert np.allclose(result, [4, 9, 1, 39, 80])

        result = ew._check_order(order, default=5)
        assert np.allclose(result, [4, 9, 1, 39, 80])

    def test_populate_wavecal_default_order(self, header):
        params = ew._parse_inputs(header, None)
        ew._setup_wavemap(params)

        for key in ['wavemap', 'wavecal', 'spatcal', 'order_mask']:
            assert np.all(np.isnan(params[key]))
        ew._populate_wavecal(params)

        for key in ['wavemap', 'wavecal', 'spatcal', 'order_mask']:
            assert not np.all(np.isnan(params[key]))

    def test_populate_wavecal_defined_order(self, header):
        params = ew._parse_inputs(header, 33)
        ew._setup_wavemap(params)

        assert np.all(np.isnan(params['wavemap']))

        ew._populate_wavecal(params)

        assert not np.all(np.isnan(params['wavemap']))
        assert len(params['wavecal']) == 1024
        assert len(params['spatcal']) == 1024

    def test_update_header_default_order(self, header):
        wavemap = np.zeros((3, 5, 5))
        wavecal = np.ones((5, 5))
        spatcal = np.ones((5, 5)) * 3
        order_mask = np.ones((5, 5)) * 4
        params = {'header': header, 'order_idx': -1, 'wavecal': wavecal,
                  'spatcal': spatcal, 'wavemap': wavemap,
                  'order_mask': order_mask}

        for key in ['WCTYPE', 'BUNIT1', 'BUNIT2']:
            assert key not in header

        ew._update_header(params)

        assert np.all(wavemap[0] == 1)
        assert np.all(wavemap[1] == 3)
        assert np.all(wavemap[2] == 4)
        assert header['WCTYPE'] == '1D'
        assert header['BUNIT1'] == 'cm-1'
        assert header['BUNIT2'] == 'arcsec'

    def test_update_header_defined_order(self, header):
        wavemap = np.zeros((3, 5, 5))
        wavecal = np.ones((5, 5))
        spatcal = np.ones((5, 5)) * 3
        order_mask = np.ones((5, 5)) * 4
        params = {'header': header, 'order_idx': 15, 'wavecal': wavecal,
                  'spatcal': spatcal, 'order_mask': order_mask,
                  'wavemap': wavemap}

        for key in ['WCTYPE', 'BUNIT1', 'BUNIT2']:
            assert key not in header

        ew._update_header(params)

        assert np.all(wavemap[0] == 0)
        assert np.all(wavemap[1] == 0)
        assert np.all(wavemap[2] == 0)
        for key in ['WCTYPE', 'BUNIT1', 'BUNIT2']:
            assert key not in header
