# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.forcast.check_readout_shift \
    import check_readout_shift
from sofia_redux.instruments.forcast.tests.resources import nmc_testdata


class TestCheckReadoutShift(object):

    def test_check_readout_shift(self):
        testdata = nmc_testdata()
        header = testdata['header']
        data = testdata['data']
        data = np.stack([data, data], axis=0)

        rand = np.random.RandomState(42)
        blank = np.zeros_like(data, dtype=float)
        blank += rand.randn(*data.shape)
        badsw = blank.copy()
        badlw = blank.copy()
        badsw[0, 36:75, (0 + 16):(2 + 16)] += rand.randn(75 - 36, 2) * 20
        badlw[0, 178: 189, (47 + 16): (61 + 16)] += \
            rand.randn(189 - 178, 61 - 47) * 20
        bad = blank.copy()
        bad[0] = badsw[0]
        bad[1] = badlw[0]
        header['SPECTEL1'] = 'FOO'
        header['SPECTEL2'] = 'FOO'
        header['SLIT'] = 'NONE'

        # test large and small regions (both setup in baddata)
        for detchan in [0, 1]:
            header['DETCHAN'] = detchan
            assert check_readout_shift(bad[detchan], header)
            assert not check_readout_shift(bad[[1, 0][detchan]], header)
            test3d = blank.copy()
            test3d[0] += bad[detchan]
            assert check_readout_shift(test3d, header)
            assert not check_readout_shift(blank, header)

        assert check_readout_shift(test3d, header)

        # check always false for x-dispersed
        for xdisp in ['FOR_XG063', 'FOR_XG111']:
            header['SPECTEL1'] = xdisp
            header['SPECTEL2'] = xdisp
            assert not check_readout_shift(test3d, header)

        # check always false for slit images
        spec_opt = ['FOR_G063', 'FOR_G111', 'FOR_G227', 'FOR_G329', 'fail']
        slits = ['NONE', 'UNKNOWN', 'fail']
        detchan = 0
        header['DETCHAN'] = detchan
        for opt in spec_opt:
            header['SPECTEL1'] = opt
            header['SPECTEL2'] = opt
            for slit in slits:
                header['SLIT'] = slit
                print((slit, opt))
                if opt == 'fail' and slit == 'fail':
                    assert not check_readout_shift(badsw, header)
                else:
                    assert check_readout_shift(badsw, header)

    def test_error_conditions(self, capsys):
        # bad data
        result = check_readout_shift(None, None)
        capt = capsys.readouterr()
        assert 'invalid data type' in capt.err
        assert result is None

        # bad header
        data = np.zeros(10)
        result = check_readout_shift(data, None)
        capt = capsys.readouterr()
        assert 'invalid header' in capt.err
        assert result is None

        # bad dimensions
        testdata = nmc_testdata()
        header = testdata['header']
        result = check_readout_shift(data, header)
        capt = capsys.readouterr()
        assert 'invalid data shape' in capt.err
        assert result is None
