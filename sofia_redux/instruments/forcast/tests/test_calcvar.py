# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.calcvar import calcvar
import sofia_redux.instruments.forcast.configuration as dripconfig


class TestCalcvar(object):

    def test_invalid(self):
        header = fits.header.Header()
        assert calcvar(np.array([0]), header) is None
        data = np.full((10, 10), 2.0)
        assert calcvar(np.full((10, 10), 2.0), header) is None
        header['DETITIME'] = 1.0
        assert calcvar(data, header) is None
        header['FRMRATE'] = 1.0
        assert calcvar(data, header) is not None

    def test_calcvar(self):
        header = fits.header.Header()
        dataval = 1.0
        data = np.full((10, 10), dataval)
        header['RN_HIGH'] = 3
        header['RN_LOW'] = 2
        header['BETA_G'] = 1
        header['EPERADU'] = 1
        header['DETITIME'] = 2
        header['FRMRATE'] = 1
        header['ILOWCAP'] = False
        dripconfig.load()
        for k, v in header.items():
            if k != 'HISTORY':
                dripconfig.configuration[k.lower()] = v
        result = calcvar(data, header)
        assert (result == 10).all()
        header['ILOWCAP'] = True
        dripconfig.configuration['ilowcap'] = True
        result = calcvar(data, header)
        dripconfig.load()
        assert (result == 5).all()

    def test_noheader(self, capsys):
        data = np.full((10, 10), 2.0)
        # returns None
        assert calcvar(data, []) is None
        # and issues error
        capt = capsys.readouterr()
        assert 'must provide valid header' in capt.err
