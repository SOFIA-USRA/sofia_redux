# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.flitecam.calcvar import calcvar


class TestCalcvar(object):

    def test_invalid(self, capsys):
        header = fits.header.Header()
        data = np.full((10, 10), 2.0)
        assert calcvar(np.array([0]), header) is None
        assert calcvar(data, 'bad') is None
        assert calcvar(data, header) is None

        header['ITIME'] = 1.0
        assert calcvar(data, header) is None
        header['COADDS'] = 1.0
        assert calcvar(data, header) is None
        header['NDR'] = 1.0
        assert calcvar(data, header) is None

        # check for error messages
        assert capsys.readouterr().err.count('Missing time keywords') == 4

        # last required keyword - should now pass
        header['TABLE_MS'] = 1.0
        assert calcvar(data, header) is not None

    def test_calcvar(self):
        header = fits.header.Header()
        dataval = 1.0
        data = np.full((10, 10), dataval)
        header['ITIME'] = 1
        header['COADDS'] = 2
        header['NDR'] = 3
        header['TABLE_MS'] = 4

        result = calcvar(data, header)
        assert np.allclose(result, 12.23810546)
