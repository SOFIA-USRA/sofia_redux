# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.instruments.forcast.getdetchan import getdetchan


class TestGetdetchan(object):

    def test_error(self):
        assert getdetchan(None) == 'SW'

    def test_lw(self):
        header = fits.header.Header()
        header['DETCHAN'] = 1
        assert getdetchan(header) == 'LW'
        header['DETCHAN'] = '1'
        assert getdetchan(header) == 'LW'
        header['DETCHAN'] = 'LW'
        assert getdetchan(header) == 'LW'

    def test_sw(self):
        header = fits.header.Header()
        header['DETCHAN'] = 0
        assert getdetchan(header) == 'SW'
        header['DETCHAN'] = 'FOO'
        assert getdetchan(header) == 'SW'
