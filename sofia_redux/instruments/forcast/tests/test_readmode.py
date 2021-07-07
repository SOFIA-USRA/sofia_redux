# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.instruments.forcast.readmode import readmode
from sofia_redux.instruments.forcast.tests.resources \
    import nmc_testdata, npc_testdata


class TestReadmode(object):

    def test_badheader(self, capsys):
        readmode(None)
        capt = capsys.readouterr()
        assert 'invalid header' in capt.err

    def test_c2nc2(self):
        header = fits.header.Header()
        header['INSTMODE'] = 'C2'
        header['C2NC2'] = 1
        header['NAXIS3'] = 2
        assert 'C2NC2' == readmode(header)

        header['INSTMODE'] = 'FOOBAR'
        header['SKYMODE'] = 'C2NC2'
        assert 'C2NC2' == readmode(header)

    def test_c2nc4(self):
        header = fits.header.Header()
        header['INSTMODE'] = 'C2'
        header['C2NC2'] = 1
        header['NAXIS3'] = 4
        assert 'C2NC4' == readmode(header)

        header['INSTMODE'] = 'C2NC2'
        assert 'C2NC4' == readmode(header)

        header['SKYMODE'] = 'test'
        assert 'C2NC4' == readmode(header)
        assert header['SKYMODE'] == 'C2NC4'

    def test_c2(self):
        header = fits.header.Header()
        header['INSTMODE'] = 'C2'
        header['C2NC2'] = 0
        assert 'C2' == readmode(header)

    def test_c2n(self):
        header = fits.header.Header()
        header['INSTMODE'] = 'C2N'
        header['C2NC2'] = 0
        # if no skymode, with zero amplitudes, is nmc
        assert 'NMC' == readmode(header)
        # if skymode, is skymode
        header['SKYMODE'] = 'SLITSCAN'
        assert 'SLITSCAN' == readmode(header)

    def test_nmc(self):
        assert readmode(nmc_testdata()['header']) == 'NMC'

    def test_npc(self):
        assert readmode(npc_testdata()['header']) == 'NPC'
