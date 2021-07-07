# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.instruments.fifi_ls.readfits import readfits
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, test_files


class TestReadfits(FIFITestCase):
    def test_success(self):
        filename = test_files()[0]
        hdul = readfits(filename, checkheader=False)
        assert isinstance(hdul, fits.HDUList)

    def test_checkhead(self, tmpdir):
        # default file passes
        filename = test_files()[0]
        hdul, success = readfits(filename, checkheader=True)
        assert isinstance(hdul, fits.HDUList)
        assert success is True

        # modify to make it not pass
        del hdul[0].header['SPECTEL1']
        badfile = str(tmpdir.join('badfile.fits'))
        hdul.writeto(badfile)
        hdul, success = readfits(badfile, checkheader=True)
        assert isinstance(hdul, fits.HDUList)
        assert success is False

    def test_errors(self, capsys, tmpdir):
        # invalid filename
        result = readfits('badfile.fits', checkheader=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

        # returns None, False if checkhead set
        result = readfits('badfile.fits', checkheader=True)
        assert result[0] is None
        assert result[1] is False
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

        # not a fifi file
        filename = test_files()[0]
        hdul = fits.open(filename)
        hdul[0].header['INSTRUME'] = 'TESTVAL'
        badfile = str(tmpdir.join('badfile.fits'))
        hdul.writeto(badfile)
        assert readfits(badfile) is None
        assert readfits(badfile, checkheader=True) == (None, False)
        capt = capsys.readouterr()
        assert 'Not a FIFI-LS file' in capt.err
