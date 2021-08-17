# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy.wcs import WCS, SingularMatrixError
import numpy as np
import pytest

from sofia_redux.instruments.flitecam.readfits import readfits
from sofia_redux.instruments.flitecam.tests.resources import raw_testdata


class TestReadfits(object):

    def test_success(self, tmpdir):
        test_hdul = raw_testdata()
        testfile = str(tmpdir.join('test.fits'))
        test_hdul.writeto(testfile, overwrite=True)

        hdul = readfits(testfile)

        # 1 extension HDUList containing FLUX
        assert len(hdul) == 1

        data = hdul[0].data
        header = hdul[0].header
        assert isinstance(data, np.ndarray)
        assert not (data == 0).all()
        assert isinstance(header, fits.header.Header)
        assert len(header) > 0

        # check basic header values
        assert header['EXTNAME'] == 'FLUX'
        assert header['BUNIT'] == 'ct'

    def test_wcs(self, tmpdir, mocker):
        test_hdul = raw_testdata()
        header = test_hdul[0].header

        # add some bad WCS keywords, present in old FLITECAM data
        header['XPIXELSZ'] = 0
        header['YPIXELSZ'] = 0
        header['RADECSYS'] = 'FK5'
        header['PC1_1'] = 1
        header['PC1_2'] = 0
        header['PC2_1'] = 0
        header['PC2_2'] = 1

        testfile = str(tmpdir.join('test.fits'))
        test_hdul.writeto(testfile, overwrite=True)

        hdul = readfits(testfile)
        header = hdul[0].header

        # bad keys are removed
        bad = ['XPIXELSZ', 'YPIXELSZ', 'RADECSYS',
               'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
               'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        for key in bad:
            assert key not in header

        # CD keys are converted to crota/cdelt
        assert 'CDELT1' in header
        assert 'CDELT2' in header
        assert 'CROTA2' in header

        # make a wcs from header - should now succeed
        assert WCS(header)

        # mock missing CD in WCS: should raise a ValueError
        mocker.patch.object(np.linalg, 'det',
                            side_effect=AttributeError)
        with pytest.raises(ValueError) as err:
            readfits(testfile)
        assert 'FITS header is unreadable' in str(err)

        # mock an error in WCS: should raise a ValueError
        mocker.patch('sofia_redux.instruments.flitecam.readfits.WCS',
                     side_effect=SingularMatrixError)
        with pytest.raises(ValueError) as err:
            readfits(testfile)
        assert 'FITS header is unreadable' in str(err)

    def test_failure(self, capsys):
        assert readfits('this file does not exist') is None
        assert 'Error loading file' in capsys.readouterr().err
