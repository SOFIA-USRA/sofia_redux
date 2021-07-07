# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import pytest

from sofia_redux.calibration.pipecal_applyphot import pipecal_applyphot
from sofia_redux.calibration.pipecal_error import PipeCalError
from sofia_redux.calibration.tests import resources


class TestApplyPhot(object):

    def test_errors(self, tmpdir, capsys, mocker):
        # test missing file
        with pytest.raises(PipeCalError):
            pipecal_applyphot('badfile.fits')
        capt = capsys.readouterr()
        assert 'Unable to open' in capt.err

        # test bad fits file
        badfile = tmpdir.join('badfile.fits')
        badfile.write('BADVAL')
        with pytest.raises(PipeCalError):
            pipecal_applyphot(str(badfile))
        capt = capsys.readouterr()
        assert 'Bad FITS file' in capt.err

        # some basic data
        hdul = resources.forcast_data()
        goodfile = str(tmpdir.join('goodfile.fits'))
        hdul.writeto(goodfile, overwrite=True)
        hdul.close()

        # test wrong instrument
        hdul[0].header['INSTRUME'] = 'BADVAL'
        badfile = str(tmpdir.join('badfile.fits'))
        hdul.writeto(badfile, overwrite=True)
        hdul.close()
        with pytest.raises(PipeCalError):
            pipecal_applyphot(badfile)
        capt = capsys.readouterr()
        assert 'Unsupported instrument' in capt.err
        hdul[0].header['INSTRUME'] = 'FORCAST'

        # negative flux - warns only
        def bad_calfac(*args, **kwargs):
            raise ValueError('bad flux')
        mocker.patch(
            'sofia_redux.calibration.pipecal_applyphot.pipecal_calfac',
            bad_calfac)
        pipecal_applyphot(goodfile)
        capt = capsys.readouterr()
        assert 'Negative flux' in capt.err

        # missing config, no srcpos -- no errors, but no refcalfac
        del hdul[0].header['SRCPOSX']
        del hdul[0].header['SRCPOSY']
        hdul.writeto(goodfile, overwrite=True)
        hdul.close()
        mocker.patch(
            'sofia_redux.calibration.pipecal_applyphot.pipecal_config',
            return_value=None)
        pipecal_applyphot(goodfile)
        capt = capsys.readouterr()
        assert 'No config found' in capt.err
        assert 'No model found' in capt.err

    def test_overwrite(self, tmpdir, mocker):
        hdul = resources.forcast_data()
        testfile = str(tmpdir.join('testfile.fits'))
        hdul.writeto(testfile, overwrite=True)
        hdul.close()

        # should write a _new file instead of overwriting
        pipecal_applyphot(testfile, overwrite=False)

        newfile = testfile.replace('.fits', '_new.fits')
        assert os.path.isfile(newfile)
        assert 'REFCALFC' in fits.getheader(newfile)
        assert 'REFCALFC' not in fits.getheader(testfile)

        # same with missing config -- no refcalfc, but stapflx
        # should be there
        testfile2 = str(tmpdir.join('testfile2.fits'))
        hdul.writeto(testfile2, overwrite=True)
        hdul.close()

        mocker.patch(
            'sofia_redux.calibration.pipecal_applyphot.pipecal_config',
            return_value=None)
        pipecal_applyphot(testfile2, overwrite=False)

        newfile = testfile2.replace('.fits', '_new.fits')
        assert os.path.isfile(newfile)
        hdr = fits.getheader(newfile)
        assert 'REFCALFC' not in hdr
        assert 'STAPFLX' in hdr
        assert 'STAPFLX' not in fits.getheader(testfile2)

    @pytest.mark.parametrize('data',
                             ['forcast_data', 'forcast_legacy_data',
                              'hawc_pol_data', 'hawc_im_data',
                              'flitecam_data', 'flipo_data',
                              'flitecam_new_data'])
    def test_instruments(self, tmpdir, capsys, data):
        # get synthetic data from resources
        func = getattr(resources, data)
        hdul = func()
        testfile = str(tmpdir.join('testfile.fits'))
        hdul.writeto(testfile, overwrite=True)
        hdul.close()

        # refcalfc not already in header
        assert 'REFCALFC' not in hdul[0].header

        # applyphot should succeed
        hdul.close()
        pipecal_applyphot(testfile)
        capt = capsys.readouterr()
        assert 'Reference Cal Factor' in capt.out

        # refcalfc is added to header
        hdr = fits.getheader(testfile)
        assert 'REFCALFC' in hdr
