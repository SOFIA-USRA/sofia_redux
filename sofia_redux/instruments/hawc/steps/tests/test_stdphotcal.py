# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepstdphotcal import StepStdPhotCal
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, scan_smp_data


class TestStdPhotCal(DRPTestCase):
    def test_siso(self, tmpdir, capsys):
        step = StepStdPhotCal()

        # cn/pol data
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        out = step(inp)
        assert isinstance(out, DataFits)
        assert out.getheadval('BUNIT') == 'Jy/pixel'

        # occasionally the fit fails on the synthetic random source --
        # don't check for stapflx in that case
        capt = capsys.readouterr()
        if not ('unable to run photometry' in capt.err.lower()):
            assert 'STAPFLX' in out.header

        # scan data
        hdul = scan_smp_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        out = step(inp)
        assert isinstance(out, DataFits)
        assert out.getheadval('BUNIT') == 'Jy/pixel'

        capt = capsys.readouterr()
        if not ('unable to run photometry' in capt.err.lower()):
            assert 'STAPFLX' in out.header

    def test_run_phot(self, tmpdir, capsys, mocker):
        hdul = scan_smp_data()

        # set the date to one with a known model flux
        hdul[0].header['DATE-OBS'] = '2019-02-20T02:00:00.000'

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        step = StepStdPhotCal()

        # default
        out = step(inp.copy())
        assert 'STAPFLX' in out.header
        assert 'MODLFLX' in out.header
        assert out.getheadval('BUNIT') == 'Jy/pixel'

        # run on an object with no model flux --
        # photometry and cal complete anyway
        inp.header['OBJECT'] = 'TESTVAL'
        out = step(inp.copy())
        assert 'STAPFLX' in out.header
        assert 'MODLFLX' not in out.header
        assert out.getheadval('BUNIT') == 'Jy/pixel'

        # mock an error in run_photometry
        def mock_phot(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch(
            'sofia_redux.instruments.hawc.steps.stepstdphotcal.run_photometry',
            mock_phot)

        # warns, but calibrates; no photometry keys
        out = step(inp.copy())
        assert out.getheadval('BUNIT') == 'Jy/pixel'
        assert 'CALFCTR' in out.header
        assert 'STAPFLX' not in out.header
        capt = capsys.readouterr()
        assert 'Unable to run photometry' in capt.err

        # mock a problem in fluxcal factor too
        mocker.patch(
            'sofia_redux.instruments.hawc.steps.stepstdphotcal.'
            'get_fluxcal_factor',
            return_value=(None, None))
        mocker.patch(
            'sofia_redux.instruments.hawc.steps.stepstdphotcal.apply_fluxcal',
            lambda *args, **kwargs: args[0])

        # warns, does not calibrate
        out = step(inp.copy())
        assert 'BUNIT' not in out.header
        capt = capsys.readouterr()
        assert 'No calibration factor found' in capt.err

    def test_phot_options(self, tmpdir, capsys):
        hdul = scan_smp_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        step = StepStdPhotCal()

        # specify readable source position --
        # photometry may fail, but calibration succeeds
        out = step(inp.copy(), srcpos='10,10')
        assert 'BUNIT' in out.header

        # bad source position
        with pytest.raises(ValueError):
            step(inp.copy(), srcpos='a,b')
        capt = capsys.readouterr()
        assert 'Invalid source position' in capt.err

        # bad source position
        with pytest.raises(ValueError):
            step(inp.copy(), srcpos='10')
        capt = capsys.readouterr()
        assert 'Invalid source position' in capt.err

    def test_nhwp1(self, tmpdir):
        hdul = pol_bgs_data()

        # make it look like chop/nod intensity data
        del hdul[0].header['NHWP']
        new_hdu = []
        for hdu in hdul:
            extname = hdu.header['EXTNAME']
            if 'Q' not in extname and 'U' not in extname:
                new_hdu.append(hdu)
        hdul = fits.HDUList(new_hdu)

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        # calibrates stokes I extensions (not bad pix mask)
        step = StepStdPhotCal()
        out = step(inp.copy())
        assert 'Jy' in out.getheader('STOKES I')['BUNIT']
        assert 'Jy' in out.getheader('ERROR I')['BUNIT']

    def test_calibrated(self, tmpdir):
        # already calibrated data
        hdul = scan_smp_data()
        hdul[0].header['BUNIT'] = 'Jy'
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        step = StepStdPhotCal()

        # runs photometry, does not calibrate
        out = step(inp)
        assert 'STAPFLX' in out.header
        assert 'CALFCTR' not in out.header
