# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the SOFIA Chooser class."""

from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.sofia.sofia_chooser import SOFIAChooser

try:
    from sofia_redux.pipeline.sofia.forcast_imaging_reduction \
        import FORCASTImagingReduction
    from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
        import FORCASTSpectroscopyReduction
    from sofia_redux.pipeline.sofia.forcast_wavecal_reduction \
        import FORCASTWavecalReduction
    from sofia_redux.pipeline.sofia.forcast_spatcal_reduction \
        import FORCASTSpatcalReduction
    from sofia_redux.pipeline.sofia.forcast_slitcorr_reduction \
        import FORCASTSlitcorrReduction
    from sofia_redux.pipeline.sofia.exes_quicklook_reduction import \
        EXESQuicklookReduction
    HAS_DRIP = True
except ImportError:
    FORCASTImagingReduction = Reduction
    FORCASTSpectroscopyReduction = Reduction
    FORCASTWavecalReduction = Reduction
    FORCASTSpatcalReduction = Reduction
    FORCASTSlitcorrReduction = Reduction
    EXESQuicklookReduction = Reduction
    HAS_DRIP = False

try:
    from sofia_redux.pipeline.sofia.hawc_reduction import HAWCReduction
    HAS_DRP = True
except ImportError:
    HAWCReduction = Reduction
    HAS_DRP = False

try:
    from sofia_redux.pipeline.sofia.fifils_reduction import FIFILSReduction
    HAS_FIFI = True
except ImportError:
    FIFILSReduction = Reduction
    HAS_FIFI = False

try:
    from sofia_redux.pipeline.sofia.flitecam_imaging_reduction \
        import FLITECAMImagingReduction
    from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
        import FLITECAMSpectroscopyReduction
    from sofia_redux.pipeline.sofia.flitecam_wavecal_reduction \
        import FLITECAMWavecalReduction
    from sofia_redux.pipeline.sofia.flitecam_spatcal_reduction \
        import FLITECAMSpatcalReduction
    from sofia_redux.pipeline.sofia.flitecam_slitcorr_reduction \
        import FLITECAMSlitcorrReduction
    HAS_FLITECAM = True
except ImportError:
    FLITECAMImagingReduction = Reduction
    FLITECAMSpectroscopyReduction = Reduction
    FLITECAMWavecalReduction = Reduction
    FLITECAMSpatcalReduction = Reduction
    FLITECAMSlitcorrReduction = Reduction
    HAS_FLITECAM = False


class TestSOFIAChooser(object):
    def make_file(self):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data('test0.fits')
        return ffile

    def make_hawc_file(self):
        """Retrieve a test FITS file for HAWC mode."""
        fitstest = FitsTestCase()
        fitstest.setup()
        fitstest.copy_file('test0.fits')
        ffile = fitstest.temp('test0.fits')
        fits.setval(ffile, 'INSTRUME', value='HAWC_PLUS')
        return ffile

    def make_forcast_file(self):
        """Retrieve a test FITS file for FORCAST mode."""
        fitstest = FitsTestCase()
        fitstest.setup()
        fitstest.copy_file('test0.fits')
        ffile = fitstest.temp('test0.fits')
        fits.setval(ffile, 'INSTRUME', value='FORCAST')
        return ffile

    def make_fifi_file(self):
        """Retrieve a test FITS file for FIFI-LS mode."""
        fitstest = FitsTestCase()
        fitstest.setup()
        fitstest.copy_file('test0.fits')
        ffile = fitstest.temp('test0.fits')
        fits.setval(ffile, 'INSTRUME', value='FIFI-LS')
        return ffile

    def test_choose_reduction(self, capsys):
        chz = SOFIAChooser()

        # null data
        ro = chz.choose_reduction()
        assert type(ro) == Reduction

        # generic data
        ffile1 = self.make_file()
        ro = chz.choose_reduction(ffile1)
        assert type(ro) == Reduction

        # hawc data
        if HAS_DRP:
            ffile2 = self.make_hawc_file()
            ro = chz.choose_reduction(ffile2)
            assert type(ro) == HAWCReduction
            assert ro.override_mode is None

            # special pipeline mode
            cfg = {'mode': 'skycal'}
            ro = chz.choose_reduction(ffile2, cfg)
            assert type(ro) == HAWCReduction
            assert ro.override_mode == 'skycal'

        # forcast data
        if HAS_DRIP:
            # forcast img data
            ffile3 = self.make_forcast_file()
            ro = chz.choose_reduction(ffile3)
            assert type(ro) == FORCASTImagingReduction

            # forcast spec data
            fits.setval(ffile3, 'DETCHAN', value='LW')
            fits.setval(ffile3, 'SPECTEL2', value='FOR_G063')
            ro = chz.choose_reduction(ffile3)
            assert type(ro) == FORCASTSpectroscopyReduction

            # wavecal config data
            cfg = {'wavecal': True}
            ro = chz.choose_reduction(ffile3, cfg)
            assert type(ro) == FORCASTWavecalReduction

            # spatcal config data
            cfg = {'spatcal': True}
            ro = chz.choose_reduction(ffile3, cfg)
            assert type(ro) == FORCASTSpatcalReduction

            # slitcorr config data
            cfg = {'slitcorr': True}
            ro = chz.choose_reduction(ffile3, cfg)
            assert type(ro) == FORCASTSlitcorrReduction

            # exes data for quicklook
            fits.setval(ffile3, 'INSTRUME', value='EXES')
            ro = chz.choose_reduction(ffile3, cfg)
            assert type(ro) == EXESQuicklookReduction

            # mismatched data
            ro = chz.choose_reduction([ffile1, ffile3])
            assert type(ro) == Reduction
            capt = capsys.readouterr()
            assert 'Files do not match' in capt.err

        # fifi data
        if HAS_FIFI:
            ffile4 = self.make_fifi_file()
            ro = chz.choose_reduction(ffile4)
            assert type(ro) == FIFILSReduction

        # flitecam data
        if HAS_FLITECAM:
            ffile5 = self.make_forcast_file()

            # flitecam spec data
            fits.setval(ffile5, 'INSTRUME', value='FLITECAM')
            fits.setval(ffile5, 'INSTCFG', value='SPECTROSCOPY')

            # standard mode
            cfg = {}
            ro = chz.choose_reduction(ffile5, cfg)
            assert type(ro) == FLITECAMSpectroscopyReduction

            # calibration modes
            cfg = {'wavecal': True}
            ro = chz.choose_reduction(ffile5, cfg)
            assert type(ro) == FLITECAMWavecalReduction

            cfg = {'spatcal': True}
            ro = chz.choose_reduction(ffile5, cfg)
            assert type(ro) == FLITECAMSpatcalReduction

            cfg = {'slitcorr': True}
            ro = chz.choose_reduction(ffile5, cfg)
            assert type(ro) == FLITECAMSlitcorrReduction

            # flitecam imaging data
            fits.setval(ffile5, 'INSTCFG', value='IMAGING')
            ro = chz.choose_reduction(ffile5)
            assert type(ro) == FLITECAMImagingReduction

    def test_choose_reduction_errors(self, capsys, tmpdir):
        chz = SOFIAChooser()

        # non-file data, non-fits data, bad fits data
        testfile1 = tmpdir.join('test.txt')
        testfile1.write('test\n')
        testfile2 = tmpdir.join('test.fits')
        testfile2.write('test\n')

        data = ['test data', str(testfile1), str(testfile2)]

        # no reduction, since none are good
        ro = chz.choose_reduction(data)
        assert ro is None
        capt = capsys.readouterr()
        assert 'no reduction' in capt.err
