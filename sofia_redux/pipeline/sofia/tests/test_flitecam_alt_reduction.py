# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FLITECAM alternate reduction classes."""

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.pipeline.reduction import Reduction

try:
    from sofia_redux.pipeline.sofia.flitecam_imaging_reduction \
        import FLITECAMImagingReduction
    from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
        import FLITECAMSpectroscopyReduction
    from sofia_redux.pipeline.sofia.flitecam_wavecal_reduction \
        import FLITECAMWavecalReduction, FLITECAMWavecalParameters
    from sofia_redux.pipeline.sofia.flitecam_spatcal_reduction \
        import FLITECAMSpatcalReduction, FLITECAMSpatcalParameters
    from sofia_redux.pipeline.sofia.flitecam_slitcorr_reduction \
        import FLITECAMSlitcorrReduction, FLITECAMSlitcorrParameters
    HAS_FLITECAM = True
except ImportError:
    HAS_FLITECAM = False
    FLITECAMImagingReduction = None
    FLITECAMSpectroscopyReduction = None
    FLITECAMWavecalReduction, FLITECAMWavecalParameters = None, None
    FLITECAMSpatcalReduction, FLITECAMSpatcalParameters = None, None
    FLITECAMSlitcorrReduction, FLITECAMSlitcorrParameters = None, None


@pytest.fixture
def flitecam_image(tmpdir):
    fname = 'F0001_FC_IMA_12345_FLTPa_CAL_001.fits'
    header = fits.Header({'INSTRUME': 'FLITECAM',
                          'PRODTYPE': 'calibrated',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.zeros((2, 10, 10)),
                                        header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.fixture
def flitecam_grism(tmpdir):
    fname = 'F0001_FC_GRI_12345_FLTC2LMFLTSS20_CMB_001.fits'
    header = fits.Header({'INSTRUME': 'FLITECAM',
                          'PRODTYPE': 'combspec',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.zeros((5, 10)),
                                        header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.mark.skipif('not HAS_FLITECAM')
class TestFLITECAMImgmapReduction(object):

    def test_imgmap(self, flitecam_image, tmpdir):
        red = FLITECAMImagingReduction()
        red.output_directory = tmpdir
        red.load(flitecam_image)
        red.load_parameters()

        # expected image file name
        outfile = tmpdir.join('F0001_FC_IMA_12345_FLTPa_CAL_001.png')
        red.step()
        assert os.path.isfile(outfile)


@pytest.mark.skipif('not HAS_FLITECAM')
class TestFLITECAMSpecmapReduction(object):

    def test_specmap(self, flitecam_grism, tmpdir):
        red = FLITECAMSpectroscopyReduction()
        red.output_directory = tmpdir
        red.load(flitecam_grism)
        red.load_parameters()

        # expected image file name
        outfile = tmpdir.join('F0001_FC_GRI_12345_FLTC2LMFLTSS20_CMB_001.png')
        red.step()
        assert os.path.isfile(outfile)


@pytest.mark.skipif('not HAS_FLITECAM')
class TestFLITECAMCalReduction(object):
    def make_file(self, tmpdir):
        """Retrieve a basic test FITS file for FLITECAM."""
        from sofia_redux.instruments.flitecam.tests.resources \
            import raw_specdata
        hdul = raw_specdata()
        fname = hdul[0].header['FILENAME']
        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)
        return ffile

    def test_startup_wavecal(self):
        red = FLITECAMWavecalReduction()
        assert isinstance(red, Reduction)

    def test_load_basic_wavecal(self, tmpdir):
        red = FLITECAMWavecalReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FLITECAMWavecalParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert 'DATE' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_wavecal_parameter(self):
        # parameter config has to include cal flags
        param = FLITECAMWavecalParameters()
        cfg = param.to_config()
        assert cfg['wavecal'] is True
        assert cfg['spatcal'] is False
        assert cfg['slitcorr'] is False

        # exercise some non-default parameters
        param.add_current_parameters('make_profiles')
        param.add_current_parameters('identify_lines')
        param.add_current_parameters('fit_lines')
        param.config = {'wavefile': 'test1', 'linefile': 'test2'}
        param.make_profiles(0)
        assert param.current[0].get_value('wavefile') == 'test1'
        param.identify_lines(1)
        assert param.current[1].get_value('wavefile') == 'test1'
        assert param.current[1].get_value('linefile') == 'test2'
        param.fit_lines(2)
        assert param.current[2].get_value('spatfile') == 'test1'

    def test_startup_spatcal(self):
        red = FLITECAMSpatcalReduction()
        assert isinstance(red, Reduction)

    def test_load_basic_spatcal(self, tmpdir):
        red = FLITECAMSpatcalReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FLITECAMSpatcalParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert 'DATE' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_spatcal_parameter(self):
        # parameter config has to include cal flags
        param = FLITECAMSpatcalParameters()
        cfg = param.to_config()
        assert cfg['wavecal'] is False
        assert cfg['spatcal'] is True
        assert cfg['slitcorr'] is False

        # exercise some non-default parameters
        param.add_current_parameters('fit_traces')
        param.config = {'wavefile': 'test'}
        param.fit_traces(0)
        assert param.current[0].get_value('wavefile') == 'test'

    def test_startup_slitcorr(self):
        red = FLITECAMSlitcorrReduction()
        assert isinstance(red, Reduction)

    def test_load_basic_slitcorr(self, tmpdir):
        red = FLITECAMSlitcorrReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FLITECAMSlitcorrParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert 'DATE' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_slitcorr_parameter(self):
        # parameter config has to include cal flags
        param = FLITECAMSlitcorrParameters()
        cfg = param.to_config()
        assert cfg['wavecal'] is False
        assert cfg['spatcal'] is False
        assert cfg['slitcorr'] is True

        # exercise some non-default parameters
        param.add_current_parameters('make_profiles')
        param.add_current_parameters('locate_apertures')

        param.config = {'wavefile': 'test'}
        param.make_profiles(0)
        assert param.current[0].get_value('wavefile') == 'test'

        param.locate_apertures(1)
        assert (param.current[1].get_value('num_aps')
                == param.default['locate_apertures']['num_aps']['value'])
