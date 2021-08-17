# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FLITECAM Spectroscopy Reduction class."""

import os
import pickle

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.reduction import Reduction
import sofia_redux.pipeline.sofia.parameters as srp
try:
    from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
        import FLITECAMSpectroscopyReduction, FLITECAMSpectroscopyParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FLITECAMSpectroscopyReduction = None
    FLITECAMSpectroscopyParameters = None


@pytest.mark.skipif('not HAS_DRIP')
class TestFLITECAMSpectroscopyReduction(object):

    @pytest.fixture(autouse=True, scope='function')
    def mock_param(self, qapp):
        # set defaults to faster options to speed up tests
        default = srp.flitecam_spectroscopy_parameters.SPECTRAL_DEFAULT
        default['trace_continuum'][2]['option_index'] = 0

    def make_file(self, tmpdir, nfiles=1, nplane=1):
        """Retrieve a basic test FITS file for FLITECAM."""
        from sofia_redux.instruments.flitecam.tests.resources \
            import raw_specdata

        ffiles = []
        for i in range(nfiles):
            dth_i = i + 1
            if i % 2 == 0:
                nod = 'A'
            else:
                nod = 'B'
            hdul = raw_specdata(dthindex=dth_i, nodbeam=nod)

            # make it look like old-style data if desired
            if nplane > 1:
                d1 = hdul[0].data.copy()
                hdul[0].data = np.array([d1] * nplane)

            fname = hdul[0].header['FILENAME']
            tmpfile = tmpdir.join(fname)
            ffile = str(tmpfile)
            hdul.writeto(ffile, overwrite=True)
            hdul.close()
            ffiles.append(ffile)

        return ffiles

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FLITECAMSpectroscopyReduction()
        ffile = self.make_file(tmpdir, nfiles)
        red.output_directory = tmpdir

        red.load(ffile)
        red.load_parameters()
        idx = red.recipe.index(step)

        # process up to current step
        for i in range(idx):
            red.step()

        if nfiles == 1:
            ffile = ffile[0]
        return ffile, red, idx

    def test_startup(self):
        red = FLITECAMSpectroscopyReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = FLITECAMSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FLITECAMSpectroscopyParameters)

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

    def test_load_intermediate(self, tmpdir, capsys):
        red = FLITECAMSpectroscopyReduction()
        ffile = self.make_file(tmpdir, nplane=3)[0]

        # Set prodtype -- now looks like an old-style
        # intermediate file with variance and expmap
        fits.setval(ffile, 'PRODTYPE', value='stacked')
        red.load(ffile)
        # check the loaded file now has 3 extensions
        assert len(red.input[0]) == 3
        assert 'PRIMARY' in red.input[0]
        assert 'ERROR' in red.input[0]
        assert 'EXPOSURE' in red.input[0]

        # keep only first two planes -- now looks like
        # flux and variance only
        hdul = fits.open(ffile, mode='update')
        hdul[0].data = hdul[0].data[0:2, :, :]
        hdul.flush()
        hdul.close()
        red.load(ffile)
        # check the loaded file now has 2 extensions
        assert len(red.input[0]) == 2
        assert 'PRIMARY' in red.input[0]
        assert 'ERROR' in red.input[0]

        # write reorganized file to disk, load it
        # again and verify it's unchanged
        # (new-style intermediate)
        red.input[0].writeto(ffile, overwrite=True)
        red.load(ffile)
        assert len(red.input[0]) == 2
        assert 'PRIMARY' in red.input[0]
        assert 'ERROR' in red.input[0]

        # test error for no further steps
        fits.setval(ffile, 'PRODTYPE', value='specmap')
        with pytest.raises(ValueError):
            red.load(ffile)
        assert 'No steps to run' in capsys.readouterr().err

    def test_register_viewers(self, mocker):
        from sofia_redux.visualization.redux_viewer import EyeViewer
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        mocker.patch.object(MatplotlibViewer, '__init__', return_value=None)
        mocker.patch.object(EyeViewer, '__init__', return_value=None)

        red = FLITECAMSpectroscopyReduction()
        vz = red.register_viewers()
        # 3 viewers -- QAD, profile, spectra
        assert len(vz) == 3
        assert isinstance(vz[0], QADViewer)
        assert isinstance(vz[1], MatplotlibViewer)
        assert isinstance(vz[2], EyeViewer)

    def test_display_data(self, tmpdir, capsys):
        red = FLITECAMSpectroscopyReduction()
        ffile = self.make_file(tmpdir)[0]

        # test for raw data
        red.load(ffile)
        red.set_display_data(raw=True)
        assert red.display_data == {'QADViewer': [ffile]}

        # test for intermediate
        red.load_fits()
        red.set_display_data(raw=False)
        assert len(red.display_data['QADViewer']) == 1
        assert isinstance(red.display_data['QADViewer'][0], fits.HDUList)

        # test for filenames instead of self.input
        red.set_display_data(raw=False, filenames=[ffile])
        assert len(red.display_data['QADViewer']) == 1
        assert red.display_data['QADViewer'] == [ffile]

        # test for bad filename -- warns, but carries on
        red.set_display_data(raw=False, filenames=['badfile.fits'])
        assert len(red.display_data['QADViewer']) == 1
        assert red.display_data['QADViewer'] == ['badfile.fits']
        assert 'not a file' in capsys.readouterr().err

    @pytest.mark.parametrize('obstype,skymode,srctype',
                             [('STANDARD_TELLURIC', 'NOD_ALONG_SLIT',
                               'POINT_SOURCE'),
                              ('STANDARD_TELLURIC', 'NOD_OFF_SLIT',
                               'POINT_SOURCE'),
                              ('OBJECT', 'NOD_OFF_SLIT', 'POINT_SOURCE'),
                              ('OBJECT', 'NOD_ALONG_SLIT', 'POINT_SOURCE'),
                              ('OBJECT', 'NOD_OFF_SLIT', 'EXTENDED_SOURCE')])
    def test_all_steps(self, tmpdir, obstype, skymode, srctype):
        # exercises nominal behavior for a typical reduction --
        # telluric standard and not
        red = FLITECAMSpectroscopyReduction()

        if skymode == 'NOD_ALONG_SLIT':
            ffile = self.make_file(tmpdir, nfiles=5)
            ffile = [ffile[0], ffile[4]]
        else:
            ffile = self.make_file(tmpdir, nfiles=4)

        for f in ffile:
            fits.setval(f, 'OBSTYPE', value=obstype)
            fits.setval(f, 'INSTMODE', value=skymode)
            fits.setval(f, 'SRCTYPE', value=srctype)
        red.load(ffile)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step in ['check_header', 'stack_dithers', 'specmap']:
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for flitecam
        red = FLITECAMSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'config')
        assert pcopy.config is not None

        assert hasattr(pcopy, 'pipecal_config')
        assert pcopy.pipecal_config is None

    @pytest.mark.parametrize('step',
                             ['make_image', 'flux_calibrate'])
    def test_save(self, tmpdir, step):
        ffile, red, idx = self.standard_setup(tmpdir, step)
        red_copy = pickle.dumps(red)

        # test no save
        parset = red.parameters.current[idx]
        parset['save']['value'] = False
        red.step()
        saw_file = False
        for fn in red.out_files:
            if red.prodnames[red.prodtype_map[step]] in fn:
                saw_file = True
                break
        assert not saw_file
        red = pickle.loads(red_copy)

        # test save
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        red.step()

        fn = ''
        for fn in red.out_files:
            if red.prodnames[red.prodtype_map[step]] in fn:
                break
        assert os.path.isfile(fn)

    def test_fluxcal(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'flux_calibrate')

        # turn all auto options off
        parset = red.parameters.current[idx]
        parset['auto_shift']['value'] = False
        parset['waveshift']['value'] = 0.0
        red_copy = pickle.dumps(red)
        input_data = red.input[0][0].data.copy()

        # default result: calibrated
        parset['save']['value'] = True
        parset['save_1d']['value'] = True
        red.flux_calibrate()
        default_data = red.input[0][0].data.copy()
        assert not np.allclose(default_data, input_data, equal_nan=True)
        assert red.input[0][0].header['PROCSTAT'] == 'LEVEL_3'
        assert np.allclose(float(red.input[0][0].header['WAVSHIFT']), 0)
        fname = red.input[0][0].header['FILENAME']
        # 2d and 1d products are saved
        assert os.path.exists(tmpdir.join(fname))
        assert os.path.exists(tmpdir.join(fname.replace('CRM', 'CAL')))

        # barycentric wavelength shifts are added
        assert np.allclose(red.input[0][0].header['BARYSHFT'], -8.7038e-05)
        assert np.allclose(red.input[0][0].header['LSRSHFT'], -2.488e-06)
        assert red.input[0][0].header['SPECSYS'] == 'TOPOCENT'
        assert red.input[0][0].header['SPECSYSA'] == 'TOPOCENT'

        # test skip cal: uncalibrated
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['skip_cal']['value'] = True
        red.flux_calibrate()
        assert 'No flux calibration' in capsys.readouterr().out
        assert np.allclose(red.input[0][0].data, input_data, equal_nan=True)
        assert red.input[0][0].header['PROCSTAT'] == 'LEVEL_2'

        # test manual waveshift: turns auto shift off, is used directly
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['waveshift']['value'] = 2.0
        parset['auto_shift']['value'] = True
        red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'Disabling auto-shift' in capt.out
        assert 'Wavelength shift applied to spectrum 1: 2.00' in capt.out
        # data is shifted, value is recorded
        assert not np.allclose(red.input[0][0].data, default_data,
                               equal_nan=True)
        assert np.allclose(float(red.input[0][0].header['WAVSHIFT']), 2)

        # now make snthresh high -- won't auto-shift
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['sn_threshold']['value'] = 5000
        parset['auto_shift']['value'] = True
        red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'too low to auto-shift' in capt.err
        assert np.allclose(float(red.input[0][0].header['WAVSHIFT']), 0)

        # test missing response: raises error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['respfile']['value'] = 'badfile.fits'
        parset['making_response']['value'] = False
        with pytest.raises(ValueError):
            red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'Bad response file' in capt.err

        # unless making a response: then proceeds, correcting ATRAN only
        parset['making_response']['value'] = True
        red.flux_calibrate()
        assert not np.allclose(default_data, input_data, equal_nan=True)
        assert red.input[0][0].header['PROCSTAT'] == 'LEVEL_2'

        # test bad resolution
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['resolution']['value'] = 'bad'
        with pytest.raises(ValueError):
            red.flux_calibrate()
        assert 'Missing spectral resolution' in capsys.readouterr().err

        # mock problem in fluxcal
        mocker.patch('sofia_redux.spectroscopy.fluxcal.fluxcal',
                     return_value=None)
        red = pickle.loads(red_copy)
        with pytest.raises(ValueError):
            red.flux_calibrate()
        assert 'Problem in flux calibration' in capsys.readouterr().err

    def test_atran_dir(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'flux_calibrate')
        red_copy = pickle.dumps(red)

        # mock smoothres for working with synthetic data
        def mock_smooth(x, y, r, siglim=5):
            return y

        mocker.patch('sofia_redux.instruments.forcast.getatran.smoothres',
                     mock_smooth)

        # specify an empty atran directory: should get the default file
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        red.flux_calibrate()
        afile = 'atran_40K_45deg_1-6mum.fits'
        assert red.input[0][0].header['ATRNFILE'] == afile

        # same for a non-existent atran directory
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = 'bad_directory'
        red.flux_calibrate()
        afile = 'atran_40K_45deg_1-6mum.fits'
        assert red.input[0][0].header['ATRNFILE'] == afile

        # mock the default directory to be empty too: should raise an error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        mocker.patch('sofia_redux.instruments.forcast.getatran.get_atran',
                     return_value=None)
        with pytest.raises(ValueError) as err:
            red.flux_calibrate()
        assert 'No matching ATRAN files' in str(err)

        # same if no initial directory is specified
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = ''
        mocker.patch('sofia_redux.instruments.forcast.getatran.get_atran',
                     return_value=None)
        with pytest.raises(ValueError) as err:
            red.flux_calibrate()
        assert 'No matching ATRAN files' in str(err)

    def test_make_image_flat(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'make_image')
        red_copy = pickle.dumps(red)

        # by default, flat is set: should succeed
        red.make_image()
        assert 'Dividing by flat' in capsys.readouterr().out

        # unset flat: should succeed
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['flatfile']['value'] = ''
        red.make_image()
        assert 'Dividing by flat' not in capsys.readouterr().out

        # set to bad file: should raise error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['flatfile']['value'] = 'badfile.fits'
        with pytest.raises(ValueError):
            red.make_image()
        assert 'Cannot find flat file' in capsys.readouterr().err
