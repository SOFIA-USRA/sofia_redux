# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Spectroscopy Reduction class."""

import os
import pickle
import shutil

from astropy.io import fits
from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.reduction import Reduction
import sofia_redux.pipeline.sofia.parameters as srp
from sofia_redux.toolkit.utilities.fits import set_log_level
try:
    from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
        import FORCASTSpectroscopyReduction
    from sofia_redux.pipeline.sofia.parameters.forcast_spectroscopy_parameters\
        import FORCASTSpectroscopyParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FORCASTSpectroscopyReduction = None
    FORCASTSpectroscopyParameters = None


@pytest.mark.skipif('not HAS_DRIP')
class TestFORCASTSpectroscopyReduction(object):

    @pytest.fixture(autouse=True, scope='function')
    def mock_param(self, qapp):
        # set defaults to faster options to speed up tests
        default = srp.forcast_spectroscopy_parameters.SPECTRAL_DEFAULT
        default['trace_continuum'][2]['option_index'] = 0
        default['flux_calibrate'][8]['value'] = False

    def make_file(self, tmpdir, fname='bFT001_0001.fits'):
        """Retrieve a basic test FITS file for FORCAST."""
        from sofia_redux.instruments.forcast.tests.resources \
            import raw_specdata
        hdul = raw_specdata()
        # set the date to one with known good response data
        hdul[0].header['DATE-OBS'] = '2018-09-01T00:00:00.000'

        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        return ffile

    def make_combspec(self, tmpdir, n_ap=1):
        fname = 'F0001_FO_GRI_12345_FORF123_CMB_001.fits'
        header = fits.Header({'INSTRUME': 'FORCAST',
                              'PRODTYPE': 'combspec',
                              'FILENAME': fname})
        if n_ap == 1:
            data = np.arange(50, dtype=float).reshape((5, 10))
        else:
            data = np.arange(n_ap * 50, dtype=float).reshape((n_ap, 5, 10))
        hdul = fits.HDUList(fits.PrimaryHDU(data=data,
                                            header=header))
        ffile = os.path.join(tmpdir, fname)
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        return ffile

    def make_cube(self, tmpdir):
        fname = 'F0001_FO_GRI_12345_FORF123_SCB_001.fits'
        header = fits.Header({'INSTRUME': 'FORCAST',
                              'PRODTYPE': 'spectral_cube',
                              'FILENAME': fname,
                              'EXTNAME': 'FLUX'})
        hdul = fits.HDUList([fits.PrimaryHDU(data=np.zeros((10, 10, 10)),
                                             header=header),
                             fits.ImageHDU(data=np.zeros((10, 10, 10)),
                                           name='ERROR'),
                             fits.ImageHDU(data=np.arange(10),
                                           name='WAVEPOS'),
                             fits.ImageHDU(data=np.arange(10),
                                           name='TRANSMISSION')])
        ffile = os.path.join(tmpdir, fname)
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        return ffile

    def make_crm(self, tmpdir, return_hdul=False, n_ap=None):
        fname = 'F0001_FO_GRI_12345_FORF123_CRM_001.fits'
        header = fits.Header({'INSTRUME': 'FORCAST',
                              'PRODTYPE': 'calibrated_spectrum',
                              'FILENAME': fname,
                              'EXTNAME': 'FLUX',
                              'CDELT1': 1, 'CDELT2': 1,
                              'CRPIX1': 1, 'CRPIX2': 1,
                              'CDELT1A': 1, 'CDELT2A': 1, 'CDELT3A': 1,
                              'CRPIX1A': 1, 'CRPIX2A': 1, 'CRPIX3A': 1})

        darr = np.zeros((10, 10))
        warr = np.arange(10)
        if n_ap is None:
            sarr = np.zeros(10)
        else:
            sarr = np.zeros((n_ap, 10))
        hdul = fits.HDUList([fits.PrimaryHDU(data=darr, header=header),
                             fits.ImageHDU(data=darr.copy(), name='ERROR'),
                             fits.ImageHDU(data=warr, name='WAVEPOS'),
                             fits.ImageHDU(data=sarr, name='SPECTRAL_FLUX'),
                             fits.ImageHDU(data=sarr.copy(),
                                           name='SPECTRAL_ERROR'),
                             fits.ImageHDU(data=sarr.copy(),
                                           name='TRANSMISSION'),
                             fits.ImageHDU(data=sarr.copy(),
                                           name='RESPONSE')])
        if return_hdul:
            return hdul

        ffile = os.path.join(tmpdir, fname)
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        return ffile

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FORCASTSpectroscopyReduction()
        ffile = []
        for i in range(nfiles):
            fn = 'bFT001_000{}.fits'.format(i + 1)
            ffile.append(self.make_file(tmpdir, fname=fn))
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
        red = FORCASTSpectroscopyReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FORCASTSpectroscopyParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_load_intermediate(self, tmpdir, capsys):
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

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

        # test error for unknown prodtype
        fits.setval(ffile, 'PRODTYPE', value='test_value')
        with pytest.raises(ValueError):
            red.load(ffile)

    def test_register_viewers(self, mocker):
        from sofia_redux.visualization.redux_viewer import EyeViewer
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        mocker.patch.object(MatplotlibViewer, '__init__', return_value=None)
        mocker.patch.object(EyeViewer, '__init__', return_value=None)

        red = FORCASTSpectroscopyReduction()
        vz = red.register_viewers()
        # 3 viewers -- QAD, profile, spectra
        assert len(vz) == 3
        assert isinstance(vz[0], QADViewer)
        assert isinstance(vz[1], MatplotlibViewer)
        assert isinstance(vz[2], EyeViewer)

    def test_display_data(self, tmpdir, capsys):
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

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
                             [('STANDARD_TELLURIC', 'NMC', 'POINT_SOURCE'),
                              ('OBJECT', 'NMC', 'POINT_SOURCE'),
                              ('OBJECT', 'NXCAC', 'POINT_SOURCE'),
                              ('OBJECT', 'NXCAC', 'EXTENDED_SOURCE'),
                              ('OBJECT', 'SLITSCAN', 'EXTENDED_SOURCE')])
    def test_all_steps(self, tmpdir, obstype, skymode, srctype):
        # exercises nominal behavior for a typical reduction --
        # standard and not
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)
        fits.setval(ffile, 'OBSTYPE', value=obstype)
        fits.setval(ffile, 'SKYMODE', value=skymode)
        fits.setval(ffile, 'SRCTYPE', value=srctype)
        red.load([ffile, ffile])
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step in ['checkhead', 'stack_dithers', 'specmap']:
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for forcast
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'drip_cal_config')
        assert pcopy.drip_cal_config is not None

        assert hasattr(pcopy, 'drip_config')
        assert pcopy.drip_config is not None

        assert not hasattr(pcopy, 'pipecal_config')

    @pytest.mark.parametrize('step',
                             ['make_profiles', 'locate_apertures',
                              'trace_continuum', 'set_apertures',
                              'subtract_background', 'extract_spectra',
                              'merge_apertures', 'flux_calibrate'])
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

    def test_stack_dithers(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'stack_dithers',
                                              nfiles=2)
        parset = red.parameters.current[idx]
        parset['skip_stack']['value'] = False
        red_copy = pickle.dumps(red)

        # stack off
        parset = red.parameters.current[idx]
        parset['skip_stack']['value'] = True
        red.stack_dithers()
        assert 'No stacking' in capsys.readouterr().out
        assert len(red.input) == 2

        # stack on, dithers match
        red = pickle.loads(red_copy)
        red.stack_dithers()
        assert 'Stacking 2 dithers' in capsys.readouterr().out
        assert len(red.input) == 1

        # stack on, dithers don't match
        red = pickle.loads(red_copy)
        red.input[1][0].header['DTHINDEX'] = 1
        red.stack_dithers()
        assert 'No repeated dithers' in capsys.readouterr().out
        assert len(red.input) == 2

        # stack on, mismatch, but ignore dither information
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['ignore_dither']['value'] = True
        red.input[1][0].header['DTHINDEX'] = 1
        red.stack_dithers()
        assert 'Stacking 2 dithers' in capsys.readouterr().out
        assert len(red.input) == 1

    def test_make_profiles(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'make_profiles')
        red_copy = pickle.dumps(red)

        # test missing calibrations

        # missing slit correction function -- warns only
        parset = red.parameters.current[idx]
        parset['slitfile']['value'] = 'badfile.fits'
        red.make_profiles()
        assert 'Missing slit correction' in capsys.readouterr().err
        assert red.input[0][0].header['SLITFILE'] == 'NONE'
        wcal = red.input[0]['WAVEPOS'].data

        # mismatched slit correction file -- raises error
        red = pickle.loads(red_copy)
        slitfile = str(tmpdir.join('wrong_shape.fits'))
        hdul = fits.HDUList([fits.PrimaryHDU(np.zeros((10, 10)))])
        hdul.writeto(slitfile, overwrite=True)

        parset = red.parameters.current[idx]
        parset['slitfile']['value'] = slitfile
        with pytest.raises(ValueError):
            red.make_profiles()
        assert 'shape (10, 10) does not match' in capsys.readouterr().err

        # missing wavecal file -- raises error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['wavefile']['value'] = 'badfile.fits'
        with pytest.raises(ValueError):
            red.make_profiles()
        assert 'Missing wavecal' in capsys.readouterr().err

        # missing order mask file -- raises error
        red = pickle.loads(red_copy)
        red.calres['maskfile'] = 'badfile.fits'
        with pytest.raises(ValueError):
            red.make_profiles()
        assert 'Missing order mask' in capsys.readouterr().err

        # set atmosthresh to get an atran file
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atmosthresh']['value'] = 0.5
        with set_log_level('DEBUG'):
            red.make_profiles()
        assert 'Using ATRAN' in capsys.readouterr().out

        # set a waveshift: should be directly applied
        red = pickle.loads(red_copy)
        red.calres['waveshift'] = 0.5
        red.make_profiles()
        assert 'Applying default waveshift of 0.5' in capsys.readouterr().out
        assert np.allclose(red.input[0]['WAVEPOS'].data, wcal + 0.5)

        # problem in rectification -- raises error
        mocker.patch('sofia_redux.spectroscopy.rectify.rectify',
                     return_value={1: None})
        red = pickle.loads(red_copy)
        with pytest.raises(ValueError):
            red.make_profiles()

    def test_parse_apertures(self, capsys):
        # test helper function for parsing apertures from
        # parameters or headers
        red = FORCASTSpectroscopyReduction()

        # nominal input
        input_position = '1,2,3;4,5,6'
        expected = [[1., 2., 3.], [4., 5., 6.]]
        assert np.allclose(red._parse_apertures(input_position, 2), expected)

        # one file: error
        with pytest.raises(ValueError):
            red._parse_apertures(input_position, 1)
        assert 'Could not read input_position' in capsys.readouterr().err

        # two files, one input: applied to all
        input_position = '1,2,3'
        expected = [[1., 2., 3.], [1., 2., 3.]]
        assert np.allclose(red._parse_apertures(input_position, 2), expected)

        # bad value in aperture: error
        input_position = '1,2,3;4,5a,6'
        with pytest.raises(ValueError):
            red._parse_apertures(input_position, 2)
        assert 'Could not read input_position' in capsys.readouterr().err

    def test_parse_bg(self, capsys):
        # test helper function for parsing background regions
        red = FORCASTSpectroscopyReduction()

        # nominal input
        input_position = '1-2,3-4;5-6'
        expected = [[[1., 2.], [3., 4.]], [[5., 6.]]]
        result = red._parse_bg(input_position, 2)
        assert len(result) == 2
        for r, e in zip(result, expected):
            assert np.allclose(r, e)

        # one file: error
        with pytest.raises(ValueError):
            red._parse_bg(input_position, 1)
        assert 'Could not read background region' in capsys.readouterr().err

        # two files, one input: applied to all
        input_position = '1-2,3-4'
        expected = [[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]
        result = red._parse_bg(input_position, 2)
        assert len(result) == 2
        for r, e in zip(result, expected):
            assert np.allclose(r, e)

        # bad value in region: error
        input_position = '1-2,3-4;5-6a'
        with pytest.raises(ValueError):
            red._parse_bg(input_position, 2)
        assert 'Could not read background region' in capsys.readouterr().err

    def test_locate_apertures(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'locate_apertures')
        red_copy = pickle.dumps(red)

        # test input options

        # fix to center
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to center'
        red.locate_apertures()
        assert 'Fixing aperture to slit center' in capsys.readouterr().out
        assert np.allclose(float(red.input[0][0].header['APPOSO01']), 99.44)

        # fix to input -- rounds to 3 decimal places
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fix to input'
        parset['input_position']['value'] = '123.45678'
        red.locate_apertures()
        assert 'Fixing aperture to input' in capsys.readouterr().out
        assert np.allclose(float(red.input[0][0].header['APPOSO01']), 123.457)

        # fit without input: should find trace near center
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['input_position']['value'] = ''
        red.locate_apertures()
        assert 'Finding aperture' in capsys.readouterr().out
        assert np.allclose(float(red.input[0][0].header['APPOSO01']),
                           98, atol=1)

        # fit without input, 3 positions: should find 3 traces
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['input_position']['value'] = ''
        parset['num_aps']['value'] = 3
        red.locate_apertures()
        assert len(red.input[0][0].header['APPOSO01'].split(',')) == 3

        # fit with close input: should find trace near center
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['input_position']['value'] = '100'
        red.locate_apertures()
        assert 'Finding aperture' in capsys.readouterr().out
        assert np.allclose(float(red.input[0][0].header['APPOSO01']),
                           98, atol=1)

        # fit with three inputs: should find 3 traces
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'auto'
        parset['input_position']['value'] = '40,100,150'
        red.locate_apertures()
        assert len(red.input[0][0].header['APPOSO01'].split(',')) == 3

    def test_trace_continuum(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'trace_continuum')
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'fit'

        # run with defaults: should work
        red.trace_continuum()
        assert 'Trace fit failed' not in capsys.readouterr().err

        # mock failure in trace
        def bad_trace(*args, info=None, **kwargs):
            if info is None:
                info = {}
            info[1] = {'trace_model': [None],
                       'mask': []}
        mocker.patch('sofia_redux.spectroscopy.tracespec.tracespec', bad_trace)
        with pytest.raises(ValueError):
            red.trace_continuum()
        assert 'Trace fit failed' in capsys.readouterr().err

        # mock a different failure in trace
        def bad_trace_2(*args, info=None, **kwargs):
            info = {}
            assert len(info) == 0
        mocker.patch('sofia_redux.spectroscopy.tracespec.tracespec',
                     bad_trace_2)
        with pytest.raises(ValueError):
            red.trace_continuum()
        assert 'Trace fit failed' in capsys.readouterr().err

    def test_set_apertures(self, tmpdir):
        ffile, red, idx = self.standard_setup(tmpdir, 'set_apertures')
        parset = red.parameters.current[idx]
        parset['full_slit']['value'] = False

        # test fixed input values
        parset['apsign']['value'] = '-1'
        parset['aprad']['value'] = '10'
        parset['psfrad']['value'] = '20'
        parset['bgr']['value'] = '0-10'
        red_copy = pickle.dumps(red)

        red.set_apertures()
        assert red.input[0][0].header['APSGNO01'] == '-1'
        assert red.input[0][0].header['APRADO01'] == '10.000'
        assert red.input[0][0].header['PSFRAD01'] == '20.000'
        assert red.input[0][0].header['BGR'] == '0.000-10.000'

        # multiple apertures, one input value: gets applied to all
        red = pickle.loads(red_copy)

        # fake two apertures in input hdul
        red.input[0][0].header['APPOSO01'] = '40,150'
        red.input[0][0].header['APFWHM01'] = '4.0,4.0'
        trace = red.input[0]['APERTURE_TRACE'].data
        red.input[0]['APERTURE_TRACE'].data = np.vstack([trace * 0 + 40,
                                                         trace * 0 + 150])

        red.set_apertures()
        assert red.input[0][0].header['APSGNO01'] == '-1,-1'
        assert red.input[0][0].header['APRADO01'] == '10.000,10.000'
        assert red.input[0][0].header['PSFRAD01'] == '20.000,20.000'
        assert red.input[0][0].header['BGR'] == '0.000-10.000'

    def test_subtract_bg(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'subtract_background')
        red_copy = pickle.dumps(red)

        # test skip bg
        parset = red.parameters.current[idx]
        parset['skip_bg']['value'] = True

        inp = red.input[0][0].data.copy()
        red.subtract_background()
        assert 'No background subtraction' in capsys.readouterr().out
        assert np.allclose(red.input[0][0].data, inp, equal_nan=True)

        # add a background and run with bg -- median value should be lower
        inp += 200.
        red = pickle.loads(red_copy)
        red.input[0][0].data = inp.copy()
        parset = red.parameters.current[idx]
        parset['skip_bg']['value'] = False
        red.subtract_background()
        assert np.nanmedian(red.input[0][0].data) < np.nanmedian(inp)

    def test_extract_spectra(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'extract_spectra')
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'optimal'
        red_copy = pickle.dumps(red)

        # test spatial map/profile settings

        # spatial map
        with set_log_level('DEBUG'):
            red.extract_spectra()
        assert 'Using spatial map' in capsys.readouterr().out
        default_img = red.input[0][0].data.copy()
        default = red.input[0]['SPECTRAL_FLUX'].data.copy()

        # median profile
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['use_profile']['value'] = True
        with set_log_level('DEBUG'):
            red.extract_spectra()
        assert 'Using median profile' in capsys.readouterr().out
        assert not np.allclose(red.input[0]['SPECTRAL_FLUX'].data, default)

        # no profile or map
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'standard'
        parset['fix_bad']['value'] = False
        with set_log_level('DEBUG'):
            red.extract_spectra()
        assert 'No profile or spatial map' in capsys.readouterr().out
        assert np.sum(np.isnan(red.input[0][0].data)) \
            > np.sum(np.isnan(default_img))

    @pytest.mark.parametrize('skymode', ['NMC', 'NXCAC'])
    def test_merge_apertures(self, tmpdir, skymode):
        # set up a reduction with 3 apertures
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)

        # set skymode to test nexp handling
        fits.setval(ffile, 'SKYMODE', value=skymode)

        red.output_directory = tmpdir
        red.load(ffile)
        red.load_parameters()

        # set ap number param
        idx = red.recipe.index('locate_apertures')
        parset = red.parameters.current[idx]
        parset['num_aps']['value'] = 3

        # set save for extract step
        idx = red.recipe.index('extract_spectra')
        parset = red.parameters.current[idx]
        parset['save_1d']['value'] = True

        # process up to merge ap
        idx = red.recipe.index('merge_apertures')
        for i in range(idx):
            red.step()

        # set param to save 2d and 1d spectrum
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        parset['save_1d']['value'] = True

        # 3 spectra before
        shape = red.input[0]['SPECTRAL_FLUX'].data.shape
        assert len(shape) == 2
        assert shape[0] == 3

        # run
        red.merge_apertures()

        # one spectrum after
        shape = red.input[0]['SPECTRAL_FLUX'].data.shape
        assert len(shape) == 1

        # 2d and 1d files exist
        fname = red.input[0][0].header['FILENAME']
        assert os.path.exists(str(tmpdir.join(fname)))
        assert os.path.exists(str(tmpdir.join(fname.replace('MGM', 'MRG'))))
        # also the 1d file from the extract step
        assert os.path.exists(str(tmpdir.join(fname.replace('MGM', 'SPC'))))

        # check exposure keyword
        if skymode == 'NMC':
            assert red.input[0][0].header['NEXP'] == 4
        else:
            assert red.input[0][0].header['NEXP'] == 3

    def test_atran_opt(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'flux_calibrate')
        parset = red.parameters.current[idx]
        parset['optimize_atran']['value'] = True
        red_copy = pickle.dumps(red)

        # mock smoothres for working with synthetic data
        def mock_smooth(x, y, r, siglim=5):
            return y
        mocker.patch('sofia_redux.instruments.forcast.getatran.smoothres',
                     mock_smooth)

        # define a quick test for non-optimization
        def test_noopt(redx):
            fnamex = str(tmpdir.join(redx.input[0][0].header['FILENAME']))
            assert not os.path.exists(
                fnamex.replace('CAL', 'PWV').replace('.fits', '.png'))
            assert not os.path.exists(
                fnamex.replace('CAL', 'OPT').replace('.fits', '.png'))

        # try without directory - won't optimize
        parset['atrandir']['value'] = ''
        red.flux_calibrate()
        assert 'Cannot optimize without ATRAN' in capsys.readouterr().err
        test_noopt(red)
        assert 'pwv' not in red.input[0][0].header['ATRNFILE']

        # same for bad directory
        parset['atrandir']['value'] = 'bad_directory'
        red.flux_calibrate()
        assert 'Cannot optimize without ATRAN' in capsys.readouterr().err
        test_noopt(red)
        assert 'pwv' not in red.input[0][0].header['ATRNFILE']

        # try with empty directory - raises error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        with pytest.raises(ValueError):
            red.flux_calibrate()
        assert 'No matching ATRAN files' in capsys.readouterr().err

        # set up atran directory for optimization
        data = np.vstack([np.arange(50, dtype=float),
                          np.random.random(50)])
        hdul = fits.HDUList(fits.PrimaryHDU(data))
        hdul.writeto(str(tmpdir.join('atran_40K_45deg_2pwv_4-50mum.fits')),
                     overwrite=True)
        data[1] *= 0.75
        hdul.writeto(str(tmpdir.join('atran_40K_45deg_4pwv_4-50mum.fits')),
                     overwrite=True)
        data[1] *= 0.5
        hdul.writeto(str(tmpdir.join('atran_40K_45deg_6pwv_4-50mum.fits')),
                     overwrite=True)

        # try with a specific atran file identified -- won't optimize
        afile = 'atran_40K_45deg_4pwv_4-50mum.fits'
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        parset['atranfile']['value'] = str(tmpdir.join(afile))
        red.flux_calibrate()
        test_noopt(red)
        assert red.input[0][0].header['ATRNFILE'] == afile

        # now make snthresh high -- won't optimize
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        parset['sn_threshold']['value'] = 5000
        parset['auto_shift']['value'] = True
        red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'too low to optimize' in capt.err
        assert 'too low to auto-shift' in capt.err
        test_noopt(red)
        assert 'pwv' not in red.input[0][0].header['ATRNFILE']
        assert red.input[0][0].header['WAVSHIFT'] == 0

        # now just add directory with appropriate files - should optimize and
        # use the least noisy pwv file
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        red.flux_calibrate()
        fname = str(tmpdir.join(red.input[0][0].header['FILENAME']))
        assert os.path.exists(
            fname.replace('CRM', 'PWV').replace('.fits', '.png'))
        assert os.path.exists(
            fname.replace('CRM', 'OPT').replace('.fits', '.png'))
        afile = 'atran_40K_45deg_2pwv_4-50mum.fits'
        assert red.input[0][0].header['ATRNFILE'] == afile

        # now write some files without pwv in a new directory
        tmpdir2 = tmpdir.join('no_pwv')
        os.makedirs(str(tmpdir2))
        hdul.writeto(str(tmpdir2.join('atran_40K_45deg_4-50mum.fits')),
                     overwrite=True)
        hdul.writeto(str(tmpdir2.join('atran_41K_45deg_4-50mum.fits')),
                     overwrite=True)

        # should raise error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir2)
        with pytest.raises(ValueError) as err:
            red.flux_calibrate()
        assert 'No matching ATRAN files' in str(err)

    def test_atran_dir(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'flux_calibrate')
        parset = red.parameters.current[idx]
        parset['optimize_atran']['value'] = False
        red_copy = pickle.dumps(red)

        # mock smoothres for working with synthetic data
        def mock_smooth(x, y, r, siglim=5):
            return y

        mocker.patch('sofia_redux.instruments.forcast.getatran.smoothres',
                     mock_smooth)

        # without optimize set, specify an empty atran directory:
        # should get the default file
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['atrandir']['value'] = str(tmpdir)
        red.flux_calibrate()
        afile = 'atran_40K_45deg_4-50mum.fits'
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

    def test_fluxcal(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'flux_calibrate')

        # turn all auto options off
        parset = red.parameters.current[idx]
        parset['optimize_atran']['value'] = False
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
        assert np.allclose(red.input[0][0].header['WAVSHIFT'], 0)
        fname = red.input[0][0].header['FILENAME']
        # 2d and 1d products are saved
        assert os.path.exists(tmpdir.join(fname))
        assert os.path.exists(tmpdir.join(fname.replace('CRM', 'CAL')))

        # wavelength shifts are added
        assert np.allclose(red.input[0][0].header['BARYSHFT'], 8.3939e-05)
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
        assert 'Wavelength shift applied: 2.00' in capt.out
        # data is shifted, value is recorded
        assert not np.allclose(red.input[0][0].data, default_data,
                               equal_nan=True)
        assert np.allclose(red.input[0][0].header['WAVSHIFT'], 2)

        # test missing response: raises error
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['respfile']['value'] = 'badfile.fits'
        parset['making_response']['value'] = False
        with pytest.raises(ValueError):
            red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'Bad response file' in capt.err
        parset['respfile']['value'] = ''
        with pytest.raises(ValueError):
            red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'Missing response file' in capt.err

        # unless making a response: then proceeds, correcting ATRAN only,
        # but turns off optimization if necessary
        parset['making_response']['value'] = True
        parset['optimize_atran']['value'] = True
        parset['atrandir']['value'] = str(tmpdir)
        red.flux_calibrate()
        capt = capsys.readouterr()
        assert 'No response file. Turning off ATRAN optimization.' in capt.err
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

    def test_combine_spectra(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'combine_spectra',
                                              nfiles=2)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'mean'
        red_copy = pickle.dumps(red)
        input_data = red.input[0][0].data.copy()

        # test one input, mean combine: saves 1d spectra only
        parset['save']['value'] = True
        red.input = [red.input[0]]
        red.combine_spectra()
        assert 'No data to combine' in capsys.readouterr().out
        assert np.allclose(red.input[0][0].data, input_data, equal_nan=True)
        fname = red.input[0][0].header['FILENAME']
        assert os.path.exists(tmpdir.join(fname.replace('COA', 'CMB')))
        assert not os.path.exists(tmpdir.join(fname))
        os.remove(tmpdir.join(fname.replace('COA', 'CMB')))

        # test one input, cube combine: saves SCB only
        red = pickle.loads(red_copy)
        red.input = [red.input[0]]
        old_wave = red.input[0]['WAVEPOS'].data

        parset = red.parameters.current[idx]
        parset['method']['value'] = 'cube'
        red.combine_spectra()

        assert 'No data to combine' not in capsys.readouterr().out
        assert red.input[0][0].data.ndim == 3
        fname = red.input[0][0].header['FILENAME']
        assert 'SCB' in fname
        assert os.path.exists(tmpdir.join(fname))
        assert not os.path.exists(tmpdir.join(fname.replace('SCB', 'CMB')))

        # check that spectral data matches cube wavelength dimension
        nwave = red.input[0]['FLUX'].data.shape[0]
        for ext in ['WAVEPOS', 'TRANSMISSION', 'RESPONSE']:
            assert red.input[0][ext].data.shape[0] == nwave

        # check that new wave is subset of old wave
        np.allclose(old_wave[4:-5], red.input[0]['WAVEPOS'].data)

        # test mismatched input wavelengths: raises error
        red = pickle.loads(red_copy)
        red.input[1]['WAVEPOS'].data += 2.0
        with pytest.raises(ValueError):
            red.combine_spectra()
        assert 'Mismatched wavelengths' in capsys.readouterr().err

        # test missing trans, response, as if fluxcal were skipped:
        # should still work
        red = pickle.loads(red_copy)
        for inp in red.input:
            del inp['TRANSMISSION']
            del inp['RESPONSE']
        red.combine_spectra()
        assert len(red.input) == 1
        assert 'TRANSMISSION' not in red.input[0]
        assert 'RESPONSE' not in red.input[0]

    def test_coadd_registration(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'combine_spectra',
                                              nfiles=2)

        # add some header keywords relevant to registration
        # and make data small for faster processing
        for i, inp in enumerate(red.input):
            inp[0].header['TGTRA'] = (10.0 + i / 3600 / 0.768) / 15.0
            inp[0].header['TGTDEC'] = 10.0 + i / 3600 / 0.768
            inp[0].header['DITHER'] = True
            inp[0].header['DTHCRSYS'] = 'SIRF'
            inp[0].header['DITHERX'] = 0.0 + i
            inp[0].header['DITHERY'] = 0.0 + i
            inp[0].header['SKY_ANGL'] = 0.0
            inp[0].data = np.full((20, 20), 1.0 + i)
            inp[1].data = np.full((20, 20), 1.0 + i)

        parset = red.parameters.current[idx]
        parset['method']['value'] = 'cube'
        red_copy = pickle.dumps(red)

        parset['registration']['value'] = 'wcs'
        red.combine_spectra()
        assert 'Using WCS as is' in capsys.readouterr().out
        r1 = red.input[0].copy()

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['registration']['value'] = 'header'
        red.combine_spectra()
        assert 'Applying dither offsets from ' \
               'the header' in capsys.readouterr().out
        r2 = red.input[0].copy()

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['registration']['value'] = 'target'
        red.combine_spectra()
        assert 'Correcting for target motion' in capsys.readouterr().out
        r3 = red.input[0].copy()

        # all reference to the same values
        assert r2[0].header['CRVAL1'] == r1[0].header['CRVAL1']
        assert r2[0].header['CRVAL2'] == r1[0].header['CRVAL2']
        assert r2[0].header['CRVAL3'] == r1[0].header['CRVAL3']
        assert r3[0].header['CRVAL1'] == r1[0].header['CRVAL1']
        assert r3[0].header['CRVAL2'] == r1[0].header['CRVAL2']
        assert r3[0].header['CRVAL3'] == r1[0].header['CRVAL3']

        # output shapes are different
        assert r1[0].data.shape == (20, 18, 11)
        assert r2[0].data.shape == (20, 20, 12)
        assert r3[0].data.shape == (20, 21, 17)

        # mock a header shift failure
        mocker.patch(
            'sofia_redux.instruments.forcast.register_datasets.get_shifts',
            return_value=[np.array([0, 0]), None])
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['registration']['value'] = 'header'
        red.combine_spectra()
        assert 'Failed to register dataset 1' in capsys.readouterr().err
        r4 = red.input[0].copy()

        # result is same as zero shift
        assert r4[0].data.shape == (20, 18, 11)
        assert np.allclose(r4[0].data, r1[0].data, equal_nan=True)

    def test_combine_multi_ap(self, tmpdir, capsys, mocker):
        # minimal setup to run the step
        red = FORCASTSpectroscopyReduction()
        red.output_directory = tmpdir
        red.parameters = FORCASTSpectroscopyParameters()
        red.parameters.add_current_parameters('combine_spectra')
        red.calres = {'spectel': 'UNKNOWN', 'gmode': 0}

        # one aperture, standard dim
        inp = []
        for i in range(3):
            hdul = self.make_crm(tmpdir, return_hdul=True)
            inp.append(hdul)
            red.filenum.append(i)
        red.input = inp
        red.combine_spectra()
        assert len(red.input) == 1
        assert red.input[0]['SPECTRAL_FLUX'].shape == (10,)

        # one apertures, stacked dimension
        inp = []
        for i in range(3):
            hdul = self.make_crm(tmpdir, return_hdul=True, n_ap=1)
            inp.append(hdul)
            red.filenum.append(i)
        red.input = inp
        red.combine_spectra()
        assert len(red.input) == 1
        assert red.input[0]['SPECTRAL_FLUX'].shape == (10,)

        # two apertures, combined
        inp = []
        for i in range(3):
            hdul = self.make_crm(tmpdir, return_hdul=True, n_ap=2)
            inp.append(hdul)
            red.filenum.append(i)
        red.input = inp
        red.parameters.current[0].set_value('combine_aps', True)
        red.combine_spectra()
        assert len(red.input) == 1
        assert red.input[0]['SPECTRAL_FLUX'].shape == (10,)

        # two apertures, not combined
        inp = []
        for i in range(3):
            hdul = self.make_crm(tmpdir, return_hdul=True, n_ap=2)
            inp.append(hdul)
            red.filenum.append(i)
        red.input = inp
        red.parameters.current[0].set_value('combine_aps', False)
        red.combine_spectra()
        assert len(red.input) == 1
        assert red.input[0]['SPECTRAL_FLUX'].shape == (2, 10)

    def test_make_response(self, tmpdir, capsys, mocker):
        # standard input data is not marked for response recipe, so
        # modify type and set up for make_response
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)
        fits.setval(ffile, 'OBSTYPE', value='STANDARD_TELLURIC')
        red.output_directory = tmpdir
        red.load(ffile)
        red.load_parameters()
        idx = red.recipe.index('make_response')
        for i in range(idx):
            red.step()

        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        red_copy = pickle.dumps(red)

        # default: makes and saves a 1d rsp file
        red.make_response()
        fname = red.input[0][0].header['FILENAME']
        assert 'RSP' in fname
        assert os.path.exists(tmpdir.join(fname))
        assert red.input[0][0].data.shape[0] == 5
        os.remove(tmpdir.join(fname))

        # test missing response extension: should warn, but work
        red = pickle.loads(red_copy)
        del red.input[0]['RESPONSE']
        del red.input[0][0].header['RP']
        red.make_response()
        assert 'No response extension found' in capsys.readouterr().err
        assert os.path.exists(tmpdir.join(fname))

        # mock a missing model: raises error
        mocker.patch('sofia_redux.instruments.forcast.getmodel.get_model',
                     return_value=None)
        red = pickle.loads(red_copy)
        with pytest.raises(ValueError):
            red.make_response()
        assert 'Cannot create response file' in capsys.readouterr().err

    def test_bunit(self, tmpdir):
        # run through all steps, checking for appropriate BUNIT keys
        red = FORCASTSpectroscopyReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps, checking bunit
        bunit = 'ct'
        exp_unit = 's'
        spatial_unit = 'arcsec'
        wave_unit = 'um'
        response_unit = 'Me/s/Jy'
        spec_unit = 'Me/s'
        unitless = ['BADMASK', 'SPATIAL_MAP', 'SPATIAL_PROFILE',
                    'APERTURE_MASK', 'TRANSMISSION']
        spatial = ['SLITPOS', 'APERTURE_TRACE']
        wave = ['WAVEPOS']
        response = ['RESPONSE', 'RESPONSE_ERROR']
        spec = ['SPECTRAL_FLUX', 'SPECTRAL_ERROR']
        for step in red.recipe:
            red.step()
            if step == 'checkhead':
                continue
            elif step == 'stack':
                bunit = 'Me/s'
            elif step == 'flux_calibrate':
                bunit = 'Jy/pixel'
                spec_unit = 'Jy'
            hdul = red.input[0]
            for hdu in hdul:
                extname = str(hdu.header['EXTNAME']).upper()
                if extname == 'EXPOSURE':
                    assert hdu.header['BUNIT'] == exp_unit
                elif extname in unitless:
                    assert hdu.header['BUNIT'] == ''
                elif extname in spatial:
                    assert hdu.header['BUNIT'] == spatial_unit
                elif extname in wave:
                    assert hdu.header['BUNIT'] == wave_unit
                elif extname in response:
                    assert hdu.header['BUNIT'] == response_unit
                elif extname in spec:
                    assert hdu.header['BUNIT'] == spec_unit
                else:
                    assert hdu.header['BUNIT'] == bunit

    def test_combine_response(self, tmpdir):
        red = FORCASTSpectroscopyReduction()

        # make a set of 1D response spectra
        rfiles = []
        flux_data = []
        for i in range(4):
            rfile = str(tmpdir.join(
                f'F0001_FO_GRI_9000000_FORG063_RSP_000{i}.fits'))
            wave = np.arange(50, dtype=float)
            flux = np.arange(50, dtype=float) * (i + 1)
            error = flux.copy()
            atran = np.ones_like(flux)
            hdr = fits.Header({'OBSTYPE': 'STANDARD_TELLURIC',
                               'PRODTYPE': 'response_spectrum',
                               'SPECTEL1': 'FOR_G063',
                               'SPECTEL2': 'OPEN',
                               'SLIT': 'FOR_LS24',
                               'DETCHAN': 0,
                               'AOR_ID': '9000000',
                               'MISSN-ID': '2020-01-01_FO_F001',
                               'FILENAME': os.path.basename(rfile)})
            hdul = fits.HDUList(fits.PrimaryHDU([wave, flux, error, atran],
                                                header=hdr))
            hdul.writeto(rfile)
            rfiles.append(rfile)
            flux_data.append(flux)

        red.load(rfiles)
        red.output_directory = str(tmpdir)
        red.load_parameters()

        # check that response files were recognized
        assert red.recipe == ['combine_response']

        # set some default parameters and keep a copy
        red.parameters.current[0]['scale_method']['value'] = 'none'
        red.parameters.current[0]['weighted']['value'] = False
        red.parameters.current[0]['robust']['value'] = False
        red.parameters.current[0]['fwhm']['value'] = 0
        red_copy = pickle.dumps(red)

        # expected output
        flux_data = np.array(flux_data)
        mean_flux = np.mean(flux_data, axis=0)
        mean_error = np.sqrt(np.sum(flux_data ** 2, axis=0)) / 4

        # test no save
        red.parameters.current[0]['save']['value'] = False
        red.step()
        assert len(red.out_files) == 0

        # check prodtype and procstat
        assert red.input[0][0].header['PROCSTAT'] == 'LEVEL_4'
        assert red.input[0][0].header['PRODTYPE'] == 'instrument_response'

        # test save
        red = pickle.loads(red_copy)
        red.parameters.current[0]['save']['value'] = True
        red.step()
        assert len(red.out_files) == 1
        assert os.path.isfile(
            str(tmpdir.join(
                'F0001_FO_GRI_9000000_FORG063_IRS_0000-0003.fits')))

        # test mismatched wavelengths
        red = pickle.loads(red_copy)
        red.input[0][0].data[0] += 10
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Mismatched wavelengths' in str(err)

        # test scale methods: flux will match either
        # the mean input spectrum or else a particular input
        red = pickle.loads(red_copy)
        red.parameters.current[0]['scale_method']['value'] = 'none'
        red.step()
        assert np.allclose(red.input[0][0].data[1], mean_flux)
        assert np.allclose(red.input[0][0].data[2], mean_error)

        red = pickle.loads(red_copy)
        red.parameters.current[0]['scale_method']['value'] = 'median'
        red.step()
        assert np.allclose(red.input[0][0].data[1], mean_flux)
        assert np.all(red.input[0][0].data[2] <= mean_error)

        red = pickle.loads(red_copy)
        red.parameters.current[0]['scale_method']['value'] = 'highest'
        red.step()
        assert np.allclose(red.input[0][0].data[1], flux_data[-1])
        assert np.all(red.input[0][0].data[2] >= mean_error)

        red = pickle.loads(red_copy)
        red.parameters.current[0]['scale_method']['value'] = 'lowest'
        red.step()
        assert np.allclose(red.input[0][0].data[1], flux_data[0])
        assert np.all(red.input[0][0].data[2] <= mean_error)

        red = pickle.loads(red_copy)
        red.parameters.current[0]['scale_method']['value'] = 'index'
        red.parameters.current[0]['scale_index']['value'] = 2
        red.step()
        assert np.allclose(red.input[0][0].data[1], flux_data[2])
        assert np.all(red.input[0][0].data[2] >= mean_error)

        # test smooth on flat data: should be same except at edges
        red = pickle.loads(red_copy)
        red.parameters.current[0]['fwhm']['value'] = 2
        red.step()
        nn = ~np.isnan(red.input[0][0].data[1])
        assert np.allclose(red.input[0][0].data[1][nn], mean_flux[nn])

        # test that smooth also reduces noise
        rand = np.random.RandomState(42)
        red = pickle.loads(red_copy)
        red.parameters.current[0]['fwhm']['value'] = 0
        noise_vals = []
        for hdul in red.input:
            noise = rand.uniform(-2, 2, 50)
            hdul[0].data[1] += noise
            noise_vals.append(noise)
        red.step()
        no_smooth = red.input[0][0].data[1].copy()

        red = pickle.loads(red_copy)
        red.parameters.current[0]['fwhm']['value'] = 2
        for i, hdul in enumerate(red.input):
            hdul[0].data[1] += noise_vals[i]
        red.step()
        smooth = red.input[0][0].data[1].copy()
        # residuals should be less for smoothed version
        nn = ~np.isnan(smooth)
        assert np.sum((smooth[nn] - mean_flux[nn])**2) \
               < np.sum((no_smooth[nn] - mean_flux[nn])**2)

        # test single input spectrum, no smoothing - should
        # just pass through
        red = pickle.loads(red_copy)
        red.input = [red.input[0]]
        red.step()
        assert np.allclose(red.input[0][0].data[1], flux_data[0])

    def test_combine_response_multi_ap(self, tmpdir):
        red = FORCASTSpectroscopyReduction()

        # make a set of 1D response spectra with 2 aps
        rfiles = []
        for i in range(4):
            rfile = str(tmpdir.join(
                f'F0001_FO_GRI_9000000_FORG063_RSP_000{i}.fits'))
            hdr = fits.Header({'OBSTYPE': 'STANDARD_TELLURIC',
                               'PRODTYPE': 'response_spectrum',
                               'SPECTEL1': 'FOR_G063',
                               'SPECTEL2': 'OPEN',
                               'SLIT': 'FOR_LS24',
                               'DETCHAN': 0,
                               'AOR_ID': '9000000',
                               'MISSN-ID': '2020-01-01_FO_F001',
                               'FILENAME': os.path.basename(rfile)})
            darr = np.arange(2 * 4 * 50, dtype=float).reshape(2, 4, 50)
            hdul = fits.HDUList(fits.PrimaryHDU(darr, header=hdr))
            hdul.writeto(rfile)
            rfiles.append(rfile)

        red.load(rfiles)
        red.output_directory = str(tmpdir)
        red.load_parameters()
        red_copy = pickle.dumps(red)

        # combine apertures: mismatched wavelengths error
        red.parameters.current[0]['combine_aps']['value'] = True
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Mismatched wavelengths' in str(err)

        # don't combine: mismatch is okay
        red = pickle.loads(red_copy)
        red.parameters.current[0]['combine_aps']['value'] = False
        red.step()
        assert len(red.input) == 1
        assert red.input[0][0].data.shape == (2, 4, 50)

        # fix mismatch and combine
        red = pickle.loads(red_copy)
        red.parameters.current[0]['combine_aps']['value'] = True
        for inp in red.input:
            inp[0].data[:, 0, :] = inp[0].data[0, 0, :]
        red.step()
        assert len(red.input) == 1
        assert red.input[0][0].data.shape == (4, 50)

    def test_specmap(self, tmpdir):
        # make some minimal spectrum-like data
        ffile = self.make_combspec(tmpdir, n_ap=1)

        # load spectrum with old-style product name -
        # should allow specmap reduction
        red = FORCASTSpectroscopyReduction()
        red.output_directory = tmpdir
        red.load(ffile)
        red.load_parameters()
        assert red.recipe == ['specmap']
        red.step()
        outfile = ffile.replace('.fits', '.png')
        assert os.path.isfile(outfile)
        shutil.copyfile(outfile, tmpdir.join('tmp0.png'))

        # same for multi-ap/order old style data
        # (including EXES, FLITECAM)
        ffile = self.make_combspec(tmpdir, n_ap=30)
        fits.setval(ffile, 'XUNITS', value='cm-1')
        red.load(ffile)
        red.load_parameters()
        assert red.recipe == ['specmap']
        red.step()
        assert os.path.isfile(outfile)
        shutil.copyfile(outfile, tmpdir.join('tmp1.png'))
        # image is not the same
        assert compare_images(tmpdir.join('tmp0.png'),
                              tmpdir.join('tmp1.png'), 0) is not None

        # current products with both spectra and image work too
        ffile = self.make_crm(tmpdir)
        red.load(ffile)
        # recipe is last two steps
        assert red.recipe == ['combine_spectra', 'specmap']
        # but overriding and just mapping it should work
        red.recipe = ['specmap']
        red.load_parameters()
        red.step()
        assert os.path.isfile(ffile.replace('.fits', '.png'))

        # including with multi-ap new style data
        ffile = self.make_crm(tmpdir, n_ap=2)
        red.load(ffile)
        red.recipe = ['specmap']
        red.load_parameters()
        red.step()
        assert os.path.isfile(ffile.replace('.fits', '.png'))

    def test_specmap_cube(self, tmpdir, capsys):
        # cube-like synthetic data
        ffile = self.make_cube(tmpdir)
        red = FORCASTSpectroscopyReduction()
        red.output_directory = tmpdir

        # expected output file
        outfile = ffile.replace('.fits', '.png')

        red.load(ffile)
        red.load_parameters()
        parset = red.parameters.current[-1]
        parset['atran_plot']['value'] = True

        assert red.recipe == ['specmap']
        with set_log_level('DEBUG'):
            red.specmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp0.png'))
        capt = capsys.readouterr()
        assert 'slice at w=0' in capt.out
        assert 'point at x=5, y=5' in capt.out

        # test override parameters
        parset['override_slice']['value'] = 5
        parset['override_point']['value'] = '[2, "3"]'
        with set_log_level('DEBUG'):
            red.specmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp1.png'))
        capt = capsys.readouterr()
        assert 'slice at w=5' in capt.out
        assert 'point at x=2, y=3' in capt.out

        assert compare_images(tmpdir.join('tmp0.png'),
                              tmpdir.join('tmp1.png'), 0) is not None

        # turn off atran plot - different image
        parset['atran_plot']['value'] = False
        red.specmap()
        shutil.move(outfile, tmpdir.join('tmp2.png'))
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp2.png'), 0) is not None

        # bad input parameters
        parset['override_slice']['value'] = 'a'
        with pytest.raises(ValueError):
            red.specmap()
        parset['override_slice']['value'] = 'none'
        parset['override_point']['value'] = 1
        with pytest.raises(ValueError):
            red.specmap()
        parset['override_point']['value'] = "1, a"
        with pytest.raises(ValueError):
            red.specmap()

    @pytest.mark.parametrize('n_ap', [1, 2])
    def test_specmap_ignore_outer(self, tmpdir, capsys, n_ap):
        # single spectrum-like data
        ffile = self.make_combspec(tmpdir, n_ap=n_ap)
        outfile = ffile.replace('.fits', '.png')

        # load spectrum with old-style product name -
        # should allow specmap reduction
        red = FORCASTSpectroscopyReduction()
        red.output_directory = tmpdir
        red.load(ffile)
        red.load_parameters()
        parset = red.parameters.current[-1]

        # set ignore outer parameter to turn off
        parset['ignore_outer']['value'] = 0.0
        with set_log_level('DEBUG'):
            red.specmap()
        assert 'Plotting between' not in capsys.readouterr().out
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp0.png'))

        # set ignore outer parameter to ignore outer 20%
        parset['ignore_outer']['value'] = 0.2
        with set_log_level('DEBUG'):
            red.specmap()
        assert 'Plotting between w=2 and w=8' in capsys.readouterr().out
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp1.png'))
        # images are different
        assert compare_images(tmpdir.join('tmp0.png'),
                              tmpdir.join('tmp1.png'), 0) is not None
