# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FIFI-LS Reduction class."""

import os
import shutil

from astropy.io import fits
from astropy import log
import dill as pickle
from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.parameters import Parameters
from sofia_redux.pipeline.gui.qad_viewer import QADViewer

try:
    from sofia_redux.pipeline.sofia.fifils_reduction \
        import FIFILSReduction
    from sofia_redux.pipeline.sofia.parameters.fifils_parameters \
        import FIFILSParameters
    HAS_FIFI = True
except ImportError:
    HAS_FIFI = False
    FIFILSReduction = Reduction
    FIFILSParameters = Parameters

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


def bad_step(*args, **kwargs):
    return None


@pytest.mark.skipif('not HAS_FIFI')
class TestFIFILSReduction(object):
    @pytest.fixture(autouse=True, scope='function')
    def mock_joblib(self, mocker):
        import types
        mock_joblib = types.ModuleType('joblib')
        mocker.patch.dict('sys.modules', {'joblib': mock_joblib})

    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def make_file(self, tmpdir, fname='00001_123456_00001_TEST_A_lw.fits',
                  nod='A', obsid='R001'):
        """Retrieve a basic test FITS file for FIFI-LS."""
        from sofia_redux.instruments.fifi_ls.tests.resources \
            import raw_testdata
        hdul = raw_testdata(nod=nod, obsid=obsid)
        hdul[0].header['FILENAME'] = fname

        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)

        return ffile

    def standard_setup(self, tmpdir, step, nfiles=2):
        red = FIFILSReduction()
        ffile = []
        for i in range(nfiles):
            obsid = 'R{:03d}'.format(i + 1)
            if i % 2 == 0:
                nod = 'A'
            else:
                nod = 'B'
            fn = '0000{}_123456_00001_TEST_{}_lw.fits.fits'.format(i + 1, nod)
            ffile.append(self.make_file(tmpdir, fname=fn,
                                        nod=nod, obsid=obsid))
        red.output_directory = str(tmpdir)

        red.load(ffile)
        red.load_parameters()
        idx = red.recipe.index(step)

        # set parallel to false for testing
        for param in red.parameters.current:
            param.set_value('parallel', False)

        # step up to specified pipeline step
        for i in range(idx):
            red.step()

        if nfiles == 1:
            ffile = ffile[0]
        return ffile, red, idx

    def test_startup(self):
        red = FIFILSReduction()
        assert isinstance(red, Reduction)

    def test_cleanup(self, capsys):
        red = FIFILSReduction()
        capsys.readouterr()

        # does nothing by default
        red.cleanup()
        capt = capsys.readouterr()
        assert capt.out == ''

    def test_load_basic(self, tmpdir):
        red = FIFILSReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FIFILSParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_load_error(self, tmpdir, capsys):
        # check that fails appropriately for bad FITS file
        red = FIFILSReduction()
        red.output_directory = str(tmpdir)
        badfile = tmpdir.join('bad.fits')
        badfile.write('badval')

        red.load(badfile)
        with pytest.raises(ValueError):
            red.load_fits()
        capt = capsys.readouterr()
        assert 'Unable to read' in capt.err
        # input is empty
        assert len(red.input) == 0

    def test_load_intermediate(self, tmpdir, capsys):
        red = FIFILSReduction()
        orig_recipe = red.recipe.copy()
        ffile = self.make_file(tmpdir)

        # set prodtype
        fits.setval(ffile, 'PRODTYPE', value='ramps_fit')
        red.load(ffile)

        # recipe should pick up after ramp fit
        assert red.recipe == orig_recipe[orig_recipe.index('fit_ramps') + 1:]

        # set to last type and verify error is raised
        red = FIFILSReduction()
        fits.setval(ffile, 'PRODTYPE', value='specmap')
        with pytest.raises(ValueError):
            red.load(ffile)
        capt = capsys.readouterr()
        assert 'No steps to run' in capt.err

        # check that load_fits is called if prodtype is not unknown
        red = FIFILSReduction()
        fits.setval(ffile, 'PRODTYPE', value='test_value')
        red.load(ffile)
        assert isinstance(red.input[0], fits.HDUList)

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        red = FIFILSReduction()
        vz = red.register_viewers()
        assert len(vz) == 1
        assert isinstance(vz[0], QADViewer)

    def test_display_data(self, tmpdir):
        red = FIFILSReduction()
        ffile = self.make_file(tmpdir)

        # test for raw data
        red.load(ffile)
        red.load_fits()
        red.set_display_data(raw=True)

        red.set_display_data(raw=False)
        assert len(red.display_data['QADViewer']) == 1
        assert isinstance(red.display_data['QADViewer'][0], fits.HDUList)

    def test_write_output(self, tmpdir):
        red = FIFILSReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.load_fits()
        hdul = red.input[0]

        red.output_directory = str(tmpdir)
        outname = 'new_name.fits'
        hdul[0].header['FILENAME'] = outname

        outpath = str(tmpdir.join(outname))
        red.write_output(hdul)
        assert os.path.isfile(outpath)
        assert outpath in red.out_files

    def test_all_steps(self, tmpdir):
        # exercises nominal behavior for a standard reduction
        red = FIFILSReduction()
        ffile1 = self.make_file(tmpdir, nod='A', obsid='R001',
                                fname='00001_123456_00001_TEST_A_sw.fits')
        ffile2 = self.make_file(tmpdir, nod='B', obsid='R002',
                                fname='00002_123456_00001_TEST_B_sw.fits')

        red.load([ffile1, ffile2])
        red.output_directory = str(tmpdir)
        red.load_parameters()

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = str(hdul[0].header['HISTORY']).lower()
        msgs = ['checkhead',
                'chops split',
                'ramps fit',
                'chops subtracted',
                'nods combined',
                'wavelength calibrated',
                'xy offsets added',
                'flat-field corrected',
                'grating scans combined',
                'telluric corrected',
                'flux calibrated',
                'wavelength shift',
                'resampled',
                'specmap']
        for i, step in enumerate(red.recipe):
            if step == 'checkhead' or step == 'specmap':
                continue
            assert msgs[i] in history

    @pytest.mark.parametrize('step',
                             ['split_grating_and_chop', 'fit_ramps',
                              'subtract_chops', 'combine_nods',
                              'lambda_calibrate', 'spatial_calibrate',
                              'apply_static_flat', 'combine_grating_scans',
                              'telluric_correct', 'flux_calibrate',
                              'resample'])
    def test_step_minimal(self, tmpdir, capsys, mocker, step):
        """
        Test standard behavior for multiple steps.

        Exercises save and general error condition.
        """
        ffile, red, idx = self.standard_setup(tmpdir, step, nfiles=1)

        # copy the reduction just before step
        red_copy = pickle.dumps(red)

        # test save
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        red.step()

        fn = None
        for fn in red.out_files:
            if red.prodnames[step] in fn:
                break
        assert os.path.isfile(fn)

        # test error in step
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.{}.{}'.format(step, step),
            bad_step)
        red = pickle.loads(red_copy)
        red.step()
        capt = capsys.readouterr()
        assert 'Problem in fifi_ls.{}'.format(step) in capt.err
        assert len(red.input) == 0

    def test_checkhead(self, tmpdir, capsys):
        # check that fails appropriately for bad header
        red = FIFILSReduction()
        ffile = self.make_file(tmpdir)
        red.output_directory = str(tmpdir)

        # passes
        red.load(ffile)
        red.load_parameters()
        red.checkhead()

        # logs error
        fits.setval(ffile, 'SPECTEL2', value='BADVAL')
        red.load(ffile)
        red.load_parameters()

        red.checkhead()
        capt = capsys.readouterr()
        assert 'Invalid headers' in capt.err
        # input is empty
        assert len(red.input) == 0

    def test_combine_nods(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'combine_nods')

        # copy just before step
        red_copy = pickle.dumps(red)

        # standard -- A and B combined
        red.step()
        capt = capsys.readouterr()
        assert 'Combined A and B nods' in capt.out

        # A only -- still propagated
        red = pickle.loads(red_copy)
        red.input = [red.input[0]]
        red.step()
        capt = capsys.readouterr()
        assert 'No B nods found' in capt.out
        assert len(red.input) == 1

        # B only -- not propagated (red.error is populated)
        red = pickle.loads(red_copy)
        red.input = [red.input[1]]
        red.step()
        capt = capsys.readouterr()
        assert 'No A nods found' in capt.err
        assert len(red.input) == 0

        # Some As did not find Bs
        red = pickle.loads(red_copy)
        # copy the A and modify to make it distinct-ish
        a_copy = [hdu.copy() for hdu in red.input[0]]
        a_copy[0].header['FILENAME'] = 'test_100.fits'
        red.input = [red.input[0], red.input[1], a_copy]
        red.step()
        capt = capsys.readouterr()
        assert '1/2 A nods did not find B nods' in capt.err
        assert len(red.input) == 1

    def test_correct_shift(self, tmpdir, capsys, mocker):
        """Test standard behavior for wave shift step."""
        step = 'correct_wave_shift'
        ffile, red, idx = self.standard_setup(tmpdir, step, nfiles=1)

        # copy the reduction just before step
        red_copy = pickle.dumps(red)

        # test save
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        red.step()

        fn = None
        for fn in red.out_files:
            if red.prodnames[step] in fn:
                break
        assert os.path.isfile(fn)

        # test error in step: in this case, it just warns and continues
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.{}.{}'.format(step, step),
            bad_step)
        red = pickle.loads(red_copy)
        red.step()
        capt = capsys.readouterr()
        assert 'No wavelength shift correction performed' in capt.err
        assert len(red.input) > 0

    def test_spatcal_parameters(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'spatial_calibrate',
                                              nfiles=1)
        # copy just before step
        red_copy = pickle.dumps(red)

        # test that flip -> True -> -
        param = red.get_parameter_set()
        param.set_value('flipsign', 'flip')
        red.step()
        capt = capsys.readouterr()

        assert 'sign convention: -' in capt.out

        # no flip -> False -> +
        red = pickle.loads(red_copy)
        param = red.get_parameter_set()
        param.set_value('flipsign', 'no flip')
        red.step()
        capt = capsys.readouterr()
        assert 'sign convention: +' in capt.out

        # default -> None -> date dependent
        # pre-2015 flips, after doesn't
        red = pickle.loads(red_copy)
        infile = red.input[0]
        infile[0].header['DATE-OBS'] = '2014-03-01T10:49:55'
        red.input = [infile]
        param = red.get_parameter_set()
        param.set_value('flipsign', 'default')
        red.step()
        capt = capsys.readouterr()
        assert 'sign convention: -' in capt.out

        red = pickle.loads(red_copy)
        infile = red.input[0]
        infile[0].header['DATE-OBS'] = '2016-03-01T10:49:55'
        red.input = [infile]
        param = red.get_parameter_set()
        param.set_value('flipsign', 'default')
        red.step()
        capt = capsys.readouterr()
        assert 'sign convention: +' in capt.out

    def test_skip_cal(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'apply_static_flat',
                                              nfiles=1)
        fname = red.input[0][0].header['FILENAME']

        # skip flat correction: input file stays the same
        param = red.get_parameter_set()
        param.set_value('skip_flat', True)
        red.step()
        capt = capsys.readouterr()
        assert 'No flat correction performed' in capt.out
        assert red.input[0][0].header['FILENAME'] == fname

        # run scan combine
        red.step()

        # skip telluric correction: atran file is attached,
        # file name is updated
        param = red.get_parameter_set()
        param.set_value('skip_tell', True)
        red.step()
        capt = capsys.readouterr()
        assert 'no correction performed' in capt.out
        fname = red.input[0][0].header['FILENAME']

        # skip flux cal: input stays the same
        param = red.get_parameter_set()
        param.set_value('skip_cal', True)
        red.step()
        capt = capsys.readouterr()
        assert 'No flux calibration performed' in capt.out
        assert red.input[0][0].header['FILENAME'] == fname

        # skip wave shift: input stays the same
        param = red.get_parameter_set()
        param.set_value('skip_shift', True)
        red.step()
        capt = capsys.readouterr()
        assert 'No wavelength shift correction performed' in capt.out
        assert red.input[0][0].header['FILENAME'] == fname

        # run the last step: should complete okay
        red.step()
        assert 'WXY' in red.input[0][0].header['FILENAME']

    def test_resample_params(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'resample',
                                              nfiles=4)
        red_copy = pickle.dumps(red)
        param = red.get_parameter_set()

        # assign thresholds to zero -- they should get converted to None
        param.set_value('fitthresh', 0)
        param.set_value('negthresh', 0)
        param.set_value('posthresh', 0)
        param.set_value('adaptive_algorithm', 'none')

        # skip coadd to make separate cubes
        param.set_value('skip_coadd', True)

        red.step()
        capt = capsys.readouterr()
        assert 'Turning off fit rejection' in capt.out
        assert 'Turning off negative rejection pass' in capt.out
        assert 'Turning off outlier rejection' in capt.out
        assert 'Turning off adaptive smoothing' in capt.out

        assert len(red.input) == 2

        # again, but set adaptive threshold to something unreadable
        red = pickle.loads(red_copy)
        param = red.get_parameter_set()
        param.set_value('adaptive_threshold', 'test')
        red.step()
        capt = capsys.readouterr()
        assert 'Turning off adaptive smoothing' in capt.out

        # now set adaptive threshold with non-gaussian smoothing width
        red = pickle.loads(red_copy)
        param = red.get_parameter_set()
        param.set_value('adaptive_algorithm', 'scaled')
        param.set_value('xy_smoothing', 2.0)
        red.step()
        capt = capsys.readouterr()
        assert 'Turning off adaptive smoothing' not in capt.out
        assert 'Setting x/y smoothing radius to Gaussian sigma' in capt.err

        # check that no warning is produced if smoothing is already
        # close enough to the gaussian width
        red = pickle.loads(red_copy)
        param = red.get_parameter_set()
        param.set_value('adaptive_algorithm', 'shaped')
        param.set_value('xy_smoothing', 0.42)
        red.step()
        capt = capsys.readouterr()
        assert 'Turning off adaptive smoothing' not in capt.out
        assert 'Setting x/y smoothing radius to Gaussian sigma' \
            not in capt.err

    def test_resample_pixsize(self, tmpdir, capsys):
        red = FIFILSReduction()
        red.output_directory = str(tmpdir)
        ffile = self.make_file(tmpdir)

        # for RED, output pixel size should be set to 3.0
        fits.setval(ffile, 'DETCHAN', value='RED')
        red.load(ffile)
        red.load_parameters()
        idx = red.recipe.index('resample')
        param = red.get_parameter_set(step_index=idx)
        pixsize = param.get_value('xy_pixel_size')
        assert np.allclose(pixsize, 3.0)

        # for BLUE, output pixel size should be set to 1.5
        fits.setval(ffile, 'DETCHAN', value='BLUE')
        red.load(ffile)
        red.load_parameters()
        param = red.get_parameter_set(step_index=idx)
        pixsize = param.get_value('xy_pixel_size')
        assert np.allclose(pixsize, 1.5)

        # for bad values, output pixel size remains unset
        fits.setval(ffile, 'DETCHAN', value='UNKNOWN')
        red.load(ffile)
        red.load_parameters()
        param = red.get_parameter_set(step_index=idx)
        pixsize = param.get_value('xy_pixel_size')
        assert str(pixsize).strip() == ''

        # verify setting to blank value still works - oversample is used
        # instead
        fits.setval(ffile, 'DETCHAN', value='BLUE')
        red.load(ffile)
        red.load_parameters()
        param = red.get_parameter_set(step_index=idx)
        param.set_value('xy_pixel_size', None)
        red.reduce()
        capt = capsys.readouterr()
        assert "Spatial oversample: " \
               "{}".format(param.get_value('xy_oversample')) in capt.out

    def test_parallel(self, tmpdir, mocker):
        # mock a 4-core machine
        mocker.patch('psutil.cpu_count', return_value=4)
        red = FIFILSReduction()
        assert red.max_cores == 2

        # mock a 2-core machine
        mocker.patch('psutil.cpu_count', return_value=2)
        red = FIFILSReduction()
        assert red.max_cores is None

        red.output_directory = str(tmpdir)
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.load_parameters()
        red_copy = pickle.dumps(red)

        # smoke test all steps with and without parallel parameter:
        # should have the same effect when max_cores is None
        for param in red.parameters.current:
            param.set_value('parallel', False)
        red.reduce()
        red = pickle.loads(red_copy)
        for param in red.parameters.current:
            param.set_value('parallel', True)
        red.reduce()

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attribute necessary for fifi-ls
        red = FIFILSReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'basehead')
        assert pcopy.basehead is not None
        assert isinstance(pcopy.basehead, fits.Header)

    def test_specmap_options(self, tmpdir, capsys):
        step = 'specmap'
        ffile, red, idx = self.standard_setup(tmpdir, step, nfiles=1)

        # copy the reduction just before step
        red.get_parameter_set(step_index=idx).set_value('atran_plot', False)
        red_copy = pickle.dumps(red)

        # expected image file name
        outfile = tmpdir.join('F0282_FI_IFS_90000101_RED_WXY_001.png')

        # test skip
        param = red.get_parameter_set(step_index=idx)
        param.set_value('skip_preview', True)
        red.step()
        capt = capsys.readouterr()
        assert 'Not making preview image' in capt.out
        assert not os.path.isfile(outfile)

        # test reference line method
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('slice_method', 'reference')
        param.set_value('point_method', 'reference')
        red.input[0][0].header['G_WAVE_R'] = 118.6
        red.step()
        capt = capsys.readouterr()
        assert 'Plotting at 118.600 um, near reference ' \
               'wavelength at 118.6 um' in capt.out
        assert 'Saved image' in capt.out
        assert os.path.isfile(outfile)
        shutil.copyfile(outfile, tmpdir.join('tmp0.png'))

        # blue channel gets g_wave_b instead of _r
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('slice_method', 'reference')
        param.set_value('point_method', 'reference')
        red.input[0][0].header['CHANNEL'] = 'BLUE'
        red.input[0][0].header['G_WAVE_B'] = 118.6
        red.step()
        capt = capsys.readouterr()
        assert 'Plotting at 118.600 um, near reference ' \
               'wavelength at 118.6 um' in capt.out
        assert 'Saved image' in capt.out

        # test bad data with reference: will reject all frames,
        # attempt peak method, then give up
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('slice_method', 'reference')
        red.input[0][1].data[:] = np.nan
        red.step()
        capt = capsys.readouterr()
        assert 'is empty; using peak method' in capt.err
        assert 'No good data' in capt.err
        assert 'Saved image' not in capt.out

        # now return a line out of range for test data
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('slice_method', 'reference')
        red.input[0][0].header['G_WAVE_R'] = 10
        red.step()
        capt = capsys.readouterr()
        assert 'is empty; using peak method' in capt.err
        assert 'Saved image' in capt.out
        shutil.copyfile(outfile, tmpdir.join('tmp1.png'))
        assert compare_images(tmpdir.join('tmp0.png'),
                              tmpdir.join('tmp1.png'), 0) is not None

        # spatial point out of range will do same
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('point_method', 'reference')
        red.input[0][0].header['OBSRA'] = 12
        red.step()
        capt = capsys.readouterr()
        assert 'out of range; using peak pixel' in capt.err
        shutil.copyfile(outfile, tmpdir.join('tmp2.png'))
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp2.png'), 0) is None

        # test peak method
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('slice_method', 'peak')
        red.step()
        capt = capsys.readouterr()
        assert 'Plotting at S/N peak 118' in capt.out
        assert 'Saved image' in capt.out
        shutil.copyfile(outfile, tmpdir.join('tmp3.png'))
        # image should be same as in failed line method
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp3.png'), 0) is None

        # test override method
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('override_slice', 20)
        param.set_value('override_point', "[10,'10']")
        red.step()
        capt = capsys.readouterr()
        assert 'index 20' in capt.out
        assert 'pixel 10,10' in capt.out
        shutil.copyfile(outfile, tmpdir.join('tmp4.png'))
        # image should be different
        assert compare_images(tmpdir.join('tmp3.png'),
                              tmpdir.join('tmp4.png'), 0) is not None

        # bad override input - will match peak
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('override_slice', 500)
        param.set_value('override_point', "[500,'500']")
        red.step()
        capt = capsys.readouterr()
        assert 'out of range' in capt.err
        shutil.copyfile(outfile, tmpdir.join('tmp5.png'))
        assert compare_images(tmpdir.join('tmp3.png'),
                              tmpdir.join('tmp5.png'), 0) is None

        # test alternate extension option
        red = pickle.loads(red_copy)
        # modify input data x-width to smoke test plot adjustment
        new_cube = np.pad(red.input[0][1].data, ((0, 0), (0, 0), (100, 100)))
        red.input[0][3].data = new_cube
        red.input[0][4].data = new_cube
        param = red.get_parameter_set(step_index=idx)
        param.set_value('extension', 'UNCORRECTED_FLUX')
        red.step()
        assert 'Displaying UNCORRECTED_FLUX' in capsys.readouterr().out

        # test bad extension
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('extension', 'WAVELENGTH')
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Invalid extension' in str(err)

        # test bad input overrides
        red = pickle.loads(red_copy)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('override_slice', 'bad')
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Bad input' in str(err)
        param.set_value('override_slice', 10)
        param.set_value('override_point', '1,bad')
        with pytest.raises(ValueError) as err:
            red.step()
        assert 'Bad input' in str(err)

    def test_specmap_old_beam(self, tmpdir, capsys):
        step = 'specmap'
        ffile, red, idx = self.standard_setup(tmpdir, step, nfiles=1)
        param = red.get_parameter_set(step_index=idx)
        param.set_value('beam', True)

        # modify the input data to have older style beam values,
        # but mismatched units
        hdul = red.input[0]
        for hdu in hdul:
            if 'BMAJ' in hdu.header:
                hdu.header['BMAJ'] = (hdu.header['BMAJ'] * 3600,
                                      'beam (degree)')
                hdu.header['BMIN'] = (hdu.header['BMIN'] * 3600,
                                      'beam (degree)')
        # beam is too big in degree units
        red.specmap()
        assert 'Beam major, minor, angle: 14280' in capsys.readouterr().out

        # set comments to proper old-style keys with arcsec values
        for hdu in hdul:
            if 'BMAJ' in hdu.header:
                hdu.header['BMAJ'] = (hdu.header['BMAJ'],
                                      'beam (arcsec)')
                hdu.header['BMIN'] = (hdu.header['BMIN'],
                                      'beam (arcsec)')
        # this case is handled - comment units match values
        red.specmap()
        assert 'Beam major, minor, angle: 3.96' in capsys.readouterr().out
