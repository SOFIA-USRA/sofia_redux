# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FLITECAM Imaging Reduction class."""

import os
import pickle

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.reduction import Reduction
import sofia_redux.pipeline.sofia.parameters as srp
try:
    from sofia_redux.pipeline.sofia.flitecam_imaging_reduction \
        import FLITECAMImagingReduction
    from sofia_redux.pipeline.sofia.parameters.flitecam_imaging_parameters \
        import FLITECAMImagingParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FLITECAMImagingReduction = None
    FLITECAMImagingParameters = None


def bad_step(*args, **kwargs):
    return None


def bad_step_2(*args, **kwargs):
    return [None, None]


@pytest.mark.skipif('not HAS_DRIP')
class TestFLITECAMImagingReduction(object):
    @pytest.fixture(autouse=True, scope='function')
    def mock_param(self, qapp):
        # set defaults to faster options to speed up tests
        default = srp.flitecam_imaging_parameters.IMAGE_DEFAULT
        default['clip_image'][1]['value'] = True

    def make_file(self, tmpdir, nfiles=1, nplane=1,
                  extname=None, obstype=None):
        """Retrieve a basic test FITS file for FLITECAM."""
        from sofia_redux.instruments.flitecam.tests.resources \
            import raw_testdata

        ffiles = []
        for i in range(nfiles):
            dth_i = i + 1
            if i % 2 == 0:
                nod = 'A'
            else:
                nod = 'B'
            hdul = raw_testdata(dthindex=dth_i, nodbeam=nod)

            # make it look like old-style data if desired
            if nplane > 1:
                d1 = hdul[0].data.copy()
                hdul[0].data = np.array([d1] * nplane)

            # modify the extension name  or obstype if desired
            if extname is not None:
                hdul[0].header['EXTNAME'] = extname
            if obstype is not None:
                hdul[0].header['OBSTYPE'] = obstype

            fname = hdul[0].header['FILENAME']
            tmpfile = tmpdir.join(fname)
            ffile = str(tmpfile)
            hdul.writeto(ffile, overwrite=True)
            hdul.close()
            ffiles.append(ffile)

        return ffiles

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FLITECAMImagingReduction()
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
        red = FLITECAMImagingReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)[0]

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FLITECAMImagingParameters)

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

    def test_load_intermediate(self, tmpdir):
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir, nplane=3)[0]

        # Set prodtype -- now looks like an old-style
        # intermediate file with variance and expmap
        fits.setval(ffile, 'PRODTYPE', value='registered')
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

        # set to last type and verify error is raised
        fits.setval(ffile, 'PRODTYPE', value='imgmap')
        with pytest.raises(ValueError) as err:
            red.load(ffile)
        assert 'No steps to run' in str(err)

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        red = FLITECAMImagingReduction()
        vz = red.register_viewers()
        assert len(vz) == 1
        assert isinstance(vz[0], QADViewer)

    def test_filenum(self):
        red = FLITECAMImagingReduction()

        # from filenames
        fn = red.getfilenum('test.fits')
        assert fn == 'UNKNOWN'
        fn = red.getfilenum('test_data.fits')
        assert fn == 'UNKNOWN'
        fn = red.getfilenum('test_1.fits')
        assert fn == '0001'
        fn = red.getfilenum('test_1.a.fits')
        assert fn == '0001'
        fn = red.getfilenum('F0001_FC_IMA_90000101_FLTJ_RAW_0001.fits')
        assert fn == '0001'

        # test if not list, string and number
        fn = 1
        assert red._catfilenum(fn) == '1'
        fn = '1'
        assert red._catfilenum(fn) == '1'

        # test if list
        fn = [1, 2]
        assert red._catfilenum(fn) == '1-2'
        fn = ['1', '2']
        assert red._catfilenum(fn) == '1-2'

        # test if list of list
        fn = [[1, 2], [3, 4]]
        assert red._catfilenum(fn) == '1-4'
        fn = [['1', '2'], [3, 4]]
        assert red._catfilenum(fn) == '1-4'

    def test_filename(self, tmpdir):
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)[0]

        # test unknown values before loading file
        fn = red.getfilename(fits.header.Header())
        assert fn == 'UNKNOWN_FC_IMA_UNKNOWN_UNKNOWN_RAW_UNKNOWN.fits'

        red.load(ffile)
        red.load_fits()

        # test default
        header = red.input[0][0].header
        fn = red.getfilename(header)
        assert fn == 'F0146_FC_IMA_90000101_FLTH_RAW_UNKNOWN.fits'
        fn = red.getfilename(header, filenum='0001')
        assert fn == 'F0146_FC_IMA_90000101_FLTH_RAW_0001.fits'

        # test prodtype and update
        fn = red.getfilename(header, filenum='0001', prodtype='PT1',
                             update=True)
        assert fn == 'F0146_FC_IMA_90000101_FLTH_PT1_0001.fits'
        assert header['FILENAME'] == fn
        fn2 = red.getfilename(header, filenum='0001', prodtype='PT2',
                              update=False)
        assert fn2 == 'F0146_FC_IMA_90000101_FLTH_PT2_0001.fits'
        assert header['FILENAME'] == fn

        # test other conditions
        badhdr = header.copy()
        badhdr['MISSN-ID'] = '2018-12-31_FO_FBAD'
        fn = red.getfilename(badhdr)
        assert fn == 'UNKNOWN_FC_IMA_90000101_FLTH_RAW_UNKNOWN.fits'

    def test_display_data(self, tmpdir):
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)[0]

        # test for raw data
        red.load(ffile)
        red.set_display_data(raw=True)
        assert red.display_data == {'QADViewer': [ffile]}

        red.load_fits()
        red.set_display_data(raw=False)
        assert len(red.display_data['QADViewer']) == 1
        assert isinstance(red.display_data['QADViewer'][0], fits.HDUList)

    def test_update_output(self, tmpdir):
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir, nplane=3)[0]
        red.load(ffile)
        red.load_fits()
        hdul = red.input[0]
        fn = red.update_output(hdul, ['0001', '0002'], 'clipped')
        assert '0001-0002' in fn
        assert 'CLP' in fn
        assert '-- Pipeline step: Clip Image' in hdul[0].header['HISTORY']
        assert hdul[0].header['PRODTYPE'] == 'clipped'

    def test_write_output(self, tmpdir):
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)[0]
        red.load(ffile)
        red.load_fits()
        hdul = red.input[0]

        red.output_directory = tmpdir
        outname = 'new_name.fits'
        outpath = str(tmpdir.join(outname))
        red.write_output(hdul, outname)
        assert os.path.isfile(outpath)
        assert outpath in red.out_files

    def test_all_steps(self, tmpdir):
        # exercises nominal behavior for a standard reduction
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir, nfiles=4, obstype='OBJECT')
        red.load(ffile)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step == 'check_header' or step == 'imgmap':
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

    @pytest.mark.parametrize('step',
                             ['correct_linearity',
                              'clip_image', 'make_flat', 'correct_gain',
                              'subtract_sky'])
    def test_step_minimal(self, tmpdir, capsys, step):
        """
        Test standard behavior for multiple steps.

        Exercises save option.
        """
        ffile, red, idx = self.standard_setup(tmpdir, step, nfiles=4)

        # copy the reduction just before step
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

        # test save
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        red.step()

        fn = None
        for fn in red.out_files:
            if red.prodnames[red.prodtype_map[step]] in fn:
                break
        assert os.path.isfile(fn)

    def test_check_header(self, tmpdir, capsys):
        # check that fails appropriately for bad header
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)[0]
        red.output_directory = tmpdir

        # passes
        red.load(ffile)
        red.load_parameters()
        parset = red.parameters.current[0]
        parset['abort']['value'] = True
        red.check_header()

        # logs error
        fits.setval(ffile, 'SPECTEL2', value='BADVAL')
        red.load(ffile)
        red.load_parameters()
        parset = red.parameters.current[0]
        parset['abort']['value'] = True

        red.check_header()
        capt = capsys.readouterr()
        assert 'Invalid headers' in capt.err
        # input is cleared
        assert len(red.input) == 0

        # raises an error when input is empty
        with pytest.raises(RuntimeError) as err:
            red.step()
        assert 'No input' in str(err)

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for flitecam
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)[0]
        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'config')
        assert pcopy.config is not None

        assert hasattr(pcopy, 'pipecal_config')
        assert pcopy.pipecal_config is not None

    def test_bunit(self, tmpdir):
        # run through all steps, checking for appropriate BUNIT keys
        red = FLITECAMImagingReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps, checking bunit
        bunit = 'ct'
        exp_unit = 's'
        bpm_unit = ''
        for step in red.recipe:
            red.step()
            if step == 'check_head':
                continue
            elif step == 'correct_linearity':
                bunit = 'ct/s'
            elif step == 'fluxcal':
                bunit = 'Jy/pixel'
            hdul = red.input[0]
            for hdu in hdul:
                print(hdu.header['EXTNAME'], hdu.header['BUNIT'])
                if hdu.header['EXTNAME'] == 'EXPOSURE':
                    assert hdu.header['BUNIT'] == exp_unit
                elif hdu.header['EXTNAME'] == 'BADMASK':
                    assert hdu.header['BUNIT'] == bpm_unit
                else:
                    assert hdu.header['BUNIT'] == bunit

    def test_lincor(self, tmpdir):
        ffile, red, idx = self.standard_setup(tmpdir, 'correct_linearity',
                                              nfiles=1)
        parset = red.parameters.current[idx]

        # no saturation can be specified as ''
        parset['saturation']['value'] = ''
        red.correct_linearity()
        assert np.all(red.input[0]['BADMASK'].data == 0)

        # error if linfile is unset or badly specified
        parset['linfile']['value'] = ''
        with pytest.raises(ValueError) as err:
            red.correct_linearity()
        assert 'No linearity file' in str(err)

        parset['linfile']['value'] = 'badfile.fits'
        with pytest.raises(ValueError) as err:
            red.correct_linearity()
        assert 'No linearity file' in str(err)

    def test_clip_image(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'clip_image',
                                              nfiles=1)
        red_copy = pickle.dumps(red)
        parset = red.parameters.current[idx]
        parset['skip_clean']['value'] = True
        red.clip_image()
        assert 'Skipping bad pixel' in capsys.readouterr().out
        f1 = red.input[0][0].data

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['skip_clean']['value'] = False
        red.clip_image()
        capt = capsys.readouterr()
        assert 'Skipping bad pixel' not in capt.out
        assert 'hot pixels' in capt.out
        assert 'cold pixels' in capt.out
        f2 = red.input[0][0].data
        assert np.sum(np.isnan(f1)) < np.sum(np.isnan(f2))

    def test_mkflat(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'make_flat',
                                              nfiles=3)
        red_copy = pickle.dumps(red)

        # skip flat: allowed, but data is not gain corrected
        # in the next step
        parset = red.parameters.current[idx]
        parset['skip_flat']['value'] = True
        red.step()
        assert 'Skipping flat generation' in capsys.readouterr().out
        assert 'FLAT' not in red.input[0]
        red.step()
        assert 'No FLAT extension present; ' \
               'not correcting data' in capsys.readouterr().err

        # make flat from source
        red = pickle.loads(red_copy)
        for hdul in red.input:
            hdul[0].header['OBSTYPE'] = 'OBJECT'
        red.step()
        assert 'FLAT' in red.input[0]
        assert capsys.readouterr().out.count('Using remaining source') == 3

        # check that current obs is not used in making the
        # flat but the others are
        for i, hdul in enumerate(red.input):
            obs = hdul['FLAT'].header['ASSC_OBS']
            for j in range(3):
                if i == j:
                    assert f'FL000{j + 1}' not in obs
                else:
                    assert f'FL000{j + 1}' in obs

        # make a separate flat file to use
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]

        # bad file: no 'FLAT' extension
        flat = self.make_file(tmpdir)[0]
        parset['flatfile']['value'] = flat
        with pytest.raises(ValueError):
            red.make_flat()

        # good file
        flat = self.make_file(tmpdir, extname='FLAT')[0]
        parset['flatfile']['value'] = flat
        red.step()
        assert 'FLAT' in red.input[0]
        assert 'Using previously generated ' \
               'flat file' in capsys.readouterr().out

    def test_subtract_sky(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'make_flat',
                                              nfiles=2)
        # modify input to make one sky, one source
        red.input[0][0].header['OBSTYPE'] = 'OBJECT'
        red.input[1][0].header['OBSTYPE'] = 'SKY'

        # step through flat and gaincor
        red.step()
        red.step()

        # keep starting data
        red_copy = pickle.dumps(red)
        idx += 2
        f1 = red.input[0][0].data
        f2 = red.input[1][0].data

        # skip sky: allowed
        parset = red.parameters.current[idx]
        parset['skip_sky']['value'] = True
        red.step()
        assert 'Skipping sky subtraction' in capsys.readouterr().out
        assert np.allclose(red.input[0][0].data, f1, equal_nan=True)
        # sky files not dropped
        assert len(red.input) == 2

        # test median and flatnorm methods
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['sky_method']['value'] = 'flatnorm'
        for hdul in red.input:
            hdul[0].header['FLATNORM'] = 1000
        red.step()
        assert np.allclose(red.input[0][0].data, f1 - 1000, equal_nan=True)
        # sky files dropped
        assert len(red.input) == 1

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['sky_method']['value'] = 'median'
        red.step()
        assert np.allclose(red.input[0][0].data, f1 - np.nanmedian(f1),
                           equal_nan=True)
        # sky files dropped
        assert len(red.input) == 1

        # only sky file: is propagated
        red = pickle.loads(red_copy)
        red.input = [red.input[1]]
        parset = red.parameters.current[idx]
        parset['sky_method']['value'] = 'median'
        red.step()
        assert np.allclose(red.input[0][0].data, f2 - np.nanmedian(f2),
                           equal_nan=True)
        assert len(red.input) == 1
        assert 'Only sky files are present' in capsys.readouterr().err

        # make a separate sky file to use
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]

        # bad file: no 'FLUX' extension
        sky = self.make_file(tmpdir, extname='BAD')[0]
        parset['skyfile']['value'] = sky
        with pytest.raises(ValueError) as err:
            red.subtract_sky()
        assert 'Bad sky file' in str(err)

        # good file
        sky = str(tmpdir.join('skyfile.fits'))
        hdul = fits.HDUList(
            fits.PrimaryHDU(np.full_like(f1, 2000),
                            header=fits.Header({'EXTNAME': 'FLUX'})))
        hdul.writeto(sky, overwrite=True)
        hdul.close()
        parset['skyfile']['value'] = sky
        red.step()
        assert 'Using previously generated ' \
               'sky file' in capsys.readouterr().out
        assert np.allclose(red.input[0][0].data, f1 - 2000,
                           equal_nan=True)
        # sky file still dropped
        assert len(red.input) == 1
