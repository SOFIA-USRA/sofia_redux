# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Imaging Reduction class."""

import os
import pickle
import shutil

from astropy.io import fits
from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.toolkit.utilities.fits import set_log_level
try:
    from sofia_redux.pipeline.sofia.forcast_imaging_reduction \
        import FORCASTImagingReduction
    from sofia_redux.pipeline.sofia.parameters.forcast_imaging_parameters \
        import FORCASTImagingParameters
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FORCASTImagingReduction = None
    FORCASTImagingParameters = None


def bad_step(*args, **kwargs):
    return None


def bad_step_2(*args, **kwargs):
    return [None, None]


@pytest.mark.skipif('not HAS_DRIP')
class TestFORCASTImagingReduction(object):
    def make_file(self, tmpdir, fname='bFT001_0001.fits'):
        """Retrieve a basic test FITS file for FORCAST."""
        from sofia_redux.instruments.forcast.tests.resources \
            import raw_testdata
        hdul = raw_testdata()

        tmpfile = tmpdir.join(fname)
        ffile = str(tmpfile)
        hdul.writeto(ffile, overwrite=True)
        hdul.close()

        return ffile

    def standard_setup(self, tmpdir, step, nfiles=1):
        red = FORCASTImagingReduction()
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
        red = FORCASTImagingReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, FORCASTImagingParameters)

        # test updated keys
        header = red.input[0][0].header
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert header['OBS_ID'].startswith('P_')
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_load_intermediate(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        # Set prodtype -- now looks like an old-style
        # intermediate file with variance and expmap
        fits.setval(ffile, 'PRODTYPE', value='merged')
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

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        red = FORCASTImagingReduction()
        vz = red.register_viewers()
        assert len(vz) == 1
        assert isinstance(vz[0], QADViewer)

    def test_filenum(self):
        red = FORCASTImagingReduction()

        fn = red.getfilenum('test.fits')
        assert fn == 'UNKNOWN'
        fn = red.getfilenum('test_data.fits')
        assert fn == 'UNKNOWN'
        fn = red.getfilenum('test_1.fits')
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

        # test if list of list of list
        fn = [[[1, 2], [3, 4]], [5, 6], 7]
        assert red._catfilenum(fn) == '1-7'

    def test_filename(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        # test error before loading file
        with pytest.raises(ValueError):
            red.getfilename(fits.header.Header())

        red.load(ffile)
        red.load_fits()

        # test default
        header = red.input[0][0].header
        fn = red.getfilename(header)
        assert fn == 'F0001_FO_IMA_90000101_FORF197_RAW_UNKNOWN.fits'
        fn = red.getfilename(header)
        assert fn == 'F0001_FO_IMA_90000101_FORF197_RAW_UNKNOWN.fits'
        fn = red.getfilename(header, filenum='0001')
        assert fn == 'F0001_FO_IMA_90000101_FORF197_RAW_0001.fits'

        # test prodtype and update
        fn = red.getfilename(header, filenum='0001', prodtype='PT1',
                             update=True)
        assert fn == 'F0001_FO_IMA_90000101_FORF197_PT1_0001.fits'
        assert header['FILENAME'] == fn
        fn2 = red.getfilename(header, filenum='0001', prodtype='PT2',
                              update=False)
        assert fn2 == 'F0001_FO_IMA_90000101_FORF197_PT2_0001.fits'
        assert header['FILENAME'] == fn

        # test other conditions
        badhdr = header.copy()
        badhdr['MISSN-ID'] = '2018-12-31_FO_FBAD'
        fn = red.getfilename(badhdr)
        assert fn == 'UNKNOWN_FO_IMA_90000101_FORF197_RAW_UNKNOWN.fits'

        red.calres['gmode'] = 0
        fn = red.getfilename(badhdr)
        assert fn == 'UNKNOWN_FO_GRI_90000101_FORF197_RAW_UNKNOWN.fits'

    def test_display_data(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        # test for raw data
        red.load(ffile)
        red.set_display_data(raw=True)
        assert red.display_data == {'QADViewer': [ffile]}

        red.load_fits()
        red.set_display_data(raw=False)
        assert len(red.display_data['QADViewer']) == 1
        assert isinstance(red.display_data['QADViewer'][0], fits.HDUList)

    def test_update_output(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.load_fits()
        hdul = red.input[0]
        fn = red.update_output(hdul, ['0001', '0002'], 'cleaned')
        assert '0001-0002' in fn
        assert 'CLN' in fn
        assert '-- Pipeline step: Clean Images' in hdul[0].header['HISTORY']
        assert hdul[0].header['PRODTYPE'] == 'cleaned'

    def test_write_output(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
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
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step == 'checkhead' or step == 'imgmap':
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

    def test_slitimage(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        # modify to make it a slit image
        fits.setval(ffile, 'SLIT', value='FOR_LS24')

        # load and check recipe
        red.load(ffile)
        red.load_fits()
        assert red.recipe == red.slit_image_recipe
        assert len(red.prodtypes) == len(red.recipe)

        # test intermediate
        fits.setval(ffile, 'PRODTYPE', value='cleaned')
        red.load(ffile)
        assert red.recipe == red.slit_image_recipe[2:]
        assert len(red.prodtypes) == len(red.recipe)

        # test error for no further steps
        fits.setval(ffile, 'PRODTYPE', value='merged')
        with pytest.raises(ValueError):
            red.load(ffile)

    def test_mosaic(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        # modify to make a TEL/CAL image

        # TEL: use full recipe
        fits.setval(ffile, 'PRODTYPE', value='telluric_corrected')
        red.load(ffile)
        assert red.recipe == red.mosaic_recipe
        assert len(red.prodtypes) == len(red.recipe)

        # CAL: use intermediate recipe (skip calibration)
        fits.setval(ffile, 'PRODTYPE', value='calibrated')
        red.load(ffile)
        assert red.recipe == red.mosaic_recipe[1:]
        assert len(red.prodtypes) == len(red.recipe)

    def test_c2nc2(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)

        # modify to make old-style C2NC2 images
        basename = os.path.splitext(os.path.basename(ffile))[0]
        nnames = [str(tmpdir.join(basename + '_{}.fits'.format(i + 1)))
                  for i in range(8)]

        hdul = fits.open(ffile)
        hdul[0].data = hdul[0].data[0:2, :, :]
        hdul[0].header['SKYMODE'] = 'C2NC2'

        for i in range(8):
            hdul[0].header['DTHINDEX'] = i + 1
            hdul.writeto(nnames[i], overwrite=True)
        hdul.close()

        # load files -- will be automatically reorganized
        # to C2NC4 style
        red.load(nnames)
        red.load_parameters()
        red.load_fits()
        assert len(red.raw_files) == 8
        assert len(red.input) == 5

        # filenum is now list of lists, combining ABAABAAB
        # with shared Bs
        fnum = [['0001', '0002'],
                ['0002', '0003'],
                ['0004', '0005'],
                ['0005', '0006'],
                ['0007', '0008']]
        assert red.filenum == fnum

    def test_filter_shift(self, tmpdir):
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)

        orig_crpix = [fits.getval(ffile, 'CRPIX1'),
                      fits.getval(ffile, 'CRPIX2')]

        # modify calres to add a pixel shift
        x, y = 10.1, 15
        red.calres['pixshiftx'] = x
        red.calres['pixshifty'] = y

        red.load_fits()
        hdul = red.input[0]
        assert hdul[0].header['CRPIX1'] - orig_crpix[0] - x < 1e-5
        assert hdul[0].header['CRPIX2'] - orig_crpix[1] - y < 1e-5

    @pytest.mark.parametrize('step',
                             ['clean', 'droop', 'stack', 'merge'])
    def test_step_minimal(self, tmpdir, capsys, mocker, step):
        """
        Test standard behavior for multiple steps.

        Exercises save and general error condition.
        """
        ffile, red, idx = self.standard_setup(tmpdir, step)

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

        # test error in step
        mocker.patch(
            'sofia_redux.instruments.forcast.{}.{}'.format(step, step),
            bad_step)
        red = pickle.loads(red_copy)
        red.step()
        capt = capsys.readouterr()
        assert 'Problem' in capt.err
        assert step in capt.err

    def test_checkhead(self, tmpdir, capsys):
        # check that fails appropriately for bad header
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
        red.output_directory = tmpdir

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
        # input is cleared
        assert len(red.input) == 0

        # raises an error when input is empty
        with pytest.raises(RuntimeError) as err:
            red.step()
        assert 'No input' in str(err)

    def test_clean(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'clean')

        # copy the reduction just before clean
        red_copy = pickle.dumps(red)

        # check readout shift parameters

        # shift first file
        parset = red.parameters.current[idx]
        parset['shiftfile']['value'] = '1'
        red.clean()
        capt = capsys.readouterr()
        assert 'Shifting file 1' in capt.out

        # reset; shift all
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['shiftfile']['value'] = 'all'
        red.clean()
        capt = capsys.readouterr()
        assert 'Shifting file 1' in capt.out

        # reset; try bad shifts
        red = pickle.loads(red_copy)

        parset = red.parameters.current[idx]
        parset['shiftfile']['value'] = '1;2'
        red.clean()
        capt = capsys.readouterr()
        assert 'out of range' in capt.err

        parset['shiftfile']['value'] = 'abc'
        red.clean()
        capt = capsys.readouterr()
        assert 'must be semicolon-separated integers' in capt.err

        # now test missing bad pix file
        parset['shiftfile']['value'] = ''
        parset['badfile']['value'] = ''
        red.clean()
        capt = capsys.readouterr()
        assert 'No bad pixel file' in capt.err

    def test_nonlin(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'nonlin')

        # test background section error
        parset = red.parameters.current[idx]
        sc = parset['secctr']['value']
        parset['secctr']['value'] = 'BADVAL'
        red.nonlin()
        capt = capsys.readouterr()
        assert 'Invalid background' in capt.err

        # reset good section
        parset['secctr']['value'] = sc

        # test save and error in step
        # (should carry on with input data)
        parset['save']['value'] = True
        mocker.patch(
            'sofia_redux.instruments.forcast.imgnonlin.imgnonlin',
            bad_step)
        red.nonlin()
        # no error
        capt = capsys.readouterr()
        assert capt.err == ''
        # file saved
        assert os.path.isfile(red.out_files[0])

    def test_stack(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'stack')

        # copy the reduction just before step
        red_copy = pickle.dumps(red)

        # test background section error
        parset = red.parameters.current[idx]
        sc = parset['secctr']['value']
        parset['secctr']['value'] = 'BADVAL'
        red.stack()
        capt = capsys.readouterr()
        assert 'Invalid background' in capt.err

        # reset good section
        parset['secctr']['value'] = sc

        # test jbclean on
        parset = red.parameters.current[idx]
        parset['jbclean']['value'] = True
        red.stack()
        capt = capsys.readouterr()
        assert 'Jailbars cleaned' in capt.out

        # test jbclean off
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['jbclean']['value'] = False
        red.stack()
        capt = capsys.readouterr()
        assert 'Jailbars not removed' in capt.out

        # test add frames
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['add_frames']['value'] = True
        red.stack()
        capt = capsys.readouterr()
        assert 'All frames added' in capt.out

    def test_undistort(self, tmpdir, capsys, mocker):
        step = 'undistort'
        ffile, red, idx = self.standard_setup(tmpdir, step)

        # copy the reduction just before the step
        red_copy = pickle.dumps(red)

        # test missing bad pix file
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['pinfile']['value'] = ''
        red.undistort()
        capt = capsys.readouterr()
        assert 'No pinhole file' in capt.err

        # test error in step
        mocker.patch(
            'sofia_redux.instruments.forcast.{}.{}'.format(step, step),
            bad_step)
        red = pickle.loads(red_copy)
        red.step()
        capt = capsys.readouterr()
        assert 'Problem in ' \
            'sofia_redux.instruments.forcast.{}'.format(step) in capt.err

    def test_register(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'register',
                                              nfiles=2)

        # copy the reduction just before step
        red_copy = pickle.dumps(red)

        # test offset parameters
        parset = red.parameters.current[idx]
        parset['offsets']['value'] = 'BADVAL'
        red.register()
        capt = capsys.readouterr()
        assert 'does not match number of images' in capt.err

        parset['offsets']['value'] = 'BADVAL1;BADVAL2'
        red.register()
        capt = capsys.readouterr()
        assert 'Must provide valid' in capt.err

        # test good offsets
        parset['offsets']['value'] = '0,0;10,10'
        red.register()
        capt = capsys.readouterr()
        assert 'CRPIX shifts used:' in capt.out
        assert '0.00,0.00;10.00,10.00' in capt.out

        # test error in step
        mocker.patch('sofia_redux.instruments.forcast.'
                     'register_datasets.get_shifts', bad_step_2)
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        # set to non-wcs -- otherwise get_shifts is not called
        parset['corcoadd']['option_index'] = 2
        red.register()
        capt = capsys.readouterr()
        assert 'Failed to register dataset' in capt.err
        assert '0.00,0.00;0.00,0.00' in capt.out

    def test_register_header(self, tmpdir, capsys):
        ffile, red, idx = self.standard_setup(tmpdir, 'register',
                                              nfiles=2)
        parset = red.parameters.current[idx]
        parset.set_value('corcoadd', 'Header shifts')

        hdr = red.input[0][0].header
        crpix1 = hdr['CRPIX1']
        crpix2 = hdr['CRPIX2']

        with set_log_level('DEBUG'):
            red.step()
        capt = capsys.readouterr()
        assert 'Delta CRPIX' in capt.out

        assert np.allclose(crpix1 - red.input[0][0].header['CRPIX1'],
                           -0.37, atol=.01)
        assert np.allclose(crpix2 - red.input[0][0].header['CRPIX2'],
                           -1.37, atol=.01)

    @pytest.mark.parametrize('step',
                             ['undistort', 'tellcor', 'fluxcal',
                              'nonlin', 'register'])
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

    def test_coadd(self, tmpdir):
        ffile, red, idx = self.standard_setup(tmpdir, 'coadd',
                                              nfiles=2)
        parset = red.parameters.current[idx]
        parset['method']['value'] = 'mean'
        red_copy = pickle.dumps(red)

        # test save off
        parset = red.parameters.current[idx]
        parset['save']['value'] = False
        red.coadd()
        saw = 0
        for fn in red.out_files:
            if red.prodnames[red.prodtype_map['coadd']] in fn:
                assert os.path.isfile(fn)
                saw += 1
        assert saw == 0

        # test save off and coadd off
        parset = red.parameters.current[idx]
        parset['save']['value'] = False
        parset['skip_coadd']['value'] = True
        red.coadd()
        saw = 0
        for fn in red.out_files:
            if red.prodnames[red.prodtype_map['coadd']] in fn:
                assert os.path.isfile(fn)
                saw += 1
        assert saw == 0

        # test save on
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        red.coadd()
        saw = 0
        for fn in red.out_files:
            if red.prodnames[red.prodtype_map['coadd']] in fn:
                assert os.path.isfile(fn)
                saw += 1
        assert saw == 1

        # test save off and mosaic mode
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['save']['value'] = False
        for hdul in red.input:
            hdul[0].header['PROCSTAT'] = 'LEVEL_3'
        red.coadd()
        saw = 0
        for fn in red.out_files:
            if 'MOS' in fn:
                assert os.path.isfile(fn)
                assert fits.getval(fn, 'PRODTYPE') == 'mosaic'
                saw += 1
        assert saw == 0

        # test save on and mosaic mode
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['save']['value'] = True
        for hdul in red.input:
            hdul[0].header['PROCSTAT'] = 'LEVEL_3'
        red.coadd()
        saw = 0
        for fn in red.out_files:
            if 'MOS' in fn:
                assert os.path.isfile(fn)
                assert fits.getval(fn, 'PRODTYPE') == 'mosaic'
                saw += 1
        assert saw == 1

    def test_coadd_reference(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'coadd',
                                              nfiles=2)

        # skip photometry for this case
        red.calres['obstype'] = 'OBJECT'

        # add some header keywords relevant to registration
        # and make data small for faster processing
        for i, inp in enumerate(red.input):
            inp[0].header['TGTRA'] = (10.0 + i / 3600 / 0.768) / 15.
            inp[0].header['TGTDEC'] = 10.0 + i / 3600 / 0.768
            inp[0].data = np.full((20, 20), 1.0 + i)
            inp[1].data = np.full((20, 20), 1.0 + i)

        parset = red.parameters.current[idx]
        red_copy = pickle.dumps(red)

        parset['reference']['value'] = 'first'
        red.coadd()
        assert 'Using first image' in capsys.readouterr().out
        r1 = red.input[0].copy()

        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['reference']['value'] = 'target'
        red.coadd()
        assert 'Correcting for target motion' in capsys.readouterr().out
        r2 = red.input[0].copy()

        # all reference to the same values
        assert r2[0].header['CRVAL1'] == r1[0].header['CRVAL1']
        assert r2[0].header['CRVAL2'] == r1[0].header['CRVAL2']

        # output shapes are different
        assert r1[0].data.shape == (21, 21)
        assert r2[0].data.shape == (22, 23)

    def test_fluxcal(self, tmpdir, capsys, mocker):
        ffile, red, idx = self.standard_setup(tmpdir, 'fluxcal')
        red_copy = pickle.dumps(red)

        # test do_phot and srcpos parameter
        parset = red.parameters.current[idx]
        parset['rerun_phot']['value'] = True
        parset['srcpos']['value'] = 'BADVAL'
        red.fluxcal()
        capt = capsys.readouterr()
        assert 'Invalid source position' in capt.err

        # set a valid source position; photometry should run
        hdr = red.input[0][0].header
        parset['srcpos']['value'] = ','.join([str(hdr['SRCPOSX']),
                                              str(hdr['SRCPOSY'])])
        red.fluxcal()
        capt = capsys.readouterr()
        assert 'Source Position (x,y)' in capt.out

        # test missing fluxcal keys: should not throw error
        def rm_modlflx(data, header, variance, config, **kwargs):
            print('rm modl')
            del header['MODLFLX']

        def rm_stapflx(data, header, variance, config, **kwargs):
            print('rm stap')
            del header['STAPFLX']

        mocker.patch(
            'sofia_redux.calibration.pipecal_util.run_photometry',
            rm_modlflx)
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['rerun_phot']['value'] = True
        red.fluxcal()
        capt = capsys.readouterr()
        assert 'Source Flux' in capt.out
        assert 'Model Flux' not in capt.out

        mocker.patch(
            'sofia_redux.calibration.pipecal_util.run_photometry',
            rm_stapflx)
        red = pickle.loads(red_copy)
        parset = red.parameters.current[idx]
        parset['rerun_phot']['value'] = True
        red.fluxcal()
        capt = capsys.readouterr()
        assert 'Source Flux' not in capt.out
        assert 'Model Flux' not in capt.out

        # test missing cal factor: should not throw error, data is not updated
        def no_calfac(*args, **kwargs):
            return {}
        mocker.patch(
            'sofia_redux.calibration.pipecal_config.pipecal_config', no_calfac)
        red = pickle.loads(red_copy)
        compare = red.input[0].copy()
        red.fluxcal()
        capt = capsys.readouterr()
        assert 'No reference flux' in capt.err
        assert 'Source Flux' not in capt.out
        assert np.allclose(red.input[0][0].data, compare[0].data,
                           equal_nan=True)
        assert red.input[0][0].header['PROCSTAT'] == 'LEVEL_2'

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary for forcast
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'drip_cal_config')
        assert pcopy.drip_cal_config is not None

        assert hasattr(pcopy, 'drip_config')
        assert pcopy.drip_config is not None

        assert hasattr(pcopy, 'pipecal_config')
        assert pcopy.pipecal_config is not None

    def test_bunit(self, tmpdir):
        # run through all steps, checking for appropriate BUNIT keys
        red = FORCASTImagingReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps, checking bunit
        bunit = 'ct'
        exp_unit = 's'
        for step in red.recipe:
            red.step()
            if step == 'checkhead':
                continue
            elif step == 'stack':
                bunit = 'Me/s'
            elif step == 'fluxcal':
                bunit = 'Jy/pixel'
            hdul = red.input[0]
            for hdu in hdul:
                if hdu.header['EXTNAME'] == 'EXPOSURE':
                    assert hdu.header['BUNIT'] == exp_unit
                else:
                    assert hdu.header['BUNIT'] == bunit

    def test_imgmap(self, tmpdir, capsys):
        step = 'imgmap'
        ffile, red, idx = self.standard_setup(tmpdir, step, nfiles=1)

        # make sure beam marker is added
        parset = red.parameters.current[idx]
        parset.set_value('beam', True)

        # expected image file name
        outfile = tmpdir.join('F0001_FO_IMA_90000101_FORF197_CAL_0001.png')
        red.imgmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp0.png'))

        # delete beam keys -- beam is generated by default anyway
        for key in ['BMAJ', 'BMIN', 'BPA']:
            del red.input[0][0].header[key]
        red.imgmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp1.png'))

        # output image should be the same
        assert compare_images(tmpdir.join('tmp0.png'),
                              tmpdir.join('tmp1.png'), 0) is None

        # unless beam is off
        parset.set_value('beam', False)
        red.imgmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp2.png'))
        assert compare_images(tmpdir.join('tmp0.png'),
                              tmpdir.join('tmp2.png'), 0) is not None

    def test_coadd_slitimage(self):
        # parameter config has to include cal flags
        param = FORCASTImagingParameters()

        # exercise some non-default parameters for coadd
        param.add_current_parameters('coadd')
        param.add_current_parameters('coadd')
        param.drip_config = {}

        # with slit boresight: coadd off by default
        param.drip_cal_config = {'boresight': 'LONGSLIT'}
        param.coadd(0)
        assert param.current[0].get_value('skip_coadd') is True

        # without: coadd on by default
        param.drip_cal_config = {'boresight': 'OTHER'}
        param.coadd(1)
        assert param.current[1].get_value('skip_coadd') is False
