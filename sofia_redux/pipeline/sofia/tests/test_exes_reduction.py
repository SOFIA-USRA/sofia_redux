# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
import os
import pickle
import re
import shutil

from astropy.io import fits
from matplotlib.testing.compare import compare_images
import numpy as np
import numpy.testing as npt
import pytest

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer
from sofia_redux.pipeline.parameters import Parameters
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.toolkit.image.adjust import rotate90
from sofia_redux.toolkit.utilities.fits import set_log_level
from sofia_redux.visualization.redux_viewer import EyeViewer

try:
    from sofia_redux.pipeline.sofia.exes_reduction \
        import EXESReduction
    from sofia_redux.pipeline.sofia.parameters.exes_parameters \
        import EXESParameters
    from sofia_redux.instruments.exes.tests import resources
    from sofia_redux.instruments.exes.wavecal import wavecal
    HAS_EXES = True
except ImportError:
    HAS_EXES = False
    EXESReduction = Reduction
    EXESParameters = Parameters
    resources = None
    wavecal = None

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.fixture
def exes_combspec(tmpdir):
    fname = 'F0001_EX_SPE_12345_EXES_CMB_001.fits'
    header = fits.Header({'INSTRUME': 'EXES',
                          'PRODTYPE': 'combspec',
                          'INSTCFG': 'HI-MED',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(
        data=np.arange(500, dtype=float).reshape((10, 5, 10)), header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.fixture
def exes_mrgordspec(tmpdir):
    fname = 'F0001_EX_SPE_12345_EXES_MRD_001.fits'
    header = fits.Header({'INSTRUME': 'EXES',
                          'PRODTYPE': 'mrgordspec',
                          'INSTCFG': 'HI-MED',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(
        data=np.arange(50, dtype=float).reshape((5, 10)),
        header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.mark.skipif('not HAS_EXES')
class TestExesReduction(object):

    def make_file(self, tmpdir, fname='test.fits'):
        if 'flat' in fname:
            header = resources.low_flat_header()
        elif 'dark' in fname:
            header = resources.low_flat_header()
            header['OBSTYPE'] = 'DARK'
        else:
            header = resources.low_header()
        header['FILENAME'] = fname

        hdul = fits.HDUList(fits.PrimaryHDU(np.ones((10, 10)),
                                            header=header))
        filename = str(tmpdir.join(fname))
        hdul.writeto(filename, overwrite=True)
        hdul.close()
        return filename

    def make_intermediate_file(self, tmpdir, fname='test.fits', nz=None,
                               match_mask=False, merge_flat=False):
        header = resources.low_header(coadded=True)
        header['FILENAME'] = fname
        header['EXTNAME'] = 'FLUX'
        header['NSPAT'] = 10
        header['NSPEC'] = 10
        shape = (10, 10)
        if nz is None:
            zshape = shape
        else:
            zshape = (nz, 10, 10)

        if merge_flat:
            flat_header = resources.single_order_flat_header()
            flat_header['NSPAT'] = 10
            flat_header['NSPEC'] = 10
            header.update(flat_header)

        hdul = fits.HDUList(fits.PrimaryHDU(np.full(zshape, 1.0),
                                            header=header))
        hdul.append(fits.ImageHDU(np.full(zshape, 0.1), name='ERROR'))
        if match_mask:
            hdul.append(fits.ImageHDU(np.full(zshape, 0), name='MASK'))
        else:
            hdul.append(fits.ImageHDU(np.full(shape, 0), name='MASK'))
        hdul.append(fits.ImageHDU(np.full(shape, 1.0), name='FLAT',
                                  header=header))
        hdul.append(fits.ImageHDU(np.full(shape, 0), name='FLAT_ERROR'))
        hdul.append(fits.ImageHDU(np.full(shape, 1), name='FLAT_ILLUMINATION'))
        hdul.append(fits.ImageHDU(np.full(shape, 0.0), name='DARK'))
        hdul.append(fits.ImageHDU(np.full(shape, 1.0), name='WAVECAL'))
        hdul.append(fits.ImageHDU(np.full(shape, 1.0), name='SPATCAL'))
        hdul.append(fits.ImageHDU(np.full(shape, 1), name='ORDER_MASK'))

        filename = str(tmpdir.join(fname))
        hdul.writeto(filename, overwrite=True)
        hdul.close()
        return filename

    def make_spec_file(self, tmpdir, fname='test.fits',
                       add_appos=False, add_apmask=False):
        ny, nx = 200, 200
        header = resources.cross_dispersed_flat_header()
        header['FILENAME'] = fname
        header['EXTNAME'] = 'FLUX'
        header['NSPAT'] = nx
        header['NSPEC'] = ny
        header['PRODTYPE'] = 'rectified_image'
        header['SLTH_ARC'] = 1.0
        header['SLTH_PIX'] = 1.0
        header['DATE-OBS'] = '2015-01-01T00:00:00.000'
        shape = (ny, nx)
        y, x = np.mgrid[:ny, :nx]

        # add aperture pos for order 1 to primary header
        if add_appos:
            header['APPOSO01'] = 2.0
            header['APFWHM01'] = 1.0
            header['APSGNO01'] = 1

        # make calibration data
        wavemap = wavecal(header)
        data = np.full(shape, 1.0)
        error = np.full(shape, 0.1)
        flat = np.full(shape, 1.0)
        wave = rotate90(wavemap[0], 3)
        spat = rotate90(wavemap[1], 3)
        mask = rotate90(wavemap[2], 3)
        hdul = fits.HDUList(fits.PrimaryHDU(data, header=header))
        hdul.append(fits.ImageHDU(error, name='ERROR'))
        hdul.append(fits.ImageHDU(flat, name='FLAT'))
        hdul.append(fits.ImageHDU(wave, name='WAVECAL'))
        hdul.append(fits.ImageHDU(spat, name='SPATCAL'))
        hdul.append(fits.ImageHDU(mask, name='ORDER_MASK'))

        for order in range(1, 4):
            ordnum = f'{order:02d}'
            omask = (mask == order)
            xmin, xmax = x[omask].min(), x[omask].max() + 1
            ymin, ymax = y[omask].min(), y[omask].max() + 1
            ny = ymax - ymin
            nx = xmax - xmin
            exthead = fits.Header({'CRPIX1': nx // 2, 'CRPIX2': ny // 2,
                                   'CRVAL1': wave[ymin, nx // 2],
                                   'CRVAL2': spat[ny // 2, xmin],
                                   'CDELT1': 1, 'CDELT2': 1,
                                   'CTYPE1': 'LINEAR', 'CTYPE2': 'LINEAR',
                                   'CUNIT1': 'cm-1', 'CUNIT2': 'arcsec',
                                   'CROTA2': 0, 'SPECSYS': 'TOPOCENT'})

            hdul.append(fits.ImageHDU(data[ymin:ymax, xmin:xmax].copy(),
                                      header=exthead,
                                      name=f'FLUX_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(error[ymin:ymax, xmin:xmax].copy(),
                                      header=exthead,
                                      name=f'ERROR_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(flat[ymin:ymax, xmin:xmax].copy(),
                                      header=exthead,
                                      name=f'FLAT_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(wave[ymin, xmin:xmax].copy(),
                                      name=f'WAVEPOS_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(spat[ymin:ymax, xmin].copy(),
                                      name=f'SLITPOS_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(spat[ymin:ymax, xmin].copy(),
                                      name=f'SPATIAL_PROFILE_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(spat[ymin:ymax, xmin:xmax].copy(),
                                      name=f'SPATIAL_MAP_ORDER_{ordnum}'))
            hdul.append(fits.ImageHDU(np.zeros((ny, nx), dtype=int),
                                      name=f'BADMASK_ORDER_{ordnum}'))

            if order == 1 and add_apmask:
                apmask = np.zeros((ny, nx), dtype=float)
                apmask[ny // 2 - 2:ny // 2 + 3, :] = -1
                apmask[ny // 2 - 1:ny // 2 + 2, :] = 1
                apmask[0, :] = np.nan
                apmask[-1, :] = np.nan
                hdul.append(
                    fits.ImageHDU(apmask, name='APERTURE_MASK_ORDER_01'))

        hdul[0].header['NORDERS'] = 3
        hdul[0].header['ORDERS'] = '3,2,1'
        hdul[0].header['ORDR_B'] = '302,198,95'
        hdul[0].header['ORDR_T'] = '397,293,190'
        hdul[0].header['ORDR_S'] = '2,2,2'
        hdul[0].header['ORDR_E'] = '992,981,969'

        filename = str(tmpdir.join(fname))
        hdul.writeto(filename, overwrite=True)
        hdul.close()
        return filename

    def make_low_files(self, tmpdir):
        hdul = resources.raw_low_nod_on()
        filename = str(tmpdir.join(hdul[0].header['FILENAME']))
        hdul.writeto(filename, overwrite=True)
        hdul.close()

        hdul = resources.raw_low_flat()
        flat = str(tmpdir.join(hdul[0].header['FILENAME']))
        hdul.writeto(flat, overwrite=True)
        hdul.close()

        return [filename, flat]

    def make_high_med_files(self, tmpdir):
        hdul = resources.raw_high_med_nod_off()
        filename = str(tmpdir.join(hdul[0].header['FILENAME']))
        hdul.writeto(filename, overwrite=True)
        hdul.close()

        hdul = resources.raw_high_med_flat()
        flat = str(tmpdir.join(hdul[0].header['FILENAME']))
        hdul.writeto(flat, overwrite=True)
        hdul.close()

        return [filename, flat]

    def make_mrm(self, tmpdir, return_hdul=False, n_ap=None):
        fname = 'F0001_EX_SPE_12345_EXES_MRM_001.fits'
        header = fits.Header({'INSTRUME': 'EXES',
                              'PRODTYPE': 'orders_merged',
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
        red = EXESReduction()
        ffile = self.make_low_files(tmpdir)
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

    def prep_reduction(self, ffile, tmpdir):
        red = EXESReduction()
        red.load(ffile)
        red.load_parameters()
        red.load_fits()
        red.output_directory = tmpdir
        return red

    def test_startup(self):
        red = EXESReduction()
        assert isinstance(red, Reduction)

    def test_load_basic(self, tmpdir):
        red = EXESReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)
        red.load_fits()
        assert len(red.input) == 1
        assert isinstance(red.input[0], fits.HDUList)
        assert isinstance(red.parameters, EXESParameters)

        # test updated keys
        header = red.input[0][0].header
        red.update_sofia_keys(header)
        assert header['PIPELINE'] == red.pipe_name
        assert header['PIPEVERS'] == red.pipe_version
        assert 'ASSC_AOR' in header
        assert 'ASSC_OBS' in header
        assert 'ASSC_MSN' in header
        assert header['PROCSTAT'] == 'LEVEL_2'

    def test_load_intermediate(self, tmpdir, capsys):
        red = EXESReduction()
        ffile = self.make_file(tmpdir)

        # set prodtype
        fits.setval(ffile, 'PRODTYPE', value='coadded')
        red.load(ffile)

        # load data is always first, then next step in the pipeline
        assert red.recipe[0] == 'load_data'
        assert red.recipe[1] == 'convert_units'

        # not a sky spectrum product
        assert red.parameters.base_header['SKYSPEC'] is False

        # test error for no further steps
        fits.setval(ffile, 'PRODTYPE', value='specmap')
        with pytest.raises(ValueError):
            red.load(ffile)
        assert 'No steps to run' in capsys.readouterr().err

        # test error for unknown prodtype
        fits.setval(ffile, 'PRODTYPE', value='test_value')
        with pytest.raises(ValueError):
            red.load(ffile)

        # test sky spectrum product
        fits.setval(ffile, 'PRODTYPE', value='sky_coadded')
        red.load(ffile)
        assert red.recipe[0] == 'load_data'
        assert red.recipe[1] == 'convert_units'
        assert red.parameters.base_header['SKYSPEC'] is True

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)
        mocker.patch.object(MatplotlibViewer, '__init__', return_value=None)
        mocker.patch.object(EyeViewer, '__init__', return_value=None)

        red = EXESReduction()
        vz = red.register_viewers()
        # 3 viewers -- QAD, profile, spectra
        assert len(vz) == 3
        assert isinstance(vz[0], QADViewer)
        assert isinstance(vz[1], MatplotlibViewer)
        assert isinstance(vz[2], EyeViewer)

    def test_display_data(self, tmpdir, capsys):
        red = EXESReduction()
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

        # test for bad filename - no data to display
        red.set_display_data(raw=False, filenames=['badfile.fits'])
        assert 'QADViewer' not in red.display_data

    def test_filenum(self):
        red = EXESReduction()

        fn = red.get_filenum('test.fits')
        assert fn == 'UNKNOWN'
        fn = red.get_filenum('test_data.fits')
        assert fn == 'UNKNOWN'
        fn = red.get_filenum('test_1.fits')
        assert fn == '0001'
        fn = red.get_filenum('test_0001_1.fits')
        assert fn == '0001_1'
        fn = red.get_filenum('test_0001_1-0002_2.fits')
        assert fn == '0002'
        fn = red.get_filenum('test_0001-0002.fits')
        assert fn == ['0001', '0002']

        # test if not list, string and number
        fn = 1
        assert red.concatenate_filenum(fn) == '1'
        fn = '1'
        assert red.concatenate_filenum(fn) == '1'

        # test if list
        fn = [1, 2]
        assert red.concatenate_filenum(fn) == '1-2'
        fn = ['1', '2']
        assert red.concatenate_filenum(fn) == '1-2'

        # test if list of list
        fn = [[1, 2], [3, 4]]
        assert red.concatenate_filenum(fn) == '1-4'
        fn = [['1', '2'], [3, 4]]
        assert red.concatenate_filenum(fn) == '1-4'

        # test if list of list of list
        fn = [[[1, 2], [3, 4]], [5, 6], 7]
        assert red.concatenate_filenum(fn) == '1-7'

        # test if list of list of list with possible serial numbers
        fn = [[['1_2', '1_3'], [3, 4]], ['5_1', 6], '7']
        assert red.concatenate_filenum(fn) == '1-7'

        # test if serial numbers but all the same file
        fn = [['1_1', '1_2'], ['1_3', '1_4']]
        assert red.concatenate_filenum(fn) == '1'

    def test_filename(self):
        red = EXESReduction()

        # test default
        header = fits.Header()
        fn = red.get_filename(header)
        assert fn == \
               'UNKNOWN_EX_SPE_UNKNOWN_UNKNOWNUNKNOWN_UNKNOWN_UNKNOWN.fits'
        assert header['FILENAME'] == fn

        # add relevant data
        header['MISSN-ID'] = '2015-01-01_EX_F001'
        header['DATATYPE'] = 'SPECTRAL'
        header['AOR_ID'] = '01_0001_01'
        header['SPECTEL1'] = 'EXE_ELON'
        header['SPECTEL2'] = 'EXE_ECHL'
        fn = red.get_filename(header)
        assert fn == \
               'F0001_EX_SPE_01000101_EXEELONEXEECHL_UNKNOWN_UNKNOWN.fits'
        assert header['FILENAME'] == fn

        # add optional prodtype and filenum
        fn = red.get_filename(header, prodtype='PTP', filenum=1)
        assert fn == 'F0001_EX_SPE_01000101_EXEELONEXEECHL_PTP_1.fits'
        assert header['FILENAME'] == fn

        # don't update header
        fn2 = red.get_filename(header, prodtype='PTP', filenum=2, update=False)
        assert fn2 == 'F0001_EX_SPE_01000101_EXEELONEXEECHL_PTP_2.fits'
        assert header['FILENAME'] == fn

        # image file
        header['DATATYPE'] = 'IMAGE'
        fn = red.get_filename(header)
        assert fn == \
               'F0001_EX_IMA_01000101_EXEELONEXEECHL_UNKNOWN_UNKNOWN.fits'

    def test_update_sofia_keys(self):
        red = EXESReduction()
        header = fits.Header({'AOR_ID': 'test_aor',
                              'OBS_ID': 'test_obs',
                              'MISSN-ID': 'test_msn'})
        red.update_sofia_keys(header)
        assert header['ASSC_AOR'] == 'TEST_AOR'
        assert header['ASSC_MSN'] == 'TEST_MSN'

        assert header['OBS_ID'] == 'P_TEST_OBS'
        assert header['ASSC_OBS'] == 'TEST_OBS'

        # for darks or flats, aor id is update from a base header;
        # other keys match the input header
        red.parameters = EXESParameters(base_header=header)
        dark_header = fits.Header({'OBSTYPE': 'dark',
                                   'AOR_ID': 'test_dark_aor',
                                   'OBS_ID': 'test_dark_obs',
                                   'MISSN-ID': 'test_dark_msn'})
        red.update_sofia_keys(dark_header)
        assert dark_header['AOR_ID'] == 'TEST_AOR'
        assert dark_header['ASSC_AOR'] == 'TEST_AOR'
        assert dark_header['ASSC_MSN'] == 'TEST_DARK_MSN'
        assert dark_header['OBS_ID'] == 'P_TEST_DARK_OBS'
        assert dark_header['ASSC_OBS'] == 'TEST_DARK_OBS'

    def test_update_output(self, tmpdir):
        red = EXESReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.load_fits()
        red.load_parameters()
        hdul = red.input[0]

        fn = red.update_output(hdul, ['0001', '0002'], 'cleaned')
        assert '0001-0002' in fn
        assert 'CLN' in fn
        history = hdul[0].header['HISTORY']
        assert '-- Pipeline step: Clean Bad Pixels' in history
        assert hdul[0].header['PRODTYPE'] == 'cleaned'

    def test_write_output(self, tmpdir):
        red = EXESReduction()
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

    @pytest.mark.parametrize('flag,nframes,obsmode,nodbeam,correct',
                             [('A', 0, 'nod_off_slit', 'B', None),
                              ('A', 1, 'nod_off_slit', 'B', [0]),
                              ('B', 1, 'nod_off_slit', 'B', None),
                              ('A', 4, 'nod_off_slit', 'B', [1, 3]),
                              ('B', 4, 'nod_off_slit', 'B', [0, 2]),
                              ('A', 5, 'nod_off_slit', 'B', [1, 3]),
                              ('A', 5, 'nod_off_slit', 'A', [0, 2]),
                              ('B', 5, 'nod_off_slit', 'B', [0, 2]),
                              ('B', 5, 'nod_off_slit', 'A', [1, 3]),
                              ('B', 5, 'nod_on_slit', 'A', [1, 3]),
                              ('A', 4, 'map', '', [0, 1, 2, 3]),
                              ('B', 4, 'map', '', None),
                              ('A', 6, 'map', '', [0, 1, 2]),
                              ('B', 6, 'map', '', [3, 4, 5]),
                              ('A', 5, 'other', '', [0, 1, 2, 3, 4]),
                              ('B', 4, 'other', '', None)])
    def test_get_beams(self, flag, nframes, obsmode, nodbeam, correct):
        header = fits.header.Header()
        header['INSTMODE'] = obsmode
        header['NODBEAM'] = nodbeam
        beams = EXESReduction._get_beams(header, flag, nframes)

        if correct is None:
            npt.assert_equal(beams, np.empty(0))
        else:
            npt.assert_equal(beams, np.array(correct))

    def test_parameter_copy(self, tmpdir):
        # test that parameter copy gets the additional
        # config attributes necessary
        red = EXESReduction()
        ffile = self.make_file(tmpdir)

        red.load(ffile)

        param = red.parameters
        pcopy = param.copy()

        assert hasattr(pcopy, 'base_header')
        assert isinstance(pcopy.base_header, fits.Header)

    def test_specmap(self, exes_combspec, exes_mrgordspec, tmpdir):
        red = EXESReduction()
        red.output_directory = tmpdir
        red.load([exes_combspec, exes_mrgordspec])
        red.load_parameters()

        # expected image file name
        red.step()
        outfile = tmpdir.join('F0001_EX_SPE_12345_EXES_CMB_001.png')
        assert os.path.isfile(outfile)
        outfile = tmpdir.join('F0001_EX_SPE_12345_EXES_MRD_001.png')
        assert os.path.isfile(outfile)

        # current products with both spectra and image work too
        ffile = self.make_mrm(tmpdir)
        red.load(ffile)
        # recipe is load data, specmap, but directly mapping should work
        assert red.recipe == ['load_data', 'specmap']
        red.recipe = ['specmap']
        red.load_parameters()
        red.load_fits()
        red.step()
        assert os.path.isfile(ffile.replace('.fits', '.png'))

        # including with multi-ap new style data
        ffile = self.make_mrm(tmpdir, n_ap=2)
        red.load(ffile)
        red.recipe = ['specmap']
        red.load_parameters()
        red.load_fits()
        red.step()
        assert os.path.isfile(ffile.replace('.fits', '.png'))

    def test_specmap_ignore_outer(self, tmpdir, capsys,
                                  exes_combspec, exes_mrgordspec):
        red = EXESReduction()
        red.output_directory = tmpdir
        red.load([exes_combspec, exes_mrgordspec])
        red.load_parameters()
        parset = red.parameters.current[-1]
        outfile1 = tmpdir.join('F0001_EX_SPE_12345_EXES_CMB_001.png')
        outfile2 = tmpdir.join('F0001_EX_SPE_12345_EXES_MRD_001.png')

        # set ignore outer parameter to turn off
        parset['ignore_outer']['value'] = 0.0
        with set_log_level('DEBUG'):
            red.specmap()
        assert 'Plotting between' not in capsys.readouterr().out
        assert os.path.isfile(outfile1)
        shutil.move(outfile1, tmpdir.join('tmp1.png'))
        assert os.path.isfile(outfile2)
        shutil.move(outfile2, tmpdir.join('tmp2.png'))

        # set ignore outer parameter to ignore outer 20%
        parset['ignore_outer']['value'] = 0.2
        with set_log_level('DEBUG'):
            red.specmap()
        assert 'Plotting between w=2 and w=8' in capsys.readouterr().out
        assert os.path.isfile(outfile1)
        shutil.move(outfile1, tmpdir.join('tmp1a.png'))
        assert os.path.isfile(outfile2)
        shutil.move(outfile2, tmpdir.join('tmp2a.png'))

        # images are the same size, different data plotted
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp1a.png'), 0) is not None
        assert compare_images(tmpdir.join('tmp2.png'),
                              tmpdir.join('tmp2a.png'), 0) is not None

    def test_specmap_atran_option(self, tmpdir, exes_combspec):
        red = EXESReduction()
        red.output_directory = tmpdir
        red.load([exes_combspec])
        red.load_parameters()
        parset = red.parameters.current[-1]
        outfile = tmpdir.join('F0001_EX_SPE_12345_EXES_CMB_001.png')

        # turn atran off
        parset['atran_plot']['value'] = False
        red.specmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp1.png'))

        # and turn on
        parset['atran_plot']['value'] = True
        red.specmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp2.png'))

        # image is different
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp2.png'), 0) is not None

    @pytest.mark.parametrize('mode', ['low', 'high_med'])
    def test_all_steps(self, tmpdir, mode):
        # exercises nominal behavior for a standard reduction
        red = EXESReduction()
        if mode == 'low':
            ffiles = self.make_low_files(tmpdir)
        else:
            ffiles = self.make_high_med_files(tmpdir)
        red.load(ffiles)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step in {'load_data', 'specmap', 'debounce', 'refine_wavecal'}:
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

        # check data products match standard set
        prodnames = set(red.default_prodnames.values())
        prodnames.update(['FLT', 'SPC', 'MRD', 'CMB'])
        for fn in red.out_files:
            ptype = fn.split('_')[-2]
            assert ptype in prodnames

    def test_sky_products(self, tmpdir):
        # exercises sky spectra reduction
        red = EXESReduction()
        ffiles = self.make_low_files(tmpdir)
        red.load(ffiles)
        red.output_directory = tmpdir
        red.load_parameters()

        # set sky reduction parameter
        parset = red.parameters.current[0]
        parset['sky_spec']['value'] = True

        # run all steps
        red.reduce()

        # check all were run, in history of last file
        hdul = red.input[0]
        history = hdul[0].header['HISTORY']
        for step in red.recipe:
            if step in {'load_data', 'specmap', 'debounce', 'refine_wavecal',
                        'subtract_background'}:
                continue
            msg = '-- Pipeline step: {}'.format(red.processing_steps[step])
            assert msg in history

        # check data products match sky set
        prodnames = set(red.sky_prodnames.values())
        prodnames.update(['FLT', 'SSP', 'SMD', 'RDC', 'SCS'])
        for fn in red.out_files:
            ptype = fn.split('_')[-2]
            assert ptype in prodnames

    def test_override_header_values(self):
        red = EXESReduction()
        red.load([])
        red.load_parameters()
        header = fits.Header()
        params = red.parameters.current[0]
        keys = ['WNO0', 'HRFL0', 'XDFL0', 'SLITROT', 'DETROT', 'HRR',
                'FLATTAMB', 'FLATEMIS']
        names = ['cent_wave', 'hrfl', 'xdfl', 'slit_rot', 'det_rot', 'hrr',
                 'flattamb', 'flatemis']
        values = [1, 2, 3, 4, 5, 6, 7, 8]

        # default: no overrides
        red._override_header_values(header, params)
        for key in keys:
            assert key not in header

        # set overrides
        for i, name in enumerate(names):
            params.set_value(name, values[i])
        red._override_header_values(header, params)
        for i, key in enumerate(keys):
            assert header[key] == values[i]
            assert isinstance(header[key], float)

    def test_load_data(self, tmpdir, capsys):
        red = EXESReduction()
        ffile = self.make_file(tmpdir)
        red.load(ffile)
        red.load_parameters()

        # succeeds with good header
        red.load_data()
        assert len(red.input) == 1
        assert red.input[0][0].header['PIPELINE'] == 'EXES_REDUX'

        # error message with bad header
        fits.setval(ffile, 'OBSTYPE', value='bad')
        red.load(ffile)
        red.load_parameters()
        red.load_data()
        assert len(red.input) == 0
        assert 'Invalid headers' in capsys.readouterr().err

    def test_check_raw_rasterized(self, tmpdir):
        red = EXESReduction()
        ffile1 = self.make_file(tmpdir, fname='test1.fits')
        ffile2 = self.make_file(tmpdir, fname='test2.fits')
        red.load([ffile1, ffile2])
        red.load_fits()

        # not a raster flat, no dark
        raster = red._check_raw_rasterized()
        assert np.all(raster['flag'] == [0, 0])
        assert raster['dark'] is None
        assert raster['dark_header'] is None

        # raster flat, no dark
        flat = red.input[0]
        flat[0].header['OBSTYPE'] = 'SKY'
        flat[0].header['CALSRC'] = 'BLACKBODY'
        flat[0].header['INSTMODE'] = 'STARE'

        raster = red._check_raw_rasterized()
        assert np.all(raster['flag'] == [1, 0])
        assert raster['dark'] is None
        assert raster['dark_header'] is None

        # raster flat, dark
        dark = red.input[1]
        dark[0].header['OBSTYPE'] = 'DARK'
        raster = red._check_raw_rasterized()

        assert np.all(raster['flag'] == [1, -1])
        assert raster['dark'] is dark[0].data
        assert raster['dark_header'] is dark[0].header

    def test_coadd_readouts_algorithms(self, tmpdir, mocker):
        ffile1 = self.make_file(tmpdir, 'test.sci.1.fits')
        ffile2 = self.make_file(tmpdir, 'test.flat.2.fits')
        ffile3 = self.make_file(tmpdir, 'test.dark.3.fits')
        red = self.prep_reduction([ffile1, ffile2, ffile3], tmpdir)

        # set step index to retrieve the right parameters
        red.step_index = 1
        params = red.get_parameter_set()

        # mock the algorithms for faster test
        m1 = mocker.patch('sofia_redux.instruments.exes.readraw.readraw',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))
        m2 = mocker.patch('sofia_redux.instruments.exes.'
                          'derasterize.derasterize',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))
        m3 = mocker.patch.object(red, '_check_raw_rasterized',
                                 return_value={'flag': [0, 0, 0], 'dark': None,
                                               'dark_header': None})

        # default parameters
        red.coadd_readouts()

        # coadd called for each file, deraster not
        assert m1.call_count == 3
        assert m2.call_count == 0
        assert m3.call_count == 1

        # set alternate algorithms
        params.set_value('algorithm', 'Last destructive only')
        red.coadd_readouts()
        assert m1.call_args[1]['algorithm'] == 0

        params.set_value('algorithm', 'Default for read mode')
        red.coadd_readouts()
        assert m1.call_args[1]['algorithm'] is None

        params.set_value('algorithm', 'First/last frame only')
        red.coadd_readouts()
        assert m1.call_args[1]['algorithm'] == 2

        params.set_value('algorithm', 'Second/penultimate frame only')
        red.coadd_readouts()
        assert m1.call_args[1]['algorithm'] == 3

    def test_coadd_readouts_raster(self, tmpdir, mocker, capsys):
        ffile1 = self.make_file(tmpdir, 'test.flat.1.fits')
        ffile2 = self.make_file(tmpdir, 'test.dark.2.fits')
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # make first file raster flat, second raster dark
        flat = red.input[0]
        flat[0].header['OBSTYPE'] = 'SKY'
        flat[0].header['CALSRC'] = 'BLACKBODY'
        flat[0].header['INSTMODE'] = 'STARE'
        dark = red.input[1]
        dark[0].header['OBSTYPE'] = 'DARK'

        # set step index to retrieve the right parameters
        red.step_index = 1

        # mock the algorithms for faster test
        m1 = mocker.patch('sofia_redux.instruments.exes.readraw.readraw',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))
        m2 = mocker.patch('sofia_redux.instruments.exes.'
                          'derasterize.derasterize',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))

        # default parameters
        red.coadd_readouts()

        # deraster called for flat file only, coadd not
        assert m1.call_count == 0
        assert m2.call_count == 1

        capt = capsys.readouterr()
        assert 'Skipping direct handling for raster dark' in capt.err
        assert 'Processing raster flat' in capt.out

    def test_coadd_readouts_processed_flat(self, tmpdir, mocker, capsys):
        ffile1 = self.make_file(tmpdir, 'test.sci.1.fits')
        ffile2 = self.make_file(tmpdir, 'test.flat.2.fits')
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # make flat processed
        sci = red.input[0]
        flat = red.input[1]
        flat[0].header['PRODTYPE'] = 'FLAT'

        # set step index to retrieve the right parameters
        red.step_index = 1

        # mock the algorithms for faster test
        m1 = mocker.patch('sofia_redux.instruments.exes.readraw.readraw',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))
        m2 = mocker.patch('sofia_redux.instruments.exes.'
                          'derasterize.derasterize',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))

        # default parameters
        red.coadd_readouts()

        # coadd called for sci file only, raster not called
        assert m1.call_count == 1
        assert m2.call_count == 0

        capt = capsys.readouterr()
        assert 'Passing processed flat' in capt.out
        assert red.input[0] is not sci
        assert red.input[1] is flat

    def test_coadd_readouts_fix_rows(self, tmpdir, mocker, capsys):
        ffile1 = self.make_file(tmpdir, 'test.sci.1.fits')
        ffile2 = self.make_file(tmpdir, 'test.flat.2.fits')
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # set step index to retrieve the right parameters
        red.step_index = 1
        params = red.get_parameter_set()

        # mock the algorithms for faster test
        m1 = mocker.patch('sofia_redux.instruments.exes.readraw.readraw',
                          return_value=(np.ones((10, 10)), np.ones((10, 10)),
                                        np.ones((10, 10), dtype=bool)))
        m2 = mocker.patch('sofia_redux.instruments.exes.'
                          'correct_row_gains.correct_row_gains',
                          return_value=np.ones((10, 10)))

        # no correction
        params.set_value('fix_row_gains', False)
        red.coadd_readouts()

        # coadd called for sci and flat,
        # gain correction not called
        assert m1.call_count == 2
        assert m2.call_count == 0

        # with correction
        params.set_value('fix_row_gains', True)
        red.coadd_readouts()

        # gain correction now called for each file
        assert m1.call_count == 4
        assert m2.call_count == 2

    def test_combine_flats(self):
        red = EXESReduction()
        hdr1 = fits.Header({'EXTNAME': 'FLUX', 'ASSC_OBS': '1'})
        hdr2 = fits.Header({'EXTNAME': 'FLUX', 'ASSC_OBS': '2'})
        flat1 = fits.HDUList([fits.PrimaryHDU(np.full((1, 10, 10), 1.0),
                                              header=hdr1),
                              fits.ImageHDU(np.full((1, 10, 10), 0.1),
                                            name='ERROR'),
                              fits.ImageHDU(np.full((10, 10), 0),
                                            name='MASK')])
        flat2 = fits.HDUList([fits.PrimaryHDU(np.full((1, 10, 10), 2.0),
                                              header=hdr2),
                              fits.ImageHDU(np.full((1, 10, 10), 0.2),
                                            name='ERROR'),
                              fits.ImageHDU(np.full((10, 10), 1),
                                            name='MASK')])

        new_flat = red._combine_flats([flat1, flat2])
        assert isinstance(new_flat, fits.HDUList)

        # output is weighted mean, so closer to 1 than 2,
        # error a little lower than .1, mask is or'd
        assert np.allclose(new_flat['FLUX'].data, 1.2)
        assert np.allclose(new_flat['ERROR'].data, 0.08944272)
        assert np.all(new_flat['MASK'].data == 1)

        # headers are merged
        assert new_flat[0].header['ASSC_OBS'] == '1,2'

        # mismatched data throws error
        flat2[0].data = np.ones((1, 5, 5))
        with pytest.raises(ValueError) as err:
            red._combine_flats([flat1, flat2])
        assert 'Flat files do not match' in str(err)

        # if flats are processed, only the first one is returned
        flat1[0].header['PRODTYPE'] = 'FLAT'
        new_flat = red._combine_flats([flat1, flat2])
        assert new_flat is flat1
        assert np.allclose(new_flat[0].data, 1.0)
        assert np.allclose(new_flat[1].data, 0.1)
        assert np.allclose(new_flat[2].data, 0)

    def test_write_flat(self, tmpdir):
        red = self.prep_reduction([], tmpdir)

        ny, nx = 30, 40
        flat_header = resources.single_order_flat_header()
        flat_header['NSPAT'] = nx
        flat_header['NSPEC'] = ny

        flat = fits.ImageHDU(np.ones((ny, nx)), flat_header, name='FLAT')
        error = fits.ImageHDU(np.zeros((ny, nx)), name='FLAT_ERROR')
        illum = np.ones((ny, nx))
        filenum = '001'
        dark = fits.ImageHDU(np.zeros((ny, nx)), name='DARK')

        hdul = red._write_flat(flat, error, illum, filenum)
        assert len(hdul) == 8
        extnames = ['FLAT', 'FLAT_ERROR', 'TORT_FLAT', 'TORT_FLAT_ERROR',
                    'ILLUMINATION', 'WAVECAL', 'SPATCAL', 'ORDER_MASK']
        for name in extnames:
            assert name in hdul
            if 'ERROR' in name or 'MASK' in name:
                assert np.allclose(np.nanmedian(hdul[name].data), 0)
            elif 'CAL' in name:
                # wave and spatcal out of range for small input data
                assert np.all(np.isnan(hdul[name].data))
            else:
                assert np.allclose(np.nanmedian(hdul[name].data), 1)

        # file is saved by default
        filename = hdul[0].header['FILENAME']
        assert filename.endswith('001.fits')
        assert os.path.isfile(str(tmpdir.join(filename)))

        # add a dark with new filenum, don't save
        filenum = '002'
        hdul = red._write_flat(flat, error, illum, filenum, dark=dark,
                               save=False)
        assert 'DARK' in hdul
        assert np.allclose(hdul['DARK'].data, 0)
        filename = hdul[0].header['FILENAME']
        assert filename.endswith('002.fits')
        assert not os.path.isfile(str(tmpdir.join(filename)))

    @pytest.mark.parametrize('method,param_val',
                             [('Derivative', 'deriv'),
                              ('Squared Derivative', 'sqderiv'),
                              ('Sobel', 'sobel')])
    def test_make_flat_edge_methods(self, tmpdir, mocker, method, param_val):
        ffile = self.make_file(tmpdir, fname='test.flat.1.fits')
        red = self.prep_reduction(ffile, tmpdir)

        # rdc-like input
        hdul = red.input[0]
        hdul[0].header['EXTNAME'] = 'FLUX'
        hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data), name='ERROR'))

        # set step index to retrieve the right parameters
        red.step_index = 2
        params = red.get_parameter_set()
        params.set_value('edge_method', method)

        # mock the algorithms for faster test
        flat_param = {
            'header': resources.single_order_flat_header(),
            'flat': np.ones((10, 10)),
            'flat_variance': np.ones((10, 10)),
            'illum': np.ones((10, 10))}
        m1 = mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                          return_value=flat_param)
        m2 = mocker.patch.object(red, '_combine_flats',
                                 return_value=hdul)
        m3 = mocker.patch.object(red, '_write_flat',
                                 return_value=hdul)

        # default params: make and write called, combine not called
        red.make_flat()
        assert m1.call_count == 1
        assert m2.call_count == 0
        assert m3.call_count == 1

        # check that edge method was passed
        assert m1.call_args[1]['edge_method'] == param_val

        # try to call again:
        # flat file is filtered out of input, so not remade
        red.make_flat()
        assert m1.call_count == 1
        assert len(red.input) == 0

    def test_make_flat_tort_options(self, tmpdir, mocker):
        ffile = self.make_file(tmpdir, fname='test.flat.1.fits')
        red = self.prep_reduction(ffile, tmpdir)

        # rdc-like input
        hdul = red.input[0]
        hdul[0].header['EXTNAME'] = 'FLUX'
        hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data), name='ERROR'))

        # set step index to retrieve the right parameters
        red.step_index = 2

        # set some non-default tort parameters
        params = red.get_parameter_set()
        params.set_value('start_rot', 0.1)
        params.set_value('predict_spacing', 100)
        params.set_value('threshold', 0.5)
        params.set_value('custom_wavemap', 'test_file')
        params.set_value('opt_rot', True)

        # mock the algorithms for faster test
        flat_param = {
            'header': resources.single_order_flat_header(),
            'flat': np.ones((10, 10)),
            'flat_variance': np.ones((10, 10)),
            'illum': np.ones((10, 10))}
        m1 = mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                          return_value=flat_param)
        m2 = mocker.patch.object(red, '_combine_flats',
                                 return_value=hdul)
        m3 = mocker.patch.object(red, '_write_flat',
                                 return_value=hdul)

        # default params: make and write called, combine not called
        red.make_flat()
        assert m1.call_count == 1
        assert m2.call_count == 0
        assert m3.call_count == 1

        # check that parameters were passed to makeflat,
        # either in header or keywords
        passed_header = m1.call_args[0][1]
        assert passed_header['KROT'] == 0.1
        assert passed_header['SPACING'] == 100
        assert passed_header['THRFAC'] == 0.5
        passed_kwargs = m1.call_args[1]
        assert passed_kwargs['custom_wavemap'] == 'test_file'
        assert passed_kwargs['fix_tort'] is False

    def test_make_flat_combine_multi(self, tmpdir, mocker):
        ffile1 = self.make_file(tmpdir, fname='test.flat.1.fits')
        ffile2 = self.make_file(tmpdir, fname='test.flat.2.fits')
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # rdc-like input
        for hdul in red.input:
            hdul[0].header['EXTNAME'] = 'FLUX'
            hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data),
                                      name='ERROR'))

        # set step index to retrieve the right parameters
        red.step_index = 2

        # mock the algorithms for faster test
        flat_param = {
            'header': resources.single_order_flat_header(),
            'flat': np.ones((10, 10)),
            'flat_variance': np.ones((10, 10)),
            'illum': np.ones((10, 10))}
        mod_flat = deepcopy(red.input[0])
        mod_flat[0].header['FILENAME'] = 'modified.fits'
        m1 = mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                          return_value=flat_param)
        m2 = mocker.patch.object(red, '_combine_flats',
                                 return_value=mod_flat)
        m3 = mocker.patch.object(red, '_write_flat',
                                 return_value=mod_flat)

        # for multi flat input, combine should be called
        red.make_flat()
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m3.call_count == 1

        # input to makeflat should be combine output
        assert m1.call_args[0][1]['FILENAME'] == 'modified.fits'

        # all flat files are filtered out of input at end of step
        assert len(red.input) == 0

    def test_make_flat_multi_dark(self, tmpdir, mocker, capsys):
        ffile1 = self.make_file(tmpdir, fname='test.flat.1.fits')
        ffile2 = self.make_file(tmpdir, fname='test.dark.2.fits')
        ffile3 = self.make_file(tmpdir, fname='test.dark.3.fits')
        red = self.prep_reduction([ffile1, ffile2, ffile3], tmpdir)

        # rdc-like input
        for hdul in red.input:
            hdul[0].header['EXTNAME'] = 'FLUX'
            hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data),
                                      name='ERROR'))

            # distinguish dark values
            if '2' in hdul[0].header['FILENAME']:
                hdul[0].data *= 2.0
            elif '3' in hdul[0].header['FILENAME']:
                hdul[0].data *= 3.0

        # set step index to retrieve the right parameters
        red.step_index = 2

        # mock the algorithms for faster test
        flat_param = {
            'header': resources.single_order_flat_header(),
            'flat': np.ones((10, 10)),
            'flat_variance': np.ones((10, 10)),
            'illum': np.ones((10, 10))}
        m1 = mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                          return_value=flat_param)
        m2 = mocker.patch.object(red, '_combine_flats',
                                 return_value=red.input[0])
        m3 = mocker.patch.object(red, '_write_flat',
                                 return_value=red.input[0])

        # for multi dark input, only the first should be used,
        # combine is not called
        red.make_flat()
        assert m1.call_count == 1
        assert m2.call_count == 0
        assert m3.call_count == 1

        assert 'More than 1 dark loaded; ' \
               'using the first' in capsys.readouterr().err

        # input to makeflat should include 1st dark data only
        assert np.allclose(m1.call_args[1]['dark'], 2.0)

        # all flat and dark files are filtered out of input at end of step
        assert len(red.input) == 0

    def test_make_flat_source_dark(self, tmpdir, mocker, capsys):
        ffile1 = self.make_file(tmpdir, fname='test.flat.1.fits')
        ffile2 = self.make_file(tmpdir, fname='test.sci.2.fits')
        ffile3 = self.make_file(tmpdir, fname='test.dark.3.fits')
        red = self.prep_reduction([ffile1, ffile2, ffile3], tmpdir)

        # rdc-like input
        for hdul in red.input:
            hdul[0].header['EXTNAME'] = 'FLUX'
            hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data),
                                      name='ERROR'))

            # distinguish sci and dark values
            if '2' in hdul[0].header['FILENAME']:
                hdul[0].data *= 2.0
            elif '3' in hdul[0].header['FILENAME']:
                hdul[0].data *= 3.0
        flat = red.input[0]

        # set step index to retrieve the right parameters
        red.step_index = 2

        # set save to keep sci file with flat appended
        param = red.get_parameter_set()
        param.set_value('save', True)

        # mock the algorithms for faster test
        flat_param = {
            'header': resources.single_order_flat_header(),
            'flat': np.ones_like(flat[0].data),
            'flat_variance': np.ones_like(flat[0].data),
            'illum': np.ones_like(flat[0].data)}
        mod_flat = deepcopy(red.input[0])
        mod_flat[0].header['FILENAME'] = 'modified.fits'
        m1 = mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                          return_value=flat_param)
        m2 = mocker.patch.object(red, '_combine_flats',
                                 return_value=mod_flat)
        m3 = mocker.patch.object(red, '_write_flat',
                                 return_value=mod_flat)

        # for one flat, dark input, combine is not called
        red.make_flat()
        assert m1.call_count == 1
        assert m2.call_count == 0
        assert m3.call_count == 1

        assert 'Using slit dark test.dark.3.fits' in capsys.readouterr().out

        # input to makeflat should include flat and dark data
        assert np.allclose(m1.call_args[0][0], 1.0)
        assert np.allclose(m1.call_args[1]['dark'], 3.0)

        # all flat and dark files are filtered out of input at end of step,
        # source remains, unmodified
        assert len(red.input) == 1
        sci = red.input[0]
        assert np.allclose(sci[0].data, 2.0)

        # flat and dark data should be appended to sci
        extnames = ['FLAT', 'FLAT_ERROR', 'FLAT_ILLUMINATION', 'DARK']
        for name in extnames:
            assert name in sci

        # flat and dark data should be added to sci header
        assert sci[0].header['SLITDRKF'] == 'test.dark.3.fits'
        assert sci[0].header['FLTFILE'] == 'modified.fits'
        assert sci[0].header['NORDERS'] == 1
        assert sci[0].header['ORDERS'] == '1'
        assert sci[0].header['ORDR_B'] == '324'
        assert sci[0].header['ORDR_T'] == '673'

        # sci file should be saved
        assert os.path.isfile(str(tmpdir.join(sci[0].header['FILENAME'])))

    def test_make_flat_preprocessed(self, tmpdir, mocker, capsys):
        ffile1 = self.make_file(tmpdir, fname='test.flat.1.fits')
        ffile2 = self.make_file(tmpdir, fname='test.sci.2.fits')
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # rdc-like input
        for hdul in red.input:
            hdul[0].header['EXTNAME'] = 'FLUX'
            hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data),
                                      name='ERROR'))
            # distinguish sci values
            if '2' in hdul[0].header['FILENAME']:
                hdul[0].data *= 2.0

        # flat is preprocessed
        flat = red.input[0]
        flat[0].header['PRODTYPE'] = 'FLAT'
        flat[0].header['EXTNAME'] = 'FLAT'
        flat[1].header['EXTNAME'] = 'FLAT_ERROR'
        flat.append(fits.ImageHDU(np.ones_like(flat[0].data),
                                  name='ILLUMINATION'))
        flat.append(fits.ImageHDU(np.full(flat[0].data.shape, 3.0),
                                  name='DARK'))

        # set step index to retrieve the right parameters
        red.step_index = 2

        # mock the algorithms
        flat_param = {
            'header': resources.single_order_flat_header(),
            'flat': np.ones_like(flat[0].data),
            'flat_variance': np.ones_like(flat[0].data),
            'illum': np.ones_like(flat[0].data)}
        mod_flat = deepcopy(red.input[0])
        mod_flat[0].header['FILENAME'] = 'modified.fits'
        m1 = mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                          return_value=flat_param)
        m2 = mocker.patch.object(red, '_combine_flats',
                                 return_value=mod_flat)
        m3 = mocker.patch.object(red, '_write_flat',
                                 return_value=mod_flat)

        # for processed flat, none of these are called
        red.make_flat()
        assert m1.call_count == 0
        assert m2.call_count == 0
        assert m3.call_count == 0

        assert 'Processed flat provided' in capsys.readouterr().out

        # flat is filtered out of input at end of step,
        # source remains
        assert len(red.input) == 1
        sci = red.input[0]

        # flat and dark data from processed file should be appended to sci
        extnames = ['FLAT', 'FLAT_ERROR', 'FLAT_ILLUMINATION', 'DARK']
        for name in extnames:
            assert name in sci

        assert np.allclose(sci['FLAT'].data, 1.0)
        assert np.allclose(sci['FLAT_ERROR'].data, 1.0)
        assert np.allclose(sci['DARK'].data, 3.0)
        assert np.allclose(sci[0].data, 2.0)

    def test_make_flat_error(self, tmpdir, mocker, capsys):
        ffile = self.make_file(tmpdir, fname='test.flat.1.fits')
        red = self.prep_reduction(ffile, tmpdir)

        # rdc-like input
        hdul = red.input[0]
        hdul[0].header['EXTNAME'] = 'FLUX'
        hdul.append(fits.ImageHDU(np.ones_like(hdul[0].data), name='ERROR'))

        # set step index to retrieve the right parameters
        red.step_index = 2

        mocker.patch('sofia_redux.instruments.exes.makeflat.makeflat',
                     side_effect=ValueError('bad'))

        # raises error
        with pytest.raises(ValueError) as err:
            red.make_flat()
        assert 'Error in makeflat' in str(err)

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_despike_errors(self, tmpdir, mocker, capsys, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz)
        red = self.prep_reduction(ffile, tmpdir)

        # set step index to retrieve the right parameters
        red.step_index = 2
        red_copy = pickle.dumps(red)

        # no input: logs error
        red.input = []
        red.despike()
        assert 'No source files loaded' in capsys.readouterr().err

        # spike factor zero: logs and returns
        red = pickle.loads(red_copy)
        red.get_parameter_set().set_value('spike_fac', 0)
        red.despike()
        assert 'Spike factor is 0' in capsys.readouterr().out

        # no good frames => all data trashed: dropped from input
        red = pickle.loads(red_copy)
        red.get_parameter_set().set_value('propagate_nan', True)
        red.get_parameter_set().set_value('mark_trash', True)
        if nz is None:
            mocker.patch('sofia_redux.instruments.exes.despike.despike',
                         return_value=(np.full((10, 10), 1.0),
                                       np.full((10, 10), True), []))
        else:
            mocker.patch('sofia_redux.instruments.exes.despike.despike',
                         return_value=(np.full((nz, 10, 10), 1.0),
                                       np.full((nz, 10, 10), True), []))
        red.despike()
        assert 'All data trashed' in capsys.readouterr().err
        assert len(red.input) == 0

        # error in despike
        red = pickle.loads(red_copy)
        mocker.patch('sofia_redux.instruments.exes.despike.despike',
                     side_effect=ValueError('bad'))
        with pytest.raises(ValueError) as err:
            red.despike()
        assert 'Error in despike' in str(err)

    def test_despike_save(self, tmpdir):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=2)
        red = self.prep_reduction(ffile, tmpdir)

        # set step index to retrieve the right parameters
        red.step_index = 2
        red.get_parameter_set().set_value('save', True)

        red.despike()
        hdul = red.input[0]
        assert os.path.isfile(str(tmpdir.join(hdul[0].header['FILENAME'])))

    @pytest.mark.parametrize('ignore_beams', [False, True])
    def test_despike_options(self, tmpdir, ignore_beams):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=2)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=2)
        red = self.prep_reduction([ffile1, ffile2], tmpdir)
        assert len(red.input) == 2

        # add a spike
        red.input[0][0].data[1, 1, 1] = 1000

        # set step index to retrieve the right parameters
        red.step_index = 2
        param = red.get_parameter_set()
        param.set_value('combine_all', True)
        param.set_value('propagate_nan', True)
        param.set_value('ignore_beams', ignore_beams)

        # all input files should be combined to single output file
        red.despike()
        assert len(red.input) == 1
        hdul = red.input[0]
        assert hdul[0].data.shape == (4, 10, 10)

        if not ignore_beams:
            # spike is marked nan in both A frames
            assert np.sum(np.isnan(hdul[0].data)) == 2
            assert np.sum(hdul['MASK'].data) == 2
        else:
            # spike is marked nan in 1 frame only
            assert np.sum(np.isnan(hdul[0].data)) == 1
            assert np.sum(hdul['MASK'].data) == 1

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_debounce(self, tmpdir, capsys, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz)
        red = self.prep_reduction(ffile, tmpdir)
        hdul = deepcopy(red.input[0])

        # set step index to retrieve the right parameters
        red.step_index = 3
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('bounce_fac', 0.0)

        # nothing happens if factor is 0
        red.debounce()
        assert 'debounce not applied' in capsys.readouterr().out
        assert red.input[0][0].header['FILENAME'] == hdul[0].header['FILENAME']

        # set bounce to something meaningful
        param.set_value('bounce_fac', 0.1)
        red.debounce()
        outfname = red.input[0][0].header['FILENAME']
        assert outfname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outfname)))
        assert red.input[0][0].data.shape == hdul[0].data.shape
        assert red.input[0][0].header['BOUNCE'] == 0.1

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_subtract_nods(self, tmpdir, capsys, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz)
        red = self.prep_reduction(ffile, tmpdir)
        hdul_copy = deepcopy(red.input[0])

        # set step index to retrieve the right parameters
        red.step_index = 4
        param = red.get_parameter_set()
        param.set_value('save', True)

        # frames subtracted only if > 1
        param.set_value('subtract_dark', False)
        red.subtract_nods()
        if nz is None or nz < 2:
            assert 'No B beams identified' in capsys.readouterr().out

        # file saved either way
        outname = red.input[0][0].header['FILENAME']
        assert 'NSB' in outname
        assert os.path.isfile(str(tmpdir.join(outname)))

        red.input[0] = deepcopy(hdul_copy)

        # try to subtract dark, but not available
        del red.input[0]['DARK']
        param.set_value('subtract_dark', True)
        red.subtract_nods()
        capt = capsys.readouterr()
        assert 'No dark frame available' in capt.err
        assert 'Skipping sky subtraction' in capt.out

        red.input[0] = deepcopy(hdul_copy)

        # modify A and B (if available) for test
        data = red.input[0][0].data
        data *= 40.0
        if nz is not None and nz > 1:
            data[1] *= 2
        data_copy = data.copy()

        # add dark value to file
        red.input[0]['DARK'].data[:] = 1
        param.set_value('subtract_dark', True)
        red.subtract_nods()
        if nz is None:
            assert np.allclose(red.input[0][0].data, data_copy - 1)
        else:
            assert np.allclose(red.input[0][0].data, data_copy[0] - 1)

    def test_subtract_nods_cirrus(self, tmpdir, mocker, capsys):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=2)
        red = self.prep_reduction(ffile, tmpdir)

        # set cirrus param
        red.step_index = 4
        red.get_parameter_set().set_value('subtract_sky', True)

        m1 = mocker.patch('sofia_redux.instruments.exes.diff_arr.diff_arr',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data,
                                        np.zeros(red.input[0][0].data.shape,
                                                 dtype=int)))
        m2 = mocker.patch('sofia_redux.instruments.exes.cirrus.cirrus',
                          return_value=red.input[0][0].data)

        # cirrus not called for nod on
        red.subtract_nods()
        assert m1.call_count == 1
        assert m2.call_count == 0
        assert 'Subtracting continuum' not in capsys.readouterr().out

        # set instmode to nod off
        red.input[0][0].header['INSTMODE'] = 'NOD_OFF_SLIT'

        # cirrus now called
        red.subtract_nods()
        assert m1.call_count == 2
        assert m2.call_count == 1
        assert 'Subtracting continuum' in capsys.readouterr().out

    def test_flat_correct(self, tmpdir, capsys):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=2)
        red = self.prep_reduction(ffile, tmpdir)
        hdul = deepcopy(red.input[0])

        # set step index to retrieve the right parameters
        red.step_index = 5
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('skip_flat', True)

        # nothing happens if skipping flat
        red.flat_correct()
        assert 'Skipping flat' in capsys.readouterr().out
        assert red.input[0][0].header['FILENAME'] == hdul[0].header['FILENAME']

        # don't skip
        param.set_value('skip_flat', False)
        red.flat_correct()
        outfname = red.input[0][0].header['FILENAME']
        assert outfname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outfname)))
        assert red.input[0][0].data.shape == hdul[0].data.shape
        assert red.input[0][0].header['BUNIT'] == 'erg s-1 cm-2 sr-1 (cm-1)-1'

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_clean_badpix(self, tmpdir, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz, match_mask=True)
        red = self.prep_reduction(ffile, tmpdir)
        hdul = deepcopy(red.input[0])

        # set step index to retrieve the right parameters
        red.step_index = 6
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('propagate_nan', True)

        red.clean_badpix()
        outfname = red.input[0][0].header['FILENAME']
        assert outfname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outfname)))
        assert red.input[0][0].data.shape == hdul[0].data.shape

        # mask is empty, so only standard bpm is marked
        assert not np.all(np.isnan(red.input[0][0].data))

        # set all bad -- all data marked nan
        red.input[0] = hdul
        red.input[0]['MASK'].data[:] = 1
        red.clean_badpix()
        assert np.all(np.isnan(red.input[0][0].data))

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_undistort(self, tmpdir, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz, merge_flat=True)
        red = self.prep_reduction(ffile, tmpdir)
        hdul = deepcopy(red.input[0])

        # set step index to retrieve the right parameters
        red.step_index = 8
        param = red.get_parameter_set()
        param.set_value('save', True)

        # remove extensions from synthetic input -
        # should be added by this step
        del red.input[0]['WAVECAL']
        del red.input[0]['SPATCAL']
        del red.input[0]['ORDER_MASK']

        red.undistort()
        outfname = red.input[0][0].header['FILENAME']
        assert outfname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outfname)))
        assert red.input[0][0].data.shape == hdul[0].data.shape

        assert 'WAVECAL' in red.input[0]
        assert 'SPATCAL' in red.input[0]
        assert 'ORDER_MASK' in red.input[0]

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_undistort_block_bad(self, tmpdir, mocker, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz, merge_flat=True)
        red = self.prep_reduction(ffile, tmpdir)

        # set an unilluminated pixel
        red.input[0]['FLAT_ILLUMINATION'].data[2, 2] = 0
        hdul = deepcopy(red.input[0])

        # mock wavecal to return good order mask
        mocker.patch('sofia_redux.instruments.exes.wavecal.wavecal',
                     return_value=np.full((3, 10, 10), 1))

        # set step index to retrieve the right parameters
        red.step_index = 8
        param = red.get_parameter_set()

        param.set_value('block_unilluminated', True)
        red.undistort()
        blocked = red.input[0][0].data.copy()
        red.input = [hdul]

        param.set_value('block_unilluminated', False)
        red.undistort()
        unblocked = red.input[0][0].data.copy()

        assert np.isnan(blocked).sum() > np.isnan(unblocked).sum()

    @pytest.mark.parametrize('nz', [1, 2, None])
    def test_correct_calibration(self, tmpdir, nz):
        ffile = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                            nz=nz)
        red = self.prep_reduction(ffile, tmpdir)
        hdul = deepcopy(red.input[0])

        # set step index to retrieve the right parameters
        red.step_index = 8
        param = red.get_parameter_set()
        param.set_value('save', True)

        red.correct_calibration()
        outfname = red.input[0][0].header['FILENAME']
        assert outfname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outfname)))
        assert red.input[0][0].data.shape == hdul[0].data.shape

        # input wavecal is flat, so all data should have the same correction
        assert np.allclose(red.input[0][0].data, hdul[0].data * 1.92222884e-05)

        # set parameter to skip correction
        red.input = [deepcopy(hdul)]
        param.set_value('skip_correction', True)
        red.correct_calibration()
        outfname = red.input[0][0].header['FILENAME']
        assert outfname == hdul[0].header['FILENAME']
        assert np.allclose(red.input[0][0].data, hdul[0].data)

    def test_blank_frames(self, tmpdir, capsys):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2)
        red = self.prep_reduction([ffile1, ffile2, ffile3], tmpdir)

        def check_nan(hdul1):
            f_nan = np.sum(np.isnan(hdul1[0].data))
            e_nan = np.sum(np.isnan(hdul1[1].data))
            return f_nan, e_nan

        # bad frames list has to match number of files but can be empty
        bad = [[], [], []]
        red._blank_frames(bad)
        assert 'No valid frame numbers passed' in capsys.readouterr().err
        for hdul in red.input:
            # no NaNs in data or error
            assert check_nan(hdul) == (0, 0)

        # same result for frames out of range (1-indexed, not 0)
        bad = [[2], [2, 3], [3, 10]]
        red._blank_frames(bad)
        assert 'No valid frame numbers passed' in capsys.readouterr().err
        for hdul in red.input:
            assert check_nan(hdul) == (0, 0)

        # frames in range are set to NaN, invalid frames are ignored
        bad = [[0, 1], [1, 2], [2, 10]]
        red._blank_frames(bad)
        capt = capsys.readouterr()
        assert 'No valid frame numbers passed' not in capt.err
        assert capt.out.count('Excluding frame') == 3
        for hdul in red.input:
            # 1 10x10 frame set to NaN
            assert check_nan(hdul) == (100, 100)

    def test_split_nod_pairs(self, tmpdir, capsys):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # split all files, putting each frame into a new hdul
        red._split_nod_pairs()
        assert len(red.input) == 7

        # single frame files are not updated, others are
        filenames = ['test.sci.1.fits', 'test.sci.2.fits',
                     'F0001_EX_SPE_01000101_NONEEXEECHL_RDC_0003_1.fits',
                     'F0001_EX_SPE_01000101_NONEEXEECHL_RDC_0003_2.fits',
                     'F0001_EX_SPE_01000101_NONEEXEECHL_RDC_0004_1.fits',
                     'F0001_EX_SPE_01000101_NONEEXEECHL_RDC_0004_2.fits',
                     'F0001_EX_SPE_01000101_NONEEXEECHL_RDC_0004_3.fits']
        filenums = ['0001', '0002', '0003_1', '0003_2',
                    '0004_1', '0004_2', '0004_3']
        for i, hdul in enumerate(red.input):
            assert np.squeeze(hdul['FLUX'].data).shape == (10, 10)
            assert np.squeeze(hdul['ERROR'].data).shape == (10, 10)
            assert hdul['FLAT'].data.shape == (10, 10)
            assert hdul['FLAT_ILLUMINATION'].data.shape == (10, 10)
            assert hdul[0].header['FILENAME'] == filenames[i]
            assert red.filenum[i] == filenums[i]

    @pytest.mark.parametrize('mask2d', [False, True])
    def test_combine_nod_pairs(self, tmpdir, capsys, mask2d):
        match_mask = not mask2d
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=match_mask)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=match_mask)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2, match_mask=match_mask)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3, match_mask=match_mask)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # combine all files, putting all frames into a single hdul
        red._combine_nod_pairs(mask2d=mask2d)
        assert len(red.input) == 1

        hdul = red.input[0]
        assert hdul['FLUX'].data.shape == (7, 10, 10)
        assert hdul['ERROR'].data.shape == (7, 10, 10)
        if mask2d:
            assert hdul['MASK'].data.shape == (10, 10)
            assert np.all(hdul['MASK'].data == 0)
        else:
            assert hdul['MASK'].data.shape == (7, 10, 10)
            assert np.all(hdul['MASK'].data == 0)
        assert hdul['FLAT'].data.shape == (10, 10)
        assert hdul['FLAT_ILLUMINATION'].data.shape == (10, 10)
        assert hdul[0].header['FILENAME'] == 'test.sci.1.fits'
        assert red.filenum[0] == ['0001', '0002', '0003', '0004']

    def test_combine_nod_pairs_errors(self, tmpdir, capsys):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=True)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=True)

        # only 1 file, nothing to do
        red = self.prep_reduction(ffile1, tmpdir)
        hdul = red.input[0]
        red._combine_nod_pairs()
        assert len(red.input) == 1
        assert red.input[0] is hdul

        # 2 files, but mismatched shape
        red = self.prep_reduction([ffile1, ffile2], tmpdir)
        red.input[1][0].data = np.ones((5, 5, 5))
        with pytest.raises(ValueError) as err:
            red._combine_nod_pairs()
        assert 'does not match dimensions' in str(err)

    @pytest.mark.parametrize('weight_method,weight_mode,stdwt',
                             [('Uniform weights', 'unweighted', True),
                              ('Weight by flat', None, False),
                              ('Weight by variance', None, True)])
    def test_coadd_pairs(self, tmpdir, capsys, mocker,
                         weight_method, weight_mode, stdwt):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=True)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=True)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2, match_mask=True)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3, match_mask=True)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # set step index and save parameters
        red.step_index = 9
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('save_intermediate', True)
        param.set_value('weight_method', weight_method)

        m1 = mocker.patch.object(red, '_blank_frames')
        m2 = mocker.patch.object(red, '_split_nod_pairs')
        m3 = mocker.patch.object(red, '_combine_nod_pairs')
        m4 = mocker.patch('sofia_redux.instruments.exes.submean.submean',
                          return_value=red.input[0][0].data)
        m5 = mocker.patch('sofia_redux.instruments.exes.'
                          'spatial_shift.spatial_shift',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))
        m6 = mocker.patch('sofia_redux.instruments.exes.coadd.coadd',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))

        red.coadd_pairs()

        # with default parameters, only coadd called, once per file
        assert m1.call_count == 0
        assert m2.call_count == 0
        assert m3.call_count == 0
        assert m4.call_count == 0
        assert m5.call_count == 0
        assert m6.call_count == 4

        # coadd called with expected weighting parameters
        assert m6.call_args[1]['weight_mode'] == weight_mode
        assert m6.call_args[1]['std_wt'] == stdwt

        # intermediate and final files should be saved
        for hdul in red.input:
            coa_outname = hdul[0].header['FILENAME']
            assert os.path.isfile(str(tmpdir.join(coa_outname)))
            coi_outname = coa_outname.replace('COA', 'COI')
            assert os.path.isfile(str(tmpdir.join(coi_outname)))

    @pytest.mark.parametrize('shift_method,sharpen',
                             [('Maximize signal-to-noise', False),
                              ('Maximize signal (sharpen)', True)])
    def test_coadd_pairs_shift(self, tmpdir, capsys, mocker,
                               shift_method, sharpen):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=True)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=True)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2, match_mask=True)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3, match_mask=True)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # set step index and shift parameter
        red.step_index = 9
        param = red.get_parameter_set()
        param.set_value('shift', True)
        param.set_value('shift_method', shift_method)

        m1 = mocker.patch.object(red, '_blank_frames')
        m2 = mocker.patch.object(red, '_split_nod_pairs')
        m3 = mocker.patch.object(red, '_combine_nod_pairs')
        m4 = mocker.patch('sofia_redux.instruments.exes.submean.submean',
                          return_value=red.input[0][0].data)
        m5 = mocker.patch('sofia_redux.instruments.exes.'
                          'spatial_shift.spatial_shift',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))
        m6 = mocker.patch('sofia_redux.instruments.exes.coadd.coadd',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))

        red.coadd_pairs()

        # shift called once per file with frames > 1, coadd called for all
        assert m1.call_count == 0
        assert m2.call_count == 0
        assert m3.call_count == 0
        assert m4.call_count == 0
        assert m5.call_count == 2
        assert m6.call_count == 4

        # shift called with expected parameters
        assert m5.call_args[1]['sharpen'] == sharpen

    @pytest.mark.parametrize('instmode,called',
                             [('NOD_OFF_SLIT', False),
                              ('MAP', False),
                              ('NOD_ON_SLIT', True)])
    def test_coadd_pairs_submean(self, tmpdir, capsys, mocker,
                                 instmode, called):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=2, match_mask=True)
        red = self.prep_reduction([ffile1], tmpdir)

        # set step index and submean parameter
        red.step_index = 9
        param = red.get_parameter_set()
        param.set_value('subtract_sky', True)

        red.input[0][0].header['INSTMODE'] = instmode

        m1 = mocker.patch.object(red, '_blank_frames')
        m2 = mocker.patch.object(red, '_split_nod_pairs')
        m3 = mocker.patch.object(red, '_combine_nod_pairs')
        m4 = mocker.patch('sofia_redux.instruments.exes.submean.submean',
                          return_value=red.input[0][0].data)
        m5 = mocker.patch('sofia_redux.instruments.exes.'
                          'spatial_shift.spatial_shift',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))
        m6 = mocker.patch('sofia_redux.instruments.exes.coadd.coadd',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))

        red.coadd_pairs()

        # submean called once per file if instmode matches
        assert m1.call_count == 0
        assert m2.call_count == 0
        assert m3.call_count == 0
        if called:
            assert m4.call_count == 1
        else:
            assert m4.call_count == 0
        assert m5.call_count == 0
        assert m6.call_count == 1

    @pytest.mark.parametrize('exclude,called,success',
                             [('', False, True),
                              ('1', True, True),
                              ('1,2,5,10', True, True),
                              ('1;;2;', True, True),
                              ('1;2;3;4', True, True),
                              ('1;2,3;;4,5,6', True, True),
                              ('1;a;3;4', True, False),
                              ('1;2', True, False)])
    def test_coadd_pairs_exclude(self, tmpdir, capsys, mocker,
                                 exclude, called, success):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=True)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=True)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2, match_mask=True)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3, match_mask=True)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # set step index and exclude parameter
        red.step_index = 9
        param = red.get_parameter_set()
        param.set_value('exclude_pairs', exclude)

        m1 = mocker.patch.object(red, '_blank_frames')
        m2 = mocker.patch.object(red, '_split_nod_pairs')
        m3 = mocker.patch.object(red, '_combine_nod_pairs')
        m4 = mocker.patch('sofia_redux.instruments.exes.submean.submean',
                          return_value=red.input[0][0].data)
        m5 = mocker.patch('sofia_redux.instruments.exes.'
                          'spatial_shift.spatial_shift',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))
        m6 = mocker.patch('sofia_redux.instruments.exes.coadd.coadd',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))

        if success:
            red.coadd_pairs()
            if called:
                assert m1.call_count == 1
            assert m2.call_count == 0
            assert m3.call_count == 0
            assert m4.call_count == 0
            assert m5.call_count == 0
            assert m6.call_count == 4
        else:
            with pytest.raises(ValueError) as err:
                red.coadd_pairs()
            assert 'Invalid position parameter' in str(err)
            assert 'Could not read exclude pairs ' \
                   'parameter' in capsys.readouterr().err
            assert m1.call_count == 0
            assert m2.call_count == 0
            assert m3.call_count == 0
            assert m4.call_count == 0
            assert m5.call_count == 0
            assert m6.call_count == 0

    @pytest.mark.parametrize('skip,combine,ncoadd',
                             [(True, False, 7),
                              (True, True, 7),
                              (False, True, 1),
                              (False, False, 4)])
    def test_coadd_pairs_skip_combine(self, tmpdir, capsys, mocker,
                                      skip, combine, ncoadd):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=True)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=True)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2, match_mask=True)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3, match_mask=True)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # set step index and skip/combine parameters
        # note: skip takes precedence
        red.step_index = 9
        param = red.get_parameter_set()
        param.set_value('skip_coadd', skip)
        param.set_value('coadd_all_files', combine)

        m1 = mocker.patch('sofia_redux.instruments.exes.coadd.coadd',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))

        red.coadd_pairs()

        # coadd called once per remaining frame
        assert m1.call_count == ncoadd

    @pytest.mark.parametrize('override,called,coadd_all,success',
                             [('', False, False, True),
                              ('1', True, False, True),
                              ('1,2,5,10', True, False, True),
                              ('1;;2;', True, False, False),
                              ('1;2;3;4', True, False, True),
                              ('1;2;3,4;5,6,7', True, False, True),
                              ('1;a;3;4', True, False, False),
                              ('1;2', True, False, False),
                              ('', False, True, True),
                              ('1', True, True, True),
                              ('1,2,5,10', True, True, True),
                              ('1;;2;', True, True, False),
                              ('1;2;3;4', True, True, True),
                              ('1;2;3,4;5,6,7', True, True, True),
                              ('1;a;3;4', True, True, False),
                              ('1;2', True, True, False),
                              ])
    def test_coadd_pairs_overrides(self, tmpdir, capsys, mocker,
                                   override, called, coadd_all, success):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None, match_mask=True)
        ffile2 = self.make_intermediate_file(tmpdir, fname='test.sci.2.fits',
                                             nz=1, match_mask=True)
        ffile3 = self.make_intermediate_file(tmpdir, fname='test.sci.3.fits',
                                             nz=2, match_mask=True)
        ffile4 = self.make_intermediate_file(tmpdir, fname='test.sci.4.fits',
                                             nz=3, match_mask=True)
        red = self.prep_reduction([ffile1, ffile2, ffile3, ffile4], tmpdir)

        # set step index and exclude parameter
        red.step_index = 9
        param = red.get_parameter_set()
        param.set_value('override_weights', override)
        param.set_value('coadd_all_files', coadd_all)

        m1 = mocker.patch.object(red, '_blank_frames')
        m2 = mocker.patch.object(red, '_split_nod_pairs')
        # m3 = mocker.patch.object(red, '_combine_nod_pairs', )
        m4 = mocker.patch('sofia_redux.instruments.exes.submean.submean',
                          return_value=red.input[0][0].data)
        m5 = mocker.patch('sofia_redux.instruments.exes.'
                          'spatial_shift.spatial_shift',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))
        m6 = mocker.patch('sofia_redux.instruments.exes.coadd.coadd',
                          return_value=(red.input[0][0].data,
                                        red.input[0][1].data))

        if success:
            red.coadd_pairs()
            assert m1.call_count == 0
            assert m2.call_count == 0
            assert m4.call_count == 0
            assert m5.call_count == 0
            if coadd_all:
                assert m6.call_count == 1
            else:
                assert m6.call_count == 4
            if called:
                assert m6.call_args[1]['weight_mode'] == 'useweights'
                assert np.allclose(np.sum(m6.call_args[1]['weights']), 1)
        else:
            with pytest.raises(ValueError) as err:
                red.coadd_pairs()
            assert 'Invalid position parameter' in str(err)
            assert 'Could not read override weights ' \
                   'parameter' in capsys.readouterr().err
            assert m1.call_count == 0
            assert m2.call_count == 0
            assert m4.call_count == 0
            assert m5.call_count == 0
            assert m6.call_count == 0

    def test_make_profiles(self, tmpdir, capsys, mocker):
        ffile1 = self.make_intermediate_file(tmpdir, fname='test.sci.1.fits',
                                             nz=None)
        red = self.prep_reduction(ffile1, tmpdir)
        red.step_index = 11
        param = red.get_parameter_set()
        param.set_value('save', True)

        # output file is saved
        red.make_profiles()
        outname = red.input[0][0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outname)))

        # error in rectify
        mocker.patch('sofia_redux.spectroscopy.rectify.rectify',
                     return_value={0: None})
        with pytest.raises(ValueError) as err:
            red.make_profiles()
        assert 'Problem in rectification' in str(err)

    @pytest.mark.parametrize('method,position,success,message',
                             [('fix to center', '', True,
                               'Fixing aperture to slit center'),
                              ('fix to input', '', False,
                               'Could not read input_position parameter'),
                              ('fix to input', '2', True,
                               'Fixing aperture to input'),
                              ('auto', '', True,
                               'Finding aperture positions from '
                               'Gaussian fits'),
                              ('auto', '2', True,
                               'Finding aperture positions from '
                               'Gaussian fits'),
                              ('auto', 'a,b,c', False,
                               'Could not read input_position parameter')])
    def test_locate_apertures(self, tmpdir, capsys, method, position,
                              success, message):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits')
        red = self.prep_reduction(ffile1, tmpdir)
        red.step_index = 1
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('method', method)
        param.set_value('input_position', position)

        if success:
            # output file is saved
            red.locate_apertures()
            outname = red.input[0][0].header['FILENAME']
            assert os.path.isfile(str(tmpdir.join(outname)))
            assert message in capsys.readouterr().out
            for hdu in red.input[0]:
                # aperture recorded in all flux extension headers
                if 'flux' in hdu.header['EXTNAME'].lower():
                    assert 'APPOSO01' in hdu.header
        else:
            with pytest.raises(ValueError):
                red.locate_apertures()
            assert message in capsys.readouterr().err

    def test_locate_apertures_exclude(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits')
        red = self.prep_reduction(ffile1, tmpdir)
        red.step_index = 1
        param = red.get_parameter_set()

        # exclude order
        param.set_value('exclude_orders', '2')
        red.locate_apertures()
        assert 'Skipping order 02' in capsys.readouterr().out
        assert 'APPOSO01' in red.input[0][0].header
        assert 'APPOSO02' not in red.input[0][0].header
        assert 'APPOSO03' in red.input[0][0].header

        # bad param
        param.set_value('exclude_orders', 'bad')
        with pytest.raises(ValueError) as err:
            red.locate_apertures()
        assert 'Invalid order exclusion' in str(err)

    @pytest.mark.parametrize('appos,full_slit,success',
                             [(False, False, False),
                              (True, True, True),
                              (True, False, True)])
    def test_set_apertures(self, tmpdir, appos, full_slit, success):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=appos)
        red = self.prep_reduction(ffile1, tmpdir)
        red.step_index = 2
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('full_slit', full_slit)

        # output file is saved
        red.set_apertures()
        outname = red.input[0][0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outname)))

        for hdu in red.input[0]:
            # aperture recorded in all flux extension headers
            name = hdu.header['EXTNAME'].lower()
            if success and name in ['flux', 'flux_order_01']:
                assert 'APPOSO01' in hdu.header
                assert 'APSGNO01' in hdu.header
                assert 'APRADO01' in hdu.header
                assert 'PSFRAD01' in hdu.header
                assert 'BGR_O01' in hdu.header
            else:
                assert 'APRADO01' not in hdu.header

    @pytest.mark.parametrize('full_slit,apsign,aprad,psfrad,'
                             'bgr,ap_start,ap_end,success,hpos,hrad',
                             [(False, '', '', '', '', '', '', True, 2, 2),
                              (True, '', '', '', '', '', '',
                               True, 2.925, 2.925),
                              (False, '-1', '', '', '', '', '', True, 2, 2),
                              (False, '-1;1', '', '', '', '', '', False, 2, 2),
                              (True, '', '', '', '', '0', '2', True, 1, 1),
                              (False, '1', '1', '1', '3-5', '', '',
                               True, 2, 1),
                              (False, '1', '1', '1', '3', '', '', False, 2, 1),
                              (False, '1', '1', '1', 'none', '', '',
                               True, 2, 1),
                              (False, '1', '1', 'a', '', '', '', False, 2, 1),
                              ])
    def test_set_apertures_manual(self, tmpdir, capsys,
                                  full_slit, apsign, aprad, psfrad,
                                  bgr, ap_start, ap_end, success, hpos, hrad):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True)
        red = self.prep_reduction(ffile1, tmpdir)
        red.step_index = 2
        param = red.get_parameter_set()

        param.set_value('full_slit', full_slit)
        param.set_value('apsign', apsign)
        param.set_value('aprad', aprad)
        param.set_value('psfrad', psfrad)
        param.set_value('bgr', bgr)
        param.set_value('ap_start', ap_start)
        param.set_value('ap_end', ap_end)

        if success:
            red.set_apertures()
            assert 'APERTURE_MASK_ORDER_01' in red.input[0]
        else:
            with pytest.raises(ValueError):
                red.set_apertures()

        for hdu in red.input[0]:
            # aperture recorded in all flux extension headers
            name = hdu.header['EXTNAME'].lower()
            if success and name in ['flux', 'flux_order_01']:
                assert 'APPOSO01' in hdu.header
                assert 'APSGNO01' in hdu.header
                assert 'APRADO01' in hdu.header
                assert 'PSFRAD01' in hdu.header
                assert 'BGR_O01' in hdu.header

                assert np.isclose(float(hdu.header['APPOSO01']), hpos)
                assert np.isclose(float(hdu.header['APRADO01']), hrad)
            else:
                assert 'APRADO01' not in hdu.header

    def test_subtract_background(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction(ffile1, tmpdir)
        hdul = deepcopy(red.input[0])

        red.step_index = 3
        param = red.get_parameter_set()
        param.set_value('save', True)

        # if skipped, nothing happens
        param.set_value('skip_bg', True)
        red.subtract_background()
        assert red.input[0][0].header['FILENAME'] == hdul[0].header['FILENAME']
        assert 'No background subtraction performed' in capsys.readouterr().out

        # stop skip; output file is saved
        param.set_value('skip_bg', False)
        red.subtract_background()
        outname = red.input[0][0].header['FILENAME']
        assert outname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outname)))

        # order 1 is now ~0 because of background subtraction;
        # others are unmodified because they were excluded earlier
        assert np.allclose(red.input[0]['FLUX_ORDER_01'].data, 0)
        assert np.allclose(red.input[0]['FLUX_ORDER_02'].data, 1)
        assert np.allclose(red.input[0]['FLUX_ORDER_03'].data, 1)

        # if no bg regions in apmask, nothing happens
        apmask = hdul['APERTURE_MASK_ORDER_01'].data
        apmask[np.isnan(apmask)] = 0
        red.input[0] = hdul
        red.subtract_background()
        assert np.allclose(red.input[0]['FLUX_ORDER_01'].data, 1)
        assert 'No background regions defined' in capsys.readouterr().out

    def test_get_atran(self, tmpdir, mocker, capsys):
        red = EXESReduction()
        header = fits.Header({'RP': 1000.0})

        # check for source installation with at least one file
        try:
            red._get_atran(header, None, None)

            # returns [awave, atrans, ...] from a default file
            atran = red._get_atran(header, None, None)
            assert 'Using PSG file' in capsys.readouterr().out
            assert atran.shape[0] == 13
        except ValueError:
            pass

        # specify a particular file
        afile = str(tmpdir.join('psg_40K_45deg_5-28um.fits'))
        hdul = fits.HDUList(fits.PrimaryHDU([np.linspace(1000, 10000, 10),
                                             np.linspace(0, 1, 10)]))
        hdul.writeto(afile, overwrite=True)
        atran = red._get_atran(header, afile, None)
        assert f'Using PSG file: {afile}' in capsys.readouterr().out
        assert atran.shape[0] == 2

        # output awave is not converted from input wavenumber
        assert np.allclose(atran[0], np.linspace(1000, 10000, 10))

        mocker.patch('sofia_redux.instruments.exes.get_atran.get_atran',
                     return_value=None)
        with pytest.raises(ValueError) as err:
            red._get_atran(header, None, None)
        assert 'No matching transmission files' in str(err)

        with pytest.raises(ValueError) as err:
            red._get_atran(header, None, str(tmpdir))
        assert 'No matching transmission files' in str(err)

    @pytest.mark.parametrize('use_profile', [True, False])
    def test_extract_spectra(self, tmpdir, use_profile):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction(ffile1, tmpdir)
        hdul = deepcopy(red.input[0])

        red.step_index = 4
        param = red.get_parameter_set()
        param.set_value('save', True)
        param.set_value('save_1d', True)
        param.set_value('use_profile', use_profile)
        # test both ways to specify a bad/missing atran dir
        if use_profile:
            param.set_value('atrandir', '')
        else:
            param.set_value('atrandir', '/bad/test/dir')

        red.extract_spectra()

        # 1d spectra and full product are saved
        outname = red.input[0][0].header['FILENAME']
        assert outname != hdul[0].header['FILENAME']
        assert os.path.isfile(str(tmpdir.join(outname)))
        assert os.path.isfile(str(tmpdir.join(outname.replace('SPM', 'SPC'))))

        # spectrum and transmission are attached to file for order 1 only
        assert 'SPECTRAL_FLUX_ORDER_01' in red.input[0]
        assert 'SPECTRAL_ERROR_ORDER_01' in red.input[0]
        assert 'TRANSMISSION_ORDER_01' in red.input[0]
        assert 'SPECTRAL_FLUX_ORDER_02' not in red.input[0]
        assert 'SPECTRAL_ERROR_ORDER_02' not in red.input[0]
        assert 'TRANSMISSION_ORDER_02' not in red.input[0]

    def test_bad_atran(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction(ffile1, tmpdir)
        hdul = red.input[0]

        # make a bad model file
        afile = str(tmpdir.join('psg_40K_45deg_5-28um.fits'))
        a_hdul = fits.HDUList(fits.PrimaryHDU([np.linspace(1000, 10000, 10),
                                               np.linspace(0, 1, 10)]))
        a_hdul.writeto(afile, overwrite=True)

        red.step_index = 4
        param = red.get_parameter_set()
        param.set_value('atranfile', afile)

        red.extract_spectra()

        capt = capsys.readouterr()
        assert f'Using PSG file: {afile}' in capt.out
        assert 'model range does not match data' in capt.err

        # transmission extension is all nans
        assert np.all(np.isnan(hdul['TRANSMISSION_ORDER_01'].data))

    @pytest.mark.parametrize('skyspec,onefile',
                             [(False, False), (False, True),
                              (True, False), (True, True)])
    def test_combine_spectra(self, tmpdir, skyspec, onefile):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        if onefile:
            red = self.prep_reduction(ffile1, tmpdir)
        else:
            ffile2 = self.make_spec_file(tmpdir, fname='test.sci.2.fits',
                                         add_appos=True, add_apmask=True)
            red = self.prep_reduction([ffile1, ffile2], tmpdir)
        red._set_sky_products(skyspec)

        # extract spectra
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5

        param = red.get_parameter_set()
        param.set_value('save', True)

        red.combine_spectra()
        assert len(red.input) == 1

        # 1d spectra and full product are saved
        outname = red.input[0][0].header['FILENAME']
        if not onefile:
            assert '0001-0002.fits' in outname
        else:
            assert '0001.fits' in outname
        if skyspec:
            assert 'SCM' in outname
            assert os.path.isfile(str(tmpdir.join(outname)))
            assert os.path.isfile(
                str(tmpdir.join(outname.replace('_SCM_', '_SCS_'))))
        else:
            assert 'COM' in outname
            assert os.path.isfile(str(tmpdir.join(outname)))
            assert os.path.isfile(
                str(tmpdir.join(outname.replace('_COM_', '_CMB_'))))

    def test_combine_spectra_multi_ap(self, tmpdir):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        ffile2 = self.make_spec_file(tmpdir, fname='test.sci.2.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # add second aperture to apmask
        red.input[0][0].header['APSGNO01'] = '1,1'
        red.input[0]['APERTURE_MASK_ORDER_01'].data[-4:, :] = 2

        # extract spectra
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5

        param = red.get_parameter_set()
        param.set_value('save', True)

        red.combine_spectra()
        assert len(red.input) == 1

        # 1d spectra and full product are saved
        outname = red.input[0][0].header['FILENAME']
        assert '0001-0002.fits' in outname
        assert 'COM' in outname
        assert os.path.isfile(str(tmpdir.join(outname)))
        assert os.path.isfile(
            str(tmpdir.join(outname.replace('_COM_', '_CMB_'))))

    def test_combine_spectra_mismatched_order(self, tmpdir):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        ffile2 = self.make_spec_file(tmpdir, fname='test.sci.2.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # add second order to file 2
        hdul = red.input[1]
        hdul[0].header['APPOSO02'] = '2'
        hdul[0].header['APSGNO02'] = '1'
        hdul.append(fits.ImageHDU(hdul['APERTURE_MASK_ORDER_01'].data,
                                  name='APERTURE_MASK_ORDER_02'))

        # extract spectra
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5

        with pytest.raises(ValueError) as err:
            red.combine_spectra()
        assert 'Mismatched wavenumbers or orders' in str(err)

    def test_combine_spectra_mismatched_wavenum(self, tmpdir):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        ffile2 = self.make_spec_file(tmpdir, fname='test.sci.2.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1, ffile2], tmpdir)

        # calibration in second value doesn't match first
        hdul = red.input[1]
        hdul['WAVEPOS_ORDER_01'].data *= 0.1

        # extract spectra
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5

        with pytest.raises(ValueError) as err:
            red.combine_spectra()
        assert 'Mismatched wavenumbers or orders' in str(err)

    def test_refine_wavecal(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1], tmpdir)
        hdul = deepcopy(red.input[0])

        # extract spectra
        red.step_index = 4
        red.extract_spectra()

        red.step_index = 6
        param = red.get_parameter_set()

        # no overrides
        red.refine_wavecal()
        assert 'No line identified' in capsys.readouterr().out

        # set a line
        param.set_value('identify_line', 100)
        param.set_value('identify_waveno', 442.0)

        # set a bad order
        param.set_value('identify_order', 'a')
        with pytest.raises(ValueError) as err:
            red.refine_wavecal()
        assert 'Invalid order number' in str(err)

        param.set_value('identify_order', 40)
        with pytest.raises(ValueError) as err:
            red.refine_wavecal()
        assert 'Invalid order number' in str(err)

        del red.input[0]['WAVEPOS_ORDER_03']
        param.set_value('identify_order', 3)
        with pytest.raises(ValueError) as err:
            red.refine_wavecal()
        assert 'Invalid order number' in str(err)

        param.set_value('identify_order', 1)
        red.refine_wavecal()
        capt = capsys.readouterr()
        assert 'Old central wavenumber: 442' in capt.out
        assert 'New central wavenumber: 435' in capt.out
        assert np.allclose(red.input[0][0].header['WNO0'], 435.1461032)
        assert not np.allclose(red.input[0]['WAVEPOS_ORDER_01'].data,
                               hdul['WAVEPOS_ORDER_01'].data)
        assert 'TRANSMISSION_ORDER_01' in red.input[0]

    def test_refine_wavecal_single_order(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1], tmpdir)
        red.input[0][0].header['INSTCFG'] = 'LOW'
        hdul = deepcopy(red.input[0])

        # extract spectra
        red.step_index = 4
        red.extract_spectra()

        red.step_index = 6
        param = red.get_parameter_set()
        param.set_value('identify_line', 100)
        param.set_value('identify_waveno', 442.0)
        param.set_value('identify_order', 1)

        red.refine_wavecal()
        capt = capsys.readouterr()
        assert 'Old central wavenumber: 442' in capt.out
        assert 'New central wavenumber: 435' in capt.out
        assert np.allclose(red.input[0][0].header['WNO0'], 435.140649)
        assert not np.allclose(red.input[0]['WAVEPOS_ORDER_01'].data,
                               hdul['WAVEPOS_ORDER_01'].data)
        assert 'TRANSMISSION_ORDER_01' in red.input[0]

    def test_parse_regions(self, capsys):
        red = EXESReduction()

        # nominal input
        region = '1:1-2,3-4;2:5-6'
        expected = {1: [[1., 2.], [3., 4.]], 2: [[5., 6.]]}
        result = red._parse_regions(region)
        assert len(result) == 2
        for r, e in zip(result, expected):
            assert np.allclose(r, e)

        # bad value in region: error
        region = '1:1-2,3-4;2:5-6a'
        with pytest.raises(ValueError):
            red._parse_regions(region)
        assert 'Could not read wavenumber region' in capsys.readouterr().err

        # bad order in region: error
        region = '1:1-2,3-4;a:5-6'
        with pytest.raises(ValueError):
            red._parse_regions(region)
        assert 'Could not read wavenumber region' in capsys.readouterr().err

    def test_trim_regions(self, tmpdir):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1], tmpdir)
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5
        red.combine_spectra()

        # extracted spectral file in expected format
        hdul = red.input[0]
        ordnum = '01'
        # 2 regions in range; out of range ignored
        nan_regions = [[449, 449.229], [449.3, 449.4], [460, 470]]

        red._trim_data(hdul, ordnum, nan_regions)

        # NaNs in image, error, spectral flux, and spectral error should match
        ny = hdul['FLUX_ORDER_01'].data.shape[0]
        assert np.sum(np.isnan(hdul['SPECTRAL_FLUX_ORDER_01'].data)) == 71
        assert np.sum(np.isnan(hdul['SPECTRAL_ERROR_ORDER_01'].data)) == 71
        assert np.sum(np.isnan(hdul['FLUX_ORDER_01'].data)) == 71 * ny
        assert np.sum(np.isnan(hdul['ERROR_ORDER_01'].data)) == 71 * ny

    def test_merge_orders(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1], tmpdir)
        hdul = red.input[0]

        # add a second order to extract
        hdul[0].header['APPOSO02'] = '2'
        hdul[0].header['APSGNO02'] = '1'
        hdul.append(fits.ImageHDU(hdul['APERTURE_MASK_ORDER_01'].data,
                                  name='APERTURE_MASK_ORDER_02'))

        # extract and combine spectra
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5
        red.combine_spectra()

        # prior to merge, 2 orders in input
        hdul = red.input[0]
        assert 'SPECTRAL_FLUX_ORDER_01' in hdul
        assert 'SPECTRAL_FLUX_ORDER_02' in hdul
        assert hdul['SPECTRAL_FLUX_ORDER_01'].data.shape == (198,)
        assert hdul['SPECTRAL_FLUX_ORDER_02'].data.shape == (198,)

        red.step_index = 7
        param = red.get_parameter_set()
        param.set_value('atrandir', '')

        red.merge_orders()

        # after merge, only 1 spectral extension
        hdul = red.input[0]
        assert 'SPECTRAL_FLUX_ORDER_01' not in hdul
        assert 'SPECTRAL_FLUX_ORDER_02' not in hdul
        assert 'SPECTRAL_FLUX' in hdul
        assert hdul['SPECTRAL_FLUX'].data.shape == (396,)

    def test_merge_orders_multi_ap(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1], tmpdir)
        hdul = red.input[0]

        # add a second order and aperture to extract
        red.input[0][0].header['APPOSO01'] = '2,4'
        red.input[0][0].header['APSGNO01'] = '1,1'
        red.input[0]['APERTURE_MASK_ORDER_01'].data[-4:, :] = 2
        hdul[0].header['APPOSO02'] = '2,4'
        hdul[0].header['APSGNO02'] = '1,1'
        hdul.append(fits.ImageHDU(hdul['APERTURE_MASK_ORDER_01'].data,
                                  name='APERTURE_MASK_ORDER_02'))

        # extract and combine spectra, leaving apertures
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5
        red.get_parameter_set().set_value('combine_aps', False)
        red.combine_spectra()

        # prior to merge, 2 orders, 2 aps in input
        hdul = red.input[0]
        assert 'SPECTRAL_FLUX_ORDER_01' in hdul
        assert 'SPECTRAL_FLUX_ORDER_02' in hdul
        assert hdul['SPECTRAL_FLUX_ORDER_01'].data.shape == (2, 198)
        assert hdul['SPECTRAL_FLUX_ORDER_02'].data.shape == (2, 198)

        red.step_index = 7
        red.merge_orders()

        # after merge, only 1 spectral extension but still has 2 aps
        hdul = red.input[0]
        assert 'SPECTRAL_FLUX_ORDER_01' not in hdul
        assert 'SPECTRAL_FLUX_ORDER_02' not in hdul
        assert 'SPECTRAL_FLUX' in hdul
        assert hdul['SPECTRAL_FLUX'].data.shape == (2, 396)

    def test_merge_orders_trim(self, tmpdir, capsys):
        ffile1 = self.make_spec_file(tmpdir, fname='test.sci.1.fits',
                                     add_appos=True, add_apmask=True)
        red = self.prep_reduction([ffile1], tmpdir)
        hdul = red.input[0]

        # add a second order to extract
        hdul[0].header['APPOSO02'] = '2'
        hdul[0].header['APSGNO02'] = '1'
        hdul.append(fits.ImageHDU(hdul['APERTURE_MASK_ORDER_01'].data,
                                  name='APERTURE_MASK_ORDER_02'))

        # extract and combine spectra
        red.step_index = 4
        red.extract_spectra()
        red.step_index = 5
        red.combine_spectra()

        red.step_index = 7
        param = red.get_parameter_set()
        param.set_value('trim_regions', '1:449-449.229;2:0-1000')

        red.merge_orders()

        # first order has some NaNs, second is all bad so ignored
        hdul = red.input[0]
        assert hdul['SPECTRAL_FLUX'].data.shape == (198,)
        assert np.sum(np.isnan(hdul['SPECTRAL_FLUX'].data)) == 53

        capt = capsys.readouterr()
        assert 'Trimming order 01 regions' in capt.out
        assert 'No good data in order 01' not in capt.err
        assert 'Trimming order 02 regions' in capt.out
        assert 'No good data in order 02' in capt.err

    @pytest.mark.parametrize('convert', [True, False])
    def test_bunit(self, tmpdir, convert):
        # run through all steps, checking for appropriate BUNIT keys
        red = EXESReduction()
        ffiles = self.make_low_files(tmpdir)
        red.load(ffiles)
        red.output_directory = tmpdir
        red.load_parameters()

        # run all steps, checking bunit
        bunit = 'ct'
        spatial_unit = 'arcsec'
        wave_unit = 'cm-1'
        response_unit = 'ct erg-1 s cm2 sr cm-1'
        spec_unit = 'erg s-1 cm-2 sr-1 (cm-1)-1'
        flat_unit = 'erg s-1 cm-2 sr-1 (cm-1)-1 ct-1'
        unitless = ['MASK', 'SPATIAL_MAP', 'SPATIAL_PROFILE',
                    'APERTURE_MASK', 'TRANSMISSION', 'FLAT_ILLUMINATION',
                    'ORDER_MASK', 'BADMASK']
        spatial = ['SLITPOS', 'SPATCAL']
        wave = ['WAVEPOS', 'WAVECAL']
        response = ['RESPONSE', 'RESPONSE_ERROR']
        spec = ['SPECTRAL_FLUX', 'SPECTRAL_ERROR']
        flat = ['FLAT', 'FLAT_ERROR']
        for step in red.recipe:

            params = red.get_parameter_set()
            params.set_value('skip_conversion', not convert)

            red.step()
            if step == 'flat_correct':
                bunit = 'erg s-1 cm-2 sr-1 (cm-1)-1'
            elif convert and step == 'convert_units':
                bunit = 'Jy/pixel'
                spec_unit = 'Jy'
                response_unit = 'ct/Jy'
                flat_unit = 'Jy/(pixel ct)'
            hdul = red.input[0]
            for hdu in hdul:
                extname = str(hdu.header.get('EXTNAME', 'UNKNOWN')).upper()
                extname = re.sub(r'_ORDER_\d+', '', extname)
                print(convert, extname, hdu.header['BUNIT'])
                if extname in unitless:
                    assert hdu.header['BUNIT'] == ''
                elif extname in spatial:
                    assert hdu.header['BUNIT'] == spatial_unit
                elif extname in wave:
                    assert hdu.header['BUNIT'] == wave_unit
                elif extname in response:
                    assert hdu.header['BUNIT'] == response_unit
                elif extname in spec:
                    assert hdu.header['BUNIT'] == spec_unit
                elif extname in flat:
                    assert hdu.header['BUNIT'] == flat_unit
                else:
                    assert hdu.header['BUNIT'] == bunit
