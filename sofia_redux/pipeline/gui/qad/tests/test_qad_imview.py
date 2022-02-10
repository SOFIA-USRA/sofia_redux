# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the QAD Image Viewer."""

import types

from astropy import log
from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase
from astropy.table import Table
from astropy.wcs import WCS

import numpy as np
import pytest

from sofia_redux.pipeline.gui.qad.qad_imview import QADImView, HAS_REGIONS
from sofia_redux.pipeline.gui.tests.test_qad_viewer import MockDS9


@pytest.fixture(scope='function')
def gaussian_data():
    # make a Gaussian source to test on
    size = 50
    fwhm = 4
    peak = 10
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    data = peak * np.exp(-4 * np.log(2)
                         * ((x - x0) ** 2 + (y - y0) ** 2)
                         / fwhm ** 2)
    return data, size, fwhm, peak, x0, y0


@pytest.fixture(scope='function')
def ds9_regions_wcs():
    reg = '# Region file format: DS9 version 4.1\n' \
          'global color=green dashlist=8 3 width=1 ' \
          'font="helvetica 10 normal roman" select=1 highlite=1 ' \
          'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n' \
          'fk5\n' \
          'circle(83.9058980,-5.4970264,62.417")\n' \
          'box(83.8851360,-5.4452279,128.396",91.713",0.00048086493)\n' \
          'polygon(83.8341123,-5.4841330,83.8012509,-5.4891015,83.8023857,' \
          '-5.5236350,83.8490243,-5.5332609)\n' \
          'ellipse(83.9292108,-5.4562630,20.380",63.179",0.00048086493)\n' \
          'annulus(83.8979271,-5.3911623,27.293",54.586")\n' \
          'box(83.9468298,-5.3985173,48.909",50.955",97.818",' \
          '101.910",0.00048086493)\n'
    return reg


@pytest.fixture(scope='function')
def ds9_regions_image():
    reg = '# Region file format: DS9 version 4.1\n' \
          'global color=green dashlist=8 3 width=1 font="helvetica 10 ' \
          'normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 ' \
          'delete=1 include=1 source=1\n' \
          'image\n' \
          'circle(83.759613,64.104134,13.718022)\n' \
          'box(100.10907,105.08872,28.218902,20.156704,1.1177834e-12)\n' \
          'polygon(140.29627,74.307014,166.17717,70.374312,165.28108,' \
          '43.051126,128.55144,35.436764)\n' \
          'ellipse(65.395114,96.354596,4.479121,13.885495,1.1177834e-12)\n' \
          'annulus(90.031207,147.86529,5.9984617,11.996923)\n' \
          'box(51.510941,142.04163,10.749231,11.198901,' \
          '21.498462,22.397803,1.1177834e-12)\n'

    return reg


@pytest.fixture(scope='function')
def header_wcs():
    hdict = {'WCSAXES': 2, 'CRPIX1': 124.5, 'CRPIX2': 114.5,
             'CDELT1': -0.0012638888888889, 'CDELT2': 0.0012638888888889,
             'CUNIT1': 'deg', 'CUNIT2': 'deg',
             'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
             'CRVAL1': 83.854168829081, 'CRVAL2': -5.4333338782771,
             'LONPOLE': 180.0, 'LATPOLE': -5.4333338782771, 'MJDREF': 0.0,
             'DATE-OBS': '2020-09-12T11:01:33.490',
             'MJD-OBS': 59104.459415394, 'RADESYS': 'FK5', 'EQUINOX': 2000.0}
    hdr = fits.Header(hdict)
    return WCS(hdr)


class QADPlotForTest(object):
    """Empty class to avoid bringing up plot windows."""
    def plot(self, data):
        print('plotting data: {}'.format(data))

    def show(self):
        pass

    def raise_(self):
        pass

    def setWindowTitle(self, *args, **kwargs):
        pass

    def set_scroll(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def isVisible(self):
        pass


class TestQADImView(object):
    """Test the QADImView class."""

    def setup_method(self):
        self.log_level = log.level

    def teardown_method(self):
        # make sure MockDS9 gets returned to normal settings
        MockDS9.reset_values()

        # reset log level
        log.setLevel(self.log_level)
        delattr(self, 'log_level')

    def mock_ds9(self, mocker):
        """Mock the pyds9 DS9 class."""
        mock_pyds9 = types.ModuleType('pyds9')
        mock_pyds9.DS9 = MockDS9
        mocker.patch.dict('sys.modules', {'pyds9': mock_pyds9})

        # also mock the plotter
        mocker.patch('sofia_redux.pipeline.gui.qad.qad_imview.MatplotlibPlot',
                     QADPlotForTest)

    def make_file(self, fname='test0.fits'):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data(fname)
        return ffile

    def make_imview(self):
        imviewer = QADImView()
        imviewer.disp_parameters = imviewer.default_parameters('display')
        imviewer.phot_parameters = imviewer.default_parameters('photometry')
        imviewer.plot_parameters = imviewer.default_parameters('plot')
        return imviewer

    def test_init(self, mocker, capsys):
        # mock missing pipecal and PyQt5
        mocker.patch('sofia_redux.pipeline.gui.qad.qad_imview.HAS_PIPECAL',
                     False)
        mocker.patch('sofia_redux.pipeline.gui.qad.qad_imview.HAS_PYQT5',
                     False)
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        capt = capsys.readouterr()
        assert 'Photometry tools are not available' in capt.err
        assert not imviewer.HAS_PIPECAL
        assert 'Plotting tools are not available' in capt.err
        assert imviewer.signals is None
        assert not imviewer.HAS_PYQT5

        # photometry and plot functions just return
        imviewer.photometry(0, 0)
        assert capsys.readouterr().out == ''
        imviewer.histogram_plot()
        assert capsys.readouterr().out == ''
        imviewer.pix2pix_plot()
        assert capsys.readouterr().out == ''
        imviewer.radial_plot()
        assert capsys.readouterr().out == ''

    def test_run_internal(self, mocker, capsys):
        """Test for error if via is not get or set in _run_internal."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        via = 'test'
        imviewer._run_internal('test command', via=via)
        capt = capsys.readouterr()
        assert 'unknown' in capt.err.lower()

        # check for wcs command without loaded data
        log.setLevel('DEBUG')
        MockDS9.get_test['fits size'] = '0'
        result = imviewer._run_internal('wcs align', via='get')
        assert result == ''
        assert 'No loaded data' in capsys.readouterr().out
        # same for an error in the size check
        MockDS9.get_test['fits size'] = ValueError('bad')
        result = imviewer._run_internal('wcs align', via='get')
        assert result == ''
        assert 'No loaded data' in capsys.readouterr().out

        # also check for type error from pyds9
        def mock_err(*args, **kwargs):
            raise TypeError('test error')
        mocker.patch.object(MockDS9, 'get', mock_err)
        imviewer.ds9 = MockDS9()
        imviewer._run_internal('test command', via='get')
        capt = capsys.readouterr()
        assert 'error in pyds9' in capt.err.lower()

    def test_defaults(self, mocker):
        """Test for default type options in default_parameters."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        default = imviewer.default_parameters('photometry')
        assert len(default) > 0
        default = imviewer.default_parameters('display')
        assert len(default) > 0
        default = imviewer.default_parameters('plot')
        assert len(default) > 0
        default = imviewer.default_parameters('test')
        assert len(default) == 0

    def test_imexam_loop(self, mocker, capsys):
        """Test keypress values in imexam loop."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # prevent infinite loops
        imviewer.break_loop = True

        # starting condition
        imviewer.phot_parameters['show_plots'] = False
        assert imviewer.ptable is None

        # press 'a' at coord 0, 0
        MockDS9.keypress = 'a 0 0'
        imviewer.imexam()
        assert isinstance(imviewer.ptable, Table)

        # help message should show on startup, and anytime 'h' is pressed
        assert capsys.readouterr().out.count('Available options') == 1
        MockDS9.keypress = 'h 0 0'
        imviewer.imexam()
        assert capsys.readouterr().out.count('Available options') == 2

        # clear data
        MockDS9.keypress = 'c 0 0'
        imviewer.imexam()
        assert imviewer.ptable is None
        assert len(imviewer.radial_data) == 0
        assert len(imviewer.histogram_data) == 0
        assert len(imviewer.p2p_data) == 0

        # quit imexam
        MockDS9.keypress = 'q 0 0'
        imviewer.imexam()
        assert imviewer.ptable is None

        # exercise a few non-standard options;
        # verify no errors are thrown and timeout is not hit

        # cs = wcs, tile = True
        MockDS9.get_test['wcs align'] = 'yes'
        MockDS9.get_test['tile'] = 'yes'
        imviewer.imexam()

        # raise a ValueError
        MockDS9.raise_error_get = True
        imviewer.imexam()
        MockDS9.raise_error_get = False

        # send a bad value from imexam
        MockDS9.keypress = 0
        imviewer.imexam()

        # send an incomprehensible value from imexam
        MockDS9.keypress = 'a'
        imviewer.imexam()

        # show radial plots
        imviewer.phot_parameters['show_plots'] = True
        MockDS9.keypress = 'a 0 0'
        imviewer.imexam()

        # make histogram or p2p plot
        MockDS9.keypress = 's 0 0'
        imviewer.imexam()
        MockDS9.keypress = 'p 0 0'
        imviewer.imexam()

        # clear data and plots
        imviewer.plotviewer = QADPlotForTest()
        MockDS9.keypress = 'c 0 0'
        imviewer.imexam()

    def test_run(self, mocker, capsys):
        """Test sending command to DS9."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # set log level to debug to check for messages
        log.setLevel('DEBUG')

        # test that restart is attempted before re-raising
        # a persistent error
        MockDS9.raise_error_get = True
        MockDS9.error_message = 'ds9 is no longer running'
        with pytest.raises(ValueError):
            imviewer.run('test')
        capt = capsys.readouterr()
        # start and restart log messages
        assert capt.out.lower().count('starting ds9') == 2

        # test another common error from pyds9: does not try startup
        MockDS9.error_message = "'nonetype' object has no attribute'"
        imviewer.run('test')
        capt = capsys.readouterr()
        assert capt.out.lower().count('starting ds9') == 0

    def test_extension_param(self, mocker):
        """Test extension number retrieval."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        imviewer.disp_parameters['extension'] = 'frame'
        assert imviewer.get_extension_param() == 'all'

        imviewer.disp_parameters['extension'] = 'cube'
        assert imviewer.get_extension_param() == 'all'

        imviewer.disp_parameters['extension'] = 'first'
        assert imviewer.get_extension_param() == 0

        imviewer.disp_parameters['extension'] = '1'
        assert imviewer.get_extension_param() == 1

        imviewer.disp_parameters['extension'] = 'SCI'
        assert imviewer.get_extension_param() == 'SCI'

        # special case for display, but not for headers -
        # if actually present, will show extension header;
        # if automagic only, will show all headers
        imviewer.disp_parameters['extension'] = 'S/N'
        assert imviewer.get_extension_param() == 'S/N'

    def test_lock(self, mocker, capsys):
        """Test DS9 frame locks."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # turn on messages from mock ds9
        MockDS9.verbose = True

        # default: cs=None, ltype=None, off=False
        # sets all 8 lock types (frame, slice, bin, etc.)
        imviewer.lock()
        capt = capsys.readouterr()
        assert capt.out.count('lock') == 8
        assert imviewer.cs == imviewer.disp_parameters['lock_image']

        # turn locks off
        imviewer.lock(off=True)
        capt = capsys.readouterr()
        assert capt.out.count('lock') == 8
        assert capt.out.count('no') == 8
        assert imviewer.cs == 'none'

        # specify an ltype
        imviewer.lock(ltype='frame')
        capt = capsys.readouterr()
        assert capt.out.count('lock') == 1
        assert imviewer.cs == imviewer.disp_parameters['lock_image']

        # specify a cs
        imviewer.lock(ltype='frame', cs='image')
        capt = capsys.readouterr()
        assert capt.out.count('lock') == 1
        assert imviewer.cs == 'image'

    def test_retrieve_data(self, mocker, capsys):
        """Test retrieving data stamp/wcs from DS9."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # set some responses to commands
        MockDS9.get_test = {'cube': '1'}

        # assuming data is a 10 x 10 image array, all zeros
        # input center is in DS9 1-based coordinates; output is 0-based
        result = imviewer.retrieve_data(5, 5)

        assert result['data'].shape == (10, 10)
        assert np.all(result['data'] == 0.0)
        assert result['window'] == 10
        assert result['xstart'] == 0
        assert result['ystart'] == 0
        assert result['xctr'] == 4
        assert result['yctr'] == 4

        # header is empty => wcs = None
        # "DS9" not aligned by wcs
        assert result['cs'] == 'image'
        assert result['wcs'] is None
        assert result['pix_scale'] == 1.0

        # throw error from WCS class, verify wcs is none
        class BadClass(object):
            def __init__(self):
                raise ValueError('test error')
        mocker.patch('astropy.wcs.WCS', BadClass)

        result = imviewer.retrieve_data(5, 5)
        assert result['wcs'] is None
        assert result['pix_scale'] == 1.0

        # restore the wcs class
        mocker.patch('astropy.wcs.WCS', WCS)

        # set a minimal header with celestial info
        hdr = fits.Header({'NAXIS1': 10,
                           'NAXIS2': 10,
                           'NAXIS3': 10,
                           'CTYPE1': 'RA---TAN',
                           'CTYPE2': 'DEC--TAN'})
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')

        # result should be a WCS object, with pixel scale
        # 1 degree (3600 arcsec)
        result = imviewer.retrieve_data(5, 5)
        assert isinstance(result['wcs'], WCS)
        assert result['pix_scale'] == 3600

        # set a cube instead; verify result is the same
        # (takes current slice only)
        MockDS9.data = np.zeros((10, 10, 10))
        result = imviewer.retrieve_data(5, 5)
        assert result['data'].shape == (10, 10)
        assert result['window'] == 10
        assert result['xstart'] == 0
        assert result['ystart'] == 0
        assert result['xctr'] == 4
        assert result['yctr'] == 4

        # but if cube keyword is set, cube is returned
        result = imviewer.retrieve_data(5, 5, cube=True)
        assert result['data'].shape == (10, 10, 10)

        # now make window smaller than image size, verify
        # sub-cube returned; center is the same
        imviewer.phot_parameters['window'] = 5
        result = imviewer.retrieve_data(5, 5)
        assert result['data'].shape == (5, 5)
        assert result['window'] == 5
        assert result['xstart'] == 1
        assert result['ystart'] == 1
        assert result['xctr'] == 4
        assert result['yctr'] == 4

        # move the center to the edge, verify full window
        # is still extracted
        result = imviewer.retrieve_data(10, 10)
        assert result['data'].shape == (5, 5)
        assert result['window'] == 5
        assert result['xstart'] == 5
        assert result['ystart'] == 5
        assert result['xctr'] == 9
        assert result['yctr'] == 9

        # get a bad slice value from DS9;
        # should still return first slice
        MockDS9.get_test = {'cube': '20'}
        imviewer.phot_parameters['window'] = 5
        result = imviewer.retrieve_data(5, 5)
        assert result['data'].shape == (5, 5)

        # make DS9 align by wcs; result should still
        # be the same, since WCS is neutral
        MockDS9.get_test = {'cube': '1',
                            'wcs align': 'yes',
                            'fits header': hdr.tostring(sep='\n')}
        result = imviewer.retrieve_data(5, 5)
        assert result['data'].shape == (5, 5)
        assert round(result['xctr']) == 4
        assert round(result['yctr']) == 4

        # set a 3D WCS, result is still the same
        hdr = fits.Header({'NAXIS1': 10,
                           'NAXIS2': 10,
                           'NAXIS3': 10,
                           'CTYPE1': 'RA---TAN',
                           'CTYPE2': 'DEC--TAN',
                           'CTYPE3': 'UNITLESS'})
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')
        result = imviewer.retrieve_data(5, 5)
        assert result['data'].shape == (5, 5)
        assert round(result['xctr']) == 4
        assert round(result['yctr']) == 4

    def test_retrieve_array_error(self, mocker, capsys):
        """Test retrieving data stamp/wcs from DS9."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        imgfile = self.make_file()
        imviewer.load([imgfile])
        mocker.patch.object(imviewer.ds9, 'get_arr2np',
                            side_effect=ValueError())
        with pytest.raises(ValueError):
            imviewer.retrieve_data(5, 5)
        capt = capsys.readouterr()
        assert 'cannot be retrieved as an array' in capt.err
        assert 'Try turning off cube' in capt.err

    def test_spec_test(self, mocker):
        """Test the check for spectral data."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # spectral data: NAXIS1 > 0, NAXIS2 < 6

        # with naxis3
        hdr = fits.Header({'NAXIS1': 10, 'NAXIS2': 10, 'NAXIS3': 10})
        hdul = fits.HDUList(fits.PrimaryHDU(header=hdr))
        assert imviewer.spec_test(hdul) == 'image'
        hdr = fits.Header({'NAXIS1': 10, 'NAXIS2': 5, 'NAXIS3': 10})
        hdul[0].header = hdr
        assert imviewer.spec_test(hdul) == 'spectrum_only'

        # without naxis3
        hdr = fits.Header({'NAXIS1': 10, 'NAXIS2': 10})
        hdul[0].header = hdr
        assert imviewer.spec_test(hdul) == 'image'
        hdr = fits.Header({'NAXIS1': 10, 'NAXIS2': 5})
        hdul[0].header = hdr
        assert imviewer.spec_test(hdul) == 'spectrum_only'

        # other end cases
        hdr = fits.Header({'NAXIS1': 0, 'NAXIS2': 0})
        hdul[0].header = hdr
        assert imviewer.spec_test(hdul) == 'image'
        hdr = fits.Header({'NAXIS1': 1, 'NAXIS2': 0})
        hdul[0].header = hdr
        assert imviewer.spec_test(hdul) == 'spectrum_only'

        # spectral flux hdul present is always 'spectrum'
        hdul.append(fits.ImageHDU(name='spectral_flux'))
        assert imviewer.spec_test(hdul) == 'spectrum'
        hdr = fits.Header({'NAXIS1': 0, 'NAXIS2': 0})
        hdul[0].header = hdr
        assert imviewer.spec_test(hdul) == 'spectrum'

    def test_set_defaults(self, mocker, capsys):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # clear any output
        capsys.readouterr()

        # turn on messages from mock ds9
        MockDS9.verbose = True

        imviewer.disp_parameters['tile'] = True
        imviewer.set_defaults()
        capt = capsys.readouterr()
        assert 'tile yes' in capt.out

        imviewer.disp_parameters['tile'] = False
        imviewer.set_defaults()
        capt = capsys.readouterr()
        assert 'tile no' in capt.out

        imviewer.disp_parameters['lock_image'] = 'wcs'
        imviewer.set_defaults()
        assert imviewer.cs == 'wcs'

        imviewer.disp_parameters['lock_image'] = 'image'
        imviewer.set_defaults()
        assert imviewer.cs == 'image'

    def test_startup(self, mocker, tmpdir, capsys):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # set files so that they are loaded on startup
        ffile = self.make_file()
        imviewer.files = ffile

        # no errors raised
        imviewer.startup()

        # files are no longer loaded on startup
        assert len(imviewer.files) == 0

        # also try with regfiles
        imviewer.regions = None
        imviewer.startup()
        assert imviewer.regions == []

        regfile = tmpdir.join('test.reg')
        regfile.write('test')
        imviewer.regions = str(regfile)
        imviewer.startup()

        assert imviewer.regions == []

        # now cause a value error on init (inaccessible DS9)
        MockDS9.raise_error_init = True
        with pytest.raises(ValueError) as err:
            imviewer.startup()
        assert 'not accessible' in str(err.value)
        MockDS9.raise_error_init = False

        # now cause an import error; verify it's not passed on
        capsys.readouterr()
        mocker.patch.dict('sys.modules', {'pyds9': None})
        imviewer.startup()
        assert 'Cannot import PyDS9' in capsys.readouterr().err
        assert not imviewer.HAS_DS9

    def test_overlays(self, mocker, capsys):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # set a FITS header with no overlay info
        hdr = fits.Header({'NAXIS1': 10,
                           'NAXIS2': 10,
                           'CTYPE1': 'RA---TAN',
                           'CTYPE2': 'DEC--TAN'})
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')

        # turn on messages from mock ds9
        MockDS9.verbose = True

        imviewer.overlay()
        imviewer.overlay_aperture(fits.Header())
        capt = capsys.readouterr()
        assert 'regions' not in capt.out

        # add photometry information
        hdr['SRCPOSX'] = 4
        hdr['SRCPOSY'] = 4
        hdr['STCENTX'] = 4
        hdr['STCENTY'] = 4
        hdr['PHOTAPER'] = 6
        hdr['PHOTSKAP'] = '7,8'
        hdr['STAPFLX'] = 10
        hdr['STAPSKY'] = 1
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')

        # test photometry overlay: should be 5 regions
        # (srcpos, ap center, ap radius, sky annulus, flux text)
        imviewer.overlay()
        capt = capsys.readouterr()
        assert capt.out.count('regions') == 5

        # aperture overlay still empty for this header
        imviewer.overlay_aperture(hdr)
        capt = capsys.readouterr()
        assert 'regions' not in capt.out

        # add spectral aperture information
        hdr['APPOSO01'] = '4'

        # try with bad wcs -- still empty
        badhdr = hdr.copy()
        badhdr['A_ORDER'] = 3
        imviewer.overlay_aperture(badhdr)
        capt = capsys.readouterr()
        assert 'regions' not in capt.out

        # test aperture overlay with good header: should be 1 region
        imviewer.overlay_aperture(hdr)
        capt = capsys.readouterr()
        assert capt.out.count('regions') == 1

        # add aprad: should be 3 regions
        # (center, ap lower, ap higher)
        hdr['APRADO01'] = '2'
        imviewer.overlay_aperture(hdr)
        capt = capsys.readouterr()
        assert capt.out.count('regions') == 3

        # add a psf rad and background regions
        # should add 2 more lines for the PSF and 4 for BGR
        hdr['PSFRAD01'] = '3'
        hdr['BGR'] = '0-1,8-9'
        imviewer.overlay_aperture(hdr)
        capt = capsys.readouterr()
        assert capt.out.count('regions') == 9

        # mismatched psfrad and appos: appos appears,
        # aprad and psfrad only for first, bgr appears
        hdr['APPOSO01'] = '4,5,6'
        imviewer.overlay_aperture(hdr)
        capt = capsys.readouterr()
        assert capt.out.count('regions') == 11

        # throw error from DS9; verify imviewer does not
        # raise an error, does not display regions
        MockDS9.raise_error_set = True
        imviewer.overlay()
        imviewer.overlay_aperture(hdr)
        capt = capsys.readouterr()
        assert 'regions' not in capt.out

    def test_radial_plot(self, mocker, capsys):
        """Test radial plot call.  Plot functionality is mocked."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        imviewer.radial_data = 'test data'
        imviewer.radial_plot()
        capt = capsys.readouterr()
        assert 'plotting' in capt.out

    def test_histogram_plot(self, mocker, capsys):
        """Test histogram plot call.  Plot functionality is mocked."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        imviewer.histogram_data = 'test data'
        imviewer.histogram_plot()
        capt = capsys.readouterr()
        assert 'plotting' in capt.out

    def test_p2p_plot(self, mocker, capsys):
        """Test p2p plot call.  Plot functionality is mocked."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        imviewer.p2p_data = 'test data'
        imviewer.pix2pix_plot()
        capt = capsys.readouterr()
        assert 'plotting' in capt.out

        # just returns if data is None
        imviewer.p2p_data = None
        imviewer.pix2pix_plot()
        capt = capsys.readouterr()
        assert 'plotting' not in capt.out

    def test_photometry(self, mocker, capsys, gaussian_data):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        data, size, fwhm, peak, x0, y0 = gaussian_data

        # set mock values for retrieve_data
        MockDS9.data = data
        MockDS9.get_test = {'cube': '1', 'frame': '1',
                            'frame active': '1'}

        # set parameters
        imviewer.phot_parameters = \
            {'model': 'gaussian',
             'window': size,
             'window_units': 'pixels',
             'fwhm': fwhm,
             'fwhm_units': 'pixels',
             'psf_radius': fwhm * 3,
             'aperture_units': 'pixels',
             'bg_inner': None,
             'bg_width': None,
             'show_plots': False}

        # define a set of tests for the results of the fit
        def values_test():
            assert abs(abs(imviewer.ptable['Peak'][0]) - peak) < 1
            assert abs(imviewer.ptable['X'][0] - (x0 + 1)) < 1
            assert abs(imviewer.ptable['Y'][0] - (y0 + 1)) < 1
            assert abs(imviewer.ptable['FWHM (px)'][0] - fwhm) < 1
            assert abs(imviewer.ptable['FWHM (")'][0] - fwhm) < 1
            # for fwhm = 4
            assert abs(abs(imviewer.ptable['Flux'][0]) - peak * 18.1) < 1

        def badfit_values_test():
            assert abs(abs(imviewer.ptable['Peak'][0]) - peak) < 1
            assert abs(imviewer.ptable['X'][0] - (x0 + 1)) < 1
            assert abs(imviewer.ptable['Y'][0] - (y0 + 1)) < 1
            assert abs(abs(imviewer.ptable['Flux'][0]) - peak * 18.1) < 1
            # above are same; fwhm is nan
            assert np.isnan(imviewer.ptable['FWHM (px)'][0])
            assert np.isnan(imviewer.ptable['FWHM (")'][0])

        # test photometry
        imviewer.photometry(x0 + 1, y0 + 1)

        # verify printed
        capt = capsys.readouterr()
        assert 'already measured' not in capt.out
        assert 'Frame' in capt.out

        # verify values
        values_test()

        # try again, verify fit is not repeated but print is
        imviewer.photometry(x0 + 1, y0 + 1)
        capt = capsys.readouterr()
        assert 'already measured' in capt.out
        assert 'Frame' in capt.out

        # now try a different model; values are still the same
        # within rounding
        imviewer.ptable = None
        imviewer.phot_parameters['model'] = 'moffat'
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()

        # clear output
        capsys.readouterr()

        # try an unknown model
        imviewer.ptable = None
        imviewer.phot_parameters['model'] = 'unknown'
        imviewer.photometry(x0 + 1, y0 + 1)
        capt = capsys.readouterr()
        assert 'Invalid profile selection' in capt.err

        # reset model
        imviewer.phot_parameters['model'] = 'gaussian'

        # set a background region; should still be the same (bg ~ 0)
        imviewer.ptable = None
        imviewer.phot_parameters['bg_inner'] = 20
        imviewer.phot_parameters['bg_width'] = 4
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()

        # set an invalid background: no error, same result
        imviewer.ptable = None
        imviewer.phot_parameters['bg_inner'] = 20
        imviewer.phot_parameters['bg_width'] = -4
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()

        # set units to arcsec -- still same, since pixscal = 1.0
        imviewer.ptable = None
        imviewer.phot_parameters['fwhm_units'] = 'arcsec'
        imviewer.phot_parameters['aperture_units'] = 'arcsec'
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()

        # cause a null result in peak find -- still same,
        # since source is at center
        mocker.patch('photutils.find_peaks', lambda x, y, npeaks=None: None)
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()

        # set non-float apertures
        # still the same -- auto set radii are similar to above
        imviewer.ptable = None
        imviewer.phot_parameters['psf_radius'] = 'other'
        imviewer.phot_parameters['bg_inner'] = 'other'
        imviewer.phot_parameters['bg_width'] = 'other'
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()

        # make the source negative -- still the same because
        # of absolute values taken
        data *= -1
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()
        data *= -1

        # add a null WCS to header, verify RA/Dec are set, rest is same
        hdr = fits.Header({'NAXIS1': size,
                           'NAXIS2': size,
                           'CDELT1': 1 / 3600,
                           'CDELT2': 1 / 3600,
                           'CTYPE1': 'RA---TAN',
                           'CTYPE2': 'DEC--TAN'})
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')
        MockDS9.get_test['tile'] = 'yes'
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()
        assert abs(imviewer.ptable['RA'][0] - (x0 + 1) / 3600) < 1
        assert abs(imviewer.ptable['Dec'][0] - (y0 + 1) / 3600) < 1

        # trigger error in RA/Dec conversion; RA/Dec should be nan
        def err_func(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(WCS, 'wcs_pix2world', err_func)
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        values_test()
        assert np.isnan(imviewer.ptable['RA'][0])
        assert np.isnan(imviewer.ptable['Dec'][0])

        # show radial plots; verify model data is stored for a good fit
        imviewer.phot_parameters['show_plots'] = True
        imviewer.ptable = None
        imviewer.radial_data = []
        imviewer.photometry(x0 + 1, y0 + 1)
        assert len(imviewer.radial_data) == 1
        assert len(imviewer.radial_data[0]) == 4
        assert len(imviewer.radial_data[0]['overplot']) == 5
        imviewer.phot_parameters['show_plots'] = False

        # clear output
        capsys.readouterr()

        # mock a bad fit
        mocker.patch('sofia_redux.calibration.'
                     'pipecal_photometry.pipecal_fitpeak',
                     side_effect=RuntimeError())

        # for both models
        imviewer.phot_parameters['model'] = 'gaussian'
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        badfit_values_test()
        capt = capsys.readouterr()
        assert 'Bad fit' in capt.err

        imviewer.phot_parameters['model'] = 'moffat'
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        badfit_values_test()
        capt = capsys.readouterr()
        assert 'Bad fit' in capt.err

        # show radial plots; verify model data is None for bad fit
        imviewer.phot_parameters['show_plots'] = True
        imviewer.ptable = None
        imviewer.radial_data = []
        imviewer.photometry(x0 + 1, y0 + 1)
        assert len(imviewer.radial_data) == 1
        assert len(imviewer.radial_data[0]) == 4
        assert len(imviewer.radial_data[0]['overplot']) == 3
        imviewer.phot_parameters['show_plots'] = False

        # return a specific model for exercising a few other branches
        fitpar = {'baseline': 0.0,
                  'dpeak': peak,
                  'col_mean': x0,
                  'row_mean': y0,
                  'col_sigma': fwhm,
                  'row_sigma': fwhm,
                  'theta': 0,
                  'beta': 3}
        sigma = fitpar.copy()

        def mock_fitpeak(*args, **kwargs):
            return fitpar.copy(), sigma, None

        mocker.patch('sofia_redux.calibration.'
                     'pipecal_photometry.pipecal_fitpeak',
                     mock_fitpeak)

        # make elliptical in the y-direction
        fitpar['col_sigma'] = 2  # x
        fitpar['row_sigma'] = 4  # y
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        assert imviewer.ptable['Ellip.'][0] == 0.5
        assert imviewer.ptable['PA'][0] == 90

        # again with a different pa
        fitpar['theta'] = np.deg2rad(-1)
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        assert imviewer.ptable['Ellip.'][0] == 0.5
        assert imviewer.ptable['PA'][0] == 89

        # make elliptical in the x-direction
        fitpar['col_sigma'] = 4  # x
        fitpar['row_sigma'] = 2  # y
        fitpar['theta'] = 0
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        assert imviewer.ptable['Ellip.'][0] == 0.5
        assert imviewer.ptable['PA'][0] == 90

        # again with a different pa
        fitpar['theta'] = np.deg2rad(-1)
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        assert imviewer.ptable['Ellip.'][0] == 0.5
        assert imviewer.ptable['PA'][0] == 89

        # make a bad fit: FWHM too high
        fitpar['col_sigma'] = 20
        fitpar['row_sigma'] = 40
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        badfit_values_test()
        capt = capsys.readouterr()
        assert 'Bad fit' in capt.err

        # again with a good fit, but centroid outside the stamp
        fitpar['col_mean'] = 100
        fitpar['row_mean'] = 100
        fitpar['col_sigma'] = fwhm
        fitpar['row_sigma'] = fwhm
        fitpar['theta'] = 0
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        # peak is as returned by model, flux is zero
        assert abs(abs(imviewer.ptable['Peak'][0]) - peak) < 1
        assert imviewer.ptable['Flux'][0] == 0

        # show plots; verify model data is stored for a good fit
        imviewer.phot_parameters['show_plots'] = True
        imviewer.ptable = None
        imviewer.radial_data = []
        imviewer.photometry(x0 + 1, y0 + 1)
        assert len(imviewer.radial_data) == 1
        assert len(imviewer.radial_data[0]) == 4
        assert len(imviewer.radial_data[0]['overplot']) == 5
        assert len(imviewer.ptable) == 1
        imviewer.phot_parameters['show_plots'] = False

        # make all data NaN: should have no result and no error
        data *= np.nan
        imviewer.ptable = None
        imviewer.photometry(x0 + 1, y0 + 1)
        assert len(imviewer.ptable) == 0

    def test_load_data(self, mocker, capsys, tmpdir):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        ffile = self.make_file()
        hdul = fits.open(ffile)
        fname = hdul[0].header['FILENAME']

        hdrs = [[hdu.header for hdu in hdul]]

        # set non-standard cmap
        imviewer.disp_parameters['cmap'] = 'heat'

        # load a list of headers
        imviewer.load(hdrs)
        assert len(imviewer.files) == 0
        assert len(imviewer.regions) == 0
        assert len(imviewer.headers) == 1
        assert imviewer.headers[fname][0] == hdul[0].header

        # remove the filename key -- name is now 'Array 0'
        hdul[0].header.remove('FILENAME')
        fname = 'Array 0'
        imviewer.load(hdrs)
        assert len(imviewer.files) == 0
        assert len(imviewer.regions) == 0
        assert len(imviewer.headers) == 1
        assert imviewer.headers[fname][0] == hdul[0].header

        # load an HDUList
        imviewer.load([hdul])
        assert len(imviewer.files) == 1
        assert imviewer.files[0] == hdul
        assert len(imviewer.regions) == 0
        assert len(imviewer.headers) == 1
        assert imviewer.headers[fname][0] == hdul[0].header

        # load a region file with the data
        regfile = tmpdir.join('test.reg')
        regfile.write('test')
        imviewer.load([hdul], regfiles=str(regfile))
        assert imviewer.files[0] == hdul
        assert len(imviewer.regions) == 1

        # turn on messages from DS9
        MockDS9.verbose = True
        MockDS9.get_test['frame'] = '1'

        # load multiframe from hdul
        imviewer.disp_parameters['extension'] = 'frame'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'multiframe' in capt.out
        assert len(imviewer.files) == 1

        # and from file
        imviewer.load(ffile)
        capt = capsys.readouterr()
        assert 'multiframe' in capt.out
        assert len(imviewer.files) == 1

        # check reload of same data
        imviewer.reload()
        capt = capsys.readouterr()
        assert 'multiframe' in capt.out
        assert len(imviewer.files) == 1

        # load cube
        imviewer.disp_parameters['extension'] = 'cube'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'mecube' in capt.out
        assert len(imviewer.files) == 1

        # load number extension
        imviewer.disp_parameters['extension'] = '1'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'frame new' in capt.out
        assert len(imviewer.files) == 1

        # load string extension
        imviewer.disp_parameters['extension'] = 'SCI'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'frame new' in capt.out
        assert len(imviewer.files) == 1

        # load bad extension: file is kept, but error is logged
        imviewer.disp_parameters['extension'] = 'TEST'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'Error loading extension' in capt.err
        assert len(imviewer.files) == 1

        # attempt automagic S/N extension: fails because FLUX
        # and ERROR not present
        imviewer.disp_parameters['extension'] = 'S/N'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'frame new' not in capt.out
        assert 'Cannot determine S/N' in capt.err
        assert len(imviewer.files) == 1

        # rename extensions
        hdul[1].header['EXTNAME'] = 'FLUX'
        hdul[2].header['EXTNAME'] = 'ERROR'
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'frame new' in capt.out
        assert len(imviewer.files) == 1

        # raise error in load: will try loading from memory,
        # then tempfile
        def err_func(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(imviewer, 'set_defaults', return_value=None)
        mocker.patch.object(imviewer, '_load_from_memory', err_func)

        log.setLevel('DEBUG')
        imviewer.disp_parameters['extension'] = 'first'

        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert "Loading from memory" in capt.out
        assert "Loading from tempfile" in capt.out

        hdul.close()

    def test_load_alternates(self, mocker, capsys):
        """Test error cases in internal load methods."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # raise error from set
        MockDS9.raise_error_set = True

        ffile = self.make_file()
        data = fits.open(ffile)

        # load from memory raises error
        with pytest.raises(ValueError):
            imviewer._load_from_memory('test cmd', 'test_file.fits', data)
        capt = capsys.readouterr()
        assert 'Cannot load image' in capt.err

        # load from tempfile just issues message
        status = imviewer._load_from_tempfile('test cmd',
                                              'test_file.fits', data)
        capt = capsys.readouterr()
        assert 'Cannot load image' in capt.err
        assert status == 0

    def test_load_spec(self, mocker, capsys):
        """Test that spec data are passed to spec viewer."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        ffile = self.make_file()

        class SpecTestClass(object):
            def start(self):
                return

            def update(self, data):
                print('loaded spec')

            def display(self):
                return

        mocker.patch('sofia_redux.pipeline.gui.qad.qad_imview.EyeViewer',
                     SpecTestClass)
        mocker.patch.object(imviewer, 'spec_test',
                            return_value='spectrum_only')
        imviewer.HAS_EYE = True

        # load as spectral data, passed as file
        imviewer.load([ffile])
        capt = capsys.readouterr()
        assert 'loaded spec' in capt.out
        assert len(imviewer.files) == 1

        # load as spectral data, passes as HDUList - should also work
        hdul = fits.open(ffile)
        imviewer.load([hdul])
        capt = capsys.readouterr()
        assert 'loaded spec' in capt.out
        assert len(imviewer.files) == 1

    def test_load_errors(self, mocker, capsys):
        """Test various error conditions on load."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        ffile = self.make_file()

        log.setLevel('DEBUG')

        # load an extension -- good
        MockDS9.verbose = True
        imviewer.disp_parameters['extension'] = 'SCI'
        imviewer.load(ffile)
        capt = capsys.readouterr()
        assert '[SCI]' in capt.out

        # raise error in run with extension
        def err_func(*args, **kwargs):
            raise ValueError('bad run')
        mocker.patch.object(QADImView, 'run', err_func)
        mocker.patch.object(imviewer, 'set_defaults', return_value=None)

        imviewer.disp_parameters['extension'] = 'TEST'
        with pytest.raises(ValueError):
            imviewer.load(ffile)
        capt = capsys.readouterr()
        assert 'Error in XPA command' in capt.err

        # raise error without extension
        imviewer.disp_parameters['extension'] = 'first'
        with pytest.raises(ValueError):
            imviewer.load(ffile)
        capt = capsys.readouterr()
        assert 'Error in XPA command' in capt.err

        # reset
        MockDS9.verbose = False
        mocker.patch.object(QADImView, 'run', QADImView.run)

        # raise IOError for fits file
        def err_func(*args, **kwargs):
            raise IOError('test error')
        mocker.patch('sofia_redux.pipeline.gui.qad.'
                     'qad_imview.fits.open', err_func)
        imviewer.load(ffile)
        capt = capsys.readouterr()
        assert 'Cannot load' in capt.err

        # try to load something unexpected
        data = {1: 2}
        imviewer.load(data)
        capt = capsys.readouterr()
        assert 'Cannot load' in capt.err

    def test_load_spec_imgs(self, mocker, capsys):
        """Test loading spectral data with image data."""
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        img_file = self.make_file()
        spec_file = self.make_file('blank.fits')

        class SpecTestClass(object):
            def start(self):
                return

            def update(self, data):
                for fname in data:
                    print('loaded {}'.format(fname))

            def display(self):
                return

        mocker.patch('sofia_redux.pipeline.gui.qad.qad_imview.EyeViewer',
                     SpecTestClass)
        imviewer.HAS_EYE = True

        imviewer.load([img_file, img_file, spec_file])
        capt = capsys.readouterr()
        assert 'loaded {}'.format(spec_file) in capt.out
        assert 'loaded {}'.format(img_file) not in capt.out
        assert len(imviewer.files) == 3

        # load specdata with image with bad WCS -- should work fine
        hdul = fits.open(img_file)
        hdul[0].header['A_ORDER'] = 3
        imviewer.load([hdul, spec_file])
        capt = capsys.readouterr()
        assert 'loaded {}'.format(spec_file) in capt.out
        assert len(imviewer.files) == 2

    def test_load_mult_reg(self, mocker, capsys, tmpdir):
        self.mock_ds9(mocker)
        MockDS9.verbose = True
        imgfile = self.make_file()
        regfile = tmpdir.join('test.reg')
        regfile.write('test')
        regfile2 = tmpdir.join('test2.reg')
        regfile2.write('test')
        imviewer = self.make_imview()

        # load one region file with multiple frames: should load into all
        imviewer.load([imgfile, imgfile], regfiles=str(regfile))
        assert 'region load all' in capsys.readouterr().out

        # load two region files with two frames: should load into each
        MockDS9.get_test['frame active'] = '1 2'
        imviewer.load([imgfile, imgfile],
                      regfiles=[str(regfile), str(regfile2)])
        capt = capsys.readouterr()
        assert 'region load all' not in capt.out
        assert capt.out.count('region load') == 2

    def test_no_regions(self, mocker):
        mocker.patch('sofia_redux.pipeline.gui.qad.'
                     'qad_imview.HAS_REGIONS', False)
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # this would fail if attempted; without HAS_REGIONS, just
        # returns None regardless
        assert imviewer._region_mask(None, None, None, None, None) is None

    @pytest.mark.skipif(not HAS_REGIONS, reason='regions not installed')
    @pytest.mark.parametrize('x,y,match',
                             [(84, 62, 'circle'),
                              (98, 104, 'rectangle'),
                              (64, 94, 'ellipse'),
                              (51, 142, 'rectangle'),
                              (80, 145, 'annulus'),
                              (154, 58, 'polygon')])
    def test_region_mask(self, mocker, capsys, ds9_regions_image,
                         ds9_regions_wcs, header_wcs, x, y, match):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()

        # same regions, recorded in image and wcs coordinates;
        # x, y is always passed in image coordinates
        img_mask = imviewer._region_mask('image', [ds9_regions_image], x, y,
                                         header_wcs)
        wcs_mask = imviewer._region_mask('wcs', [ds9_regions_wcs], x, y,
                                         header_wcs)

        assert wcs_mask is not None
        assert img_mask is not None
        assert match in capsys.readouterr().out.lower()

        # trigger an error in the parser
        log.setLevel('DEBUG')
        mocker.patch('sofia_redux.pipeline.gui.qad.'
                     'qad_imview.ar.Regions.parse',
                     side_effect=ValueError())
        bad_mask = imviewer._region_mask('image', [ds9_regions_image], x, y,
                                         header_wcs)
        assert bad_mask is None
        assert 'Region parser error' in capsys.readouterr().out

    def test_histogram(self, mocker, capsys, gaussian_data):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        data, size, fwhm, peak, x0, y0 = gaussian_data

        # set mock values for retrieve_data
        MockDS9.data = data
        MockDS9.get_test = {'cube': '1', 'frame': '1',
                            'frame active': '1'}

        # starting parameters
        imviewer.plot_parameters['separate_plots'] = True
        imviewer.plot_parameters['window'] = 20
        imviewer.plot_parameters['hist_limits'] = [0, peak]

        # test histogram
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 1

        # run again: adds another data set
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 2

        # run again with separate_plots = False: adds to last plot
        imviewer.plot_parameters['separate_plots'] = False
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 2

        # verify stats are printed each time
        capt = capsys.readouterr()
        assert capt.out.count(f'Histogram at: {x0 + 1},{y0 + 1}') == 3
        assert capt.out.count('Using the analysis window') == 3
        assert capt.out.count('Total pixels: 400') == 3
        assert capt.out.count('Min, max, sum') == 3

        # reset and run again with full image
        imviewer.histogram_data = []
        imviewer.plot_parameters['window'] = None
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 1
        capt = capsys.readouterr()
        assert 'Using the full image' in capt.out
        assert 'Total pixels: 2500' in capt.out

        # remaining tests require regions
        if not HAS_REGIONS:
            return

        # add a region under the cursor
        log.setLevel('DEBUG')
        MockDS9.get_test['regions -system image'] = \
            f'image\ncircle({x0 + 1},{y0 + 1},10)'
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 1
        capt = capsys.readouterr()
        assert 'Contained in CirclePixelRegion' in capt.out
        assert 'Total pixels: 305' in capt.out

        # add a null WCS to header, verify region still works
        hdr = fits.Header({'NAXIS1': size,
                           'NAXIS2': size,
                           'CRVAL1': x0,
                           'CRVAL2': y0,
                           'CRPIX1': x0,
                           'CRPIX2': y0,
                           'CDELT1': 1,
                           'CDELT2': 1,
                           'CTYPE1': 'RA---TAN',
                           'CTYPE2': 'DEC--TAN'})
        MockDS9.verbose = True
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')
        MockDS9.get_test['fits size'] = '50 50'
        MockDS9.get_test['wcs align'] = 'yes'
        MockDS9.get_test['tile'] = 'yes'
        imviewer.plot_parameters['window'] = 20 * 3600
        imviewer.plot_parameters['window_units'] = 'arcsec'

        # add a WCS region that contains the point
        MockDS9.get_test['regions -system wcs'] = \
            f'fk5\ncircle({x0 + 1},{y0 + 1},10)'

        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 1
        capt = capsys.readouterr()
        assert 'Contained in CirclePixelRegion' in capt.out
        assert 'Total pixels: 314' in capt.out

        # mock imexam input: assumes pixel position, then will
        # just take a window at the top of the image, since
        # the center values are out of range
        imviewer.histogram(x0 + 1000, y0 + 1000)
        capt = capsys.readouterr()
        assert 'WCS position retrieval failed' in capt.out
        assert 'Total pixels: 400' in capt.out

        # mock an unloaded frame: nothing happens
        imviewer.histogram_data = []
        MockDS9.get_test['fits size'] = '0 0'
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 0
        assert 'Total pixels' not in capsys.readouterr().out

        # mock an error in retrieving data: warns only
        MockDS9.get_test['fits size'] = '50 50'
        mocker.patch.object(imviewer, 'retrieve_data',
                            side_effect=ValueError())
        imviewer.histogram(x0 + 1, y0 + 1)
        assert len(imviewer.histogram_data) == 0
        capt = capsys.readouterr()
        assert 'Error in retrieving data' in capt.out
        assert 'Total pixels' not in capt.out

    def test_histogram_summary_stats(self, mocker, capsys, gaussian_data):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        data, size, fwhm, peak, x0, y0 = gaussian_data

        # set mock values for retrieve_data
        MockDS9.data = data
        MockDS9.get_test = {'cube': '1', 'frame': '1',
                            'frame active': '1'}

        # starting parameters
        imviewer.plot_parameters['separate_plots'] = True
        imviewer.plot_parameters['window'] = 20
        imviewer.plot_parameters['hist_limits'] = [0, peak]

        # expected values for simulated data
        expected = {'mean': 0.45324,
                    'median': 1.2831e-4,
                    'clipped mean': 0.011499,
                    'clipped median': 2.6974e-5}

        # test histogram overplot for each possible stat
        for stat in expected.keys():
            imviewer.histogram_data = []
            imviewer.plot_parameters['summary_stat'] = stat
            imviewer.histogram(x0 + 1, y0 + 1)
            assert len(imviewer.histogram_data) == 1
            overplot = imviewer.histogram_data[0]['overplot']
            assert np.allclose(overplot[1]['args'][0], expected[stat],
                               rtol=1e-4)

    def test_make_s2n(self, mocker):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        arr = np.arange(10, dtype=float) + 1

        # S/N directly present
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=arr.copy(),
                            header=fits.Header({'EXTNAME': 'S/N'})),
            fits.ImageHDU(data=arr.copy(), name='ERROR')])
        s2n = imviewer._make_s2n('test.fits', hdul)
        assert len(s2n) == 2
        assert s2n[0].data is None
        assert np.allclose(s2n[1].data, arr)

        # FLUX and ERROR extensions
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=arr.copy(),
                            header=fits.Header({'EXTNAME': 'FLUX'})),
            fits.ImageHDU(data=arr.copy(), name='ERROR')])
        s2n = imviewer._make_s2n('test.fits', hdul)
        assert len(s2n) == 2
        assert s2n[0].data is None
        assert np.allclose(s2n[1].data, 1)

        # FLUX and STDDEV extensions
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=arr.copy(),
                            header=fits.Header({'EXTNAME': 'FLUX'})),
            fits.ImageHDU(data=arr.copy(), name='STDDEV')])
        s2n = imviewer._make_s2n('test.fits', hdul)
        assert len(s2n) == 2
        assert s2n[0].data is None
        assert np.allclose(s2n[1].data, 1)

        # STOKES I and ERROR extensions
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=arr.copy(),
                            header=fits.Header({'EXTNAME': 'STOKES I'})),
            fits.ImageHDU(data=arr.copy(), name='ERROR I')])
        s2n = imviewer._make_s2n('test.fits', hdul)
        assert len(s2n) == 2
        assert s2n[0].data is None
        assert np.allclose(s2n[1].data, 1)

        # clip s/n data
        hdul[0].data **= 2
        imviewer.disp_parameters['s2n_range'] = 2, 5
        s2n = imviewer._make_s2n('test.fits', hdul)
        assert np.allclose(s2n[1].data, [np.nan, 2, 3, 4, 5, np.nan,
                                         np.nan, np.nan, np.nan, np.nan],
                           equal_nan=True)

    def test_pix2pix(self, mocker, capsys, gaussian_data):
        self.mock_ds9(mocker)
        imviewer = self.make_imview()
        data, size, fwhm, peak, x0, y0 = gaussian_data

        # set mock values for retrieve_data
        MockDS9.data = data
        MockDS9.get_test = {'cube': '1', 'frame': '1',
                            'frame active': '1'}

        # starting parameters
        imviewer.plot_parameters['separate_plots'] = True
        imviewer.plot_parameters['window'] = 20
        imviewer.plot_parameters['p2p_reference'] = 1

        # test plot with one frame: no plot made
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 0
        assert 'Pixel comparison requires 2' in capsys.readouterr().out

        # add another frame: now makes plot
        MockDS9.get_test = {'cube': '1', 'frame': '1',
                            'frame active': '1 2',
                            'tile': 'yes'}
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 1
        # data is the same for the mock ds9
        d1, d2 = imviewer.p2p_data[0]['overplot'][0]['args']
        assert np.allclose(d1, d2)

        # run again: adds a new plot to data
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 2

        # run again with separate_plots = False: adds to last plot
        imviewer.plot_parameters['separate_plots'] = False
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 2

        # verify message are printed each time for each frame
        capt = capsys.readouterr()
        assert capt.out.count(f'Pixel comparison at: {x0 + 1},{y0 + 1}') == 6
        assert capt.out.count('Using the analysis window') == 6

        # reset and run again with full image
        imviewer.p2p_data = []
        imviewer.plot_parameters['window'] = None
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 1
        capt = capsys.readouterr()
        assert 'Using the full image' in capt.out

        # remaining tests require regions
        if not HAS_REGIONS:
            return

        # add a region under the cursor
        log.setLevel('DEBUG')
        MockDS9.get_test['regions -system image'] = \
            f'image\ncircle({x0 + 1},{y0 + 1},10)'
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 1
        capt = capsys.readouterr()
        assert 'Contained in CirclePixelRegion' in capt.out

        # add a null WCS to header, verify region still works
        hdr = fits.Header({'NAXIS1': size,
                           'NAXIS2': size,
                           'CRVAL1': x0,
                           'CRVAL2': y0,
                           'CRPIX1': x0,
                           'CRPIX2': y0,
                           'CDELT1': 1,
                           'CDELT2': 1,
                           'CTYPE1': 'RA---TAN',
                           'CTYPE2': 'DEC--TAN'})
        MockDS9.verbose = True
        MockDS9.get_test['fits header'] = hdr.tostring(sep='\n')
        MockDS9.get_test['fits size'] = '50 50'
        MockDS9.get_test['wcs align'] = 'yes'
        MockDS9.get_test['tile'] = 'yes'
        imviewer.plot_parameters['window'] = 20 * 3600
        imviewer.plot_parameters['window_units'] = 'arcsec'

        # add a WCS region that contains the point
        MockDS9.get_test['regions -system wcs'] = \
            f'fk5\ncircle({x0 + 1},{y0 + 1},10)'

        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 1
        capt = capsys.readouterr()
        assert 'Contained in CirclePixelRegion' in capt.out

        # mock imexam input: assumes pixel position, then will
        # just take a window at the top of the image, since
        # the center values are out of range
        imviewer.pix2pix(x0 + 1000, y0 + 1000)
        capt = capsys.readouterr()
        assert 'WCS position retrieval failed' in capt.out
        assert f'Pixel comparison at: {x0 + 1000},{y0 + 1000}' in capt.out

        # mock an unloaded frame: nothing happens
        imviewer.p2p_data = []
        MockDS9.get_test['fits size'] = '0 0'
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 0

        # mock an error in retrieving data: warns only
        MockDS9.get_test['fits size'] = '50 50'
        mocker.patch.object(imviewer, 'retrieve_data',
                            side_effect=ValueError())
        imviewer.pix2pix(x0 + 1, y0 + 1)
        assert len(imviewer.p2p_data) == 0
        capt = capsys.readouterr()
        assert 'Error in retrieving data' in capt.out
