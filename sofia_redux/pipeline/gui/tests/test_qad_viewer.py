# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the QAD Viewer class."""

import os
import shutil
import types

import numpy as np
from astropy import log
from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase
import pytest

from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.gui.qad.qad_imview import QADImView

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


class MockDS9(object):
    """Mock DS9 display."""

    # these variables are class level for
    # easy modification as needed in tests

    # flag to log commands
    verbose = False

    # flag to raise error instead of returning
    # a value
    raise_error_init = False
    raise_error_get = False
    raise_error_set = False
    error_message = 'test error'

    # general return value for get
    get_return_value = ''

    # return values for specific commands
    get_test = {}

    # status value for set
    set_return_value = 1

    # keypress value to return for imexam
    keypress = 'q 0.0 0.0'

    # data to return for get_arr2np
    data = np.zeros((10, 10))

    def __init__(self):
        if self.raise_error_init:
            raise ValueError(self.error_message)

    def get(self, *args):
        if self.raise_error_get:
            raise ValueError(self.error_message)
        value = self.get_return_value
        try:
            cmd = ' '.join(args)
            if self.verbose:
                if len(cmd) < 200:
                    log.info('get: {}'.format(cmd))
                else:
                    log.info('get: {} ...'.format(cmd[0:200]))
            if 'imexam' in cmd:
                value = self.keypress
            elif cmd in self.get_test:
                value = self.get_test[cmd]
                if isinstance(value, Exception):
                    raise value
        except ValueError:
            pass
        return value

    def set(self, *args):
        if self.raise_error_set:
            raise ValueError(self.error_message)
        try:
            args = [str(a) for a in args]
            if self.verbose:
                cmd = ' '.join(args)
                if len(cmd) < 200:
                    log.info('set: {}'.format(cmd))
                else:
                    log.info('set: {} ...'.format(cmd[0:200]))
        except (ValueError, TypeError):
            pass
        return self.set_return_value

    def get_arr2np(self):
        return self.data

    @staticmethod
    def reset_values():
        MockDS9.verbose = False
        MockDS9.raise_error_init = False
        MockDS9.raise_error_get = False
        MockDS9.raise_error_set = False
        MockDS9.error_message = 'test error'
        MockDS9.get_return_value = ''
        MockDS9.get_test = {}
        MockDS9.set_return_value = 1
        MockDS9.keypress = 'q 0.0 0.0'
        MockDS9.data = np.zeros((10, 10))


@pytest.mark.skipif("not HAS_PYQT5")
class TestQADViewer(object):
    """Test the QADViewer and QADViewerSettings classes."""
    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def setup_method(self):
        self.log_level = log.level

    def teardown_method(self):
        # make sure MockDS9 gets returned to normal settings
        MockDS9.reset_values()
        # reset log level
        log.setLevel(self.log_level)
        delattr(self, 'log_level')

    def make_window(self, qtbot):
        """Make and register a QAD viewer."""
        view = QADViewer()
        view.start(parent=qtbot)
        return view

    def make_file(self, fname='test0.fits'):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data(fname)
        return ffile

    def mock_ds9(self, mocker):
        mock_pyds9 = types.ModuleType('pyds9')
        mock_pyds9.DS9 = MockDS9
        mocker.patch.dict('sys.modules', {'pyds9': mock_pyds9})

    def test_start(self, qtbot, mocker):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)
        assert isinstance(view.imviewer, QADImView)

    def test_close(self, qtbot, mocker, capsys):
        MockDS9.verbose = True

        self.mock_ds9(mocker)
        view = self.make_window(qtbot)
        assert isinstance(view.imviewer, QADImView)
        view.imviewer.startup()
        assert isinstance(view.imviewer.ds9, MockDS9)

        view.close()

        capt = capsys.readouterr()
        assert 'quit' in capt.out

        # raise error; verify nothing happens
        MockDS9.raise_error_set = True
        view.close()
        capt = capsys.readouterr()
        assert capt.err == ''

    def test_no_op(self, qtbot, mocker):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        # set imviewer to None and verify no errors
        view.imviewer = None
        assert view.display() is None
        assert view.update('test data') is None
        assert view.reset() is None

    def test_reset(self, qtbot, mocker):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        # display a file
        ffile = self.make_file()
        view.update(ffile)

        # reset
        view.reset()

        # check that display_data is empty
        assert len(view.imviewer.files) == 0

    def test_display_files(self, qtbot, mocker, tmpdir, capsys):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        # reset display parameters
        view.imviewer.phot_parameters = \
            view.imviewer.default_parameters('photometry')
        view.imviewer.disp_parameters = \
            view.imviewer.default_parameters('display')

        # get a test fits file
        ffile = self.make_file()

        # update viewer
        view.update(ffile)

        # wait until load is finished
        # and test image was displayed
        def test():
            QtWidgets.QApplication.processEvents()
            assert view.settings.load_worker is None
            assert ffile in view.imviewer.files
        qtbot.waitUntil(test)

        # send .reg file
        regfile = tmpdir.join('test.reg')
        regfile.write('test')
        view.imviewer.files = []
        view.update([ffile, str(regfile)])

        qtbot.waitUntil(test)
        assert str(regfile) in view.imviewer.regions

        # send other file; verify no errors raised, nothing happens
        def log_cmd(*args, **kwargs):
            print(args)
            return b'value'
        mocker.patch('subprocess.check_output', log_cmd)
        txtfile = tmpdir.join('test.txt')
        txtfile.write('test')

    def test_display_data(self, qtbot, mocker, capsys):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        # get a test fits file
        ffile = self.make_file()
        hdul = fits.open(ffile)

        # update viewer
        view.update(hdul)

        def test():
            QtWidgets.QApplication.processEvents()
            assert view.settings.load_worker is None
        qtbot.waitUntil(test)
        assert isinstance(view.imviewer.files[0], fits.HDUList)

        # trigger error in load; verify it gets logged
        def err_cmd(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(QADImView, 'load', err_cmd)
        view.update(hdul)
        qtbot.waitUntil(test)
        capt = capsys.readouterr()
        assert 'test error' in capt.err

    def test_settings_getvalue(self, qtbot, mocker, capsys, tmpdir):
        # ignore any user override parameters
        mocker.patch.object(os.path, 'expanduser', return_value=str(tmpdir))

        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        orig_disp = view.imviewer.disp_parameters.copy()
        orig_phot = view.imviewer.phot_parameters.copy()
        orig_plot = view.imviewer.plot_parameters.copy()

        # call getvalue; verify unchanged
        view.settings.getDispValue()
        view.settings.getPhotValue()
        view.settings.getPlotValue()
        assert view.imviewer.disp_parameters == orig_disp
        assert view.imviewer.phot_parameters == orig_phot
        assert view.imviewer.plot_parameters == orig_plot

        # change some widget values and verify change is passed on
        view.settings.tileBox.toggle()
        if view.settings.modelTypeBox.currentIndex() == 1:
            view.settings.modelTypeBox.setCurrentIndex(0)
        else:
            view.settings.modelTypeBox.setCurrentIndex(1)
        view.settings.apradBox.setText('test value')
        view.settings.plotColorBox.setText('test value')
        view.settings.getDispValue()
        view.settings.getPhotValue()
        view.settings.getPlotValue()
        assert view.imviewer.disp_parameters != orig_disp
        assert view.imviewer.phot_parameters != orig_phot
        assert view.imviewer.plot_parameters != orig_plot
        assert view.imviewer.disp_parameters['tile'] == \
            view.settings.tileBox.isChecked()
        assert view.imviewer.phot_parameters['model'] == \
            str(view.settings.modelTypeBox.currentText()).lower()
        assert view.imviewer.plot_parameters['color'] == \
            str(view.settings.plotColorBox.text()).lower()

        # reset values to original
        view.settings.resetDisp()
        view.settings.resetPhot()
        view.settings.resetPlot()
        view.settings.getDispValue()
        view.settings.getPhotValue()
        view.settings.getPlotValue()
        assert view.imviewer.disp_parameters == orig_disp
        assert view.imviewer.phot_parameters == orig_phot
        assert view.imviewer.plot_parameters == orig_plot

        # restore defaults
        view.settings.restoreDisp()
        view.settings.restorePhot()
        view.settings.restorePlot()
        view.settings.getDispValue()
        view.settings.getPhotValue()
        view.settings.getPlotValue()

        # for display parameters, the ds9 disable should match
        # the pipeline value; all others match default
        default_par = view.imviewer.default_parameters('display')
        for key in view.imviewer.disp_parameters:
            if key == 'ds9_viewer':
                assert view.imviewer.disp_parameters[key] == \
                    view.imviewer.disp_parameters['ds9_viewer_pipeline']
            else:
                assert view.imviewer.disp_parameters[key] == default_par[key]

        assert view.imviewer.phot_parameters == \
            view.imviewer.default_parameters('photometry')
        assert view.imviewer.plot_parameters == \
            view.imviewer.default_parameters('plot')

        # put bad values in float boxes; verify unchanged
        view.settings.resetPhot()
        view.settings.windowSizeBox.setText('test')
        view.settings.fwhmBox.setText('test')
        view.settings.getPhotValue()
        assert view.imviewer.phot_parameters == orig_phot
        view.settings.resetPlot()
        view.settings.histBinBox.setText('test')
        view.settings.histLimitsBox.setText('test')
        view.settings.p2pReferenceBox.setText('test')
        view.settings.getPlotValue()
        assert view.imviewer.plot_parameters == orig_plot

        # verify these get set to auto or None if not float
        view.settings.apradBox.setText('test')
        view.settings.bgrinBox.setText('test')
        view.settings.bgwidBox.setText('test')
        view.settings.getPhotValue()
        assert view.imviewer.phot_parameters['psf_radius'] == 'auto'
        assert view.imviewer.phot_parameters['bg_inner'] == 'auto'
        assert view.imviewer.phot_parameters['bg_width'] == 'auto'
        view.settings.bgrinBox.setText('none')
        view.settings.bgwidBox.setText('none')
        view.settings.getPhotValue()
        assert view.imviewer.phot_parameters['bg_inner'] is None
        assert view.imviewer.phot_parameters['bg_width'] is None
        view.settings.plotWindowSizeBox.setText('test')
        view.settings.getPlotValue()
        assert view.imviewer.plot_parameters['window'] is None

        def test():
            QtWidgets.QApplication.processEvents()
            assert view.settings.load_worker is None
        qtbot.waitUntil(test)

        # trigger error in reload at end of getDispValue;
        # verify it is logged
        ffile = self.make_file()
        view.imviewer.files = [ffile]

        def err_cmd(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(QADImView, 'load', err_cmd)

        # trigger reload with changed parameter
        view.settings.tileBox.toggle()
        view.settings.getDispValue()

        qtbot.waitUntil(test)
        capt = capsys.readouterr()
        assert 'test error' in capt.err

    def test_settings_setvalue(self, qtbot, mocker):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        disp = view.imviewer.disp_parameters.copy()

        # set non-default extension
        disp['extension'] = '2'
        view.settings.setDispValue(disp)
        assert view.settings.extensionBox.itemText(3) == '2'

        view.settings.getDispValue()
        assert view.imviewer.disp_parameters == disp

        # remove some keys from the settings
        prev = view.imviewer.disp_parameters.copy()
        del disp['cmap']
        del disp['zoom_fit']
        del disp['tile']
        del disp['overplots']
        del disp['ds9_viewer']
        del disp['ds9_viewer_pipeline']
        disp['extension'] = 's/n'
        disp['s2n_range'] = 'test'
        view.settings.setDispValue(disp)
        view.settings.getDispValue()
        for key, val in view.imviewer.disp_parameters.items():
            if key in disp:
                if key == 's2n_range':
                    # invalid value set to default
                    assert val is None
                else:
                    assert val == disp[key]
            else:
                assert val == prev[key]

        # same for phot -- set only passed ones
        phot = {'model': 'gaussian'}
        prev = view.imviewer.phot_parameters.copy()
        view.settings.setPhotValue(phot)
        view.settings.getPhotValue()
        for key, val in view.imviewer.phot_parameters.items():
            if key in phot:
                assert val == phot[key]
            else:
                assert val == prev[key]

        # same for plot -- set only valid passed ones
        plot = {'share_axes': 'x', 'hist_limits': 'test'}
        prev = view.imviewer.plot_parameters.copy()
        view.settings.setPlotValue(plot)
        view.settings.getPlotValue()
        for key, val in view.imviewer.plot_parameters.items():
            if key in plot and key != 'hist_limits':
                assert val == plot[key]
            else:
                assert val == prev[key]

    def test_status(self, qtbot, mocker):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        view.settings.setStatus('test')
        assert view.settings.status.text() == 'test'

    def test_display_header(self, qtbot, mocker, tmpdir):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        # mock the show function
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)

        # no headers loaded
        view.settings.onDisplayHeader()
        assert 'no headers' in view.settings.status.text().lower()

        # update viewer
        ffile = self.make_file()
        view.update(ffile)

        def test():
            QtWidgets.QApplication.processEvents()
            assert view.settings.load_worker is None
        qtbot.waitUntil(test)

        view.settings.onDisplayHeader()
        assert 'headers displayed' in view.settings.status.text().lower()
        assert os.path.basename(ffile) in \
            view.settings.headviewer.windowTitle()
        assert os.path.basename(ffile) in \
            view.settings.headviewer.textEdit.toPlainText()

        # multiple files
        ffile2 = str(tmpdir.join('ffile2.fits'))
        shutil.copyfile(ffile, ffile2)
        fits.setval(ffile2, 'EXTNAME', 1, value='TEST')
        view.update([ffile, ffile2])
        qtbot.waitUntil(test)
        view.settings.onDisplayHeader()
        assert '...' in \
            view.settings.headviewer.windowTitle()
        ptext = view.settings.headviewer.textEdit.toPlainText()
        assert os.path.basename(ffile) in ptext
        assert os.path.basename(ffile2) in ptext

        # trigger error in extension retrieval;
        # verify all extensions shown
        del view.imviewer.disp_parameters['extension']
        view.settings.onDisplayHeader()
        ptext = view.settings.headviewer.textEdit.toPlainText()
        assert ptext.count('Extension') == \
            (1 + fits.getval(ffile, 'NEXTEND')) * 2

        # choose one extension instead: present in one, other shows all
        view.imviewer.disp_parameters['extension'] = 'TEST'
        view.settings.onDisplayHeader()
        ptext = view.settings.headviewer.textEdit.toPlainText()
        assert ptext.count('Extension') == 2 + fits.getval(ffile, 'NEXTEND')

        # try a garbage extension; all should display
        view.imviewer.disp_parameters['extension'] = '20'
        view.settings.onDisplayHeader()
        ptext = view.settings.headviewer.textEdit.toPlainText()
        assert ptext.count('Extension') == \
            (1 + fits.getval(ffile, 'NEXTEND')) * 2

    def test_imexam(self, qtbot, mocker, capsys):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)
        ffile = self.make_file()
        view.update(ffile)
        view.imviewer.HAS_DS9 = True

        log.setLevel('DEBUG')
        view.settings.onImExam()

        def test1():
            assert 'imexam' in view.settings.status.text().lower()
        qtbot.waitUntil(test1)

        # mock ds9 will just quit
        def test2():
            QtWidgets.QApplication.processEvents()
            assert view.settings.imexam_worker is None
        qtbot.waitUntil(test2)

        # trigger error from ds9: caught in imexam loop
        MockDS9.raise_error_get = True
        view.settings.onImExam()
        qtbot.waitUntil(test1)
        qtbot.waitUntil(test2)
        capt = capsys.readouterr()
        assert 'Error in ImExam' in capt.err
        MockDS9.raise_error_get = False

        # trigger error in imexam itself
        def err_cmd(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(QADImView, 'imexam', err_cmd)
        view.settings.onImExam()
        qtbot.waitUntil(test1)
        qtbot.waitUntil(test2)
        capt = capsys.readouterr()
        assert 'test error' in capt.err

    def test_save_param(self, qtbot, mocker, capsys, tmpdir):
        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        # test no directory
        view.settings.cfg_dir = None
        view.settings.onSave()
        capt = capsys.readouterr()
        assert 'not saving' in capt.err
        assert 'saved' not in view.settings.status.text()

        # test non-existent directory
        view.settings.cfg_dir = str(tmpdir.join('config'))
        view.settings.onSave()
        capt = capsys.readouterr()
        assert 'not saving' in capt.err
        assert 'saved' not in view.settings.status.text()

        # test valid directory
        os.makedirs(view.settings.cfg_dir, exist_ok=True)
        view.settings.onSave()
        capt = capsys.readouterr()
        assert 'not saving' not in capt.err
        assert 'saved' in view.settings.status.text()

    def test_load_param(self, qtbot, mocker, tmpdir):
        # mock expanduser to return tmpdir
        mocker.patch('os.path.expanduser',
                     return_value=str(tmpdir))
        os.mkdir(str(tmpdir.join('.qad')))
        cfg1 = tmpdir.join('.qad', 'photometry.cfg')
        cfg1.write("model = 'lorentzian'\n")
        cfg2 = tmpdir.join('.qad', 'display.cfg')
        cfg2.write("cmap = 'heat'\n")
        cfg3 = tmpdir.join('.qad', 'plot.cfg')
        cfg3.write("color = 'tab20b'\n")

        self.mock_ds9(mocker)
        view = self.make_window(qtbot)

        assert view.imviewer.disp_parameters['cmap'] == 'heat'
        assert view.imviewer.phot_parameters['model'] == 'lorentzian'
        assert view.imviewer.plot_parameters['color'] == 'tab20b'
