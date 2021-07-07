# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the QAD Main Panel class."""

import os
import shutil
import types

from astropy import log
from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase
import pytest

from sofia_redux.pipeline.gui.qad.qad_main_panel import QADMainWindow
from sofia_redux.pipeline.gui.qad.qad_imview import QADImView
from sofia_redux.pipeline.gui.qad.qad_dialogs \
    import DispSettingsDialog, PhotSettingsDialog, PlotSettingsDialog
from sofia_redux.pipeline.gui.tests.test_qad_viewer import MockDS9

try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    QtWidgets, QtCore = None, None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.mark.skipif("not HAS_PYQT5")
class TestQADMain(object):
    """Test the QADMainWindow class"""

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

    def make_window(self, qtbot, tmpdir):
        """Make and register a main window with new files displayed."""
        mw = QADMainWindow()
        qtbot.addWidget(mw)

        # set a root path with two fits file in it
        mw.rootpath = str(tmpdir)

        ffile = self.make_file()
        self.newfile1 = str(tmpdir.join('file1.fits'))
        shutil.copyfile(ffile, self.newfile1)

        self.newfile2 = str(tmpdir.join('file2.fits'))
        shutil.copyfile(self.newfile1, self.newfile2)
        fits.setval(self.newfile2, 'EXTNAME', 1, value='TEST')

        mw.setRoot()
        return mw

    def make_window_simple(self, qtbot):
        """Make and register a main window, no files displayed."""
        mw = QADMainWindow()
        qtbot.addWidget(mw)
        return mw

    def make_file(self, fname='test0.fits'):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data(fname)
        return ffile

    def mock_ds9(self, mocker):
        """Mock the pyds9 DS9 class."""
        mock_pyds9 = types.ModuleType('pyds9')
        mock_pyds9.DS9 = MockDS9
        mocker.patch.dict('sys.modules', {'pyds9': mock_pyds9})

    def select_files(self, mainwindow, fname=None):
        """
        Select files from the treeview.

        Parameters
        ----------
        all : bool
            If True, all files selected.  If False, only the first.
        """
        mw = mainwindow

        # select the named file
        model = mw.treeView.model()
        selection = mw.treeView.selectionModel()
        selection.clearSelection()
        if fname is not None:
            idx1 = model.index(fname, 0)
            idx2 = model.index(fname, model.columnCount() - 1)
            itemsel = QtCore.QItemSelection(idx1, idx2)
            selection.select(itemsel, QtCore.QItemSelectionModel.Select)
        else:
            # or else select them all
            for fname in os.listdir(mw.rootpath):
                fname = os.path.join(mw.rootpath, fname)
                idx1 = model.index(fname, 0)
                idx2 = model.index(fname, model.columnCount() - 1)
                itemsel = QtCore.QItemSelection(idx1, idx2)
                selection.select(itemsel,
                                 QtCore.QItemSelectionModel.Select)

        selected = []
        idx = mw.treeView.selectionModel().selectedRows()
        for i in idx:
            fpath = os.path.abspath(model.filePath(i))
            selected.append(fpath)
        return selected

    def test_startup_error(self, qtbot, mocker, tmpdir, capsys):
        # test for error messages if imviewer can't start

        # mock the show and exec function
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=None)

        # mock the imviewer class
        class BadClass(object):
            def __init__(self):
                raise ValueError('test error')
        mocker.patch(
            'sofia_redux.pipeline.gui.qad.qad_imview.QADImView', BadClass)

        # make a window
        self.mock_ds9(mocker)
        mw = self.make_window(qtbot, tmpdir)

        # verify error was logged
        capt = capsys.readouterr()
        assert 'test error' in capt.err

        # try a few functions -- should not throw errors
        self.select_files(mw, fname=self.newfile1)
        mw.onSaveSettings()
        mw.onDisplayHeader()
        mw.onRow()
        mw.onDisplaySettings()
        mw.onPhotometrySettings()
        mw.onPlotSettings()

    def test_directory_setting(self, qtbot, mocker, tmpdir):
        self.mock_ds9(mocker)

        # start in a current directory
        with tmpdir.as_cwd():
            mocker.patch('sys.argv', ['test'])
            mw = self.make_window_simple(qtbot)
            assert mw.rootpath == str(tmpdir)

        # start with a path in the argument
        mocker.patch('sys.argv', ['test', str(tmpdir)])
        mw = self.make_window_simple(qtbot)
        assert mw.rootpath == str(tmpdir)

        # now go home
        home_dir = os.path.expanduser('~')
        mw.onHome()
        assert mw.rootpath == home_dir

        # now go up a directory
        mw.onPrevious()
        assert mw.rootpath == os.path.dirname(home_dir)

        # now return to previous directory
        mw.onNext()
        assert mw.rootpath == home_dir

        # now open the tmpdir again, via the file dialog
        mocker.patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                            return_value=str(tmpdir))
        mw.onOpen()
        assert mw.rootpath == str(tmpdir)

    def test_display_header(self, qtbot, mocker, tmpdir, capsys):
        self.mock_ds9(mocker)
        mw = self.make_window(qtbot, tmpdir)

        # mock the show function
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)

        # no files selected
        mw.onDisplayHeader()

        def test():
            assert 'no fits files' in mw.statusBar.currentMessage().lower()
        qtbot.waitUntil(test)

        # select the first file
        self.select_files(mw, fname=self.newfile1)

        # display headers
        mw.onDisplayHeader()
        assert 'headers displayed' in mw.statusBar.currentMessage().lower()
        assert os.path.basename(self.newfile1) in \
            mw.headviewer.windowTitle()
        assert os.path.basename(self.newfile1) in \
            mw.headviewer.textEdit.toPlainText()

        # select both files
        self.select_files(mw)

        mw.onDisplayHeader()
        assert '...' in \
               mw.headviewer.windowTitle()
        ptext = mw.headviewer.textEdit.toPlainText()
        assert os.path.basename(self.newfile1) in ptext
        assert os.path.basename(self.newfile2) in ptext

        # trigger error in extension retrieval;
        # verify all extensions shown
        del mw.imviewer.disp_parameters['extension']
        mw.onDisplayHeader()
        ptext = mw.headviewer.textEdit.toPlainText()
        assert ptext.count('Extension') == \
            (1 + fits.getval(self.newfile1, 'NEXTEND')) * 2

        # choose one extension instead
        mw.imviewer.disp_parameters['extension'] = 'TEST'
        mw.onDisplayHeader()
        ptext = mw.headviewer.textEdit.toPlainText()
        assert ptext.count('Extension') == \
               2 + fits.getval(self.newfile1, 'NEXTEND')

        # try a garbage extension; all should display
        mw.imviewer.disp_parameters['extension'] = '20'
        mw.onDisplayHeader()
        ptext = mw.headviewer.textEdit.toPlainText()
        assert ptext.count('Extension') == \
            (1 + fits.getval(self.newfile1, 'NEXTEND')) * 2
        assert 'Cannot load' not in capsys.readouterr().err

        # mock a garbage file; should ignore and keep old text
        mocker.patch('astropy.io.fits.open', side_effect=OSError('bad file'))
        mw.onDisplayHeader()
        ptext2 = mw.headviewer.textEdit.toPlainText()
        assert ptext2 == ptext
        assert 'Cannot load' in capsys.readouterr().err

    def test_display_files(self, qtbot, mocker, tmpdir, capsys):
        self.mock_ds9(mocker)
        mw = self.make_window(qtbot, tmpdir)

        # reset display preferences
        mw.imviewer.phot_parameters = \
            mw.imviewer.default_parameters('photometry')
        mw.imviewer.disp_parameters = \
            mw.imviewer.default_parameters('display')
        mw.imviewer.plot_parameters = \
            mw.imviewer.default_parameters('plot')

        # select the first file
        self.select_files(mw, self.newfile1)

        # display it
        mw.onRow()

        # test image was displayed
        assert self.newfile1 in mw.imviewer.files

        # send .reg file
        regfile = tmpdir.join('test.reg')
        regfile.write('test')

        # select and display all files
        self.select_files(mw)
        mw.onRow()

        assert self.newfile1 in mw.imviewer.files
        assert self.newfile2 in mw.imviewer.files
        assert str(regfile) in mw.imviewer.regions

        # display non-fits file; verify no errors raised

        # make a file and mock system calls
        def log_cmd(*args, **kwargs):
            print(args)
            return b'value'
        mocker.patch('subprocess.check_output', log_cmd)
        txtfile = tmpdir.join('test.txt')
        txtfile.write('test')

        # clear previous output
        capsys.readouterr()

        # qad calls open for Mac
        mocker.patch('sys.platform', 'darwin')
        mocker.patch('subprocess.call', log_cmd)

        self.select_files(mw, fname=str(txtfile))
        mw.onRow()

        capt = capsys.readouterr()
        assert "'open'" in capt.out
        assert os.path.basename(txtfile) in capt.out

        # raise an error in subprocess, verify nothing happens
        def err_cmd(*args, **kwargs):
            raise RuntimeError
        mocker.patch('subprocess.call', err_cmd)
        self.select_files(mw, fname=str(txtfile))
        mw.onRow()
        capt = capsys.readouterr()
        assert capt.out == ''
        assert capt.err == ''

        # tries to call xdg-open otherwise
        mocker.patch('sys.platform', 'other')
        mocker.patch('subprocess.call', log_cmd)
        self.select_files(mw, fname=str(txtfile))
        mw.onRow()
        capt = capsys.readouterr()
        assert "'xdg-open'" in capt.out
        assert os.path.basename(txtfile) in capt.out

        # raise error in check output instead
        mocker.patch('subprocess.check_output', err_cmd)
        self.select_files(mw, fname=str(txtfile))
        mw.onRow()
        capt = capsys.readouterr()
        assert capt.out == ''
        assert capt.err == ''

        # make a new directory and select it instead
        newdir = str(tmpdir.join('newdir'))
        os.makedirs(newdir, exist_ok=True)
        self.select_files(mw, fname=newdir)
        mw.onRow()
        assert mw.rootpath == newdir

        # trigger error in imview load on a fits file;
        # verify it gets logged
        def err_cmd(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(QADImView, 'load', err_cmd)
        self.select_files(mw, fname=self.newfile1)
        mw.onRow()
        assert 'cannot open' in mw.statusBar.currentMessage().lower()

    def test_keypress(self, qtbot, mocker, tmpdir):
        self.mock_ds9(mocker)
        mw = self.make_window(qtbot, tmpdir)

        # select the first file
        self.select_files(mw, fname=self.newfile1)

        # mock the hasFocus function
        mocker.patch.object(QtWidgets.QTreeView, 'hasFocus',
                            return_value=True)

        # press enter
        qtbot.keyPress(mw, QtCore.Qt.Key_Return)

        # verify file is displayed
        assert self.newfile1 in mw.imviewer.files

    def test_settings_dialogs(self, qtbot, mocker):
        self.mock_ds9(mocker)
        mw = self.make_window_simple(qtbot)

        # get a settings dict and modify it
        orig_dict = mw.imviewer.disp_parameters.copy()
        mod_dict = mw.imviewer.disp_parameters.copy()
        test_key = 'extension'
        mod_dict[test_key] = 'test'

        # mock the exec and getValue functions
        mocker.patch.object(DispSettingsDialog, 'exec_',
                            return_value=1)
        mocker.patch.object(DispSettingsDialog, 'getValue',
                            return_value=mod_dict)

        # call the dialog
        mw.onDisplaySettings()

        # verify the value was changed and others were not
        new_dict = mw.imviewer.disp_parameters
        for key in orig_dict:
            if key == test_key:
                assert new_dict[key] != orig_dict[key]
                assert new_dict[key] == mod_dict[key]
            else:
                assert new_dict[key] == orig_dict[key]
                assert new_dict[key] == mod_dict[key]

        # same for the photometry settings
        orig_dict = mw.imviewer.phot_parameters.copy()
        mod_dict = mw.imviewer.phot_parameters.copy()
        test_key = 'model'
        mod_dict[test_key] = 'test'

        mocker.patch.object(PhotSettingsDialog, 'exec_',
                            return_value=1)
        mocker.patch.object(PhotSettingsDialog, 'getValue',
                            return_value=mod_dict)
        mw.onPhotometrySettings()

        new_dict = mw.imviewer.phot_parameters
        for key in orig_dict:
            if key == test_key:
                assert new_dict[key] != orig_dict[key]
                assert new_dict[key] == mod_dict[key]
            else:
                assert new_dict[key] == orig_dict[key]
                assert new_dict[key] == mod_dict[key]

        # same for the plot settings
        orig_dict = mw.imviewer.plot_parameters.copy()
        mod_dict = mw.imviewer.plot_parameters.copy()
        test_key = 'color'
        mod_dict[test_key] = 'test'

        mocker.patch.object(PlotSettingsDialog, 'exec_',
                            return_value=1)
        mocker.patch.object(PlotSettingsDialog, 'getValue',
                            return_value=mod_dict)
        mw.onPlotSettings()

        new_dict = mw.imviewer.plot_parameters
        for key in orig_dict:
            if key == test_key:
                assert new_dict[key] != orig_dict[key]
                assert new_dict[key] == mod_dict[key]
            else:
                assert new_dict[key] == orig_dict[key]
                assert new_dict[key] == mod_dict[key]

    def test_filter(self, qtbot, mocker, tmpdir):
        self.mock_ds9(mocker)
        mw = self.make_window(qtbot, tmpdir)

        # select all files
        sel = self.select_files(mw)
        assert self.newfile1 in sel
        assert self.newfile2 in sel

        # set a filter to include only the second file
        filt = os.path.basename(self.newfile2)
        mw.fileFilterBox.setText(filt)

        mw.onFilter()

        assert mw.file_filter == [filt]

        def test():
            QtWidgets.QApplication.processEvents()

            # verify file2 shows, file1 is hidden
            idx = mw.treeView.indexAt(mw.treeView.rect().topLeft())
            fpaths = []
            while idx.isValid():
                fpaths.append(os.path.abspath(mw.model.filePath(idx)))
                idx = mw.treeView.indexBelow(idx)
            assert fpaths == [self.newfile2]

        qtbot.waitUntil(test)

        # set an empty filter and make sure both appear
        filt = ''
        mw.fileFilterBox.setText(filt)
        mw.onFilter()
        assert mw.file_filter == ['*']

        def test():
            QtWidgets.QApplication.processEvents()
            idx = mw.treeView.indexAt(mw.treeView.rect().topLeft())
            fpaths = []
            while idx.isValid():
                fpaths.append(os.path.abspath(mw.model.filePath(idx)))
                idx = mw.treeView.indexBelow(idx)
            assert self.newfile1 in fpaths
            assert self.newfile2 in fpaths
        qtbot.waitUntil(test)

    def test_imexam(self, qtbot, mocker, capsys):
        self.mock_ds9(mocker)
        mw = self.make_window_simple(qtbot)

        mw.onImExam()

        def test1():
            assert 'imexam' in mw.statusBar.currentMessage().lower()
        qtbot.waitUntil(test1)

        # mock ds9 will just quit
        def test2():
            QtWidgets.QApplication.processEvents()
            assert mw.imexam_worker is None
        qtbot.waitUntil(test2)

        # trigger error from ds9: caught in imexam loop
        MockDS9.raise_error_get = True
        mw.onImExam()
        qtbot.waitUntil(test1)
        qtbot.waitUntil(test2)
        capt = capsys.readouterr()
        assert 'Error in ImExam' in capt.err
        MockDS9.raise_error_get = False

        # trigger error in imexam itself
        def err_cmd(*args, **kwargs):
            raise ValueError('test error')
        mocker.patch.object(QADImView, 'imexam', err_cmd)
        mw.onImExam()
        qtbot.waitUntil(test1)
        qtbot.waitUntil(test2)
        capt = capsys.readouterr()
        assert 'test error' in capt.err

    def test_save_param(self, qtbot, mocker, capsys, tmpdir):
        self.mock_ds9(mocker)
        mw = self.make_window_simple(qtbot)

        # test no directory
        mw.cfg_dir = None
        mw.onSaveSettings()
        capt = capsys.readouterr()
        assert 'not saving' in capt.err
        assert 'saved' not in mw.statusBar.currentMessage().lower()

        # test non-existent directory
        mw.cfg_dir = str(tmpdir.join('config'))
        mw.onSaveSettings()
        capt = capsys.readouterr()
        assert 'not saving' in capt.err
        assert 'saved' not in mw.statusBar.currentMessage().lower()

        # test valid directory
        os.makedirs(mw.cfg_dir, exist_ok=True)
        mw.onSaveSettings()
        capt = capsys.readouterr()
        assert 'not saving' not in capt.err
        assert 'saved' in mw.statusBar.currentMessage().lower()

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

        view = self.make_window_simple(qtbot)

        assert view.imviewer.disp_parameters['cmap'] == 'heat'
        assert view.imviewer.phot_parameters['model'] == 'lorentzian'
        assert view.imviewer.plot_parameters['color'] == 'tab20b'

    def test_imexam_load(self, qtbot, mocker, tmpdir, capsys):
        self.mock_ds9(mocker)

        # mock warning dialog
        warn_mock = mocker.patch('PyQt5.QtWidgets.QMessageBox.warning')
        mw = self.make_window(qtbot, tmpdir)

        # select the first file and display it
        self.select_files(mw, self.newfile1)
        mw.onRow()
        assert self.newfile1 in mw.imviewer.files
        assert self.newfile2 not in mw.imviewer.files
        assert not warn_mock.called

        # if imexam worker is running, load will throw a warning
        # dialog instead
        mw.imexam_worker = 'test'
        self.select_files(mw, self.newfile2)
        mw.onRow()
        # no change, but warning was displayed
        assert self.newfile1 in mw.imviewer.files
        assert self.newfile2 not in mw.imviewer.files
        assert warn_mock.called
