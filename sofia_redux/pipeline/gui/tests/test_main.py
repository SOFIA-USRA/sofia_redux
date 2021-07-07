# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux GUI Main Window class."""

import os
import pickle

from astropy.io.fits.tests import FitsTestCase
import pytest

from sofia_redux.pipeline.gui.main import ReduxMainWindow
from sofia_redux.pipeline.application import Application
from sofia_redux.pipeline.chooser import Chooser
from sofia_redux.pipeline.configuration import Configuration
from sofia_redux.pipeline.parameters import Parameters
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.viewer import Viewer
from sofia_redux.pipeline.gui.widgets import RemoveFilesDialog, ParamView, \
    ConfigView, EditParam, StepRunnable

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


# establish some non-default classes for testing
# aspects of the GUI not available from the parent
# classes

class ViewerClassForTest(Viewer):
    def __init__(self):
        super().__init__()
        self.embedded = True


class ParametersClassForTest(Parameters):
    def log_input(self, idx):
        self.current[idx].set_value('message', 'test_message')


class ReductionClassForTest(Reduction):
    def __init__(self):
        super().__init__()
        # check for missing display names
        self.processing_steps = {}
        # add a parameter for the log_input step
        self.parameters = ParametersClassForTest()

    def register_viewers(self):
        # check for embedded viewer
        viewers = [ViewerClassForTest()]
        return viewers


class ChooserClassForTest(Chooser):
    def choose_reduction(self, data=None, config=None):
        return ReductionClassForTest()


@pytest.mark.skipif("not HAS_PYQT5")
class TestMainWindow(object):
    """Test the ReduxMainWindow class"""
    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def make_window(self, qtbot, mocker, show=False):
        """Make and register a main window, with recipe modification."""

        # make an Application interface and load in test data
        # set a recipe with two steps
        app = Application(
            Configuration({'recipe': ['log_input', 'log_input']}))
        app.start(['test data'])

        # mock a dialog box to always confirm
        mocker.patch.object(QtWidgets.QMessageBox, 'question',
                            return_value=QtWidgets.QMessageBox.Yes)
        mocker.patch.object(QtWidgets.QMessageBox, 'warning',
                            return_value=QtWidgets.QMessageBox.Ok)

        # make main window
        mw = ReduxMainWindow(app)

        # register the widget
        qtbot.addWidget(mw)

        # show if necessary
        if show:
            mw.show()

        return mw

    def make_window_simple(self, qtbot, mocker, show=False):
        """Make and register a main window."""

        # make an Application interface
        app = Application()

        # mock a dialog box to always confirm
        mocker.patch.object(QtWidgets.QMessageBox, 'question',
                            return_value=QtWidgets.QMessageBox.Yes)
        mocker.patch.object(QtWidgets.QMessageBox, 'warning',
                            return_value=QtWidgets.QMessageBox.Ok)

        # make main window
        mw = ReduxMainWindow(app)

        # register the widget
        qtbot.addWidget(mw)

        # show if necessary
        if show:
            mw.show()

        return mw

    def make_file(self, fname='test0.fits'):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data(fname)
        return ffile

    def test_no_op(self, qtbot, mocker):
        mw = self.make_window(qtbot, mocker)

        # close reduction
        mw.onCloseReduction()
        assert mw.interface.reduction is None

        # make sure the following don't throw errors
        # when reduction is None
        assert mw.resetSteps() is None
        assert mw.onCloseReduction() is None
        assert mw.onDisplayParameters() is None
        assert mw.onLoadParameters() is None
        assert mw.onResetParameters() is None
        assert mw.onEdit(0) is None
        assert mw.stopReduction() is None
        assert mw.saveInputManifest() is None
        assert mw.saveOutputManifest() is None
        assert mw.saveParameters() is None
        assert mw.setOutputDirectory() is None
        assert mw.step() is None
        assert mw.openFinish((None, None)) is None

    def test_open_steps(self, qtbot, mocker, capsys):
        """Test open reduction."""
        mw = self.make_window(qtbot, mocker)

        # load the test data
        mw.openFinish((None, None))

        # wait until load is finished
        def test():
            assert mw.statusbar.currentMessage() == \
                'New reduction loaded.'
        qtbot.waitUntil(test)

        # test that a generic pipe step was loaded
        item = mw.pipeStepListWidget.item(0)
        widget = mw.pipeStepListWidget.itemWidget(item)
        assert widget.isEnabled()
        assert widget.pipeStepLabel.text() == 'Log Input Files'

        # mock a mismatch between parameters and steps
        # verify no error is thrown
        param = Parameters()
        param.add_current_parameters('log_input')
        param.stepnames = ['log_input', 'log_input']
        mw.openFinish((param, None))

        # mock an error in loading
        status = ('error', 'test_error', 'test_error')
        mw.resetPipeline()
        mw.openFinish(status)
        capt = capsys.readouterr()
        assert 'test_error' in capt.err
        assert not mw.reduceButton.isEnabled()

    def test_open_file(self, qtbot, mocker, capsys):
        mw = self.make_window_simple(qtbot, mocker)

        # test if no file selected
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[]])
        mw.onOpenReduction()
        assert mw.interface.reduction is None

        # test if file selected
        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onOpenReduction()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)
        assert isinstance(mw.interface.reduction, Reduction)

    def test_add_file(self, qtbot, mocker, capsys, tmpdir):
        mw = self.make_window_simple(qtbot, mocker)

        # add a non-parent test reduction class
        mw.interface.configuration.chooser = ChooserClassForTest()

        # test if no file selected
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[]])
        mw.onAddFiles()
        assert mw.interface.reduction is None

        # test if file selected; no existing reduction
        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onAddFiles()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)
        old_red = mw.interface.reduction
        old_files = mw.loaded_files.copy()
        assert isinstance(old_red, ReductionClassForTest)

        # test if same file selected again with non-default directory
        mw.interface.reduction.output_directory = str(tmpdir)
        mw.onAddFiles()
        qtbot.waitUntil(test)
        assert isinstance(mw.interface.reduction, ReductionClassForTest)
        assert type(mw.interface.reduction) == type(old_red)
        assert mw.loaded_files == old_files
        assert mw.interface.reduction.output_directory == str(tmpdir)

        # test if non-matching file added
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New files do not match' in capt.err

        mw.interface.configuration.chooser = Chooser()
        mw.interface.chooser = Chooser()

        ffile = self.make_file(fname='blank.fits')
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])

        mw.onAddFiles()
        qtbot.waitUntil(test)
        assert isinstance(mw.interface.reduction, ReductionClassForTest)
        assert type(mw.interface.reduction) == type(old_red)
        assert mw.loaded_files == old_files

    def test_remove_file(self, qtbot, mocker, capsys):
        mw = self.make_window_simple(qtbot, mocker)

        ffile1 = self.make_file()
        ffile2 = self.make_file(fname='blank.fits')
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile1, ffile2]])
        mw.onOpenReduction()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)

        orig_files = mw.loaded_files.copy()

        # test if remove dialog canceled
        mocker.patch.object(RemoveFilesDialog, 'exec_',
                            return_value=QtWidgets.QDialog.Rejected)
        mw.onRemoveFiles()
        assert mw.loaded_files == orig_files

        # class for mocking selected items in the dialog
        class ItemClassForTest(object):
            def __init__(self, value):
                self.value = value

            def text(self):
                return os.path.basename(self.value)

        # test if no file selected
        mocker.patch.object(RemoveFilesDialog, 'exec_',
                            return_value=QtWidgets.QDialog.Accepted)
        mocker.patch.object(QtWidgets.QListWidget, 'selectedItems',
                            return_value=[])
        mw.onRemoveFiles()
        assert mw.loaded_files == orig_files

        # test if one file selected
        mocker.patch.object(QtWidgets.QListWidget, 'selectedItems',
                            return_value=[ItemClassForTest(ffile1)])
        mw.onRemoveFiles()
        qtbot.wait(100)

        def test():
            capt = capsys.readouterr()
            assert 'Removing' in capt.err
            assert mw.loaded_files == [ffile2]
            assert mw.interface.reduction is not None
        qtbot.waitUntil(test)

        # test if all files removed
        mocker.patch.object(QtWidgets.QListWidget, 'selectedItems',
                            return_value=[ItemClassForTest(ffile2)])
        mw.onRemoveFiles()

        def test():
            assert mw.loaded_files == []
            assert mw.interface.reduction is None
        qtbot.waitUntil(test)

    def test_quit(self, qtbot, mocker):
        mw = self.make_window(qtbot, mocker, show=True)

        # mock the message box to cancel the exit
        mocker.patch.object(QtWidgets.QMessageBox, 'question',
                            return_value=QtWidgets.QMessageBox.No)

        mw.close()
        assert mw.isVisible()

        mw.closeEvent('test')
        assert mw.isVisible()

        # mock the message box to confirm the exit
        mocker.patch.object(QtWidgets.QMessageBox, 'question',
                            return_value=QtWidgets.QMessageBox.Yes)
        mw.close()
        assert not mw.isVisible()

    def test_log(self, qtbot, mocker):
        mw = self.make_window(qtbot, mocker)

        # log a test message
        msg = "test message"
        mw.addLog(msg)

        def test():
            assert msg in mw.logTextEdit.toPlainText()
        qtbot.waitUntil(test)

    def test_step(self, qtbot, mocker, tmpdir, capsys):
        mw = self.make_window(qtbot, mocker)

        # load the reduction, run a step
        mw.openFinish((None, None))
        mw.step()

        # some useful tests
        def step_test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'Pipeline step complete.' in capt.out

        def undo_test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'Undid' in capt.err

        def reset_test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'steps reset' in capt.out

        # wait until step is done
        qtbot.waitUntil(step_test)
        assert mw.interface.reduction.step_index == 1

        # input logged
        msg = "test data"
        assert msg in mw.logTextEdit.toPlainText()

        assert mw.pickled_reduction is not None
        assert mw.allow_undo is True
        assert mw.last_step == 0

        # undo step
        mw.undo()
        qtbot.waitUntil(undo_test)

        assert mw.pickled_reduction is None
        assert mw.allow_undo is True
        assert mw.last_step == -1

        # run again but skip save
        mw.step(skip_save=True)
        qtbot.waitUntil(step_test)
        assert mw.pickled_reduction is None
        assert mw.allow_undo is False
        assert mw.last_step == 0
        assert mw.interface.reduction.step_index == 1

        # reset reduction
        mw.resetSteps()
        qtbot.waitUntil(reset_test)
        assert mw.interface.reduction.step_index == 0
        assert mw.last_step == -1

        # run the next step via the onRun function
        mw.onRun(0)
        qtbot.waitUntil(step_test)
        assert mw.interface.reduction.step_index == 1
        assert mw.last_step == 0

        # reset again and reduce to run both steps
        mw.resetSteps()
        qtbot.waitUntil(reset_test)
        mw.reduce()
        qtbot.waitUntil(step_test)
        assert mw.interface.reduction.step_index == 2
        assert mw.last_step == 0

        # reset again and induce a pickle error -- the step
        # still runs but undo is not allowed
        mw.resetSteps()
        qtbot.waitUntil(reset_test)

        def err_func(*args, **kwargs):
            raise pickle.PicklingError('test error')
        try:
            mocker.patch('dill.dumps', err_func)
        except ImportError:
            pass
        mocker.patch('pickle.dumps', err_func)
        mw.step()
        qtbot.waitUntil(step_test)
        assert mw.pickled_reduction is None
        assert mw.allow_undo is False
        assert mw.last_step == 0

        # try to call undo anyway -- nothing happens
        mw.undo()
        assert mw.last_step == 0

    def test_one_step_undo(self, qtbot, mocker, capsys):
        mw = self.make_window_simple(qtbot, mocker)
        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onOpenReduction()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)

        mw.step()

        # wait for step to finish
        def step_test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'Pipeline step complete.' in capt.out
        qtbot.waitUntil(step_test)

        # undo
        mw.undo()

        def undo_test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'Undid' in capt.err
        qtbot.waitUntil(undo_test)
        assert mw.interface.reduction.step_index == 0
        assert mw.last_step == -1

    def test_step_finish(self, qtbot, mocker, capsys):
        mw = self.make_window(qtbot, mocker)

        # load the reduction, run a step
        mw.openFinish((None, None))

        # call step finish with an error
        status = ('error', 'test_error', 'test_error')
        mw.stepFinish(status)
        capt = capsys.readouterr()
        assert 'test_error' in capt.err

    def test_file_summary(self, qtbot, mocker, capsys):
        mw = self.make_window_simple(qtbot, mocker)

        # load in the max number of files
        maxnum = 6
        fnames = ['{}.fits'.format(n) for n in range(maxnum)]
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[fnames])
        mw.onOpenReduction()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)

        sumtext = mw.fileSummaryTextEdit.toPlainText()
        for f in fnames:
            assert f in sumtext

        # load in one more: should now be files 1-5, then ... and last file
        one_more = 'one_more.fits'
        fnames.append(one_more)
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[one_more]])
        mw.onAddFiles()
        qtbot.waitUntil(test)

        sumtext = mw.fileSummaryTextEdit.toPlainText()
        for i, f in enumerate(fnames):
            if i < maxnum - 1 or i == len(fnames) - 1:
                assert f in sumtext
            else:
                assert f not in sumtext

    def test_display_parameters(self, qtbot, mocker, capsys):
        mw = self.make_window_simple(qtbot, mocker)
        mw.interface.configuration.chooser = ChooserClassForTest()

        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onOpenReduction()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)

        # mock the show, exec_, and isvisible functions
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=QtWidgets.QDialog.Accepted)
        mocker.patch.object(QtWidgets.QDialog, 'isVisible',
                            return_value=True)

        mw.onDisplayParameters()

        # check that the test parameter is shown
        msg = 'test_message'

        def test():
            QtWidgets.QApplication.processEvents()
            assert isinstance(mw.param_view, ParamView)
            assert 'message = {}'.format(msg) in \
                   mw.param_view.textEdit.toPlainText()
        qtbot.waitUntil(test)

        # modify parameter and check that display is updated
        pset = mw.interface.reduction.get_parameter_set(0)
        msg = 'new_message'
        pset.set_value('message', msg)
        mocker.patch.object(EditParam, 'getValue',
                            return_value=pset)
        mw.onEdit(0)

        # test for new value of msg in text browser
        qtbot.waitUntil(test)

    def test_display_config(self, qtbot, mocker):
        mw = self.make_window_simple(qtbot, mocker)
        mw.interface.configuration.chooser = ChooserClassForTest()
        mw.interface.configuration.test_key = 'test_value'

        # mock the show, exec_, and isvisible functions
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=QtWidgets.QDialog.Accepted)
        mocker.patch.object(QtWidgets.QDialog, 'isVisible',
                            return_value=True)

        mw.onDisplayConfig()

        # check that the test value is shown
        val = 'test_value'

        def test():
            QtWidgets.QApplication.processEvents()
            assert isinstance(mw.config_view, ConfigView)
            assert f'test_key = {val}' in \
                   mw.config_view.textEdit.toPlainText()
        qtbot.waitUntil(test)

        # modify parameter and check that display is updated
        val = 'new_value'
        mw.interface.configuration.test_key = val
        mw.updateConfigView()

        # test for new value of msg in text browser
        qtbot.waitUntil(test)

    def test_edit_config(self, qtbot, mocker, capsys):
        mw = self.make_window_simple(qtbot, mocker)
        mw.interface.configuration.chooser = ChooserClassForTest()
        mw.interface.configuration.test_key = 'test_value'

        # mock the show, exec_, and isvisible functions
        mocker.patch.object(QtWidgets.QDialog, 'show',
                            return_value=None)
        mocker.patch.object(QtWidgets.QDialog, 'exec_',
                            return_value=QtWidgets.QDialog.Accepted)
        mocker.patch.object(QtWidgets.QDialog, 'isVisible',
                            return_value=True)

        # load in config view
        mw.onDisplayConfig()

        def test():
            QtWidgets.QApplication.processEvents()
            assert isinstance(mw.config_view, ConfigView)
            assert 'test_key = test_value' in \
                   mw.config_view.textEdit.toPlainText()
        qtbot.waitUntil(test)

        # edit the text in the display and save
        mw.config_view.textEdit.setText('test_key = new_value')
        mw.onEditConfiguration()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'Configuration edited' in capt.out
            assert mw.interface.configuration.test_key == 'new_value'
        qtbot.waitUntil(test)

        # edit with a syntax error - should log error and
        # not update config
        mw.config_view.textEdit.setText('test_key')
        mw.onEditConfiguration()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'Configuration badly formatted' in capt.err
        qtbot.waitUntil(test)
        assert mw.interface.configuration.test_key == 'new_value'

    def test_load_parameters(self, qtbot, mocker, capsys, tmpdir):
        mw = self.make_window_simple(qtbot, mocker)
        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onOpenReduction()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)

        # keep a copy of the original parameters
        orig = mw.interface.reduction.parameters.copy()

        # test no file selected
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[[]])
        mw.onLoadParameters()
        QtWidgets.QApplication.processEvents()
        assert orig.to_text() == mw.interface.reduction.parameters.to_text()

        # test no-op file selected
        infile = tmpdir.join('test1.cfg')
        infile.write('[no op parameters]\n')
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[str(infile)])
        mw.onLoadParameters()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert str(infile) in capt.out
        qtbot.waitUntil(test)
        assert orig.to_text() == mw.interface.reduction.parameters.to_text()

        # test parameter update
        infile = tmpdir.join('test2.cfg')
        infile.write('[log_input]\nmessage = new_value\n')
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[str(infile)])
        mw.onLoadParameters()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert str(infile) in capt.out
        qtbot.waitUntil(test)
        new_text = mw.interface.reduction.parameters.to_text()
        assert orig.to_text() != new_text
        assert "message = new_value" in '\n'.join(new_text)

        # reset parameters
        mw.onResetParameters()
        assert orig.to_text() == mw.interface.reduction.parameters.to_text()

    def test_stop_reduction(self, qtbot, mocker):
        mw = self.make_window_simple(qtbot, mocker)

        # mock a step worker
        mw.worker = StepRunnable(lambda: None, 1)
        assert not mw.worker.stop

        # stop the worker
        mw.stopReduction()
        assert mw.worker.stop
        assert not mw.progress.stopButton.isEnabled()

    def test_save(self, qtbot, mocker, tmpdir, capsys):
        mw = self.make_window_simple(qtbot, mocker)
        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onOpenReduction()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)
        mw.interface.configuration.load({'test_key': 'test_value'})
        mw.interface.reduction.out_files = ['test_file.fits']

        # test if no output file selected - nothing should happen
        mocker.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                            return_value=[[]])
        mw.saveInputManifest()
        mw.saveOutputManifest()
        mw.saveParameters()
        mw.saveConfiguration()

        # method to save input, output, and parameter files
        def save_all(suffix):
            save_file = ''

            def test():
                QtWidgets.QApplication.processEvents()
                assert os.path.isfile(str(save_file))

            save_file = tmpdir.join(f'test_input_{suffix}.txt')
            mocker.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                                return_value=[str(save_file)])
            mw.saveInputManifest()
            qtbot.waitUntil(test)

            save_file = tmpdir.join(f'test_output_{suffix}.txt')
            mocker.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                                return_value=[str(save_file)])
            mw.saveOutputManifest()
            qtbot.waitUntil(test)

            save_file = tmpdir.join(f'test_param_{suffix}.txt')
            mocker.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                                return_value=[str(save_file)])
            mw.saveParameters()
            qtbot.waitUntil(test)

            save_file = tmpdir.join(f'test_config_{suffix}.txt')
            mocker.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                                return_value=[str(save_file)])
            mw.saveConfiguration()
            qtbot.waitUntil(test)

        # save all with no default names
        save_all(1)

        # set default names and save again
        mw.interface.configuration.input_manifest = 'infiles.txt'
        mw.interface.configuration.output_manifest = 'outfiles.txt'
        mw.interface.configuration.parameter_file = 'parameters.txt'
        mw.interface.configuration.config_file_name = 'config.txt'

        save_all(2)

        # set absolute default names and save again
        mw.interface.configuration.input_manifest = \
            str(tmpdir.join('infiles.txt'))
        mw.interface.configuration.output_manifest = \
            str(tmpdir.join('outfiles.txt'))
        mw.interface.configuration.parameter_file = \
            str(tmpdir.join('parameters.txt'))
        mw.interface.configuration.config_file_name = \
            str(tmpdir.join('config.txt'))

        save_all(3)

        # this may or may not be helping with timing issues in
        # resolving other tests
        QtWidgets.QApplication.processEvents()

    def test_save_configuration_startup(self, qtbot, mocker, capsys, tmpdir):
        # start window but don't load reduction
        mw = self.make_window_simple(qtbot, mocker)

        # set a configuration value
        mw.interface.configuration.load({'test_key': 'test_value'})
        save_file = tmpdir.join('test_config.txt')

        def test():
            QtWidgets.QApplication.processEvents()
            assert os.path.isfile(str(save_file))

        mocker.patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                            return_value=[str(save_file)])
        mw.saveConfiguration()
        qtbot.waitUntil(test)

    def test_set_output_directory(self, qtbot, mocker, tmpdir):
        mw = self.make_window(qtbot, mocker)
        mw.openFinish((None, None))

        # test no file selected
        mocker.patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                            return_value='')
        mw.setOutputDirectory()

        # test output file
        out_dir = str(tmpdir.join('out'))
        mocker.patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                            return_value=out_dir)

        mw.setOutputDirectory()
        assert mw.interface.reduction.output_directory == out_dir
        assert mw.save_directory == out_dir
        assert os.path.isdir(out_dir)

    def test_display_options(self, qtbot, mocker):
        # mock a dialog box to always confirm
        mocker.patch.object(QtWidgets.QMessageBox, 'question',
                            return_value=QtWidgets.QMessageBox.Yes)
        mocker.patch.object(QtWidgets.QMessageBox, 'warning',
                            return_value=QtWidgets.QMessageBox.Ok)

        # test defaults set from initial config

        # no display, no intermediate
        conf = Configuration({'update_display': False,
                              'display_intermediate': False})
        app = Application(conf)
        mw = ReduxMainWindow(app)
        qtbot.addWidget(mw)
        assert not mw.actionUpdateDisplays.isChecked()
        assert not mw.actionDisplayIntermediate.isChecked()

        # do display, do intermediate
        conf = Configuration({'update_display': True,
                              'display_intermediate': True})
        app = Application(conf)
        mw = ReduxMainWindow(app)
        qtbot.addWidget(mw)
        assert mw.actionUpdateDisplays.isChecked()
        assert mw.actionDisplayIntermediate.isChecked()

    def test_toggle_display(self, qtbot, mocker, capsys):
        # add a reduction that has an embedded viewer
        # display is on by default
        mw = self.make_window_simple(qtbot, mocker)
        mw.interface.configuration.chooser = ChooserClassForTest()

        ffile = self.make_file()
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        mw.onOpenReduction()

        # wait for load to finish
        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert 'New reduction loaded.' in capt.out
        qtbot.waitUntil(test)

        # data view tab should exist
        assert mw.interface.has_embedded_viewers()
        assert mw.dataTabWidget.tabText(0) == "Data View"

        # box is still checked -- tab should stay
        mw.toggleDisplay()
        assert mw.dataTabWidget.tabText(0) == "Data View"
        assert mw.interface.configuration.update_display

        # uncheck box -- tab should go
        mw.actionUpdateDisplays.setChecked(False)
        mw.toggleDisplay()
        assert mw.dataTabWidget.tabText(0) != "Data View"
        assert not mw.interface.configuration.update_display

        # attempt to update viewers -- verify no error
        mw.interface.update_viewers()
        assert len(mw.interface.viewers) == 0

    def test_toggle_intermediate(self, qtbot, mocker):
        mw = self.make_window_simple(qtbot, mocker)
        mw.interface.configuration.load({})

        # check box
        mw.actionDisplayIntermediate.setChecked(True)
        mw.toggleDisplayIntermediate()
        assert mw.interface.configuration.display_intermediate

        # uncheck box
        mw.actionDisplayIntermediate.setChecked(False)
        mw.toggleDisplayIntermediate()
        assert not mw.interface.configuration.display_intermediate

    def test_load_configuration(self, qtbot, mocker, capsys, tmpdir):
        # start window but don't load reduction
        mw = self.make_window(qtbot, mocker)
        orig = mw.interface.configuration.config.write()
        assert mw.interface.configuration.config['recipe'] \
            == ['log_input', 'log_input']

        # test no file selected
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[[]])
        mw.onLoadConfiguration()
        QtWidgets.QApplication.processEvents()
        assert orig == mw.interface.configuration.config.write()

        # test no-op file selected
        infile = tmpdir.join('test1.cfg')
        infile.write('# no op config\n')
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[str(infile)])
        mw.onLoadConfiguration()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert str(infile) in capt.out
        qtbot.waitUntil(test)
        # GUI adds two things to config -- settings for update
        # display and intermediate
        updated = mw.interface.configuration.config.write()
        assert orig != updated
        assert 'update_display' in mw.interface.configuration.config
        assert 'display_intermediate' in mw.interface.configuration.config

        # test parameter update
        infile = tmpdir.join('test2.cfg')
        infile.write('test_key = test_value\n')
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                            return_value=[str(infile)])
        mw.onLoadConfiguration()

        def test():
            QtWidgets.QApplication.processEvents()
            capt = capsys.readouterr()
            assert str(infile) in capt.out
        qtbot.waitUntil(test)
        assert updated != mw.interface.configuration.config.write()
        assert mw.interface.configuration.config['recipe'] \
            == ['log_input', 'log_input']
        assert mw.interface.configuration.config['test_key'] == 'test_value'

        # reset configuration
        mw.onResetConfiguration()
        assert updated == mw.interface.configuration.config.write()
