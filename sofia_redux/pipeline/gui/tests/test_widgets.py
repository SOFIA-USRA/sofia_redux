# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux GUI widgets."""

import logging
from copy import deepcopy

from astropy import log
import pytest

from sofia_redux.pipeline.parameters import Parameters
from sofia_redux.pipeline.gui.widgets import EditParam, TextEditLogger, \
    ParamView, StepRunnable, LoadRunnable, GeneralRunnable

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.mark.skipif("not HAS_PYQT5")
class TestWidgets(object):
    """Test the Redux GUI widgets"""

    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def make_default_param(self):
        """Make a set of parameters from a default dictionary."""
        # make at least one of every type of widget
        # as well as one hidden, and one unknown type
        default = {'test_step': [{'key': 'test_key_1',
                                  'value': 'test_value',
                                  'description': 'test description'},
                                 {'key': 'test_key_2',
                                  'name': 'Test Key 2',
                                  'dtype': 'int',
                                  'options': [1, 2],
                                  'description': 'test description',
                                  'option_index': 1},
                                 {'key': 'test_key_3',
                                  'options': [1, 2],
                                  'wtype': 'radio_button',
                                  'description': 'test_description'},
                                 {'key': 'test_key_4',
                                  'wtype': 'check_box',
                                  'value': True},
                                 {'key': 'test_key_5',
                                  'value': False,
                                  'description': 'test_description',
                                  'dtype': 'bool'},
                                 {'key': 'test_key_6',
                                  'value': 'test_value',
                                  'wtype': 'pick_file'},
                                 {'key': 'test_key_7',
                                  'value': 'test_value',
                                  'description': 'test_description',
                                  'wtype': 'pick_directory'},
                                 {'key': 'test_key_8',
                                  'value': 'test',
                                  'hidden': True},
                                 {'key': 'test_key_9',
                                  'value': 'test',
                                  'wtype': 'unknown'},
                                 {'key': 'test_key_10',
                                  'description': 'test_description',
                                  'wtype': 'group'},
                                 {'key': 'test_key_11'}
                                 ]}
        param = Parameters(default)
        param.add_current_parameters('test_step')
        return param

    def test_edit_param(self, qtbot, mocker, tmpdir):
        param = self.make_default_param()
        param_text = '\n'.join(param.to_text())
        current = param.current[0]
        default = deepcopy(current)

        edpar = EditParam(name='test_name', current=current,
                          default=default, directory=str(tmpdir))
        qtbot.addWidget(edpar)

        # verify hidden parameters and unknown widget types
        # do not appear, shown parameters do
        assert edpar.container.findChild(
            QtWidgets.QWidget, 'test_key_7') is not None
        assert edpar.container.findChild(
            QtWidgets.QWidget, 'test_key_8') is None
        assert edpar.container.findChild(
            QtWidgets.QWidget, 'test_key_9') is None

        # check that group box appears and test_key_11 is a
        # child of it
        # no value is set/retrieved from the group, but
        # is for its child
        gwid = edpar.container.findChild(
            QtWidgets.QWidget, 'test_key_10')
        assert isinstance(gwid, QtWidgets.QGroupBox)
        assert gwid.findChild(
            QtWidgets.QWidget, 'test_key_11') is not None

        # get value and verify it is the same as input
        newparset = edpar.getValue()
        param.current[0] = newparset
        assert '\n'.join(param.to_text()) == param_text
        assert 'test_key_10' not in param_text
        assert 'test_key_11' in param_text

        # test pick file with no file selected
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[]])
        edpar.pickFile('test_key_6')
        newparset = edpar.getValue()
        assert newparset.get_value('test_key_6') == 'test_value'

        # now with a selected file
        ffile = 'test_file.txt'
        mocker.patch.object(QtWidgets.QFileDialog, 'getOpenFileNames',
                            return_value=[[ffile]])
        edpar.pickFile('test_key_6')
        newparset = edpar.getValue()
        assert newparset.get_value('test_key_6') == ffile

        # same for pick directory
        mocker.patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                            return_value='')
        edpar.pickDirectory('test_key_7')
        newparset = edpar.getValue()
        assert newparset.get_value('test_key_7') == 'test_value'

        ffile = 'test_dir'
        mocker.patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                            return_value=ffile)
        edpar.pickDirectory('test_key_7')
        newparset = edpar.getValue()
        assert newparset.get_value('test_key_7') == ffile

        # now reset all parameters and verify it is
        # back to the original
        edpar.restore()
        newparset = edpar.getValue()
        param.current[0] = newparset
        assert '\n'.join(param.to_text()) == param_text

        QtWidgets.QApplication.processEvents()

    def test_logger(self):
        def test(msg):
            assert 'test message' in msg

        orig_level = log.level
        log.setLevel('DEBUG')

        logger = TextEditLogger()
        logger.signals.finished.connect(test)

        record = logging.LogRecord('test', logging.INFO, 'test', 0,
                                   'test message', None, None)
        logger.emit(record)

        record = logging.LogRecord('test', logging.DEBUG, 'test', 0,
                                   'test message', None, None)
        logger.emit(record)

        log.setLevel(orig_level)

    def test_paramview(self, qtbot):
        try:
            import pandas
            log.debug('Pandas version: {}'.format(pandas.__version__))
            has_pandas = True
        except ImportError:
            has_pandas = False

        pv = ParamView()
        qtbot.addWidget(pv)

        param = self.make_default_param()

        # add a few more steps, with a little variation
        param.add_current_parameters('test_step')
        param.add_current_parameters('test_step')
        del param.current[1]['test_key_1']
        del param.current[2]['test_key_2']

        param_text = param.to_text()

        # add a few extra test lines
        param_text = ['# Test comment'] + param_text + ['bad line']

        pv.load(param_text)

        # test the table function
        start = '{}\n'.format('\n'.join(param_text))
        start = start.replace('[', '\n[')

        # with no filter text
        pv.table()
        assert pv.textEdit.toPlainText() == start
        assert 'table' not in pv.textEdit.toHtml()

        # with filter text
        pv.findText.setText('test_key_1, test_key_2')
        pv.table()
        assert pv.textEdit.toPlainText() != start
        html = pv.textEdit.toHtml()
        if has_pandas:
            assert '<table' in html
            assert html.count('<tr>') == 4
            assert html.count('<td>') == 4 * 4

        # reset
        pv.findText.setText('')
        pv.table()
        assert pv.textEdit.toPlainText() == start

    def test_step_runnable(self, capsys):
        def success():
            print('success')

        def failure():
            raise RuntimeError('error')

        def progress():
            print('progress')

        def finished(result):
            print(result)
            print('finished')

        # run the success function 5 times
        worker = StepRunnable(success, 5)
        worker.signals.progress.connect(progress)
        worker.signals.finished.connect(finished)

        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('success') == 5
        assert capt.out.count('progress') == 5
        assert capt.out.count('finished') == 1

        # run again but set stop first
        worker.stop = True
        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('success') == 0
        assert capt.out.count('progress') == 0
        assert capt.out.count('finished') == 1

        # run the failure function 5 times (will stop after 1)
        worker.stop = False
        worker.step = failure
        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('progress') == 0
        assert capt.out.count('finished') == 1
        assert 'error' in capt.out

    def test_load_runnable(self, capsys):
        def success(data):
            print(data)
            print('success')

        def failure(data):
            raise RuntimeError('error')

        def finished(result):
            print(result)
            print('finished')

        # run the success function
        worker = LoadRunnable(success, 'test data', None, None)
        worker.signals.finished.connect(finished)

        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('success') == 1
        assert capt.out.count('finished') == 1
        assert 'test data' in capt.out

        # run the failure function
        worker.load = failure
        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('finished') == 1
        assert 'error' in capt.out

    def test_general_runnable(self, capsys):
        def success():
            print('success')

        def failure():
            raise RuntimeError('error')

        def finished(result):
            print(result)
            print('finished')

        # run the success function
        worker = GeneralRunnable(success)
        worker.signals.finished.connect(finished)

        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('success') == 1
        assert capt.out.count('finished') == 1

        # run the failure function
        worker.run_function = failure
        worker.run()
        capt = capsys.readouterr()
        assert capt.out.count('finished') == 1
        assert 'error' in capt.out
