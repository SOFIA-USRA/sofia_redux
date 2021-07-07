# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging
import pytest

try:
    from PyQt5 import QtWidgets
except ImportError:
    HAS_PYQT5 = False
    QtWidgets = None
else:
    HAS_PYQT5 = True

from sofia_redux.visualization.utils import logger


class TestEyeLogger(object):
    def make_record(self, level='INFO'):
        elog = logger.EyeLogger('test')
        args = ['name', logging.getLevelName(level),
                'pathname', 1, f'test {level}', [], {}]
        record = elog.makeRecord(*args)
        return record

    def test_eye_logger_record(self, mocker):
        elog = logger.EyeLogger('test')

        # check origin handling
        args = ['name', 0, 'pathname', 1, 'message', [], {}]
        record = elog.makeRecord(*args)

        # by default retrieves this module name
        assert 'test_logger' in record.origin

        # mock missing origin
        mocker.patch('sofia_redux.visualization.utils.'
                     'logger.find_current_module', return_value=None)
        record = elog.makeRecord(*args)
        assert record.origin == 'unknown'

    def test_stream_handler(self, capsys):
        hand = logger.StreamLogger()

        # info, debug, other to stdout
        out_level = ['INFO', 'DEBUG', 'NOTSET']
        for level in out_level:
            msg = self.make_record(level=level)
            hand.emit(msg)
            assert f'{level}: test {level}' in capsys.readouterr().out

        # warning, error, critical to stderr
        err_level = ['WARNING', 'ERROR', 'CRITICAL']
        for level in err_level:
            msg = self.make_record(level=level)
            hand.emit(msg)
            assert f'{level}: test {level}' in capsys.readouterr().err

    @pytest.mark.skipif(not HAS_PYQT5, reason='missing dependencies')
    def test_status_handler(self, qtbot):
        status_bar = QtWidgets.QStatusBar()
        hand = logger.StatusLogger(status_bar)

        def shown():
            assert 'test' in status_bar.currentMessage()

        def not_shown():
            assert status_bar.currentMessage() == ''

        # info level message is displayed
        msg = self.make_record(level='INFO')
        hand.emit(msg)
        qtbot.wait_until(shown)
        status_bar.clearMessage()

        # all others ignored
        err_level = ['DEBUG', 'WARNING', 'ERROR', 'CRITICAL', 'NOTSET']
        for level in err_level:
            msg = self.make_record(level=level)
            hand.emit(msg)
            assert status_bar.currentMessage() == ''

    @pytest.mark.skipif(not HAS_PYQT5, reason='missing dependencies')
    def test_dialog_handler(self, qtbot, mocker):
        # mock dialogs
        warn_mock = mocker.patch.object(logger.QtWidgets.QMessageBox,
                                        'warning')
        error_mock = mocker.patch.object(logger.QtWidgets.QMessageBox,
                                         'critical')

        parent = QtWidgets.QWidget()
        qtbot.add_widget(parent)
        hand = logger.DialogLogger(parent)

        # nothing displayed if parent is not visible
        msg = self.make_record(level='WARNING')
        hand.emit(msg)
        assert not warn_mock.called

        # warning or error level messages are displayed if parent
        # is visible
        mocker.patch.object(parent, 'isVisible', return_value=True)
        hand.emit(msg)
        assert warn_mock.called

        msg = self.make_record(level='ERROR')
        hand.emit(msg)
        assert error_mock.called

        # all others ignored
        warn_mock.called = False
        error_mock.called = False
        err_level = ['DEBUG', 'INFO', 'CRITICAL', 'NOTSET']
        for level in err_level:
            msg = self.make_record(level=level)
            hand.emit(msg)
            assert not warn_mock.called
            assert not error_mock.called
