# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Application class."""

import pytest
from sofia_redux.pipeline.application import Application

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None

pytest.importorskip('PyQt5')


class TestApplication(object):
    def test_run(self, mocker, qapp):
        """Test run method."""
        mocker.patch.object(QtWidgets.QMessageBox, 'question',
                            return_value=QtWidgets.QMessageBox.Yes)
        mocker.patch.object(QtWidgets.QApplication, 'exec_', return_value=None)
        mocker.patch.object(QtWidgets.QMainWindow, 'show', return_value=None)
        mocker.patch.object(QtWidgets, 'QApplication', return_value=qapp)
        application = Application()

        with pytest.raises(SystemExit):
            application.run()
