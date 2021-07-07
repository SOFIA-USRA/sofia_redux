# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux GUI Text View class."""

import pytest

from sofia_redux.pipeline.gui.textview import TextView

try:
    from PyQt5 import QtGui, QtWidgets
except ImportError:
    QtGui, QtWidgets = None, None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.mark.skipif("not HAS_PYQT5")
class TestTextView(object):
    """Test the TextView class"""

    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def make_window(self, qtbot):
        """Make and register a text window."""
        self.tv = TextView()

        # register the widget
        qtbot.addWidget(self.tv)

    def test_load(self, qtbot):
        """Test open reduction."""
        self.make_window(qtbot)

        msg = 'test data'
        self.tv.load(msg)
        assert msg in self.tv.textEdit.toPlainText()

        self.tv.load(None)
        assert self.tv.textEdit.toPlainText().strip() == ''

    def test_title(self, qtbot):
        self.make_window(qtbot)

        msg = 'test'
        self.tv.setTitle(msg)
        assert self.tv.windowTitle() == msg

    def test_find(self, qtbot):
        self.make_window(qtbot)

        msg = 'test data'
        self.tv.load([msg, msg])

        # find string in loaded text
        find_text = 'test'
        self.tv.findText.setText(find_text)
        self.tv.find()
        cursor = self.tv.textEdit.textCursor()
        cursor.select(QtGui.QTextCursor.WordUnderCursor)
        assert cursor.selectedText() == find_text

        # reset
        find_text = ''
        self.tv.findText.setText(find_text)
        self.tv.find()
        cursor = self.tv.textEdit.textCursor()
        assert cursor.atStart()

        # find string not in loaded text
        find_text = 'not here'
        self.tv.findText.setText(find_text)
        self.tv.find()
        cursor = self.tv.textEdit.textCursor()
        assert cursor.atStart()
        cursor.select(QtGui.QTextCursor.WordUnderCursor)
        assert cursor.selectedText() != find_text

    def test_filter(self, qtbot):
        self.make_window(qtbot)

        msg1 = 'test data one'
        msg2 = 'test data two'
        msg3 = 'test data three'
        self.tv.load([msg1, msg2, msg3])

        # find string in loaded text
        find_text = 'two'
        self.tv.findText.setText(find_text)
        self.tv.filter()
        ptext = self.tv.textEdit.toPlainText()
        assert msg1 not in ptext
        assert msg2 in ptext
        assert msg3 not in ptext

        # reset
        find_text = ''
        self.tv.findText.setText(find_text)
        self.tv.filter()
        ptext = self.tv.textEdit.toPlainText()
        assert msg1 in ptext
        assert msg2 in ptext
        assert msg3 in ptext

        # find two strings
        find_text = 'one, two'
        self.tv.findText.setText(find_text)
        self.tv.filter()
        ptext = self.tv.textEdit.toPlainText()
        assert msg1 in ptext
        assert msg2 in ptext
        assert msg3 not in ptext
