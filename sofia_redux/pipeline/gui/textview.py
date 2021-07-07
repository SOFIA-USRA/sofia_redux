# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Text viewer widget for use with QAD and Redux."""

try:
    from PyQt5 import QtWidgets, QtGui, QtCore
    from sofia_redux.pipeline.gui.ui import ui_textview
except ImportError:
    HAS_PYQT5 = False
    QtGui = None

    # duck type parents to allow class definition
    class QtWidgets:
        class QDialog:
            pass

    class ui_textview:
        class Ui_TextWindow:
            pass
else:
    HAS_PYQT5 = True


class TextView(QtWidgets.QDialog, ui_textview.Ui_TextWindow):
    """
    View, find, and filter text.

    All attributes and methods for this class are intended for internal
    use, in response to user actions within a Qt5 application.
    """
    def __init__(self, parent=None):
        """
        Initialize the text viewer widget.

        Parameters
        ----------
        parent : `QWidget`
            Parent widget.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        QtWidgets.QDialog.__init__(self, parent)

        # set up UI from Designer generated file
        self.setupUi(self)

        # make sure there's a close button
        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.WindowMaximizeButtonHint
                            | QtCore.Qt.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowCloseButtonHint)

        # connect buttons
        self.findButton.clicked.connect(self.find)
        self.filterButton.clicked.connect(self.filter)
        self.tableButton.setEnabled(False)

        # hide save button: should only appear if text is editable
        self.saveButton.setVisible(False)

        # last loaded text
        self.text = None
        self.html = None

    def load(self, text):
        """
        Load text into the widget.

        Parameters
        ----------
        text : list of str
            The text to display.
        """
        if text is None:
            text = ''
        if type(text) is not list:
            text = [text]

        self.text = text

        # format to HTML
        self.html = self.format()

        # set text in text window
        self.textEdit.setHtml(self.html)
        self.textEdit.repaint()

    def setTitle(self, title):
        """Set the window title."""
        self.setWindowTitle(title)

    def find(self):
        """Find a string within the text."""

        # read text to find
        # whole field will be matched; comma-separated fields not allowed
        find_text = self.findText.text().strip()

        cursor = self.textEdit.textCursor()
        if find_text == '':
            # set cursor back to beginning of document
            cursor.movePosition(QtGui.QTextCursor.Start)
            self.textEdit.setTextCursor(cursor)
        else:
            # find next instance
            found = self.textEdit.find(find_text)
            if not found:
                # wrap back to beginning and try one more find
                cursor.movePosition(QtGui.QTextCursor.Start)
                self.textEdit.setTextCursor(cursor)
                self.textEdit.find(find_text)
        self.textEdit.repaint()

    def filter(self):
        """Filter text to lines containing a specified string."""

        # read text to filter
        # may be comma-separated substrings
        find_text = self.findText.text().strip()

        if find_text == '':
            # clear previous filter / table
            self.textEdit.setHtml(self.html)
        else:
            # allow multiple filter keys
            sep = find_text.split(',')

            # split text on lines
            header = self.html.split('<br>')

            # find substrings in lines
            filtered = []
            for line in header:
                added = False

                # keep anchored lines -- filename headers, <pre> tags
                if '<a name="anchor">' in line:
                    filtered.append(line)
                else:
                    for text in sep:
                        if added:
                            break
                        if text.strip().lower() in line.lower():
                            filtered.append(line)
                            added = True
            self.textEdit.setHtml('<br>'.join(filtered))

        # repaint required for some versions of Qt/OS
        self.textEdit.repaint()

    def format(self):
        """
        Format the input text to HTML for display.

        Returns
        -------
        str
            HTML-formatted text.
        """
        anchor = '<a name="anchor"></a>'
        br = '<br>'

        text_str = br.join(self.text)
        html = '<pre>' + anchor + br + text_str + br + '</pre>' + anchor

        return html
