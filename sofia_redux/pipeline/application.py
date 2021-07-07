# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Run Redux reduction objects from a GUI interface."""

import sys

from sofia_redux.pipeline.interface import Interface
from sofia_redux.pipeline.gui.main import ReduxMainWindow

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
except ImportError:
    HAS_PYQT5 = False
    QtWidgets, QtCore, QtGui = None, None, None
else:
    HAS_PYQT5 = True


class Application(Interface):
    """
    Graphical interface to Redux reduction objects.

    This class provides a Qt5 GUI that allows interactive parameter
    setting and reduction step running.  Intermediate data viewers
    are also supported.  Most functionality is inherited from the
    `Interface` class.

    Attributes
    ----------
    app: QApplication
        A top-level Qt widget.
    """

    def __init__(self, configuration=None):
        """
        Initialize the application, with an optional configuration.

        Parameters
        ----------
        configuration : `Configuration`, optional
            Configuration items to be used for all reductions
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')
        super().__init__(configuration)
        self.app = None

    def run(self):
        """Start up the application."""

        # Start application
        self.app = QtWidgets.QApplication(sys.argv)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/redux_icon.png"),
                       QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.app.setWindowIcon(icon)
        self.app.setApplicationName('Redux')

        # Start a timer to allow the python interpreter to run occasionally
        # (without this, ctrl-c is swallowed by the event loop)
        timer = QtCore.QTimer()
        timer.start(200)
        timer.timeout.connect(lambda: None)

        # Start up the main window and event loop
        mw = ReduxMainWindow(self)
        mw.show()
        mw.raise_()
        sys.exit(self.app.exec_())


def main():
    """Run the Redux GUI."""
    Application.tidy_log()
    app = Application()
    app.run()
