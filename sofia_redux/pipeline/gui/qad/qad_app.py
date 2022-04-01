# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Standalone front-end for QAD display tool."""

import argparse
import logging
import sys
import warnings

from astropy import log

from sofia_redux.pipeline.interface import Interface
from sofia_redux.pipeline.gui.qad.qad_main_panel import QADMainWindow

try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    HAS_PYQT5 = False
    QtCore, QtGui = None, None
else:
    HAS_PYQT5 = True


def main():
    """
    Quality Analysis and Display tool (QAD).

    Start the QAD front end as a standalone tool with the command::

        qad

    From the GUI window, files may be selected for display in DS9
    (for FITS images or region files), the Eye of SOFIA (for FITS spectra),
    or the system default application (all other files).  Highlight
    the desired files and double-click or press enter to select
    them.

    The starting directory may be modified via the File menu,
    or by clicking the Home button.  The enclosing directory may be
    reached by clicking the up arrow; the last visited directory by
    clicking the down arrow.

    Display and photometry parameters may be set and saved from the
    Settings menu.

    Clicking the ImExam button (scissors icon) launches an event loop
    in DS9.  After launching it, bring the DS9 window forward, then
    type 'a' over a source in the image to perform photometry at that
    location.  Typing 'c' will clear any previous results and 'q' will
    quit the ImExam loop.

    Clicking the Header button (magnifying glass icon) opens a new
    window that displays headers from selected FITS files in text form.
    The extensions displayed depends on the Extension setting selected
    (in Settings -> Display Settings).  If a particular extension is
    selected, only that header will be displayed.  If all extensions
    are selected (either for cube or multi-frame display), all extension
    headers will be displayed.  The buttons at the bottom of the window
    may be used to find or filter the header text, or generate a table
    of header keywords.  For filter or table display, a comma-separated
    list of keys may be entered in the text box.
    """
    if not HAS_PYQT5:  # pragma: no cover
        raise ImportError('PyQt5 package is required for QAD.')

    # suppress all runtime warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(
        description='Interactively display FITS data.')
    parser.add_argument('-l', '--loglevel', dest='loglevel', type=str,
                        action='store', default='INFO',
                        help='Log level.')
    parser.add_argument('-o', '--logfile', dest='logfile', type=str,
                        action='store', default=None,
                        help='Log file name.')
    parser.add_argument('-f', '--logformat', dest='logformat', type=str,
                        action='store', default=None,
                        help="Log format. Default is '%%(message)s' for "
                             "INFO level, '%%(asctime)s - %%(origin)s - "
                             "%%(levelname)s - %%(message)s' for any other "
                             "level.")

    args = parser.parse_args()

    # format the log for pretty-printing to the terminal
    log_level = args.loglevel.upper()
    Interface.tidy_log(log_level)

    # add a file handler if desired
    if args.logfile is not None:
        fhand = logging.FileHandler(args.logfile, 'at')
        fhand.setLevel(log_level)

        # set default log format by level
        log_format = args.logformat
        if log_format is None:
            if log_level == 'INFO':
                log_format = '%(message)s'
            else:
                log_format = '%(asctime)s - %(origin)s - ' \
                             '%(levelname)s - %(message)s'

        fhand.setFormatter(logging.Formatter(log_format))
        log.addHandler(fhand)

        # log the file name
        log.info("Log file: {}".format(args.logfile))

    # start application
    app = QtWidgets.QApplication(sys.argv)

    # start a timer to allow the python interpreter to run occasionally
    # (without this, ctrl-c is swallowed by the event loop)
    timer = QtCore.QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    # start up the main window and event loop
    mw = QADMainWindow()
    mw.show()
    mw.raise_()
    sys.exit(app.exec_())
