# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Standalone front-end for Eye of SOFIA display tool."""

import sys
import argparse
from typing import List
import warnings

from sofia_redux.visualization import eye

try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    HAS_PYQT5 = False
    QtWidgets, QtCore = None, None
else:
    HAS_PYQT5 = True

__all__ = ['main', 'parse_args', 'check_args']


def main():
    """
    The Eye of SOFIA spectral viewer.

    Start the Eye front end as a standalone tool with the command::

        eospec

    Optionally, a list of files to load into the viewer may be provided
    on the command line. From the GUI window, files may be loaded from the
    File Choice control panel, using the Add File button.  When the filename
    is displayed in the list, double-click it to display it in a plot pane.

    Multiple spectra may be displayed together in a pane, as long as
    their units are compatible.  Incompatible spectra may be displayed
    in separate panes; use the Add Pane button in the Panes panel to
    create a new plot window for display.

    Plot display can be modified or updated with controls in the Orders,
    Axis, and Plot control panels.

    Additionally, when a plot window has focus, some keyboard controls
    trigger display and analysis tasks.  In the plot window, press *x*, *y*,
    or *z* to start zoom mode in the x-direction, y-direction, or box
    mode, respectively. Click twice on the plot to set the new limits.
    Press *w* to reset the plot limits to defaults. Press *f* to start
    fitting mode, to fit a Gaussian + baseline to a spectral feature.
    Press *c* to clear any current zoom modes or plot overlays.
    """
    if not HAS_PYQT5:  # pragma: no cover
        raise ImportError('PyQt5 package is required for the Eye.')

    args = parse_args(sys.argv[1:])

    app = QtWidgets.QApplication(sys.argv)

    # Start a timer to allow the python interpreter to run occasionally
    # (without this, ctrl-c is swallowed by the event loop)
    timer = QtCore.QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    eye_window = eye.Eye(args)
    eye_window.open_eye()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sys.exit(app.exec_())


def parse_args(args: List) -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser('eospec')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='*',
                        help='File names to display')
    parser.add_argument('--log', dest='system_logs',
                        action='store_true', default=False,
                        help='Store system logs for debugging')
    parser.add_argument('--level', dest='log_level',
                        action='store', default='CRITICAL',
                        help='Log level for terminal log')

    args = parser.parse_args(args)
    return args


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Check arguments for validity.

    No checks currently implemented; this function is a placeholder
    for future implementation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    argparse.Namespace
    """
    return args
