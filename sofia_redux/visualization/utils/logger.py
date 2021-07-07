# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module defines a logging class based on the astropy logging module."""

import sys
import logging
from logging import Logger

from astropy.utils import find_current_module

try:
    from PyQt5 import QtWidgets
except ImportError:
    HAS_PYQT5 = False
    QtWidgets = None
else:
    HAS_PYQT5 = True

__all__ = ['log', 'EyeLogger', 'StreamLogger',
           'StatusLogger', 'DialogLogger']


# Initialize by calling init_log()
log = None


def _init_log():
    """
    Initializes the Eye log.

    In most circumstances, this is called automatically when
    importing sofia_redux.visualization.
    """
    global log

    orig_logger_cls = logging.getLoggerClass()
    logging.setLoggerClass(EyeLogger)
    try:
        log = logging.getLogger('eye')
        log._set_defaults()
    finally:
        logging.setLoggerClass(orig_logger_cls)

    return log


class EyeLogger(Logger):
    """
    Set up the Eye logging.

    This class is based on the astropy logger, but keeping only the
    record handling and some default setting functionality.
    """
    def makeRecord(self, name, level, pathname, lineno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra is None:
            extra = {}
        if 'origin' not in extra:
            current_module = find_current_module(1, finddiff=[True, 'logging'])
            if current_module is not None:
                extra['origin'] = current_module.__name__
            else:
                extra['origin'] = 'unknown'
        return Logger.makeRecord(self, name, level, pathname, lineno, msg,
                                 args, exc_info, func=func, extra=extra,
                                 sinfo=sinfo)

    def _set_defaults(self):
        """Reset logger to its initial state."""

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set levels
        self.setLevel('DEBUG')

        # Set up the stdout handler
        sh = StreamLogger()
        self.addHandler(sh)


class StreamLogger(logging.StreamHandler):
    """
    Log handler for logging messages to stdout or stderr streams.

    A specialized StreamHandler that logs INFO and DEBUG messages to
    stdout, and all other messages to stderr.  Also provides color
    coding of the output.
    """

    def emit(self, record):
        """
        Emit log messages to terminal.

        Parameters
        ----------
        record : `logging.LogRecord`
           The log record, with an additional 'origin' attribute
           attached by `EyeLogger.makeRecord`.
        """
        if record.levelno <= logging.INFO:
            stream = sys.stdout
        else:
            stream = sys.stderr

        if record.levelno < logging.DEBUG:
            print(record.levelname, end='', file=stream)
        else:
            # Import utils.console only if necessary and at the latest because
            # the import takes a significant time [#4649]
            from astropy.utils.console import color_print
            if record.levelno < logging.INFO:
                color_print(record.levelname, 'magenta', end='', file=stream)
            elif record.levelno < logging.WARN:
                color_print(record.levelname, 'green', end='', file=stream)
            elif record.levelno < logging.ERROR:
                color_print(record.levelname, 'brown', end='', file=stream)
            else:
                color_print(record.levelname, 'red', end='', file=stream)
        record.message = f"{record.msg} [{record.origin:s}]"
        print(": " + record.message, file=stream)


class StatusLogger(logging.Handler):
    def __init__(self, status_bar):
        """
        Log handler for logging info messages to a status bar.

        Parameters
        ----------
        status_bar : `PyQt5.QtWidgets.QStatusBar`
           Status bar widget to display to.
        """

        super().__init__()
        self.status_bar = status_bar

    def emit(self, record):
        """
        Display an INFO message in the status bar.

        Parameters
        ----------
        record : `logging.LogRecord`
           The log record, with an additional 'origin' attribute
           attached by `astropy.log`.
        """
        if record.levelno == logging.INFO:
            msg = str(record.msg)
            try:
                self.status_bar.showMessage(msg, 5000)
            except RuntimeError:  # pragma: no cover
                # can happen in race conditions, if app is
                # closed before log completes
                pass


class DialogLogger(logging.Handler):
    def __init__(self, parent):
        """
        Log handler for logging error messages to a dialog box.

        Parameters
        ----------
        parent : PyQt5.QtWidgets.QWidget
        """
        super().__init__()
        self.parent = parent

    def emit(self, record):
        """
        Display a WARNING or ERROR message in a dialog box.

        Parameters
        ----------
        record : `logging.LogRecord`
           The log record, with an additional 'origin' attribute
           attached by `EyeLogger.makeRecord`.
        """
        # no op if no pyqt
        if not HAS_PYQT5:  # pragma: no cover
            return

        # don't show dialog box if parent isn't up yet
        if not self.parent.isVisible():
            return

        msg = str(record.msg)
        if record.levelno == logging.WARNING:
            QtWidgets.QMessageBox.warning(self.parent, 'WARNING', msg)
        elif record.levelno == logging.ERROR:
            QtWidgets.QMessageBox.critical(self.parent, 'ERROR', msg)
