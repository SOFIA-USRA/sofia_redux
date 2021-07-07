# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Main GUI window for the QAD standalone tool."""

import mimetypes
import os
import signal
import subprocess
import sys

import astropy.io.fits as pf
from astropy import log
import configobj

from sofia_redux.pipeline.gui.qad import qad_dialogs
from sofia_redux.pipeline.gui.qad import qad_headview
from sofia_redux.pipeline.gui.qad import qad_imview
from sofia_redux.pipeline.gui.widgets import GeneralRunnable

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from sofia_redux.pipeline.gui.qad.ui import ui_qad_main
except ImportError:
    HAS_PYQT5 = False
    QtCore, QtGui = None, None

    # duck type parents to allow class definition
    class QtWidgets:
        class QMainWindow:
            pass

    class ui_qad_main:
        class Ui_MainWindow:
            pass
else:
    HAS_PYQT5 = True


class QADMainWindow(QtWidgets.QMainWindow, ui_qad_main.Ui_MainWindow):
    """
    QAD Qt5 GUI main window.

    All attributes and methods for this class are intended for
    internal use, to support the main GUI event loop and operations.
    This class is normally instantiated from the
    `sofia_redux.pipeline.gui.qad.main` function; all methods are
    triggered by user interaction only.

    The UI for this application is built in Qt Designer: see the
    `designer` folder for the Designer input files; the compiled
    Python scripts are in the `ui` module.  All `ui_*.py` files
    should not be edited manually, as they are automatically generated.
    See the `designer/compile_ui` file for the sequence of commands
    required to rebuild the UI from Designer files.
    """
    def __init__(self):
        """Build the QAD GUI window."""
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for QAD.')

        # parent initialization
        QtWidgets.QMainWindow.__init__(self)

        # set up UI from Designer generated file
        self.setupUi(self)

        # Establish signal handler to catch ctrl-C
        signal.signal(signal.SIGINT, self.cleanup)

        # connect GUI signals to slots

        # menu and toolbar
        self.actionOpenDirectory.triggered.connect(self.onOpen)
        self.actionGoPrevious.triggered.connect(self.onPrevious)
        self.actionGoNext.triggered.connect(self.onNext)
        self.actionGoHome.triggered.connect(self.onHome)
        self.actionImExam.triggered.connect(self.onImExam)
        self.actionDisplayHeader.triggered.connect(self.onDisplayHeader)
        self.actionSaveSettings.triggered.connect(self.onSaveSettings)
        self.actionDisplaySettings.triggered.connect(self.onDisplaySettings)
        self.actionPhotometrySettings.triggered.connect(
            self.onPhotometrySettings)
        self.actionPlotSettings.triggered.connect(
            self.onPlotSettings)

        # other widgets
        self.fileFilterBox.editingFinished.connect(self.onFilter)
        self.treeView.doubleClicked.connect(self.onRow)

        # set directory model in tree view widget
        self.model = QtWidgets.QFileSystemModel(self)
        self.treeView.setModel(self.model)
        self.treeView.sortByColumn(0, 0)

        # for file browser tree:
        # default root is current working directory
        # default file filter is *.fits
        self.file_filter = ['*.fits']
        self.fileFilterBox.setText(self.file_filter[0])
        self.setFilter()

        if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
            self.rootpath = sys.argv[1].rstrip(os.path.sep)
        else:
            self.rootpath = os.path.abspath(QtCore.QDir.currentPath())
        self.lastpath = []
        self.setRoot()

        # placeholder for header display
        self.headviewer = None

        # startup imviewer
        self.imviewer = None
        self.startupImViewer()

        # read settings if available
        self.cfg_dir = os.path.join(os.path.expanduser('~'), '.qad')
        os.makedirs(self.cfg_dir, exist_ok=True)

        if self.imviewer is not None:
            disp_cfg = os.path.join(self.cfg_dir, 'display.cfg')
            if os.path.isfile(disp_cfg):
                config = configobj.ConfigObj(disp_cfg, unrepr=True)
                self.imviewer.disp_parameters.update(config.dict())
            phot_cfg = os.path.join(self.cfg_dir, 'photometry.cfg')
            if os.path.isfile(phot_cfg):
                config = configobj.ConfigObj(phot_cfg, unrepr=True)
                self.imviewer.phot_parameters.update(config.dict())
            plot_cfg = os.path.join(self.cfg_dir, 'plot.cfg')
            if os.path.isfile(plot_cfg):
                config = configobj.ConfigObj(plot_cfg, unrepr=True)
                self.imviewer.plot_parameters.update(config.dict())

        # placeholder for imexam runnable
        self.imexam_worker = None

        # set status bar message
        self.setStatus('QAD Ready.')

    # override functions

    def closeEvent(self, event):
        """
        Clean up processes, then close the application.

        Parameters
        ----------
        event : QEvent
            Close event.
        """
        self.cleanup()

    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts.

        Parameters
        ----------
        event : QEvent
            Keypress event.
        """
        if type(event) == QtGui.QKeyEvent:
            if (self.treeView.hasFocus()
                    and event.key() == QtCore.Qt.Key_Return):
                self.onRow()

    # event handlers

    def onDisplayHeader(self):
        """
        Start up header display window and set text.

        This method uses the display settings to determine
        which extension(s) to retrieve headers for.  If a
        particular extension is selected, only that header will be
        displayed.  If all extensions are selected (either for cube
        or multi-frame display), all extension headers will be
        displayed.
        """
        index = self.treeView.selectionModel().selectedRows()
        headers = {}
        title = None
        for i in index:
            fpath = os.path.abspath(self.model.filePath(i))
            if os.path.isfile(fpath) and fpath.endswith('.fits'):
                try:
                    hdul = pf.open(fpath)
                except (OSError, ValueError, TypeError):
                    log.error(f'Cannot load {fpath} as FITS; ignoring')
                else:
                    headers[fpath] = []
                    # check the display settings -- if a particular extension
                    # is specified, only retrieve its header.  Otherwise,
                    # get them all
                    try:
                        exten = self.imviewer.get_extension_param()
                    except (ValueError, TypeError, IndexError,
                            AttributeError, KeyError):
                        exten = 'all'

                    if str(exten) != 'all':
                        try:
                            headers[fpath].append(hdul[exten].header)
                        except (ValueError, IndexError, TypeError,
                                AttributeError, KeyError):
                            log.warning(f'No extension {exten} found for '
                                        f'{fpath}; displaying all headers')
                            exten = 'all'
                    if str(exten) == 'all':
                        for hdu in hdul:
                            headers[fpath].append(hdu.header)

                    title = os.path.basename(fpath)

        if len(headers) == 0:
            self.setStatus("No FITS files selected.")
            return
        elif len(headers) > 1:
            title += '...'

        if self.headviewer is None or not self.headviewer.isVisible():
            self.headviewer = qad_headview.HeaderViewer(self)

        self.headviewer.load(headers)
        self.headviewer.show()
        self.headviewer.raise_()
        self.headviewer.setTitle("Header for: {}".format(title))
        self.setStatus("FITS headers displayed.")

    def onDisplaySettings(self):
        """Set general display parameters from user dialog."""
        if self.imviewer is None:
            self.setStatus('Cannot modify settings without ImView.')
            return
        current = self.imviewer.disp_parameters
        default = self.imviewer.default_parameters('display')
        dialog = qad_dialogs.DispSettingsDialog(self, current, default)
        retval = dialog.exec_()
        if retval == 1:
            self.imviewer.disp_parameters = dialog.getValue()

    def onFilter(self):
        """
        Filter displayed files.

        Multiple filters may be specified in a comma-separated list.
        The wildcard '*' may be used in any filter.  An empty filter
        box will display all files.
        """
        self.file_filter = self.fileFilterBox.text().split(',')
        self.file_filter = [str(f).strip() for f in self.file_filter]
        if (len(self.file_filter) == 1
                and str(self.file_filter[0]).strip() == ''):
            self.file_filter[0] = '*'
        self.setFilter()

    def onHome(self):
        """Set the home directory as the root."""
        self.rootpath = os.path.expanduser('~')
        self.lastpath = []
        self.setRoot()

    def onImExam(self):
        """Start imexam in a new thread."""
        self.actionImExam.setEnabled(False)
        self.setStatus("Starting imexam in DS9; press 'q' to quit.")
        self.imviewer.break_loop = False
        threadpool = QtCore.QThreadPool.globalInstance()
        self.imexam_worker = GeneralRunnable(self.imviewer.imexam)
        self.imexam_worker.signals.finished.connect(self.imexamFinish)
        threadpool.start(self.imexam_worker)

    def imexamFinish(self, status):
        """
        ImExam callback.

        Parameters
        ----------
        status : None or tuple
            If not None, contains an error message to log.
        """
        if status is not None:
            # log the error
            log.error("\n{}".format(status[2]))
        self.imexam_worker = None
        self.imviewer.break_loop = True
        self.setStatus('')
        self.actionImExam.setEnabled(True)

    def onNext(self):
        """Set the last recorded directory as the root."""
        if len(self.lastpath) > 0:
            self.rootpath = self.lastpath.pop()
            self.setRoot()

    def onOpen(self):
        """Select a new directory as the root."""
        newpath = os.path.abspath(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Select Directory'))
        if newpath.strip() != '':
            self.lastpath = []
            self.rootpath = newpath
            self.resetModel()

    def onPhotometrySettings(self):
        """Set parameters for photometry from a user dialog."""
        if self.imviewer is None:
            self.setStatus('Cannot modify settings without ImView.')
            return
        current = self.imviewer.phot_parameters
        default = self.imviewer.default_parameters('photometry')
        dialog = qad_dialogs.PhotSettingsDialog(self, current, default)
        retval = dialog.exec_()
        if retval == 1:
            self.imviewer.phot_parameters = dialog.getValue()

    def onPlotSettings(self):
        """Set parameters for plots from a user dialog."""
        if self.imviewer is None:
            self.setStatus('Cannot modify settings without ImView.')
            return
        current = self.imviewer.plot_parameters
        default = self.imviewer.default_parameters('plot')
        dialog = qad_dialogs.PlotSettingsDialog(self, current, default)
        retval = dialog.exec_()
        if retval == 1:
            self.imviewer.plot_parameters = dialog.getValue()

    def onPrevious(self):
        """Set the enclosing directory as the root."""
        if self.rootpath != '/':
            self.lastpath.append(self.rootpath)
            self.rootpath = os.path.dirname(self.rootpath)
            self.setRoot()

    def onRow(self):
        """Determine the selected files and schedule them for display."""
        # check for an open imexam first
        if self.imexam_worker is not None:
            QtWidgets.QMessageBox.warning(
                self, 'Load Files',
                'Please quit ImExam before loading new files.')
            return

        index = self.treeView.selectionModel().selectedRows()
        nsel = len(index)
        files_to_open = []
        if nsel == 1 and os.path.isdir(self.model.filePath(index[0])):
            self.lastpath = []
            self.rootpath = os.path.abspath(self.model.filePath(index[0]))
            self.setRoot()
        else:
            for i in index:
                fpath = self.model.filePath(i)
                if os.path.isfile(fpath):
                    files_to_open.append(os.path.abspath(fpath))
        if len(files_to_open) > 0:
            self.openFiles(files_to_open)

    def onSaveSettings(self):
        """
        Save current parameters to disk.

        Parameters are saved to files in the hidden .qad
        folder in the user's home directory.  On startup, these
        files are read and used as the default QAD settings.
        """
        if self.imviewer is None:
            self.setStatus('Cannot save settings without ImView.')
            return
        if self.cfg_dir is None or not os.path.isdir(self.cfg_dir):
            log.error('No config directory available; not saving parameters')
            return

        # display parameters
        config = configobj.ConfigObj(self.imviewer.disp_parameters,
                                     unrepr=True)
        config.filename = os.path.join(self.cfg_dir, 'display.cfg')
        config.write()

        # photometry parameters
        config = configobj.ConfigObj(self.imviewer.phot_parameters,
                                     unrepr=True)
        config.filename = os.path.join(self.cfg_dir, 'photometry.cfg')
        config.write()

        # plot parameters
        config = configobj.ConfigObj(self.imviewer.plot_parameters,
                                     unrepr=True)
        config.filename = os.path.join(self.cfg_dir, 'plot.cfg')
        config.write()

        self.setStatus('Settings saved to {:s}'.format(self.cfg_dir))

    def cleanup(self, *args):
        """Close the application."""
        # make sure any dangling imexam threads close
        if self.imviewer is not None:
            self.imviewer.break_loop = True
            self.imviewer.quit()
        QtWidgets.QApplication.quit()

    def openFiles(self, filelist):
        """
        Display selected files.

        This method dispatches FITS files, DS9 region files, and
        data arrays to the QAD image viewer.  Any other files are
        opened with the system default application for their file type.

        Parameters
        ----------
        filelist : list of str
            Full paths to files for display.
        """
        fits_files = []
        reg_files = []
        other_files = []
        for fpath in filelist:
            if fpath.endswith('.reg'):
                # ds9 region files: pass to imviewer
                reg_files.append(fpath)
            elif fpath.endswith('.fits'):
                # FITS files: pass to imviewer
                fits_files.append(fpath)
            else:
                # other files: pass to OS
                other_files.append(fpath)

        if len(fits_files) > 0:
            # make sure imexam loop is broken before new load
            if self.imviewer is not None:
                self.imviewer.break_loop = True
            try:
                self.imviewer.load(fits_files, regfiles=reg_files)
                self.setStatus('Loaded FITS files')
            except (ValueError, AttributeError):
                self.setStatus('Cannot open ImViewer')
                return

        if len(other_files) > 0:
            # if not FITS related, try xdg-open (for Linux)
            # or open (for Mac)
            # TODO - check windows
            from sys import platform
            if platform == 'darwin':
                cmd = ['open']
            else:
                cmd = ['xdg-open']
                openable = []
                for fname in other_files:
                    mtype = mimetypes.guess_type(fname)
                    try:
                        out = subprocess.check_output(
                            'xdg-mime query default {}'.format(mtype[0]),
                            shell=True)
                        if out.decode('utf-8').strip() != '':
                            openable.append(fname)
                    except Exception:
                        # never mind if there's an error
                        continue
                other_files = openable

            if len(other_files) > 0:
                for fname in other_files:
                    try:
                        subprocess.call(cmd + [fname])
                    except Exception:
                        # ignore it if anything goes wrong
                        pass

    def setFilter(self):
        """Set the file filter in the TreeView model."""
        self.model.setNameFilters(self.file_filter)
        self.model.setNameFilterDisables(False)

    def resetModel(self):
        """Reset the TreeView model."""
        self.model = QtWidgets.QFileSystemModel(self)
        self.treeView.setModel(self.model)
        self.setFilter()
        self.setRoot()

    def setRoot(self):
        """Set the TreeView model root path."""
        # set model root
        self.model.setRootPath(self.rootpath)

        # set tree root and hide unnecessary columns
        self.treeView.setRootIndex(self.model.index(self.rootpath))
        self.treeView.hideColumn(1)
        self.treeView.hideColumn(2)

        # resize to contents
        self.treeView.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        self.treeView.resizeColumnToContents(0)

        # enable/disable next button
        if len(self.lastpath) > 0:
            self.actionGoNext.setEnabled(True)
        else:
            self.actionGoNext.setEnabled(False)

        # set current directory name
        self.dirLabel.setText(self.rootpath)
        self.dirLabel.repaint()

        # set status message
        self.setStatus('Opened {:s}'.format(self.rootpath))

    def setStatus(self, msg):
        """
        Set a status message.

        Parameters
        ----------
        msg : str
            Message to display.
        """
        self.statusBar.showMessage(msg, 5000)

    def startupImViewer(self):
        """Start the DS9 image viewer."""
        try:
            self.imviewer = qad_imview.QADImView()
        except (ValueError, OSError) as e:
            log.error(e)
            self.setStatus('Cannot start ImViewer')
