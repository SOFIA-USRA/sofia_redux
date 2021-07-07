# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Main window for Redux Qt5 GUI."""

import os
import signal

from astropy import log

import sofia_redux.pipeline
from sofia_redux.pipeline.gui import widgets

try:
    # dill allows pickling of more complex
    # software structures.
    # Undo may be unavailable for some reduction
    # objects if dill is not installed.
    import dill as pickle
except ImportError:
    import pickle

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from sofia_redux.pipeline.gui.ui import ui_main
except ImportError:
    HAS_PYQT5 = False
    QtCore, QtGui = None, None

    # duck type parents to allow class definition
    class QtWidgets:
        class QMainWindow:
            pass

    class ui_main:
        class Ui_MainWindow:
            pass
else:
    HAS_PYQT5 = True


class ReduxMainWindow(QtWidgets.QMainWindow, ui_main.Ui_MainWindow):
    """
    Redux Qt5 GUI main window.

    All attributes and methods for this class are intended for
    internal use, to support the main GUI event loop and operations.
    This class is normally instantiated from a `redux.Application`
    object; all methods are triggered by user interaction only.

    The UI for this application is built in Qt Designer: see the
    `designer` folder for the Designer input files; the compiled
    Python scripts are in the `ui` module.  All `ui_*.py` files
    should not be edited manually, as they are automatically generated.
    See the `designer/compile_ui` file for the sequence of commands
    required to rebuild the UI from Designer files.
    """
    def __init__(self, interface):
        """
        Start up the main GUI window.

        Parameters
        ----------
        interface : redux.Application
            Reduction interface object.  This class chooses, instantiates,
            and runs appropriate reduction objects for input data, as well
            as controlling any associated viewers.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        QtWidgets.QMainWindow.__init__(self)

        # store parent interface
        self.interface = interface

        # place holder for a text view dialog
        self.param_view = None
        self.config_view = None

        # set up UI from Designer generated file
        self.setupUi(self)

        # establish signal handler to catch ctrl-C
        signal.signal(signal.SIGINT, self.cleanup)

        # connect GUI signals to slots

        # menu events
        self.actionOpenNewReduction.triggered.connect(
            self.onOpenReduction)
        self.actionCloseReduction.triggered.connect(
            self.onCloseReduction)
        self.actionAddFiles.triggered.connect(
            self.onAddFiles)
        self.actionRemoveFiles.triggered.connect(
            self.onRemoveFiles)
        self.actionSetOutputDirectory.triggered.connect(
            self.setOutputDirectory)
        self.actionSaveInputManifest.triggered.connect(
            self.saveInputManifest)
        self.actionSaveOutputManifest.triggered.connect(
            self.saveOutputManifest)
        self.actionDisplayAllParameters.triggered.connect(
            self.onDisplayParameters)
        self.actionLoadParameters.triggered.connect(
            self.onLoadParameters)
        self.actionSaveParameters.triggered.connect(
            self.saveParameters)
        self.actionResetAllParameters.triggered.connect(
            self.onResetParameters)
        self.actionDisplayConfiguration.triggered.connect(
            self.onDisplayConfig)
        self.actionLoadConfiguration.triggered.connect(
            self.onLoadConfiguration)
        self.actionSaveConfiguration.triggered.connect(
            self.saveConfiguration)
        self.actionResetConfiguration.triggered.connect(
            self.onResetConfiguration)
        self.actionQuit.triggered.connect(
            self.closeEvent)

        # set default values for display menu options
        if self.interface.configuration.update_display is False:
            self.actionUpdateDisplays.setChecked(False)
        else:
            self.actionUpdateDisplays.setChecked(True)
        if self.interface.configuration.display_intermediate is False:
            self.actionDisplayIntermediate.setChecked(False)
        else:
            self.actionDisplayIntermediate.setChecked(True)

        # then connect the display updates to slots
        self.actionUpdateDisplays.triggered.connect(
            self.toggleDisplay)
        self.actionDisplayIntermediate.triggered.connect(
            self.toggleDisplayIntermediate)

        # other widgets
        self.stepButton.clicked.connect(self.step)
        self.undoButton.clicked.connect(self.undo)
        self.resetButton.clicked.connect(self.resetSteps)
        self.reduceButton.clicked.connect(self.reduce)

        # set the in-application log handler
        logger = widgets.TextEditLogger()
        logger.signals.finished.connect(self.addLog)
        # set it to the same level as the terminal handler
        logger.setLevel(log.handlers[0].level)
        log.addHandler(logger)

        # get a threadpool for running pipeline steps
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self.worker = None

        # add a hidden progress widget to show when necessary
        self.progress = widgets.ProgressFrame(self.stepControls)
        self.stepControls.layout().addWidget(self.progress)
        self.progress.hide()
        self.progress.stopButton.clicked.connect(self.stopReduction)

        # reduction information
        self.base_directory = str(QtCore.QDir.currentPath())
        self.param_directory = self.base_directory
        self.save_directory = self.base_directory
        self.nsteps = 0
        self.last_step = -1
        self.loaded_files = []
        self.default_param = None
        self.default_config = self.interface.configuration.to_text()
        self.pickled_reduction = None
        self.allow_undo = True

        # set Redux version in label
        self.setPipeName()

        # disable most controls on startup
        self.initial_startup = True
        self.enableReduction(False)

    def closeEvent(self, event):
        """
        Confirm before closing the application.

        Parameters
        ----------
        event : QEvent
            Close event.
        """
        response = QtWidgets.QMessageBox.question(
            self, 'Quit', 'Quit Redux?')
        if response == QtWidgets.QMessageBox.Yes:
            self.cleanup()
        else:
            try:
                event.ignore()
            except AttributeError:
                pass

    def cleanup(self, *args):
        """Quit the application."""
        # close the current reduction
        self.resetPipeline()
        for hand in log.handlers:
            if isinstance(hand, widgets.TextEditLogger):
                log.removeHandler(hand)
        # quit
        QtWidgets.QApplication.quit()

    # display functions

    def addLog(self, msg):
        """
        Add a log message to the log widget.

        Callback for the TextEditLogger.  Messages are appended
        to the end of the TextEdit window, in HTML format.

        Parameters
        ----------
        msg : str
            HTML formatted string to display in the TextEdit widget.
        """
        # move cursor to the end
        cursor = self.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.logTextEdit.setTextCursor(cursor)

        # insert the message as HTML
        self.logTextEdit.insertHtml("<pre>{}<br></pre>".format(msg))

        # move the cursor to the end again
        cursor.movePosition(QtGui.QTextCursor.End)
        self.logTextEdit.setTextCursor(cursor)

        # repaint to refresh view
        self.logTextEdit.repaint()

    def setPipeName(self, msg=None):
        """
        Set the pipeline name in a label widget.

        Parameters
        ----------
        msg : str, optional
            Pipeline version label.  If not provided, the Redux
            version label will be displayed.
        """
        if msg is None:
            msg = "Redux v{}".format(sofia_redux.pipeline.__version__)
        self.pipelineVersionLabel.setText(msg)
        self.pipelineVersionLabel.repaint()

    def setStatus(self, msg):
        """
        Set a status message.

        Messages are displayed in the status bar and as an
        INFO level log message.

        Parameters
        ----------
        msg : str
            Status message.
        """
        self.statusbar.showMessage(msg, 5000)
        log.info(' == {} =='.format(msg))

    def resetView(self):
        """Remove the Data View tab."""
        if self.dataTabWidget.tabText(0) == "Data View":
            # delete the viewer widget, then remove the tab
            data_widget = self.dataTabWidget.widget(0)
            data_widget.deleteLater()
            self.dataTabWidget.removeTab(0)

    def setFileSummary(self):
        """Set a loaded file summary message."""
        nfiles = len(self.loaded_files)
        maxdisp = 6
        msg = ["Loaded files:"]
        for i in range(maxdisp + 1):
            val = ''
            if i >= nfiles:
                val = '  '
            elif i <= maxdisp - 2:
                val = '  ' + os.path.basename(self.loaded_files[i])
            elif i == maxdisp - 1:
                if nfiles == maxdisp:
                    val = '  ' + os.path.basename(self.loaded_files[i])
                else:
                    val = '  ...'
            elif i == maxdisp:
                val = '  ' + os.path.basename(self.loaded_files[-1])
            msg.append(val)
        self.fileSummaryTextEdit.setText('\n'.join(msg))
        self.fileSummaryTextEdit.repaint()

    def updateConfigView(self):
        """Update config viewer widget with new values."""
        if self.config_view is not None and self.config_view.isVisible():
            title = 'Redux Configuration'
            text = self.interface.save_configuration()
            self.config_view.load(text)
            self.config_view.setTitle(title)

    def updateParamView(self):
        """Update parameter viewer widget with new values."""
        if self.param_view is not None and self.param_view.isVisible():
            title = 'Reduction Parameters'
            text = self.interface.save_parameters()
            self.param_view.load(text)
            self.param_view.setTitle(title)

    # reduction functions
    def enableControls(self, enable=True):
        """
        Toggle reduction-related controls.

        These controls (add files, edit parameters, run step, etc.)
        are disabled while a reduction step is running.

        Parameters
        ----------
        enable : bool
            If True, controls are enabled.  If False they are disabled.
        """
        # dis/enable most menu items
        self.actionOpenNewReduction.setEnabled(enable)
        self.actionAddFiles.setEnabled(enable)
        self.actionRemoveFiles.setEnabled(enable)
        self.actionLoadParameters.setEnabled(enable)
        self.actionSaveParameters.setEnabled(enable)
        self.actionCloseReduction.setEnabled(enable)
        self.actionResetAllParameters.setEnabled(enable)
        self.actionSetOutputDirectory.setEnabled(enable)
        self.actionSaveInputManifest.setEnabled(enable)
        self.actionSaveOutputManifest.setEnabled(enable)

        # dis/enable the reduction controls
        self.stepFrame.setEnabled(enable)
        self.stepThroughFrame.setEnabled(enable)
        self.pipeStepFrame.setEnabled(enable)

        # enable the controls box either way,
        # so that progress bar is available
        self.controlsBox.setEnabled(True)

        # enable selection in list widget if controls are disabled
        if enable:
            self.pipeStepListWidget.clearSelection()
            self.pipeStepListWidget.setSelectionMode(
                QtWidgets.QAbstractItemView.NoSelection)
        else:
            self.pipeStepListWidget.setSelectionMode(
                QtWidgets.QAbstractItemView.SingleSelection)

    def enableReduction(self, enable=True):
        """
        Toggle all non-default controls.

        These controls are disabled when no data is loaded.
        Only opening a new reduction or quitting is allowed in
        this condition.

        Parameters
        ----------
        enable : bool
            If True, enable the controls.  If False, disable them.
        """
        # always allow a new reduction to be opened and
        # Redux to quit
        self.actionOpenNewReduction.setEnabled(True)
        self.actionQuit.setEnabled(True)

        # dis/enable most other menu items
        self.actionAddFiles.setEnabled(enable)
        self.actionRemoveFiles.setEnabled(enable)
        self.actionSetOutputDirectory.setEnabled(enable)
        self.actionSaveInputManifest.setEnabled(enable)
        self.actionSaveOutputManifest.setEnabled(enable)

        self.actionDisplayAllParameters.setEnabled(enable)
        self.actionLoadParameters.setEnabled(enable)
        self.actionSaveParameters.setEnabled(enable)
        self.actionCloseReduction.setEnabled(enable)
        self.actionResetAllParameters.setEnabled(enable)

        # dis/enable the top-level reduction controls;
        # always enable the sub-widgets
        self.controlsBox.setEnabled(enable)
        self.stepFrame.setEnabled(True)
        self.stepThroughFrame.setEnabled(True)
        self.pipeStepFrame.setEnabled(True)

    def reduce(self):
        """Run all remaining reduction steps."""
        self.step(skip_save=True, run_all=True)

    def resetPipeline(self):
        """
        Reset the pipeline when a reduction is closed.

        All data items, parameter values, and pipeline
        steps are removed.
        """
        self.initial_startup = False
        self.interface.close_viewers()
        self.interface.clear_reduction()
        self.logTextEdit.clear()
        self.resetView()
        self.pipeStepListWidget.clear()
        self.fileSummaryTextEdit.clear()
        self.setPipeName()
        self.fileTableView.setModel(None)

        # clear steps out of combo box,
        # then add a blank item (i.e. no step-through)
        self.stepThroughComboBox.clear()
        self.stepThroughComboBox.addItem('')

        self.stepButton.setEnabled(False)
        self.undoButton.setEnabled(False)
        self.reduceButton.setEnabled(False)
        self.resetButton.setEnabled(False)

        self.last_step = -1

    def resetSteps(self):
        """
        Reset reduction steps to re-start a loaded reduction.

        Parameters are not reset from edited values.  The raw data
        is re-loaded into the reduction.  Viewers are reset.
        """
        if self.interface.reduction is None:
            return

        # reset reduction, then restore current parameters
        saved_param = self.interface.reduction.parameters.copy()
        self.interface.reset_reduction(self.loaded_files)
        self.interface.reset_viewers()
        self.interface.reduction.parameters = saved_param

        # reset step controls
        self.logTextEdit.clear()
        self.stepButton.setEnabled(True)
        self.undoButton.setEnabled(False)
        self.reduceButton.setEnabled(True)
        self.resetButton.setEnabled(False)

        for i in range(self.nsteps):
            item = self.pipeStepListWidget.item(i)
            widget = self.pipeStepListWidget.itemWidget(item)
            widget.setEnabled(True)
            if i == 0:
                widget.enableRun()
            else:
                widget.enableRun(False)
        self.last_step = -1

        # update any associated viewers
        self.interface.update_viewers()
        self.setStatus("Pipeline steps reset.")

    def stepFinish(self, status):
        """
        Update GUI after a reduction step finishes.

        Callback for the `widgets.StepRunnable`, which runs
        a reduction step in a separate thread.

        Parameters
        ----------
        status : tuple or str
            If tuple, `status` is assumed to be an unexpected exception
            message, with values (type, message, traceback).  If str,
            it is assumed to be an error caught by the reduction
            object and recorded as a return value for the step.  Either
            way, if the message is not an empty string, it is displayed
            to the user in a QMessageBox warning and reduction is halted.

        Returns
        -------
        bool
            True if reduction step succeeded; False if it threw an error.
        """
        if type(status) is tuple:
            # log the error
            log.error("\n{}".format(status[2]))
            msg = status[1]
        else:
            msg = status

        # update viewers from last step
        self.interface.update_viewers()

        self.enableControls(True)
        self.controlsBox.unsetCursor()
        self.progress.hide()
        self.stepFrame.show()
        self.stepThroughFrame.show()

        last = self.last_step
        current = self.interface.reduction.step_index
        for step in range(last, current):
            # disable the steps just run
            item = self.pipeStepListWidget.item(step)
            try:
                self.pipeStepListWidget.itemWidget(item).setEnabled(False)
            except AttributeError:
                # ignore it if the item isn't available
                pass

        # enable the undo and reset buttons
        if self.allow_undo:
            self.undoButton.setEnabled(True)
        else:
            self.undoButton.setEnabled(False)
        self.resetButton.setEnabled(True)

        if msg != '':
            # error from pipeline -- disable steps
            self.stepButton.setEnabled(False)
            self.reduceButton.setEnabled(False)
            QtWidgets.QMessageBox.warning(
                self, 'Pipeline Step',
                "Error from pipeline: {}".format(msg))
            return False

        # enable run for the next step
        item = self.pipeStepListWidget.item(current)
        try:
            self.pipeStepListWidget.itemWidget(item).enableRun()
        except AttributeError:
            # no more steps -- disable step and reduce
            self.stepButton.setEnabled(False)
            self.reduceButton.setEnabled(False)

        self.setStatus("Pipeline step complete.")
        self.repaint()
        return True

    def highlightStep(self, idx):
        """
        Highlight the currently running step.

        Parameters
        ----------
        idx : int
            Step index number.
        """
        if -1 < idx < self.nsteps:
            item = self.pipeStepListWidget.item(idx)
            self.pipeStepListWidget.setCurrentItem(item)

    def stepProgress(self, value):
        """
        Update the progress widget.

        Callback for the `widgets.StepRunnable` for intermediate
        progress.
        """
        self.progress.setProgress(value)
        self.highlightStep(self.interface.reduction.step_index)

        # check whether intermediate files are displayed
        if self.actionDisplayIntermediate.isChecked():
            self.interface.update_viewers()

    def step(self, skip_save=False, skip_break=False, run_all=False):
        """
        Launch a reduction step, or series of steps.

        Reduction steps are run in a separate thread from a thread pool,
        via `QtCore.QThreadPool`.  The worker for this thread is
        `widgets.StepRunnable`.

        The number of steps to run is determined by the widget from
        which this method was triggered.  If the Reduce button
        is pressed, all remaining steps are run (run_all=True).
        If the Run button next to a step is pressed, only that step is run
        (skip_break=True).  If the Step button is pressed, all steps up
        through the value selected in the Step Through box are run (default).

        Parameters
        ----------
        skip_save : bool, optional
            If True, the reduction object will not be pickled before
            running the step.  This effectively disables Undo for the
            step.
        skip_break : bool, optional
            If True, the value of the stepThroughComboBox will be ignored
            and only the next step will be run.
        run_all : bool, optional
            If True, all remaining steps will be run.
        """
        if self.interface.reduction is None:
            return

        # save the last step index
        self.last_step = self.interface.reduction.step_index

        # pickle the current reduction
        # Check the internal allow_undo here -- some steps may
        # be undoable, some not.
        if self.interface.reduction.allow_undo and not skip_save:
            try:
                self.pickled_reduction = pickle.dumps(self.interface.reduction)
                self.allow_undo = True
            except (TypeError, AttributeError, pickle.PicklingError):
                # raise error
                log.warning("Reduction object is not serializable; "
                            "'undo' will not be available.")
                self.allow_undo = False
                self.pickled_reduction = None
        else:
            self.pickled_reduction = None
            self.allow_undo = False

        # get the step-through value
        if run_all:
            step_through = self.nsteps
        elif skip_break:
            step_through = -1
        else:
            step_through = self.stepThroughComboBox.currentIndex()

        if step_through > 0 and step_through > self.last_step:
            nsteps = step_through - self.last_step
        else:
            nsteps = 1

        # turn off the whole controls box and menus
        # while steps are running
        self.enableControls(False)
        self.controlsBox.setCursor(QtCore.Qt.BusyCursor)
        self.highlightStep(self.interface.reduction.step_index)

        # make a runnable object for threading and connect it
        # to its callback
        self.setStatus("Running pipeline step.")
        self.worker = widgets.StepRunnable(self.interface.step, nsteps)
        self.worker.signals.finished.connect(self.stepFinish)

        # show a progress widget if running more than one step
        if nsteps > 1:
            self.progress.resetProgress()
            self.progress.setNStep(nsteps)

            self.stepFrame.hide()
            self.stepThroughFrame.hide()
            self.progress.show()

            # connect the runnable to the progress widget
            self.worker.signals.progress.connect(self.stepProgress)

        # start the reduction
        self.threadpool.start(self.worker)
        self.repaint()

    def undo(self):
        """
        Undo the last reduction step.

        This method unpickles the last reduction object saved in the
        `pickled_reduction` attribute.  This normally happens at the
        beginning of the `step` method, but a pickled reduction may
        be unavailable for either of the following reasons:

        * the reduction object does not allow undo operations (because
          the object is too large to save, for example)
        * serialization failed (the reduction object currently holds
          a reference to a file pointer, for example)

        It is recommended that the user install the dill module, as it
        allows serialization for more complex software structures than
        the built-in pickle module.
        """
        if self.pickled_reduction is None:
            QtWidgets.QMessageBox.warning(
                self, 'Undo', "Cannot undo the last step.")
            return

        # get the step index from the current reduction
        current = self.interface.reduction.step_index

        # restore the reduction from a pickle
        self.interface.reduction = pickle.loads(self.pickled_reduction)
        self.pickled_reduction = None

        # enable/disable buttons
        self.undoButton.setEnabled(False)
        self.stepButton.setEnabled(True)
        self.reduceButton.setEnabled(True)

        for i in range(current, self.last_step - 1, -1):
            item = self.pipeStepListWidget.item(i)
            try:
                step_widget = self.pipeStepListWidget.itemWidget(item)
                step_widget.setEnabled(True)
                if i == self.last_step:
                    step_widget.enableRun(True)
                else:
                    step_widget.enableRun(False)
                name = step_widget.pipeStepLabel.text()
                if i < current:
                    log.warning("Undid step {}: {}".format(i + 1, name))
            except AttributeError:
                # ignore if the item isn't available
                # (e.g. if there is one step, it has been run, and then
                # undone)
                pass

        self.last_step = -1

        # update viewers from last data if possible
        self.interface.update_viewers()

    # slot functions

    def onOpenReduction(self, add=False, remove_files=None):
        """
        Open a new reduction.

        Called when the following File menu items are selected:

        * Open New Reduction
        * Add Files (`add` = True)
        * Remove Files (`remove_files` = True)

        Any of these three options will reset the reduction and
        load the data into a new reduction object.  If files are being
        removed or added, the parameters and output directory
        from the old reduction object will be saved and restored
        into the new one.

        Files are loaded in a new thread from a thread pool,
        via `QtCore.QThreadPool`.  The worker for this thread is
        `widgets.LoadRunnable`.

        Parameters
        ----------
        add : bool
            Add more files to the current reduction.
        remove_files : bool
            Remove files from the current reduction
        """
        # save the parameters to avoid resetting them
        # them if adding or removing files
        if (add or remove_files) and self.interface.reduction is not None:
            saved_param = self.interface.reduction.parameters.copy()
            saved_dir = self.interface.reduction.output_directory
        else:
            saved_param = None
            saved_dir = None

        if remove_files is None:
            # file dialog widget to choose new files for add or open
            newpath = QtWidgets.QFileDialog.getOpenFileNames(
                self, caption="Select Data File(s)",
                directory=self.base_directory,
                filter="FITS files (*.fits);;"
                       "Input manifests (*.txt);;"
                       "All files (*)")
            if len(newpath[0]) == 0:
                # do nothing if no files were selected.
                return

            # store the path from the first file
            if add and self.interface.reduction is not None:
                # if adding, test that the new files can be reduced
                # with the same object
                data_files = self.loaded_files.copy()
                for fname in newpath[0]:
                    if fname not in data_files:
                        data_files.append(fname)
                test_reduction = \
                    self.interface.chooser.choose_reduction(data_files)
                if not isinstance(test_reduction,
                                  type(self.interface.reduction)):
                    msg = 'New files do not match old files; cannot add them.'
                    log.warning(msg)
                    QtWidgets.QMessageBox.warning(self, 'Add Files', msg)
                    return
            else:
                # otherwise, just use the new files
                data_files = newpath[0]
                self.base_directory = os.path.dirname(data_files[0])
        else:
            # if removing, filter out any selected files
            data_files = []
            for fname in self.loaded_files.copy():
                if fname not in remove_files:
                    data_files.append(fname)
                else:
                    log.warning('Removing {}'.format(os.path.basename(fname)))
            if not data_files:
                # if all files removed, reset the pipeline and return
                self.resetPipeline()
                self.loaded_files = []
                return

        # reset the pipeline
        self.resetPipeline()

        # load the data in a new thread
        loader = widgets.LoadRunnable(self.interface.start,
                                      data_files, saved_param,
                                      saved_dir)
        loader.signals.finished.connect(self.openFinish)
        self.threadpool.start(loader)
        self.setCursor(QtCore.Qt.BusyCursor)
        self.enableReduction(False)

    def openFinish(self, result):
        """
        Finish loading a new reduction.

        This method builds the pipeline step widgets, updates display
        widgets and viewers, and enables reduction controls.

        This method is the callback for the `widgets.LoadRunnable` class,
        which loads new data from disk in a separate thread.

        Parameters
        ----------
        result : tuple
            If `result` is a three-element tuple, it is assumed to
            be an unexpected exception, with elements
            (type, message, traceback).  If it is a two-element
            tuple, the first element is a set of saved parameters
            from the previous reduction and the second element is
            the output directory from the old reduction. If either
            element is non-null, it will be restored to the new
            reduction.
        """
        # load the reduction from the files
        self.unsetCursor()
        if type(result) is tuple and len(result) == 3:
            # log the error
            log.error("\n{}".format(result[2]))
            QtWidgets.QMessageBox.warning(
                self, 'Open Reduction',
                "Error loading data: {}".format(result[1]))
            self.enableReduction(False)
            return
        else:
            saved_param, saved_dir = result

        if self.interface.reduction is None:
            QtWidgets.QMessageBox.warning(
                self, 'Open Reduction',
                "No data loaded.")
            self.enableReduction(False)
            return

        self.enableReduction(True)

        self.loaded_files = self.interface.reduction.raw_files
        if saved_param is not None \
            and (self.interface.reduction.parameters.stepnames
                 == saved_param.stepnames):
            log.debug('Loading saved parameters.')
            self.interface.reduction.parameters = saved_param
        else:
            self.default_param = self.interface.reduction.parameters.copy()
        if saved_dir is not None:
            self.interface.reduction.output_directory = saved_dir
            self.interface.set_log_file()
        self.allow_undo = self.interface.reduction.allow_undo

        # set the pipeline description
        msg = self.interface.reduction.description
        self.setPipeName(msg)
        recipe = self.interface.reduction.recipe
        step_dict = self.interface.reduction.processing_steps
        self.nsteps = len(recipe)

        # set the file summary
        self.setFileSummary()

        # register viewers for the reduction;
        # if they are embedded, add them to a splitter widget
        parent = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.interface.register_viewers(parent)
        if self.interface.has_embedded_viewers():
            self.dataTabWidget.insertTab(0, parent, 'Data View')
            self.dataTabWidget.setCurrentIndex(0)

        # read the data id keys and set them in the file info table
        self.interface.load_data_id()
        model = widgets.DataTableModel(self.interface.reduction.data_id,
                                       self.fileTableView)
        proxy = QtCore.QSortFilterProxyModel()
        proxy.setSourceModel(model)
        self.fileTableView.setModel(proxy)

        # size the columns to content, then make them interactive
        header = self.fileTableView.horizontalHeader()
        for column in range(self.fileTableView.model().columnCount()):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents)
            width = header.sectionSize(column)
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.Interactive)
            header.resizeSection(column, width)

        # add row numbers
        self.fileTableView.verticalHeader().show()

        # load the pipe steps
        for idx, step in enumerate(recipe):
            try:
                stepname = step_dict[step]
            except KeyError:
                stepname = step
            step_widget = widgets.PipeStep(stepname=stepname,
                                           index=idx)
            step_widget.connectRun(self.onRun)
            step_widget.connectEdit(self.onEdit)

            # enable run for the first step; leave the others disabled
            if idx == 0:
                step_widget.enableRun()

            # disable edit if no parameters to set
            try:
                param = self.interface.reduction.get_parameter_set(idx)
            except IndexError:
                param = None
            if param:
                step_widget.enableEdit()
            else:
                step_widget.enableEdit(False)

            w_item = QtWidgets.QListWidgetItem(self.pipeStepListWidget)
            w_item.setSizeHint(step_widget.sizeHint())
            self.pipeStepListWidget.addItem(w_item)
            self.pipeStepListWidget.setItemWidget(w_item, step_widget)

            # also add step name to the "Step Through" box
            self.stepThroughComboBox.addItem(
                "{}. {}".format(idx + 1, stepname))

        self.stepButton.setEnabled(True)
        self.undoButton.setEnabled(False)
        self.reduceButton.setEnabled(True)
        self.resetButton.setEnabled(False)

        # update any associated viewers after load
        self.interface.update_viewers()
        self.updateParamView()

        self.setStatus('New reduction loaded.')
        self.repaint()

    def onCloseReduction(self):
        """
        Close the current reduction.

        Called when the user selects Close Reduction from the File menu.
        """
        if self.interface.reduction is None:
            return

        response = QtWidgets.QMessageBox.question(
            self, 'Close Reduction', 'Close current reduction?')
        if response == QtWidgets.QMessageBox.Yes:
            self.resetPipeline()
            self.enableReduction(False)
            self.setStatus('Reduction closed.')
            self.repaint()

    def onAddFiles(self):
        """Add new files to the current reduction."""
        self.onOpenReduction(add=True)

    def onRemoveFiles(self):
        """Remove files from the current reduction."""
        dialog = widgets.RemoveFilesDialog(self, self.loaded_files)
        retval = dialog.exec_()
        if retval == QtWidgets.QDialog.Accepted:
            remove_files = dialog.getValue()
            self.onOpenReduction(remove_files=remove_files)

    def onDisplayConfig(self):
        """Display all current configuration in a new widget."""
        if self.config_view is None or not self.config_view.isVisible():
            self.config_view = widgets.ConfigView(self)
            self.config_view.saveButton.clicked.connect(
                self.onEditConfiguration)

        self.config_view.show()
        self.updateConfigView()

    def onDisplayParameters(self):
        """Display all current parameters in a new widget."""
        if self.interface.reduction is None:
            return

        if self.param_view is None or not self.param_view.isVisible():
            self.param_view = widgets.ParamView(self)

        self.param_view.show()
        self.updateParamView()

    def onLoadConfiguration(self, default=False, config_edit=False):
        """Load persistent configuration from a file."""
        if self.initial_startup:
            # clear startup image
            self.resetPipeline()

        if not default and not config_edit:
            newpath = QtWidgets.QFileDialog.getOpenFileName(
                self, caption="Select input file path for configuration",
                directory=self.param_directory,
                filter="Config files (*.cfg);;All files (*)")
            if len(newpath[0]) == 0:
                # do nothing if no files were selected.
                return

            infile = newpath[0]
            self.param_directory = os.path.dirname(infile)
            self.interface.update_configuration(infile)
            self.setStatus("Configuration updated from {}.".format(infile))
        elif default:
            config_str = '\n'.join(self.default_config)
            self.interface.configuration.load(config_str)
            self.setStatus("All settings reset to default values.")
        elif config_edit:
            if self.config_view is not None:
                new_config = self.config_view.text
                try:
                    self.interface.configuration.load(new_config)
                except (OSError, SyntaxError):
                    log.error('Configuration badly formatted; not saved.')
                    return
                self.setStatus("Configuration edited.")

        # check for GUI settings in new config
        update = bool(self.interface.configuration.update_display)
        intermed = bool(self.interface.configuration.display_intermediate)
        self.actionUpdateDisplays.setChecked(update)
        self.actionDisplayIntermediate.setChecked(intermed)
        self.toggleDisplay()
        self.toggleDisplayIntermediate()

    def onLoadParameters(self):
        """Load parameters from a configuration file."""
        if self.interface.reduction is None:
            return

        newpath = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Select input file path for parameters",
            directory=self.param_directory,
            filter="Config files (*.cfg);;All files (*)")
        if len(newpath[0]) == 0:
            # do nothing if no files were selected.
            return

        infile = newpath[0]
        self.param_directory = os.path.dirname(infile)
        self.interface.load_parameters(infile)
        self.default_param = self.interface.reduction.parameters.copy()

        self.updateParamView()
        self.setStatus("Parameters updated from {}.".format(infile))

    def onResetConfiguration(self):
        """Reset configuration to default values."""
        self.onLoadConfiguration(default=True)

    def onResetParameters(self):
        """Reset all parameters to default values."""
        if self.interface.reduction is None:
            return

        self.interface.load_parameters()
        self.default_param = self.interface.reduction.parameters.copy()
        self.updateParamView()
        self.setStatus("All reduction parameters reset to default values.")

    def onRun(self, index):
        """
        Run the next reduction step.

        Called when a Run button in a `widgets.PipeStep` is pushed.

        Parameters
        ----------
        index : int
            Reduction step index.  This value is stored in the
            step widget (`widgets.PipeStep`).  `index` is currently ignored,
            since the Run buttons are enabled for the next step only.
        """
        self.step(skip_break=True)

    def onEdit(self, index):
        """
        Edit parameters for a reduction step.

        Called when an Edit button in a `widgets.PipeStep` is pushed.

        Parameters
        ----------
        index : int
            Reduction step index.  This value is stored in the
            step widget (`widgets.PipeStep`).
        """
        if self.interface.reduction is None:
            return
        param = self.interface.reduction.get_parameter_set(index)
        default = self.default_param.current[index]
        try:
            name = self.interface.reduction.processing_steps[
                self.interface.reduction.recipe[index]]
        except KeyError:
            name = self.interface.reduction.recipe[index]
        dialog = widgets.EditParam(self, name,
                                   param, default,
                                   self.base_directory)
        retval = dialog.exec_()
        if retval == QtWidgets.QDialog.Accepted:
            # change params
            param = dialog.getValue()
            self.interface.reduction.set_parameter_set(param, index)
            self.updateParamView()

    def onEditConfiguration(self):
        self.onLoadConfiguration(config_edit=True)

    def stopReduction(self):
        """
        Cancel reduction in thread after current step.

        Called when the Stop Reduction button is pressed in the
        progress widget.
        """
        if self.worker is None:
            return
        self.worker.stop = True
        self.progress.stopButton.setEnabled(False)

    def saveInputManifest(self):
        """
        Save an input manifest to disk.

        Called by a File menu option.  Calls the
        `Interface.save_input_manifest` method.
        """
        if self.interface.reduction is None:
            return

        default_name = self.interface.configuration.input_manifest
        if default_name is None:
            dirname = self.save_directory
        else:
            if not os.path.isabs(default_name):
                dirname = os.path.join(self.save_directory, default_name)
            else:
                dirname = default_name
        newpath = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Select output file path for input manifest",
            directory=dirname, filter="Text files (*.txt);;All files (*)")
        if len(newpath[0]) == 0:
            # do nothing if no files were selected.
            return

        outfile = newpath[0]
        self.save_directory = os.path.dirname(outfile)
        self.interface.save_input_manifest(outfile)

    def saveOutputManifest(self):
        """
        Save an output manifest to disk.

        Called by a File menu option.  Calls the
        `Interface.save_output_manifest` method.
        """
        if self.interface.reduction is None:
            return

        default_name = self.interface.configuration.output_manifest
        if default_name is None:
            dirname = self.save_directory
        else:
            if not os.path.isabs(default_name):
                dirname = os.path.join(self.save_directory, default_name)
            else:
                dirname = default_name
        newpath = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Select output file path for output manifest",
            directory=dirname, filter="Text files (*.txt);;All files (*)")
        if len(newpath[0]) == 0:
            # do nothing if no files were selected.
            return

        outfile = newpath[0]
        self.save_directory = os.path.dirname(outfile)
        self.interface.save_output_manifest(outfile)

    def saveConfiguration(self):
        """
        Save configuration settings to disk.

        Called by a Settings menu option.  Calls the
        `Interface.save_configuration` method.
        """
        if self.initial_startup:
            # clear startup image
            self.resetPipeline()

        default_name = self.interface.configuration.config_file_name
        if default_name is None:
            dirname = self.save_directory
        else:
            if not os.path.isabs(default_name):
                dirname = os.path.join(self.save_directory, default_name)
            else:
                dirname = default_name
        newpath = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Select output file path for configuration",
            directory=dirname,
            filter="Config files (*.cfg);;All files (*)")
        if len(newpath[0]) == 0:
            # do nothing if no files were selected.
            return

        outfile = newpath[0]
        self.save_directory = os.path.dirname(outfile)
        self.interface.save_configuration(outfile)

    def saveParameters(self):
        """
        Save reduction parameters to disk.

        Called by a Parameters menu option.  Calls the
        `Interface.save_parameters` method.
        """

        if self.interface.reduction is None:
            return

        default_name = self.interface.configuration.parameter_file
        if default_name is None:
            dirname = self.save_directory
        else:
            if not os.path.isabs(default_name):
                dirname = os.path.join(self.save_directory, default_name)
            else:
                dirname = default_name
        newpath = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Select output file path for parameters",
            directory=dirname,
            filter="Config files (*.cfg);;All files (*)")
        if len(newpath[0]) == 0:
            # do nothing if no files were selected.
            return

        outfile = newpath[0]
        self.save_directory = os.path.dirname(outfile)
        self.interface.save_parameters(outfile)

        self.setStatus("Current parameters saved to {}.".format(outfile))

    def setOutputDirectory(self):
        """
        Set an output directory for the reduction.

        Called by a File menu option.  Calls the
        `Interface.set_output_directory` method.
        """
        if self.interface.reduction is None:
            return

        newpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="Select Directory",
            directory=self.save_directory)
        if len(newpath) == 0:
            return
        else:
            # set new output directory for reduction and log
            self.interface.set_output_directory(newpath)
            self.interface.set_log_file()
            self.save_directory = newpath
            self.setStatus("Set output directory to {}".format(newpath))

    def toggleDisplay(self):
        """Toggle viewers."""
        # set the option in the configuration
        if self.actionUpdateDisplays.isChecked():
            self.interface.configuration.update_display = True
        else:
            self.interface.configuration.update_display = False

        # if there is a current reduction, register its viewers
        parent = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.interface.register_viewers(parent)

        # create or close a Data View tab
        self.resetView()
        if self.interface.has_embedded_viewers():
            self.dataTabWidget.insertTab(0, parent, 'Data View')
            self.dataTabWidget.setCurrentIndex(0)

        self.updateConfigView()

    def toggleDisplayIntermediate(self):
        """Toggle intermediate display setting."""
        # set the option in the configuration
        if self.actionDisplayIntermediate.isChecked():
            self.interface.configuration.display_intermediate = True
        else:
            self.interface.configuration.display_intermediate = False
        self.updateConfigView()
