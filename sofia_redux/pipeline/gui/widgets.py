# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Widgets and other useful classes for support of the Redux GUI."""

from contextlib import contextmanager
from copy import deepcopy
import html
import logging
import os
import sys
import traceback

from astropy import log

from sofia_redux.pipeline.parameters import Parameters, FALSY
from sofia_redux.pipeline.gui import textview

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from sofia_redux.pipeline.gui.ui import ui_pipe_step
    from sofia_redux.pipeline.gui.ui import ui_edit_param
    from sofia_redux.pipeline.gui.ui import ui_remove_files
    from sofia_redux.pipeline.gui.ui import ui_progress
except ImportError:
    HAS_PYQT5 = False
    QtGui = None

    # duck type parents to allow class definition
    class QtWidgets:
        class QWidget:
            pass

        class QDialog:
            pass

    class QtCore:
        class QAbstractTableModel:
            pass

        class Qt:
            class DisplayRole:
                pass

        class QObject:
            pass

        class QRunnable:
            pass

        @staticmethod
        def pyqtSignal(x):
            pass

        @staticmethod
        @contextmanager
        def pyqtSlot():
            pass

    class ui_pipe_step:
        class Ui_Form:
            pass

    class ui_progress:
        class Ui_ProgressFrame:
            pass

    class ui_edit_param:
        class Ui_Dialog:
            pass

    class ui_remove_files:
        class Ui_Dialog:
            pass

else:
    HAS_PYQT5 = True


class PipeStep(QtWidgets.QWidget, ui_pipe_step.Ui_Form):
    """
    Reduction step widget.

    Includes a Run and Edit button for the step.
    """
    def __init__(self, parent=None, stepname=None, index=0):
        """
        Build the widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        stepname : str, optional
            Reduction step name to display.
        index : int, optional
            Reduction step index.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        super().__init__(parent)

        self.setupUi(self)

        if stepname is not None:
            self.pipeStepLabel.setText(stepname)
        self.index = index
        self.indexLabel.setText(" {}. ".format(index + 1))

    def connectRun(self, slot):
        """
        Connect Run button to callback.

        Parameters
        ----------
        slot : function
            The `onRun` function from the main application.  Must accept
            the step index as an argument.
        """
        self.runButton.clicked.connect(lambda: slot(self.index))

    def connectEdit(self, slot):
        """
        Connect Edit button to callback.

        Parameters
        ----------
        slot : function
            The `onEdit` function from the main application.  Must accept
            the step index as an argument.
        """
        self.editButton.clicked.connect(lambda: slot(self.index))

    def enableRun(self, enable=True):
        """
        Enable the Run button.

        Parameters
        ----------
        enable : bool
            If True, the button is enabled.  If False, it is disabled.
        """
        self.runButton.setEnabled(enable)

    def enableEdit(self, enable=True):
        """
        Enable the Edit button.

        Parameters
        ----------
        enable : bool, optional
            If True, the button is enabled.  If False, it is disabled.
        """
        self.editButton.setEnabled(enable)


class ProgressFrame(QtWidgets.QWidget, ui_progress.Ui_ProgressFrame):
    """Progress bar widget for reduction steps."""
    def __init__(self, parent=None):
        """
        Build the widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        super().__init__(parent)
        self.setupUi(self)

    def setNStep(self, nsteps):
        """
        Set the number of steps expected to run.

        Parameters
        ----------
        nsteps : int
            The total number of steps.
        """
        self.progressBar.setMaximum(nsteps)

    def setProgress(self, value):
        """
        Set the progress value.

        Parameters
        ----------
        value : int
            Steps completed.
        """
        self.progressBar.setValue(value)

    def resetProgress(self):
        """Reset progress bar."""
        self.progressBar.setValue(0)
        self.stopButton.setEnabled(True)


class EditParam(QtWidgets.QDialog, ui_edit_param.Ui_Dialog):
    """Edit parameters for a reduction step."""
    def __init__(self, parent=None, name=None,
                 current=None, default=None, directory=None):
        """
        Build the dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        name : str, optional
            Reduction step display name.
        current : `redux.parameters.ParameterSet`, optional
            Current parameter definitions.
        default : `redux.parameters.ParameterSet`, optional
            Default parameter definitions.
        directory : str
            Base directory for pick_file and pick_directory
            widgets.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        super().__init__(parent)

        self.setupUi(self)

        # hook up the reset and restore buttons
        reset = self.buttonBox.button(
            QtWidgets.QDialogButtonBox.Reset)
        reset.clicked.connect(self.reset)
        restore = self.buttonBox.button(
            QtWidgets.QDialogButtonBox.RestoreDefaults)
        restore.clicked.connect(self.restore)

        # store the initial parameters and base directory
        self.parameters = current
        self.default = default
        self.directory = directory

        # set the title
        if name is not None:
            self.setWindowTitle("Edit Parameters: {}".format(name))

        # set the layout and load the widgets corresponding to
        # the parameter definition
        self.groupBox = self.container
        self.setFormLayout()
        self.setWidgets()

    def addComboBox(self, key, param):
        """
        Add a combo box widget to the form.

        Parameters
        ----------
        key : str
            Object name for the widget.
        param : dict
            Parameter dictionary.  Must have 'name', 'options',
            and 'option_index' defined.  If 'description' is present,
            it will be used as a tooltip.
        """
        # expanding box
        comboBox = QtWidgets.QComboBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed)
        comboBox.setSizePolicy(sizePolicy)
        comboBox.setObjectName(key)
        if param["description"] is not None:
            comboBox.setToolTip(param["description"])

        # set options in box
        if param['options'] is not None:
            opt = [str(val) for val in param['options']]
            comboBox.addItems(opt)
        comboBox.setCurrentIndex(param['option_index'])

        # add to layout with label
        self.groupBox.layout().addRow(param['name'], comboBox)

    def addTextBox(self, key, param):
        """
        Add a text box widget to the form.

        Parameters
        ----------
        key : str
            Object name for the widget.
        param : dict
            Parameter dictionary.  Must have 'name' and
            'value' defined.  If 'description' is present,
            it will be used as a tooltip.
        """
        # new line edit
        textBox = QtWidgets.QLineEdit(self.groupBox)
        textBox.setObjectName(key)

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Fixed)
        textBox.setSizePolicy(sizePolicy)
        textBox.setMinimumWidth(200)

        # set default value
        textBox.setText(str(param['value']))
        if param["description"] is not None:
            textBox.setToolTip(param["description"])

        # add to layout with label
        self.groupBox.layout().addRow(param['name'], textBox)

    def addRadioButtons(self, key, param):
        """
        Add a set of radio buttons to the form.

        Parameters
        ----------
        key : str
            Object name for the widget.
        param : dict
            Parameter dictionary.  Must have 'name', 'options', and
            'option_index' defined.  If 'description' is present,
            it will be used as a tooltip.
        """
        groupBox = QtWidgets.QGroupBox(self.groupBox)
        groupBox.setObjectName(key)
        if param["description"] is not None:
            groupBox.setToolTip(param["description"])

        vbox = QtWidgets.QVBoxLayout(groupBox)
        buttonGroup = QtWidgets.QButtonGroup(groupBox)
        buttonGroup.setObjectName('radio')

        if param['options'] is not None:
            for idx, option in enumerate(param['options']):
                radio = QtWidgets.QRadioButton(groupBox)
                radio.setText(str(option))
                if idx == param['option_index']:
                    radio.setChecked(True)
                vbox.addWidget(radio)
                buttonGroup.addButton(radio)
                buttonGroup.setId(radio, idx)

        self.groupBox.layout().addRow(param['name'], groupBox)

    def addCheckBox(self, key, param):
        """
        Add a check box widget.

        Parameters
        ----------
        key : str
            Object name for the widget.
        param : dict
            Parameter dictionary.  Must have 'name' and 'value'
            defined.  If 'description' is present,
            it will be used as a tooltip.
        """
        # new check box
        checkBox = QtWidgets.QCheckBox(self.groupBox)
        checkBox.setObjectName(key)
        if str(param['value']).lower().strip() in FALSY:
            val = False
        else:
            val = True
        checkBox.setChecked(val)
        if param["description"] is not None:
            checkBox.setToolTip(param["description"])

        # center the check box vertically
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(checkBox)
        vbox.setAlignment(QtCore.Qt.AlignHCenter)

        self.groupBox.layout().addRow(param['name'], vbox)

    def addPick(self, key, param, pick_type='file'):
        """
        Add a pick file/directory button and text widget.

        Parameters
        ----------
        key : str
            Object name for the widget.
        param : dict
            Parameter dictionary.  Must have 'name' and 'value'
            defined.  If 'description' is present,
            it will be used as a tooltip.
        pick_type : {'directory', 'file'}
            Type of file item to select.
        """
        pickButton = QtWidgets.QPushButton(self.groupBox)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Tango/16x16/document-open.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        pickButton.setIcon(icon)
        pickButton.setText(param['name'])

        # start a file dialog if clicked
        if pick_type == 'directory':
            pickButton.clicked.connect(lambda: self.pickDirectory(key))
        else:
            pickButton.clicked.connect(lambda: self.pickFile(key))

        pickTextBox = QtWidgets.QLineEdit(self.groupBox)
        pickTextBox.setObjectName(key)
        pickTextBox.setText(str(param['value']))
        if param["description"] is not None:
            pickTextBox.setToolTip(param["description"])

        self.groupBox.layout().addRow(pickButton, pickTextBox)

    def getValue(self):
        """Get new parameter values from all the widgets."""
        for param_key, param in self.parameters.items():
            widget = self.container.findChild(QtWidgets.QWidget, param_key)
            if widget is None or param['wtype'] == 'group':
                # for hidden parameters, no widget is made
                # and no modification is needed
                continue
            elif param['wtype'] == 'combo_box':
                param['option_index'] = widget.currentIndex()
                value = widget.currentText()
            elif param['wtype'] == 'radio_button':
                buttonGroup = widget.findChild(
                    QtWidgets.QButtonGroup, 'radio')
                param['option_index'] = buttonGroup.checkedId()
                value = buttonGroup.checkedButton().text()
            elif param['wtype'] == 'check_box':
                value = widget.isChecked()
            else:
                value = widget.text()
            param['value'] = Parameters.fix_param_type(value, param['dtype'])

        return self.parameters

    def pickFile(self, key):
        """
        Use a File Dialog to pick a set of files.

        Parameters
        ----------
        key : str
            Object name for the widget.
        """
        newpath = QtWidgets.QFileDialog.getOpenFileNames(
            self, caption="Select File(s)",
            directory=self.directory)
        if len(newpath[0]) == 0:
            # do nothing if no files were selected.
            return
        else:
            widget = self.container.findChild(QtWidgets.QWidget, key)
            widget.setText(', '.join(newpath[0]))
            widget.repaint()

    def pickDirectory(self, key):
        """
        Use a File Dialog to pick a directory.

        Parameters
        ----------
        key : str
            Object name for the widget.
        """
        newpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="Select Directory",
            directory=self.directory)
        if len(newpath) == 0:
            return
        else:
            widget = self.container.findChild(QtWidgets.QWidget, key)
            widget.setText(newpath)
            widget.repaint()

    def reset(self):
        """Reset all parameter values to an initial set."""
        for param_key, param in self.parameters.items():
            widget = self.container.findChild(QtWidgets.QWidget, param_key)
            try:
                if param['wtype'] == 'combo_box':
                    widget.setCurrentIndex(param['option_index'])
                elif param['wtype'] == 'radio_button':
                    button = widget.layout().itemAt(param['option_index'])
                    button.widget().setChecked(True)
                elif param['wtype'] == 'check_box':
                    widget.setChecked(param['value'])
                else:
                    widget.setText(str(param['value']))
            except AttributeError:
                # if the parameter set has changed for some reason,
                # the widget may be None.  Pass this condition
                # quietly.
                continue
            widget.repaint()

    def restore(self):
        """Restore all parameter values to a default set."""
        self.parameters = deepcopy(self.default)
        self.reset()

    def setFormLayout(self):
        """Assign a form layout to the current groupBox."""
        layout = QtWidgets.QFormLayout()
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.groupBox.setLayout(layout)

    def setWidgets(self):
        """Add parameter widgets according to their definitions."""
        for param_key, param in self.parameters.items():
            # special widget type for grouping parameters together
            if param['wtype'] == 'group':
                self.groupBox = QtWidgets.QGroupBox(self.container)
                self.container.layout().addWidget(self.groupBox)
                self.setFormLayout()
                self.groupBox.setObjectName(param_key)
                if param["description"] is not None:
                    self.groupBox.setToolTip(param['description'])
                self.groupBox.setTitle(param['name'])
                continue

            # don't make a widget if it's hidden from user
            if param['hidden']:
                continue

            if param['wtype'] == 'combo_box':
                self.addComboBox(param_key, param)
            elif param['wtype'] == 'text_box':
                self.addTextBox(param_key, param)
            elif param['wtype'] == 'radio_button':
                self.addRadioButtons(param_key, param)
            elif param['wtype'] == 'check_box':
                self.addCheckBox(param_key, param)
            elif param['wtype'] == 'pick_file':
                self.addPick(param_key, param, pick_type='file')
            elif param['wtype'] == 'pick_directory':
                self.addPick(param_key, param, pick_type='directory')
            else:
                log.warning('Unknown widget type: {}'.format(param['wtype']))
                continue

        # adjust size to contents, then set minimum height and width
        self.adjustSize()
        self.setMinimumHeight(self.height())
        self.setMinimumWidth(self.width())


class DataTableModel(QtCore.QAbstractTableModel):
    """Model for the File Information table view."""
    def __init__(self, data, parent=None):
        """
        Initialize the data model.

        Parameters
        ----------
        data : `OrderedDict`
            Keys are the column headers; values are the row items.
            Format is that of the `redux.reduction.Reduction` `data_id`
            attribute.
        parent : QWidget, optional
            Parent widget.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__(parent)
        self._data = data
        self._keys = list(data.keys())

    def rowCount(self, parent=None):
        """
        Return the total number of rows of data.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.

        Returns
        -------
        int
            The number of rows in the table.
        """
        return len(self._data[self._keys[0]])

    def columnCount(self, parent=None):
        """
        Return the total number of columns of data.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.

        Returns
        -------
        int
            The number of columns in the table.
        """
        return len(self._keys)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """
        Retrieve table data by index.

        Parameters
        ----------
        index : QModelIndex
            Data index to retrieve.
        role : int, optional
            Qt display role.

        Returns
        -------
        str or None
            If index is valid, a string representation of the data at the
            index value is returned.  If not, None is returned.
        """
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data[self._keys[index.column()]][index.row()])
        return None

    def headerData(self, col, orientation, role=QtCore.Qt.DisplayRole):
        """
        Retrieve column data.

        Parameters
        ----------
        col : int
            Column name to retrieve.
        orientation : int
            Qt orientation.
        role : int, optional
            Qt display role.

        Returns
        -------
        str or None
            Column name if orientation and role are valid; otherwise None.
        """
        if orientation == QtCore.Qt.Horizontal and \
                role == QtCore.Qt.DisplayRole:
            return self._keys[col]
        elif orientation == QtCore.Qt.Vertical and \
                role == QtCore.Qt.DisplayRole:
            return col + 1
        return None


class CustomSignals(QtCore.QObject):
    """Custom signals for widgets to emit."""

    finished = QtCore.pyqtSignal(object)
    """Signal to emit when finished."""

    progress = QtCore.pyqtSignal(object)
    """Signal to emit for intermediate progress."""


class TextEditLogger(logging.Handler):
    """
    Log handler for a TextEdit-based logging window.

    Uses `CustomSignals` to emit a 'finished' signal to
    a display slot.
    """
    def __init__(self):
        """Initialize the logger."""
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__()
        self.signals = CustomSignals()

    def emit(self, record):
        """
        Emit a log message.

        Messages are formatted to HTML with font colors for
        the level string (green for DEBUG, orange for WARNING,
        red for ERROR).  A finished signal (from `CustomSignals`)
        is emitted, containing the formatted message.

        Parameters
        ----------
        record : `logging.LogRecord`
           The log record, with an additional 'origin' attribute
           attached by `astropy.log`.
        """
        if not hasattr(record, 'origin'):
            record.origin = ''
        if record.levelno == logging.INFO:
            msg = html.escape(str(record.msg))
        else:
            if record.levelno < logging.INFO:
                color = 'limegreen'
            elif record.levelno < logging.ERROR:
                color = 'orange'
            else:
                color = 'red'
            msg = "<font color={}>{}</font>: {} [{}]".format(
                color, record.levelname,
                html.escape(str(record.msg)), record.origin)

        self.signals.finished.emit(msg)


class RemoveFilesDialog(QtWidgets.QDialog,
                        ui_remove_files.Ui_Dialog):
    """Dialog to select files for removal."""
    def __init__(self, parent=None, loaded_files=None):
        """
        Build the dialog.

        Parameters
        ----------
        parent : QWidget
            Parent widget.
        loaded_files : `list` of str, optional
            List of file paths that are currently loaded.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        # parent initialization
        QtWidgets.QDialog.__init__(self, parent)

        # set up UI from Designer generated file
        self.setupUi(self)

        if loaded_files is not None:
            if len(loaded_files) == 1:
                common_path = os.path.dirname(loaded_files[0]) + os.sep
            else:
                common_path = os.path.commonpath(loaded_files) + os.sep
            self.commonPath.setText(common_path + ' :')
            for item in loaded_files:
                basename = '  ' + item.split(common_path)[-1]
                self.listWidget.addItem(basename)

    def getValue(self):
        """
        Get the selected files from the widget.

        Returns
        -------
        list of str
            File paths to remove from the reduction.
        """
        common_path = self.commonPath.text().rstrip(' :')
        items = self.listWidget.selectedItems()
        remove_files = [common_path + str(i.text()).strip() for i in items]
        return remove_files


class ConfigView(textview.TextView):
    """View and edit current configuration values."""
    def __init__(self, parent=None):
        """Build the widget."""
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__(parent=parent)

        # hide table and filter buttons
        self.tableButton.setVisible(False)
        self.filterButton.setVisible(False)

        # show save button
        self.saveButton.setVisible(True)
        self.saveButton.setFocus()

        # make editable
        self.textEdit.setReadOnly(False)
        self.textEdit.textChanged.connect(self.update)

    def update(self):
        self.text = self.textEdit.toPlainText().split('\n')
        self.html = self.format()


class ParamView(textview.TextView):
    """View and filter current parameter values."""
    def __init__(self, parent=None):
        """Build the widget."""
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__(parent=parent)

        self.tableButton.setEnabled(True)
        self.tableButton.clicked.connect(self.table)

    def format(self):
        """
        Format the parameter text.

        Returns
        -------
        str
            HTML-formatted text.
        """
        # some useful strings
        anchor = '<a name="anchor"></a>'
        br = '<br>'

        # add anchors to pipeline step sections
        text_strs = []
        for line in self.text:
            if line.startswith('['):
                line = br + anchor + line
            text_strs.append(line)

        text_str = br.join(text_strs)
        html = '<pre>' + anchor + text_str + br + '</pre>' + anchor

        return html

    def table(self):
        """
        Format selected parameters into a table.

        Uses comma-separated filter values as keys to display
        from each parameter section.

        Requires `pandas` to display.
        """
        # read text to filter
        # may be comma-separated keys (no substrings)
        find_text = self.findText.text().strip()

        if find_text == '':
            # clear previous filter / table
            self.textEdit.setHtml(self.html)
        else:
            # check for pandas
            try:
                import pandas as pd
            except ImportError:
                msg = '(install pandas for table display)'
                self.textEdit.setPlainText(msg)
                return

            # split field on commas for multiple keys
            sep = find_text.upper().split(',')
            sep = [s.strip() for s in sep]

            # find keys in pipeline parameters
            data = {'Pipe step': []}
            for key in sep:
                data[key] = []
            seen = {}
            for line in self.text:
                if line.startswith('#'):
                    continue
                elif line.startswith('['):
                    for key in sep:
                        if key in seen and not seen[key]:
                            data[key].append(None)
                        seen[key] = False
                    data['Pipe step'].append(line.strip('[]'))
                else:
                    try:
                        testkey, val = line.split('=', 1)
                    except ValueError:
                        continue
                    for key in sep:
                        if not seen[key] and key == testkey.strip().upper():
                            data[key].append(val)
                            seen[key] = True
            for key in sep:
                if key in seen and not seen[key]:
                    data[key].append(None)

            # pandas dataframe for table display
            df = pd.DataFrame(data, columns=['Pipe step'] + sep)
            htmltable = df.to_html(max_rows=None, max_cols=None, border=1)
            htmltable = htmltable.replace('<table',
                                          '<table cellpadding="10"', 1)
            self.textEdit.setHtml(htmltable)

        # repaint required for some versions of Qt/OS
        self.textEdit.repaint()


class StepRunnable(QtCore.QRunnable):
    """
    Worker class to run a reduction step.

    This class is intended to be the worker for running
    reductions in a thread separate from the main GUI thread.
    It uses `CustomSignals` to emit a 'progress' signal after
    each step and a 'finished' signal when all steps are
    complete.

    Attributes
    ----------
    step : function
        Reduction step function to run (`redux.Interface.step`).
    nsteps : int
        Number of steps to run before returning.
    signals : `CustomSignals`
        Signals to emit.
    stop : bool
        Flag to stop before processing the next step.  May be
        set by the main GUI to interrupt reduction gracefully.
    """
    def __init__(self, step_function, nsteps):
        """
        Initialize the runner.

        Parameters
        ----------
        step_function : function
            Reduction step function to run (`redux.Interface.step`).
        nsteps : int
            Number of steps to run.
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__()
        self.step = step_function
        self.nsteps = nsteps
        self.signals = CustomSignals()
        self.stop = False

    @QtCore.pyqtSlot()
    def run(self):
        """Run the reduction steps."""
        result = ''
        for step_num in range(self.nsteps):
            if self.stop:
                break
            try:
                result = self.step()
            except Exception:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self.signals.finished.emit((exctype, value,
                                            traceback.format_exc()))
                return
            self.signals.progress.emit(step_num + 1)
        self.signals.finished.emit(result)


class LoadRunnable(QtCore.QRunnable):
    """
    Worker class to load reduction data.

    This class is intended to be the worker for loading
    reduction data in a thread separate from the main GUI thread.
    It uses `CustomSignals` to emit a 'finished' signal when the
    data loading is complete.

    Attributes
    ----------
    load : function
        Reduction load function to run (`redux.Interface.start`).
    data : list of str
        Input data file names.
    param : `redux.Parameters`
        Parameters to save and pass to the new reduction object.
    dirname : str
        Output directory to save and pass to the new reduction
        object.
    """
    def __init__(self, load_function, data, param, dirname):
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__()
        self.load = load_function
        self.data = data
        self.param = param
        self.dirname = dirname
        self.signals = CustomSignals()

    @QtCore.pyqtSlot()
    def run(self):
        """Run the load function."""
        try:
            self.load(self.data)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.finished.emit((exctype, value,
                                        traceback.format_exc()))
            return

        self.signals.finished.emit((self.param, self.dirname))


class GeneralRunnable(QtCore.QRunnable):
    """
    Worker class to run a general function.

    This class is intended to be the worker for running any
    function that does not require specific input or output
    in a separate thread.  It uses `CustomSignals` to emit a
    'finished' signal when the function is complete.

    Attributes
    ----------
    run_function : function
        Function to run.
    """
    def __init__(self, run_function, *args, **kwargs):
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__()
        self.run_function = run_function
        self.signals = CustomSignals()
        self.args = args
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):
        """Run the function."""
        try:
            self.run_function(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.finished.emit((exctype, value,
                                        traceback.format_exc()))
        else:
            self.signals.finished.emit(None)
