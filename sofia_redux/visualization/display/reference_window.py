
import os
from typing import Any

from sofia_redux.visualization import log
from sofia_redux.visualization.display.text_view import TextView

try:
    from PyQt5 import QtWidgets, QtGui, QtCore
    from sofia_redux.visualization.display.ui import reference_data as rd
except ImportError:  # pragma: no cover
    HAS_PYQT5 = False
    QtWidgets, QtGui, QtCore = None, None, None

    class QtWidgets:
        class QDialog:
            pass

    class rd:
        class Ui_Dialog:
            pass
else:
    HAS_PYQT5 = True


__all__ = ['ReferenceWindow']


class ReferenceWindow(QtWidgets.QDialog, rd.Ui_Dialog):
    """
    GUI framework for displaying the spectral lines

    It pops up a `Reference Data` window from the dropdown option of
    `Analysis`, where a user can interact directly. It includes following
    buttons:
    -- Load List - allows to load in reference data files, only one at a time.
    However more files can be loaded in by clicking on this button again.
    -- Clear Lists - it removes all the loaded files as well lines and
    labels from plots.

    Each loaded file can be opened by double clicking on a selection.

    Two check boxes include:
    - unload labels -- This removes the labels from the plot
    - unload lines -- This removed labels and lines from the plot

    Parameters
    ---------
    connections: reference_window.Reference_window
        Method to establish connection with the buttons and checkboxes
    signals: sofia_redux.visualization.signals.Signals
        Custom signals recognized by the Eye interface, used
        to trigger callbacks from user events.
    ref_models: reference_model.ReferenceData
        The line list data
    visibility: Dict
         keys are `ref_line` and `ref_label`
    textview: sofia_redux.visualization.display.text_view.TextView()
        Text viewer widget
    """

    def __init__(self, parent: Any) -> None:
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)
        self.setModal(0)
        self.connections()

        self.signals = parent.signals
        self.ref_models = parent.reference_models
        self.visibility = {'ref_line': False,
                           'ref_label': False}
        self.textview = None

    def connections(self):
        """
        Establish connection with the buttons: load_lines and clear_lists
        and
        checkboxes: `Show line` and `Show label`.
        """
        self.load_file_button.clicked.connect(self.load_lines)
        self.show_lines_box.toggled.connect(
            lambda: self.toggle_visibility('ref_line'))
        self.show_labels_box.toggled.connect(
            lambda: self.toggle_visibility('ref_label'))
        self.clear_lists_button.clicked.connect(self.clear_lists)
        self.loaded_files_list.itemDoubleClicked.connect(self.show_text)

    def load_lines(self) -> bool:
        try:
            filename = QtWidgets.QFileDialog.getOpenFileName(
                self, caption="Select Line List")[0]
        except IndexError:
            return False
        if not filename:
            return False
        try:
            result = self.ref_models.add_line_list(filename)
        except IOError:
            result = False
        if result:
            self.visibility['ref_line'] = True
            self.visibility['ref_label'] = True
            buttons = [self.show_lines_box,
                       self.show_labels_box]
            for button in buttons:
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)

            self.set_status(f'Loaded {os.path.basename(filename)}')

            # add file name to list widget
            item = QtWidgets.QListWidgetItem(os.path.basename(filename))
            item.setData(QtCore.Qt.UserRole, filename)
            item.setToolTip(filename)
            self.loaded_files_list.addItem(item)

            self.signals.update_reference_lines.emit()
        else:
            self.set_status(f'Unable to parse '
                            f'{os.path.basename(filename)}')
        return result

    def toggle_visibility(self, target):
        if target == 'ref_line':
            state = self.show_lines_box.checkState()
            self.ref_models.set_visibility(target, state)
            self.signals.update_reference_lines.emit()

        elif target == 'ref_label':
            state = self.show_labels_box.checkState()
            self.ref_models.set_visibility(target, state)
            self.signals.update_reference_lines.emit()

        else:
            state = None
            log.debug(f'Invalid visibility target {target}')

        if state is not None:
            log.debug(f'Updated visibility of {target} to '
                      f'{self.visibility[target]}')

    def show_text(self, item):
        if self.textview is None or not self.textview.isVisible():
            self.textview = TextView(self)
            self.textview.tableButton.hide()
        filename = item.data(QtCore.Qt.UserRole)
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [ln.strip() for ln in lines]
        self.textview.load(lines)
        self.textview.show()
        self.textview.raise_()
        self.textview.setTitle(os.path.basename(filename))

    def set_status(self, message):
        self.status.setText(message)

    def clear_status(self):
        self.status.setText('')

    def clear_lists(self):
        """Remove all lines and labels from display."""
        # unload data
        self.signals.unload_reference_model.emit()

        # reset visibility states
        self.visibility['ref_line'] = False
        self.visibility['ref_label'] = False
        buttons = [self.show_lines_box, self.show_labels_box]
        for button in buttons:
            button.blockSignals(True)
            button.setChecked(False)

        # clear file list table
        self.loaded_files_list.clear()

        self.set_status('Cleared line lists')
        self.signals.update_reference_lines.emit()
