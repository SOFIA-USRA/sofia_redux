# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from contextlib import contextmanager
import os
import uuid
from typing import (Dict, List, Optional, Sequence,
                    Union, Tuple, TypeVar)

from more_itertools import unique_everseen, consecutive_groups
import numpy as np
import matplotlib.backend_bases as mbb

from sofia_redux.visualization import log
from sofia_redux.visualization import signals as vs
from sofia_redux.visualization.display import (figure, pane, fitting_results,
                                               reference_window)
from sofia_redux.visualization.models import high_model, reference_model
from sofia_redux.visualization.utils.eye_error import EyeError

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QTreeWidgetItem
    from sofia_redux.visualization.display.ui import simple_spec_viewer as ssv
    from sofia_redux.visualization.display import cursor_location as cl
except ImportError:
    HAS_PYQT5 = False
    QtGui, cl, QTreeWidgetItem = None, None, None
    Event = TypeVar('Event', mbb.MouseEvent, mbb.LocationEvent)

    # duck type parents to allow class definition
    class QtWidgets:

        class QMainWindow:
            pass

        class QWidget:
            pass

    class ssv:
        class Ui_MainWindow:
            pass

    class QtCore:

        @staticmethod
        @contextmanager
        def pyqtSlot():
            pass
else:
    HAS_PYQT5 = True
    Event = TypeVar('Event', mbb.MouseEvent, mbb.LocationEvent,
                    QtCore.QEvent)

__all__ = ['View']

MT = TypeVar('MT', bound=high_model.HighModel)
Num = TypeVar('Num', int, float)
PaneID = TypeVar('PaneID', str, List[int], None, List[pane.Pane])
PT = TypeVar('PT', bound=pane.Pane)
IDT = TypeVar('IDT', str, uuid.UUID)


class View(QtWidgets.QMainWindow, ssv.Ui_MainWindow):
    """
    Interactively display data.

    The View class holds data models and displays and handles
    user interaction events.

    Parameters
    ----------
    signals : sofia_redux.visualization.signals.Signals
        Custom signals recognized by the Eye interface, used
        to trigger callbacks from user events.

    Attributes
    ----------
    stale : bool
        Flag to indicate that view is stale and should be refreshed.
    stale_background_full : bool
        Flag to indicate that plot backgrounds and axis limits are stale
        and should be refreshed.
    stale_background_partial : bool
        Flag to indicate that plot backgrounds are stale and
        should be refreshed, but axis scales remain the same.
    figure : sofia_redux.visualization.display.figure.Figure
        The main plot display associated with the View.
    signals : sofia_redux.visualization.signals.Signals
        Custom signals recognized by the Eye interface, used
        to trigger callbacks from user events.
    cid : dict
        Matplotlib event CIDs.
    model_collection : list of HighModel
        Loaded data models, available for display.
    timer : QtCore.QTimer
        Refresh loop timer.
    fit_results : fitting_results.FittingResults
        Dialog window for table display of fit parameters.
    cursor_location_window : cursor_location.CursorLocation
        Dialog window for table display of fit parameters.
    """

    def __init__(self, signals: vs.Signals) -> None:
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')

        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.stale = False
        self.stale_background_full = False
        self.stale_background_partial = False
        self.figure = figure.Figure(self.figure_widget, signals)
        self.signals = signals
        self.cid = dict()
        self.model_collection = dict()
        self.reference_models = reference_model.ReferenceData()
        self.reference_window = None
        self.timer = None
        self.fit_results = None
        self.cursor_location_window = None
        self._cursor_popout = False

    def keyPressEvent(self, event: Event) -> None:
        """
        Handle keyboard shortcuts.

        Parameters
        ----------
        event : QEvent
            Keypress event.
        """
        if type(event) == QtGui.QKeyEvent:
            try:
                name = QtGui.QKeySequence(
                    event.modifiers() | event.key()).toString().encode('utf-8')
                name = name.decode('utf-8')
            except UnicodeEncodeError:  # pragma: no cover
                name = 'UNKNOWN'
            log.debug(f'Key pushed in view: {name}')
            if event.key() == QtCore.Qt.Key_F:
                # F fits a gaussian to a selected region
                fit_mode = self._parse_fit_mode()
                self.start_selection(mode=fit_mode)
            elif event.key() == QtCore.Qt.Key_W:
                # W resets any axes range changes
                self.reset_zoom()
            elif event.key() == QtCore.Qt.Key_X:
                # X zooms in on selected x-range
                self.start_selection(mode='x_zoom')
            elif event.key() == QtCore.Qt.Key_Y:
                # Y zooms in on selected y-range
                self.start_selection(mode='y_zoom')
            elif event.key() == QtCore.Qt.Key_Z:
                # Z zooms in on selected box
                self.start_selection(mode='b_zoom')
            elif event.key() == QtCore.Qt.Key_C:
                # C clears zoom/fit status
                self.clear_selection()
                self.clear_fit()
            elif (event.key() == QtCore.Qt.Key_Return
                  or event.key() == QtCore.Qt.Key_Enter):
                # enter in the file table displays selected models
                if self.loaded_files_table.hasFocus():
                    # send display_model signal
                    self.signals.model_selected.emit()
            elif (event.key() == QtCore.Qt.Key_Delete
                  or event.key() == QtCore.Qt.Key_Backspace):
                if self.loaded_files_table.hasFocus():
                    # delete key in the file table removes selected models
                    self.signals.model_removed.emit()
                elif self.figure_widget.hasFocus():
                    # delete key in the pane table or figure removes
                    # selected pane
                    self.remove_pane()
                elif self.filename_table.hasFocus():
                    # delete key in filename table removes the selected
                    # filenames from the selected pane
                    self.remove_file_from_pane()
            elif event.key() == QtCore.Qt.Key_A:
                self.print_current_artists()

    def open_eye(self) -> None:
        """Open the view window and start the refresh timer."""
        if not self.timer:
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh_loop)
            self.timer.start(100)

        if not self.isVisible():
            log.info("The Eye is open.")
            self.show()
        self.raise_()

    def close(self) -> None:
        """Close the view window."""
        self.reset()
        super().close()

    def reset(self) -> None:
        """Remove all loaded models and panes."""
        self.figure.remove_all_panes()
        self.model_collection = dict()

    def refresh_loop(self) -> None:
        """Refresh the view."""
        if not self.timer:
            # catch for unopened eye
            return
        tests = [self.stale, self.stale_background_full,
                 self.stale_background_partial]
        if any(tests):
            self.timer.stop()
            log.debug('View is stale, refreshing all.')
            bg_full = self.stale_background_full
            bg_partial = self.stale_background_partial
            if bg_full or bg_partial:
                log.debug('Background is stale.')

            # reset flags before refreshing, so that delays in
            # refresh don't overwrite later flags
            self.stale = False
            self.stale_background_full = False
            self.stale_background_partial = False

            # Plot
            self.figure.refresh(bg_full=bg_full, bg_partial=bg_partial)

            # Controls
            self.refresh_controls()

            self.timer.start()

    def hold_atrophy(self):
        """Disconnect atrophy signals."""
        self.signals.atrophy.disconnect()
        self.signals.atrophy_bg_full.disconnect()
        self.signals.atrophy_bg_partial.disconnect()

    def release_atrophy(self):
        """Reconnect atrophy signals."""
        self.signals.atrophy.connect(self.atrophy)
        self.signals.atrophy_bg_full.connect(self.atrophy_background_full)
        self.signals.atrophy_bg_partial.connect(
            self.atrophy_background_partial)

    def refresh_controls(self) -> None:
        """Refresh all control widgets."""
        self.refresh_file_table()
        self.update_controls()
        self.populate_order_selectors()
        self.signals.controls_updated.emit()

    ####
    # Signals
    ####
    @QtCore.pyqtSlot()
    def atrophy(self) -> None:
        """Mark the view as stale."""
        log.debug('Received atrophy signal, '
                  'marking figure as stale')
        self.stale = True

    @QtCore.pyqtSlot()
    def atrophy_controls(self) -> None:
        """Mark the control panel as stale."""
        log.debug('Figure has been updated. '
                  'Controls no longer in sync')
        # clear selection: resets zoom states
        self.clear_selection()
        self.update_controls()

    @QtCore.pyqtSlot()
    def atrophy_background_full(self) -> None:
        """Mark the figure background as stale."""
        log.debug('Figure background has been updated.')
        self.stale_background_full = True

    @QtCore.pyqtSlot()
    def atrophy_background_partial(self) -> None:
        """Mark the figure background as partially stale."""
        log.debug('Figure background has been updated without rescaling.')
        self.stale_background_partial = True

    @QtCore.pyqtSlot()
    def current_pane_changed(self) -> None:
        """Set a new pane as the active pane."""
        self.figure.set_pane_highlight_flag(True)
        self.clear_selection()
        self.update_controls()
        self.populate_order_selectors(True)
        self.signals.atrophy.emit()

    @QtCore.pyqtSlot()
    def axis_limits_changed(self) -> None:
        """Change axis limits."""
        try:
            limits = self._pull_limits_from_gui()
        except ValueError:
            log.debug('Illegal limits entered')
        else:
            targets = self.selected_target_axis()
            self.figure.change_axis_limits(limits, target=targets)
        self.signals.atrophy_bg_partial.emit()

    def _pull_limits_from_gui(self) -> Dict[str, List[float]]:
        """
        Retrieve limit selection from GUI controls.

        Used if the user changes values for the limits on
        the GUI and the plots need to be updated to
        reflect the new values.

        Returns
        -------
        limits : dict
            Keys are 'x', 'y'; values are [low, high] limits
            for the axis.
        """
        xlim = [float(self.x_limit_min.text()),
                float(self.x_limit_max.text())]
        ylim = [float(self.y_limit_min.text()),
                float(self.y_limit_max.text())]
        limits = {'x': xlim}

        target = self.selected_target_axis()
        if target['axis'] in ['alt', 'all']:
            limits['y_alt'] = ylim
        if target['axis'] in ['primary', 'all']:
            limits['y'] = ylim
        return limits

    @QtCore.pyqtSlot()
    def toggle_overplot(self):
        """Enable or disable overplots for panes."""
        state = self.enable_overplot_checkbox.checkState()
        targets = self.selected_target_axis()
        self.figure.set_overplot_state(state, target=targets)
        self.signals.atrophy_bg_full.emit()

    @QtCore.pyqtSlot()
    def axis_scale_changed(self) -> None:
        """Change axis scales."""
        scales = self._pull_scale_from_gui()
        targets = self.selected_target_axis()
        self.figure.set_scales(scales, target=targets)
        self.signals.atrophy_bg_partial.emit()

    def _pull_scale_from_gui(self) -> Dict[str, str]:
        """
        Retrieve scale selection from GUI controls.

        Used if the user changes scale for an axis on
        the GUI and the plots need to be updated to
        reflect the new setting.

        Returns
        -------
        scale : dict
            Keys are 'x', 'y'; values are [linear, log]
            for the axis.
        """
        if self.x_scale_linear_button.isChecked():
            x = 'linear'
        else:
            x = 'log'
        if self.y_scale_linear_button.isChecked():
            y = 'linear'
        else:
            y = 'log'
        scale = {'x': x, 'y': y}
        return scale

    @QtCore.pyqtSlot()
    def axis_unit_changed(self) -> None:
        """Change axis units."""
        log.debug('Received signal axis unit changed.')
        units = self._pull_units_from_gui()
        targets = self.selected_target_axis()
        self.figure.change_axis_unit(units=units, target=targets)
        if self.fit_results:
            panes, _ = self.figure.parse_pane_flag(targets)
            new_fits = self.fit_results.change_units(units, panes,
                                                     return_new=True)
            self.figure.stale_fit_artists(new_fits)

        self.signals.atrophy_bg_full.emit()

    def _pull_units_from_gui(self) -> Dict[str, str]:
        """
        Retrieve unit selection from GUI controls.

        Used if the user changes the units for an axis on
        the GUI and the plots need to be updated to
        reflect the new setting.

        Returns
        -------
        units : dict
            Keys are 'x', 'y'; values are unit strings
            for the axes.
        """
        units = {'x': str(self.x_unit_selector.currentText()),
                 'y': str(self.y_unit_selector.currentText())}
        return units

    @QtCore.pyqtSlot()
    def axis_field_changed(self) -> None:
        """Change the axis field displayed."""
        log.debug('Received signal axis field changed.')
        fields = self._pull_fields_from_gui()
        targets = self.selected_target_axis()
        self.figure.change_axis_field(fields=fields, target=targets)
        # clear any active selection states
        self.clear_selection()
        self.signals.atrophy_bg_full.emit()

    def _pull_fields_from_gui(self) -> Dict[str, str]:
        """
        Retrieve field selection from GUI controls.

        Used if the user changes the fields to plot for an
        axis on the GUI and the plots need to be updated to
        reflect the new setting.

        Returns
        -------
        fields : dict
            Keys are 'x', 'y'; values are field strings
            for the axes.
        """
        x = self._parse_field_name(self.x_property_selector.currentText())
        y = self._parse_field_name(self.y_property_selector.currentText())
        fields = {'x': x, 'y': y}
        return fields

    @QtCore.pyqtSlot()
    def current_cursor_location(self, event: Event) -> None:
        """
        Update the cursor location displays.

        Parameters
        ----------
        event : QEvent
            Keypress event.
        """
        if self.cursor_checkbox.isChecked():
            if event.inaxes and self.figure.populated():

                data_points = self.figure.data_at_cursor(event)
                idxes = self.figure.determine_selected_pane(event.inaxes)
                for idx in idxes:
                    cursor_position = self.figure.panes[
                        idx].xy_at_cursor(event)

                    if self._cursor_popout:
                        self._update_cursor_loc_window(data_points,
                                                       cursor_position)
                    else:
                        self._update_cursor_loc_labels(data_points,
                                                       cursor_position)
                self.signals.atrophy.emit()

    def _update_cursor_loc_labels(self,
                                  data_coords: Dict[str, List[Dict[str, Num]]],
                                  cursor_coords: Sequence[Num]) -> None:
        """
        Update cursor location labels in the small in-window display.

        Parameters
        ----------
        data_coords : dict
            Keys are filenames; values are lists of dicts
            containing 'order', 'bin', 'bin_x', 'bin_y',
            'x_field', 'y_field', 'color', and 'visible'
            values to display.
        cursor_coords : tuple or list
            Current cursor (x, y) coordinates.
        """
        self.cursor_x_label.setText(f'{cursor_coords[0]:.2f}')
        self.cursor_y_label.setText(f'{cursor_coords[1]:.2f}')

        x_data, y_data, bin_data = list(), list(), list()
        for i, model_data_coords in enumerate(data_coords.values()):
            for values in model_data_coords:
                # skip invisible and overplot data for summary
                if values['visible'] and not values['alt']:
                    x_data.append(values["bin_x"])
                    y_data.append(values["bin_y"])
                    bin_data.append(values["bin"])

        # average values for quick look
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if x_data:
                x_bin_label = f'{np.nanmean(x_data):.3g}'
            else:
                x_bin_label = '-'
            if y_data:
                y_bin_label = f'{np.nanmean(y_data):.3g}'
            else:
                y_bin_label = '-'
            if bin_data:
                bin_label = f'{np.nanmean(bin_data):.3g}'
            else:
                bin_label = '-'

        self.cursor_wave_label.setText(x_bin_label)
        self.cursor_flux_label.setText(y_bin_label)
        self.cursor_column_label.setText(bin_label)

    def _update_cursor_loc_window(self,
                                  data_coords: Dict[str, List[Dict[str, Num]]],
                                  cursor_coords: Sequence[Num]) -> None:
        """
        Update cursor location labels in the pop-out display.

        Parameters
        ----------
        data_coords : dict
            Keys are filenames; values are lists of dicts
            containing 'order', 'bin', 'bin_x', 'bin_y',
            'x_field', 'y_field', 'color', and 'visible'
            values to display.
        cursor_coords : tuple or list
            Current cursor (x, y) coordinates.
        """
        cursor_labels = [self.cursor_x_label, self.cursor_y_label,
                         self.cursor_wave_label, self.cursor_flux_label,
                         self.cursor_column_label]
        for label in cursor_labels:
            label.setText('-')

        self.cursor_location_window.update_points(data_coords, cursor_coords)

    @QtCore.pyqtSlot()
    def leave_axes(self, event: Union[Event, None]) -> None:
        """
        Reset cursor labels when the mouse leaves a plot.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Mouse motion event.
        """
        self.cursor_x_label.setText('-')
        self.cursor_y_label.setText('-')
        self.cursor_wave_label.setText('-')
        self.cursor_flux_label.setText('-')
        self.cursor_column_label.setText('-')
        self.figure.reset_data_points()
        self.signals.atrophy.emit()

    @QtCore.pyqtSlot()
    def figure_clicked(self, event: Event) -> None:
        """
        Select a pane by mouse click.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Mouse click event.
        """
        if event.inaxes:
            log.debug('Figure was clicked on inside the axis')
            pane_index = self.figure.determine_selected_pane(event.inaxes)
            if pane_index is not None and not self.figure.recording:
                self.all_panes_checkbox.blockSignals(True)
                self.all_panes_checkbox.setChecked(False)
                self.all_panes_checkbox.blockSignals(False)
                self.set_current_pane(pane_index)

    def enable_cursor_position(self) -> None:
        """Enable cursor position displays."""
        if self.cursor_checkbox.isChecked():
            # if cursor is enabled, hook up to mouse motion events
            self.cid['cursor_loc'] = self.figure_widget.canvas.mpl_connect(
                'motion_notify_event', self.current_cursor_location)
            self.cid['cursor_axis_leave'] = \
                self.figure_widget.canvas.mpl_connect(
                    'axes_leave_event', self.leave_axes)
        else:
            # otherwise, unhook cids
            self.clear_cids('cursor')
            self.leave_axes(None)

    def popout_cursor_position(self) -> None:
        """Pop out a cursor position display window."""
        if self.cursor_location_window is None:
            self.cursor_location_window = cl.CursorLocation(self)
            self.cursor_location_window.show()
        else:
            self.cursor_location_window.raise_()
        self.cursor_checkbox.setChecked(True)
        self.enable_cursor_position()
        self._cursor_popout = True

    def closed_cursor_popout(self) -> None:
        """Close the cursor position display window."""
        self.cursor_location_window = None
        self.cursor_checkbox.setChecked(False)
        self.clear_cids('cursor')
        self.leave_axes(None)
        self._cursor_popout = False

    @QtCore.pyqtSlot()
    def toggle_fit_visibility(self):
        """Show or hide fit parameters"""
        if not self.fit_results:
            return
        fits = self.fit_results.gather_models()
        self.figure.toggle_fits_visibility(fits)
        self.signals.atrophy.emit()

    @QtCore.pyqtSlot()
    def filename_table_selection_changed(self):
        """Set selection for a cell in the filename table."""
        self.all_filenames_checkbox.blockSignals(True)
        self.all_filenames_checkbox.setChecked(False)
        self.all_filenames_checkbox.blockSignals(False)

        self._update_order_selector()

    @QtCore.pyqtSlot()
    def all_filenames_selection_changed(self):
        """Set selection for all files in the filename table."""
        state = bool(self.all_filenames_checkbox.checkState())
        self._update_filename_table_selection(all_=state)
        self._update_order_selector()

    ####
    # Setup
    ####
    def setup_property_selectors(self,
                                 panes: Optional[Union[List[PT], PT]] = None,
                                 target: Optional[Dict] = None) -> None:
        """
        Set up plot field selectors based on loaded data.

        Parameters
        ----------
        panes : pane.Pane or list of pane.Pane, optional
            The panes containing loaded data. If not
            provided, apply to the current pane.
        target : dict, optional
            Keys are `axis`, `pane` and values can be str, int or list.
        """
        log.debug(f'Updating property selector for pane {panes}')
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass
        selectors = {'x': self.x_property_selector,
                     y_label: self.y_property_selector}
        for selector in selectors.values():
            selector.clear()
        if panes is None:
            panes = self.figure.get_current_pane()
        elif not isinstance(panes, list):
            panes = [panes]
        if panes is None:
            # No panes present
            log.debug('No panes present, clearing property selectors')
        else:
            all_fields = list()
            current_fields = {k: list() for k in selectors.keys()}
            for pane_ in panes:
                fields = pane_.fields
                if pane_.models:
                    if pane_.plot_kind == 'spectrum':
                        order = list(pane_.models.values())[0].retrieve(
                            order=0, level='high')
                        if order:
                            fields = order.data.keys()
                        else:  # pragma: no cover
                            fields = ['-']
                    elif pane_.plot_kind == 'image':  # pragma: no cover
                        fields = list(pane_.models.values())[0].images.keys()
                else:
                    fields = ['-']
                all_fields.extend(fields)

                for ax in selectors.keys():
                    current_fields[ax].append(pane_.fields[ax])

            fields = list(unique_everseen(all_fields))
            for selector in selectors.values():
                fields = [self._format_field_name(f) for f in fields]
                selector.addItems(fields)

            for ax, selector in selectors.items():
                try:
                    current_field = list(set(current_fields[ax]))[0]
                except IndexError:  # pragma: no cover
                    index = 0
                else:
                    index = selector.findText(self._format_field_name(
                        current_field))
                if index > 0:
                    selector.setCurrentIndex(index)

    @staticmethod
    def _format_field_name(field):
        """Modify the input field name."""
        field = str(field).split('_order')[0]
        field = field.replace('wavepos', 'wavelength')
        field = field.replace('_', ' ').title()
        return field

    @staticmethod
    def _parse_field_name(field):
        """Parse the input field name."""
        field = field.lower().replace(' ', '_')
        field = field.replace('wavelength', 'wavepos')
        return field

    def setup_unit_selectors(self,
                             panes: Optional[Union[List[PT], PT]] = None,
                             target: Optional[Dict] = None) -> None:
        """
        Set up plot unit selectors based on loaded data.

        Parameters
        ----------
        panes : pane.Pane or list of pane.Pane, optional
            The panes containing loaded data. If not
            provided, apply to the current pane.
        target : Dict, optional
            Keys are `axis`, `pane` and values can be str, int or list
        """
        log.debug(f'Updating unit selectors for {panes}.')
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass
        unit_selectors = {'x': self.x_unit_selector,
                          y_label: self.y_unit_selector}
        if panes is None:
            panes = self.figure.get_current_pane()
        elif not isinstance(panes, list):
            panes = [panes]
        if panes is None:
            log.debug('No panes currently exist')
        else:
            for axis, selector in unit_selectors.items():
                selector.clear()
                string_units = list()
                current_selections = list()
                for pane_ in panes:
                    current_selections.append(pane_.current_units()[axis])
                    units = pane_.possible_units()
                    string_units.extend([str(i) for i in units[axis]])
                string_units = list(unique_everseen(string_units))
                selector.addItems(string_units)
                current_selections = list(set(current_selections))

                if len(current_selections) == 0:
                    idx = 0
                elif current_selections[0] in string_units:
                    idx = string_units.index(current_selections[0])
                else:
                    idx = 0
                selector.setCurrentIndex(idx)
                log.debug(f'Set unit selectors to {string_units}, '
                          f'selected {idx}')

    @staticmethod
    def merge_dicts(dol1, dol2):
        """
        Merge two dictionaries.

        Parameters
        ----------
        dol1 : dict
            First dictionary.
        dol2 : dict
            Second dictionary.

        Returns
        -------
        dict
            Merged dictionary.
        """
        keys = set(dol1).union(dol2)
        no = list()
        return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)

    def setup_axis_limits(self,
                          panes: Optional[Union[List[PT], PT]] = None,
                          target: Optional[Dict] = None) -> None:
        """
        Set up axis limit text boxes based on loaded data.

        Parameters
        ----------
        panes : pane.Pane or list of pane.Pane, optional
            The panes containing loaded data. If not provided
            apply to the current pane.
        target : Dict, optional
            Keys are `axis`, `pane` and values can be str, int or list
        """
        log.debug(f'Updating axis limits for {panes}.')
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass
        if panes is None:
            panes = self.figure.get_current_pane()
        elif not isinstance(panes, list):
            panes = [panes]
        if panes is None:
            log.debug('No pane currently present')
        else:
            limit_displays = {'x': [self.x_limit_min, self.x_limit_max],
                              y_label: [self.y_limit_min, self.y_limit_max]}
            full_limits = {k: list() for k in limit_displays.keys()}
            for pane_ in panes:
                limits = pane_.get_axis_limits()
                for ax in limit_displays.keys():
                    full_limits[ax].append(limits[ax])
                log.debug(f'New limits: {limits}.')

            for ax, limit_display in limit_displays.items():
                try:
                    limits = full_limits[ax][0]
                except IndexError:  # pragma: no cover
                    limits = None
                if limits:
                    lim = [f'{val:0.3f}' for val in limits]
                else:
                    lim = ['', '']
                for display, value in zip(limit_display, lim):
                    display.setText(value)

    def setup_initial_scales(self,
                             panes: Optional[Union[List[PT], PT]] = None,
                             target: Optional[Dict] = None) -> None:
        """
        Set up plot scale selectors based on loaded data.

        Parameters
        ----------
        panes : pane.Pane or list of pane.Pane, optional
            The pane containing loaded data. If not provided,
            apply to the current pane.
        target : Dict, optional
            Keys are `axis`, `pane` and values can be str, int or list
        """
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass

        if panes is None:
            panes = self.figure.get_current_pane()
        elif not isinstance(panes, list):
            panes = [panes]
        if panes is None:
            log.debug('No pane currently present')
        else:
            self.buttonGroup.setExclusive(False)
            self.buttonGroup_2.setExclusive(False)
            self._block_scale_signals(True)
            scale_displays = {'x': [self.x_scale_linear_button,
                                    self.x_scale_log_button],
                              y_label: [self.y_scale_linear_button,
                                        self.y_scale_log_button]}
            scales = list()
            for pane_ in panes:
                scales.append(pane_.get_axis_scale())
            for axis, displays in scale_displays.items():
                if len(set([s[axis] for s in scales])) == 1:
                    scale = scales[0]
                    if scale[axis] == 'linear' or scale[axis] is None:
                        displays[0].setChecked(True)
                        displays[1].setChecked(False)
                    elif scale[axis] == 'log':
                        displays[0].setChecked(False)
                        displays[1].setChecked(True)
                else:
                    displays[0].setChecked(False)
                    displays[1].setChecked(False)
            self.buttonGroup.setExclusive(True)
            self.buttonGroup_2.setExclusive(True)
            self._block_scale_signals(False)

    def setup_overplot_flag(
            self, panes: Optional[Union[List[PT], PT]] = None) -> None:
        """
        Set up overplot display flags.

        Parameters
        ----------
        panes : pane.Pane or list of pane.Pane, optional
            The panes containing loaded data. If not provided,
            apply to the current pane.
        """
        if panes is None:
            panes = self.figure.get_current_pane()
        elif not isinstance(panes, list):
            panes = [panes]
        if panes is None:
            log.debug('No pane currently present')
        else:
            self.enable_overplot_checkbox.blockSignals(True)
            states = list(set([p.overplot_state() for p in panes]))
            if len(states) == 1:
                self.enable_overplot_checkbox.setChecked(states[0])
            else:
                self.enable_overplot_checkbox.setChecked(False)
            self.enable_overplot_checkbox.blockSignals(False)

    def add_filename(self, model_id: IDT, filename: str) -> None:
        """
        Add a file that has been read in to the model collection.

        Logs a warning if file is already in the loaded list.

        Parameters
        ----------
        model_id: str, uuid.UUID
            Unique ID for model.
        filename : str
            Name of the file that was added to models.
        """
        if model_id not in self.model_collection:
            self.model_collection[model_id] = filename
        else:
            log.warning(f'Model {model_id} ({filename}) is already in '
                        f'display model list.')

    def remove_model_id(self, model_id: str) -> None:
        """
        Remove a file from the list.

        Parameters
        ----------
        model_id : str
            ID of the model to remove.
        """
        try:
            del self.model_collection[model_id]
        except KeyError:
            # could happen in race conditions
            pass

    def select_color_cycle(self, text: str) -> None:
        """
        Select the color cycle for the plot.

        Parameters
        ----------
        text : {'spectral', 'tableau', 'accessible'}
            Color cycle to set.
        """
        updates = self.figure.set_color_cycle(text)
        if self.fit_results:
            self.fit_results.update_colors(updates)

    def select_plot_type(self, text: str) -> None:
        """
        Select the plot type.

        Parameters
        ----------
        text : {'line', 'step', 'scatter'}
            The plot type to set.
        """
        self.figure.set_plot_type(text)

    def toggle_markers(self, state: bool) -> None:
        """
        Set the marker visibility in all panes.

        Parameters
        ----------
        state: bool
            If True, markers will be shown.  If False, they
            will be hidden.
        """
        self.figure.set_markers(state)

    def toggle_grid(self, state: bool) -> None:
        """
        Set the grid visibility in all panes.

        Parameters
        ----------
        state: bool
            If True, gridlines will be shown.  If False, they
            will be hidden.
        """
        self.figure.set_grid(state)

    def toggle_error(self, state: bool) -> None:
        """
        Set the error range visibility in all panes.

        Parameters
        ----------
        state: bool
            If True, error ranges will be shown.  If False, they
            will be hidden.
        """
        self.figure.set_error(state)

    def toggle_dark_mode(self, state: bool) -> None:
        """
        Set a dark background in all panes.

        Parameters
        ----------
        state: bool
            If True, dark mode will be enabled.  If False, it
            will be disabled.
        """
        self.figure.set_dark_mode(state)

    ####
    # Saving
    ####
    def save(self, filename: str, **kwargs) -> None:
        """
        Save the current figure to a file.

        Parameters
        ----------
        filename : str
            Full file path to save the image to.
        kwargs : dict, optional
            Additional keyword arguments to pass to
            `matplotlib.figure.Figure.savefig`.
        """
        self.figure.save(filename, **kwargs)

    ####
    # Files/Data
    ####
    def refresh_file_table(self) -> None:
        """Refresh the file list from currently loaded data."""
        self._clear_file_table()
        col_index = 0
        self.loaded_files_table.setRowCount(len(self.model_collection))
        for row_index, (model_id, filename) in enumerate(
                self.model_collection.items()):
            # display base name
            item = QtWidgets.QTableWidgetItem(os.path.basename(filename))
            # store model_id in data and set full file path as tooltip
            item.setData(QtCore.Qt.UserRole, model_id)
            item.setToolTip(filename)
            self.loaded_files_table.setItem(row_index, col_index, item)
        self.loaded_files_table.resizeRowsToContents()
        self.loaded_files_table.resizeColumnsToContents()
        self.loaded_files_table.show()

    def _clear_file_table(self) -> None:
        """Clear the file table widget."""
        self.loaded_files_table.setRowCount(0)
        self.loaded_files_table.setColumnCount(1)

    def current_files_selected(self) -> Union[List[str], None]:
        """
        Retrieve all files currently selected in the file list.

        Returns
        -------
        filenames : list of str or None
            The selected filenames. If no files are selected, None is
            returned.
        """
        items = self.loaded_files_table.selectedItems()
        if not items:
            return
        model_ids = [item.data(QtCore.Qt.UserRole) for item in items]
        return model_ids

    ####
    # Panes
    ####
    def add_pane(self) -> None:
        """Add a new pane."""
        self.figure.add_panes(n_dims=0, n_panes=1)
        self.all_panes_checkbox.blockSignals(True)
        self.all_panes_checkbox.setChecked(False)
        self.all_panes_checkbox.blockSignals(False)
        self.signals.current_pane_changed.emit()

    def add_panes(self, n_panes: int, kind: List[str],
                  layout: Optional[str] = 'grid') -> None:
        """
        Add new plot panes.

        Parameters
        ----------
        n_panes : int
            The number of panes to add.
        kind : {'spectrum', 'onedim', 'image', 'twodim'}
            The kind of pane to add.
        layout : {'grid', 'rows', 'columns'}, optional
            The layout method for the new panes.
        """
        n_dims = list()
        if len(kind) == 1:
            kind = kind * n_panes
        for k in kind:
            if k in ['spectrum', 'onedim']:
                n_dims.append(1)
            elif k in ['image', 'twodim']:
                n_dims.append(2)
            else:
                raise EyeError('Valid pane kinds are: spectrum, image')
        if layout in ['grid', 'rows', 'columns']:
            self.figure.set_layout_style(layout)
        self.figure.add_panes(n_panes=n_panes, n_dims=n_dims)

    def remove_pane(self) -> None:
        """Remove current selected pane."""
        self.figure.remove_pane()
        self.signals.atrophy_bg_partial.emit()
        # self.signals.atrophy.emit()

    def remove_panes(self, panes: Union[str, List[int]]) -> None:
        """
        Remove specified panes.

        Parameters
        ----------
        panes : str or list of int
            If 'all', all panes will be removed.  Otherwise,
            specify a list of pane indices to remove.
        """
        if isinstance(panes, str) and panes == 'all':
            self.figure.remove_all_panes()
        else:
            self.figure.remove_pane(panes)
        self.signals.atrophy_bg_partial.emit()

    def pane_count(self) -> int:
        """
        Retrieve the current number of panes.

        Returns
        -------
        int
            The pane count.
        """
        return self.figure.pane_count()

    def pane_layout(self) -> Union[None, Tuple[int, int]]:
        """
        Retrieve the current pane layout.

        Returns
        -------
        geometry : tuple of int, or None
            If there is an active layout, (nrow, ncol) is returned.
            Otherwise, None.
        """
        return self.figure.pane_layout()

    def remove_data_from_all_panes(self, model_id: str) -> None:
        """
        Remove loaded data from the view.

        Parameters
        ----------
        model_id : str
            The Unique id for the file to remove.
        """
        self.figure.remove_model_from_pane(model_id=model_id)
        self.remove_model_id(model_id)
        # clear any active selection states
        self.clear_selection()

    def display_model(self, model: MT,
                      pane_: Optional[PT] = None) -> None:
        """
        Display a model in a plot pane.

        Parameters
        ----------
        model : high_model.HighModel
            The model to display.
        pane_ : pane.Pane, optional
            The pane to display to.  If not provided, the current
            pane will be used.
        """
        self.figure.add_model_to_pane(model, pane_)
        # clear any active selection states
        self.clear_selection()
        self.signals.atrophy_bg_full.emit()
        self.populate_order_selectors()
        self.update_controls()

    def assign_models(self, mode: str, models: Dict[str, MT],
                      indices: List[int]) -> None:
        """
        Assign models to panes.

        Parameters
        ----------
        mode : {'split', 'first', 'last', assigned'}
            Specifies how to arrange the models on the panes.
            'Split' divides the models as evenly as possible
            across all present panes. 'First' assigns all the
            models to the first pane, while 'last' assigns all
            the models to the last pane. 'Assigned' attaches
            each model to the pane index provided in `indices`.
        models : dict
            Dictionary of models to add. Keys are the model ID,
            with the values being the models themselves.
        indices : list of int
            A list of integers with the same length of `models`.
            Only used for `assigned` mode. Specifies the index
            of the desired pane for the model.

        Raises
        ------
        RuntimeError :
            If an invalid mode is provided.
        """
        errors = self.figure.assign_models(mode=mode, models=models,
                                           indices=indices)
        if errors > 0:
            log.warning(f'Failed to add {errors} model'
                        f'{"s" if errors > 1 else ""}.')

    def models_per_pane(self) -> List[int]:
        """
        Retrieve the number of models in each pane.

        Returns
        -------
        count : list of int
            The number of models in each existing pane.
        """
        return self.figure.models_per_pane()

    def set_current_pane(self, pane_index: Union[List[int], int]) -> None:
        """
        Set the current pane.

        Parameters
        ----------
        pane_index : int or list of int
            The panes to activate.
        """
        if not isinstance(pane_index, list):
            pane_index = [pane_index]
        pane_change = self.figure.change_current_pane(pane_index)
        if pane_change:
            log.debug(f'Setting current pane to {pane_index}')
            self.signals.current_pane_changed.emit()

    def select_pane(self, item: QTreeWidgetItem) -> None:
        """
        Select a pane as current from the pane list widget.

        Parameters
        ----------
        item : QtWidgets.QTreeWidgetItem
            The pane item selected from the tree widget. Should
            have the pane index stored as data, under the
            UserRole tag.
        """
        pane_id = item.data(0, QtCore.Qt.UserRole)
        if isinstance(pane_id, int):
            self.set_current_pane(pane_id)

    def enable_model(self, item: QTreeWidgetItem) -> None:
        """
        Enable a model from the pane list.

        Parameters
        ----------
        item : QtWidgets.QTreeWidgetItem
            The model item associated with the model.
            Should have pane_id and model_id stored as data,
            under the UserRole tag.
        """
        pane_id, model_id = item.data(0, QtCore.Qt.UserRole)
        if not model_id or isinstance(model_id, bool):
            return
        state = bool(item.checkState(0))
        self.figure.set_enabled(pane_id, model_id, state)

    def enable_all_models(self) -> None:
        """Enable all loaded models from the pane list widget."""
        button = self.sender()
        pane_id, state = button.property('id')
        self.figure.set_all_enabled(pane_id, state)
        if state:
            button.setProperty('id', (pane_id, False))
            button.setText('Hide all')
        else:
            button.setProperty('id', (pane_id, True))
            button.setText('Show all')

    def selected_target_axis(self) -> Dict[str, str]:
        """
        Parse the state of the axis selection widget.

        Returns
        -------
        target : dict
            Keys are 'axis' or 'pane'; values can be str, int or list.
        """
        text = str(self.axes_selector.currentText()).lower()
        target = dict()
        if text == 'all':
            target['axis'] = 'all'
        else:
            if 'primary' in text:
                target['axis'] = 'primary'
            elif 'overplot' in text:
                target['axis'] = 'alt'
            elif 'both' in text:
                target['axis'] = 'all'
            else:
                raise EyeError(f'Unknown target axis selected: {text}')
        _, pane_index = self._parse_filename_table_selection()
        target['pane'] = pane_index
        return target

    ####
    # Orders
    ####
    @QtCore.pyqtSlot()
    def on_orders_changed(self):
        """Handle a change in order selection."""
        on_orders = self.decode_orders(self.on_orders_selector.text())
        self._update_orders_from_gui(on_orders, True)

    @QtCore.pyqtSlot()
    def enable_all_orders(self):
        """Enable all orders."""
        log.info('Enabling all orders')
        self.on_orders_selector.blockSignals(True)
        self.off_orders_selector.blockSignals(True)

        self.on_orders_selector.setText('*')
        self.off_orders_selector.setText('-')

        self.on_orders_selector.blockSignals(False)
        self.off_orders_selector.blockSignals(False)

        self.on_orders_changed()

    @QtCore.pyqtSlot()
    def disable_all_orders(self):
        """Disable all orders."""
        self.on_orders_selector.blockSignals(True)
        self.off_orders_selector.blockSignals(True)

        self.on_orders_selector.setText('-')
        self.off_orders_selector.setText('*')

        self.on_orders_selector.blockSignals(False)
        self.off_orders_selector.blockSignals(False)

        self.off_orders_changed()

    @QtCore.pyqtSlot()
    def off_orders_changed(self):
        """Disable selected orders."""
        off_orders = self.decode_orders(self.off_orders_selector.text())
        self._update_orders_from_gui(off_orders, False)

    @QtCore.pyqtSlot()
    def filename_order_selector_changed(self):
        """Handle a change in the filename order selector."""
        self.all_filenames_checkbox.blockSignals(True)
        self.all_filenames_checkbox.setChecked(False)
        self.all_filenames_checkbox.blockSignals(False)
        self._update_pane_selector()
        self._update_order_selector()

    @QtCore.pyqtSlot()
    def all_filenames_checking(self):
        """Update the order selector."""
        self._update_order_selector()

    @QtCore.pyqtSlot()
    def pane_order_selector_changed(self):
        """Handle a change to the pane selector."""
        self.all_panes_checkbox.blockSignals(True)
        self.all_panes_checkbox.setChecked(False)
        self.all_panes_checkbox.blockSignals(False)
        try:
            selection = [int(self.pane_selector.currentText()) - 1]
        except ValueError:
            return
        self.set_current_pane(selection)
        self._update_filename_table()
        self._update_order_selector()

    @QtCore.pyqtSlot()
    def all_panes_checking(self):
        """Handle a change to the all panes check box."""
        if self.all_panes_checkbox.isChecked():
            self.set_current_pane(list(range(self.figure.pane_count())))
        else:
            if self.figure.pane_count() > 0:
                self.set_current_pane([0])
            else:
                self.set_current_pane(list())
        self.signals.current_pane_changed.emit()

    @QtCore.pyqtSlot()
    def remove_file_from_pane(self):
        """Handle a file removal request."""
        model_id, pane_index = self._parse_filename_table_selection()
        self.figure.remove_model_from_pane(model_id=model_id, panes=pane_index)

    def _update_orders_from_gui(self, orders, enable):
        pane_text = self.pane_selector.currentText()
        all_checked = self.all_panes_checkbox.isChecked()
        if all_checked:
            update = dict().fromkeys(range(self.figure.pane_count()),
                                     orders)
        else:
            try:
                update = {int(pane_text) - 1: orders}
            except ValueError:
                # Panes not populated yet
                return

        # TODO: This needs reworking for if aperture and orders are
        #  available on multiple panes
        apertures = 'aperture' in self.enabled_orders_label.text().lower()
        self.figure.set_orders(update, enable, aperture=apertures)

        self.signals.atrophy.emit()

    def populate_order_selectors(self,
                                 set_current_pane: Optional[bool] = False):
        """
        Populate order selectors for currently loaded data.

        Parameters
        ----------
        set_current_pane : bool, optional
           If set, the current index of the selector will be set to
            current pane after populating.
        """
        self._update_pane_selector(set_current_pane)
        self._update_filename_table()
        self._update_order_selector()

    def _update_pane_selector(self, set_current_pane: Optional[bool] = False):
        """
        Populate pane selector with current panes.

        Parameters
        ----------
        set_current_pane : bool, optional
            If set, the current index of the selector will be set to
            current pane after populating.
        """
        self.pane_selector.blockSignals(True)
        initial_pane = self.pane_selector.currentText()
        self.pane_selector.clear()
        for i in range(self.figure.pane_count()):
            self.pane_selector.addItem(f'{i + 1}')
        if set_current_pane:
            try:
                pane_number = f'{self.figure.current_pane[0] + 1}'
            except IndexError:  # pragma: no cover
                pane_number = '0'
        else:
            pane_number = initial_pane
        index = max(self.pane_selector.findText(pane_number), 0)
        self.pane_selector.setCurrentIndex(index)

        self.pane_selector.blockSignals(False)

    def _update_filename_table(self):
        """Update filename table from current data."""

        table_labels = ['marker', 'color', 'orders', 'filename']
        fn_index = table_labels.index('filename')
        current_row = self.filename_table.currentRow()
        current_ids = list()
        if current_row >= 0:
            if not isinstance(current_row, list):  # pragma: no cover
                current_row = [current_row]
            for row in current_row:
                mid = self.filename_table.item(row, fn_index).data(
                    QtCore.Qt.UserRole)
                current_ids.append(mid)

        self.filename_table.blockSignals(True)
        self.filename_table.setUpdatesEnabled(False)
        self.filename_table.clearContents()

        alignments = [QtCore.Qt.AlignCenter, QtCore.Qt.AlignCenter,
                      QtCore.Qt.AlignCenter, QtCore.Qt.AlignLeft]
        size_policies = [QtWidgets.QHeaderView.ResizeToContents,
                         QtWidgets.QHeaderView.ResizeToContents,
                         QtWidgets.QHeaderView.ResizeToContents,
                         QtWidgets.QHeaderView.ResizeToContents]

        self.filename_table.setColumnCount(len(table_labels))
        self.filename_table.setHorizontalHeaderLabels(
            [label.capitalize() for label in table_labels])
        self.filename_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.filename_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.filename_table.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.filename_table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)

        if self.all_panes_checkbox.isChecked():
            panes = self.figure.panes
        else:
            try:
                pane_index = int(self.pane_selector.currentText()) - 1
                panes = [self.figure.panes[pane_index]]
            except (ValueError, IndexError):  # pragma: no cover
                panes = list()
        cells = dict()
        for pane_ in panes:
            for model_id, model in pane_.models.items():
                if model_id in cells:  # pragma: no cover
                    continue
                marker = pane_.markers[model_id]
                marker_item = QtWidgets.QTableWidgetItem(f'{marker}')

                colors = pane_.colors[model_id]
                color_item = ApertureColors(colors=colors)

                name_item = QtWidgets.QTableWidgetItem(
                    f'{os.path.basename(model.filename)}')
                name_item.setData(QtCore.Qt.UserRole, model_id)
                name_item.setToolTip(model.filename)

                orders = model.num_orders
                orders_item = QtWidgets.QTableWidgetItem(f'{orders:d}')

                cells[model_id] = {'marker': marker_item,
                                   'color': color_item,
                                   'orders': orders_item,
                                   'filename': name_item}
        self.filename_table.setRowCount(len(cells))
        for row, cell in enumerate(cells.values()):
            for col, label in enumerate(table_labels):
                item = cell[label.lower()]
                if label == 'color':
                    self.filename_table.setCellWidget(row, col, item)
                else:
                    item.setTextAlignment(alignments[col])
                    self.filename_table.setItem(row, col, item)

        header = self.filename_table.horizontalHeader()
        for i, policy in enumerate(size_policies):
            header.setSectionResizeMode(i, policy)

        for row in range(self.filename_table.rowCount()):
            item = self.filename_table.item(row, fn_index)
            if item.data(QtCore.Qt.UserRole) in current_ids:
                self.filename_table.selectRow(row)

        self.filename_table.setUpdatesEnabled(True)
        self.filename_table.blockSignals(False)

    def _parse_filename_table_selection(self) -> Tuple[List[IDT], List[int]]:
        """
        Obtain lists of models and pane indices from the filename table.

        Returns
        -------
        models_ids, pane_index : list, list
            List of selected Unique model ids; List of selected pane
            indexes.
        """
        all_panes = self.all_panes_checkbox.isChecked()
        all_filenames = self.all_filenames_checkbox.isChecked()
        headers = [self.filename_table.horizontalHeaderItem(i).text().lower()
                   for i in range(self.filename_table.columnCount())]
        try:
            fn_index = headers.index('filename')
        except ValueError:
            return list(), list()

        if all_panes:
            pane_index = list(range(self.figure.pane_count()))
        else:
            pane_choice = self.pane_selector.currentText()
            try:
                pane_index = [int(pane_choice) - 1]
            except ValueError:
                pane_index = [-1]

        if self.filename_table.rowCount() == 0:
            model_ids = list()
        elif all_filenames:
            model_ids = list()
            for index in pane_index:
                try:
                    pane_ids = list(self.figure.panes[index].models.keys())
                except IndexError:  # pragma: no cover
                    continue
                else:
                    model_ids.extend(pane_ids)
        else:
            items = list()
            selected_items = self.filename_table.selectedItems()
            if len(selected_items) == 0:
                items.append(self.filename_table.item(0, fn_index))
            else:
                for item in self.filename_table.selectedItems():
                    items.append(
                        self.filename_table.item(item.row(), fn_index))
            model_ids = [item.data(QtCore.Qt.UserRole) for item in items]
        model_ids = list(set(model_ids))

        return model_ids, pane_index

    def _update_filename_table_selection(self, all_: Optional[bool] = False,
                                         rows: Optional[List[int]] = None):
        self.filename_table.blockSignals(True)
        self.filename_table.clearSelection()
        if rows:
            for row in rows:
                self.filename_table.selectRow(row)
        elif all_:
            self.filename_table.selectAll()
        self.filename_table.blockSignals(False)

    def _update_order_selector(self):
        """Update the order selection."""
        model_ids, pane_index = self._parse_filename_table_selection()
        self.on_orders_selector.blockSignals(True)
        self.off_orders_selector.blockSignals(True)
        self.on_orders_selector.clear()
        self.off_orders_selector.clear()

        if len(model_ids) > 0 and pane_index != -1:
            ap_order_state = self._determine_ap_order_state(pane_index,
                                                            model_ids)
            self._populate_enabled_disabled_orders(pane_index, model_ids,
                                                   ap_order_state)

        self.on_orders_selector.blockSignals(False)
        self.off_orders_selector.blockSignals(False)

    def _determine_ap_order_state(self, pane_index, model_ids):
        state = dict()

        for model_id in model_ids:
            n_ap, n_or = self.figure.ap_order_state(pane_index, model_id)
            state[model_id] = {'aper': self._maximum(n_ap, model_id),
                               'order': self._maximum(n_or, model_id)}
        return state

    @staticmethod
    def _maximum(count: Dict, model_id: IDT) -> int:
        maximum = max([p[model_id] if p[model_id] is not None else 0
                       for p in count.values()])
        return maximum

    def _populate_enabled_disabled_orders(self, panes, model_ids,
                                          ap_order_state):
        """
        Display enabled and disabled order values.

        Parameters
        ----------
        panes : list of int
            Index of panes for which orders are being considered.
        model_ids : list of uuid.UUID
            Unique IDs for the models being considered.
        """
        enabled = list()
        disabled = list()

        self.multi_order = any([ap_order_state[m]['order'] > 1
                                for m in model_ids])
        self.multi_aper = any([ap_order_state[m]['aper'] > 1
                               for m in model_ids])
        self._configure_order_selector_labels()
        for p in panes:
            for model_id in model_ids:
                e, d = self._enabled_disabled_orders(p, model_id)
                enabled.append(set(e))
                disabled.append(set(d))
        if len(enabled) == 0 and len(disabled) == 0:
            return
        enabled = [e for e in enabled if len(e) > 0]
        if len(enabled) > 0:
            enabled = sorted(list(set.intersection(*enabled)))
        d_len = [len(d) for d in disabled if len(d) > 0]
        if all([length == d_len[0] for length in d_len]):
            disabled = [d for d in disabled if len(d) > 0]
            if len(disabled) > 0:
                disabled = sorted(list(set.intersection(*disabled)))
        else:
            disabled = list()
        enabled_string = self.format_orders_pairs(enabled)
        disabled_string = self.format_orders_pairs(disabled)
        self.on_orders_selector.setText(enabled_string)
        self.off_orders_selector.setText(disabled_string)

    def _configure_order_selector_labels(self):
        if self.multi_aper and not self.multi_order:
            self.enabled_orders_label.setText('Enabled Apertures')
            self.hidden_orders_label.setText('Hidden Apertures')
        else:
            self.enabled_orders_label.setText('Enabled Orders')
            self.hidden_orders_label.setText('Hidden Orders')

    def _enabled_disabled_orders(self, pane_index, model_id):
        """
        Obtain enabled and disabled orders.

        Parameters
        ----------
        pane_index : int
            Index of pane for which orders are being considered.
        model_id : uuid.UUID
            Unique id of the model being considered.

        Returns
        -------
        enabled_orders : list
            List of enabled orders.
        disabled_orders : list
            List of disabled orders.
        """
        if pane_index == -1:
            return list(), list()
        if self.multi_order:
            kind = 'order'
        elif self.multi_aper:
            kind = 'aperture'
        else:
            kind = 'order'
        if self.multi_aper:
            if self.multi_order:
                kind = 'order'
                # kind = 'all'
            else:
                kind = 'aperture'
        else:
            kind = 'order'
        args = {'target': pane_index, 'model_id': model_id,
                'group_by_model': False, 'kind': kind}
        enabled_orders = self.figure.get_orders(enabled_only=True, **args)
        enabled_orders = enabled_orders[pane_index]
        all_orders = self.figure.get_orders(enabled_only=False, **args)
        all_orders = all_orders[pane_index]
        disabled_orders = sorted(list(set(all_orders) - set(enabled_orders)))
        return enabled_orders, disabled_orders

    @staticmethod
    def format_orders_pairs(orders):
        """
        Format a list of orders selection display.

        Parameters
        ----------
        orders : list of int
            Order list to format

        Returns
        -------
        str
            String representation summarizing the list of orders.
        """
        s = list()
        # Change from zero-based indexing used by models to
        # one-based indexing used by people
        orders = [o + 1 for o in orders]
        for groups in consecutive_groups(orders):
            groups = list(groups)
            if len(groups) == 1:
                s.append(f'{groups[0]}')
            else:
                s.append(f'{groups[0]}-{groups[-1]}')
        if s:
            return ','.join(s)
        else:
            return '-'

    def decode_orders(self, orders: str) -> Dict:
        """
        Parse the text from an order selector.

        Parameters
        ----------
        orders : str
            Comma-separated list of order ranges.  A '*' indicates all
            orders; a '-' indicates none.

        Returns
        -------
        dict
            Keys are model_ids and values are lists of index numbers.
        """
        ind = dict()
        model_ids, pane_index = self._parse_filename_table_selection()
        if orders == '-':
            return ind
        elif orders == '*':
            if pane_index is None:
                return ind
            for model_id in model_ids:
                all_orders = self.figure.get_orders(pane_index,
                                                    enabled_only=False,
                                                    model_id=model_id,
                                                    group_by_model=False,
                                                    kind='all')
                index = set()
                for i in pane_index:
                    index.update(all_orders[i])
                index = sorted(list(index))
                ind[model_id] = index
            return ind
        else:
            index = list()
            for section in orders.split(','):
                if section.startswith('-') or '--' in section:
                    log.info(f'Invalid order notation: {section}')
                    continue
                try:
                    limits = [int(i) for i in section.split('-')]
                except ValueError:
                    log.info(f'Invalid order number: {section}')
                    continue
                if len(limits) == 2:
                    index.extend(list(range(limits[0], limits[1] + 1)))
                else:
                    index.append(limits[0])
            index.sort()
            # Convert one-based indexing used by people to
            # zero-based indexing used by models
            index = [i - 1 for i in index]
            ind = dict().fromkeys(model_ids, index)
            return ind

    def add_order(self) -> None:
        """Add a new order."""
        raise NotImplementedError

    def remove_order(self) -> None:
        """Remove an order."""
        raise NotImplementedError

    ####
    # Axis Controls
    ####
    def model_backup(self, models: Dict[IDT, MT]):
        """
        Obtain the backup model.

        Parameters
        ----------
        models : high_model.HighModel
            The model for which we want to obtain its backup copy.
        """
        targets = self.selected_target_axis()
        self.figure.model_backup(models, target=targets)

    def update_controls(self) -> None:
        """Update control widgets in the Axis section from loaded data."""
        panes = self.figure.get_current_pane()
        target = self.selected_target_axis()
        self.setup_unit_selectors(panes=panes, target=target)
        self.setup_property_selectors(panes=panes, target=target)
        self.setup_axis_limits(panes=panes, target=target)
        self.setup_initial_scales(panes=panes, target=target)
        self.setup_overplot_flag(panes=panes)

    def set_field(self) -> None:
        """Set a new axis field from the control widget."""
        self.signals.axis_field_changed.emit()

    def set_fields(self, fields: Dict[str, str], panes: PaneID) -> None:
        """
        Set new axis fields from the API.

        Parameters
        ----------
        fields : dict
            Should contain axis name keys ('x', 'y'), specifying
            the field strings as values.
        panes : str, None, or list of int, optional
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indexes to modify.
        """
        self.figure.change_axis_field(fields=fields, target=panes)

    def get_fields(self, panes: PaneID) -> List:
        """
        Get the fields associated with a given pane selection.

        Parameters
        ----------
        panes : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        fields : list
            List of fields for corresponding to pane selection
            provided.
        """
        return self.figure.get_fields(target=panes)

    def set_unit(self) -> None:
        """Set a new unit from the widget controls."""
        self.signals.axis_unit_changed.emit()

    def set_units(self, units: Dict[str, str], panes: PaneID) -> None:
        """
        Change the axis unit for specified panes.

        If incompatible units are specified, the current units
        are left unchanged.

        Parameters
        ----------
        units : dict
            Keys are 'x', 'y'.  Values are the units to convert
            to.
        panes : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
        """
        self.figure.change_axis_unit(units=units, target=panes)

    def get_units(self, panes: PaneID) -> List:
        """
        Get the units associated with a given pane selection.

        Parameters
        ----------
        panes : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        units : list
            List of units for corresponding to pane selection
            provided.
        """
        return self.figure.get_units(target=panes)

    def set_orders(self, orders: Dict[int, Dict]) -> None:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are indices for the panes to update. Values
            are dicts, with model ID keys, order list values.
        """
        self.figure.set_orders(orders=orders)

    def get_orders(self, panes: PaneID) -> Dict[int, Dict]:
        """
        Get the orders associated with a given pane selection.

        Parameters
        ----------
        panes : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        orders : dict
            Dictionary of orders for corresponding to pane selection
            provided. Keys are the indices of the panes.
        """
        return self.figure.get_orders(panes)

    def set_scale(self) -> None:
        """Set an axis scale from the widget controls."""
        self.signals.axis_scale_changed.emit()

    def set_limits(self) -> None:
        """Set axis limits from the widget controls."""
        self.signals.axis_limits_changed.emit()

    def set_scales(self, scales: Dict[str, str], panes: PaneID) -> None:
        """
        Set the axis scale for specified panes.

        Parameters
        ----------
        scales : dict
            Keys are 'x', 'y'.  Values are 'linear' or 'log'.
        panes : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
        """
        self.figure.set_scales(scales, panes)
        self.signals.atrophy_bg_partial.emit()

    def get_scales(self, panes: PaneID) -> List:
        """
        Get the axes scales associated with a given pane selection.

        Parameters
        ----------
        panes : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        scales : list
            List of axes scales for corresponding to pane selection
            provided.
        """
        return self.figure.get_scales(panes)

    def _block_scale_signals(self, block: bool) -> None:
        """
        Block scale signals.

        Parameters
        ----------
        block : bool
            If True, signals are blocked.  If False, they are
            unblocked.
        """
        self.x_scale_linear_button.blockSignals(block)
        self.y_scale_linear_button.blockSignals(block)
        self.x_scale_log_button.blockSignals(block)
        self.y_scale_log_button.blockSignals(block)

    ####
    # Mouse events
    ####
    def clear_guides(self) -> None:
        """Clear all guide artists."""
        targets = self.selected_target_axis()
        self.figure.clear_lines(flags='a', all_panes=targets)
        self.signals.atrophy.emit()

    def reset_zoom(self) -> None:
        """Reset axis limits to full range."""
        self.clear_selection()
        targets = self.selected_target_axis()
        self.figure.reset_zoom(targets=targets)
        self.signals.atrophy_bg_partial.emit()
        log.debug('Zoom reset')

    def start_selection(self, mode: str) -> None:
        """
        Start a user range selection, for zoom or fit.

        Parameters
        ----------
        mode : {'x_zoom', 'y_zoom', 'b_zoom', 'fit'}
            The mode to start.
        """
        self.clear_selection()
        self.figure.set_cursor_mode(mode)
        self.cid['zoom_crosshair'] = self.figure_widget.canvas.mpl_connect(
            'motion_notify_event', self.figure.crosshair)
        self.cid[mode] = self.figure_widget.canvas.mpl_connect(
            'button_press_event', self.figure.record_cursor_location)
        self.signals.atrophy.emit()
        log.info(f'Starting {mode} mode')

    def end_selection(self) -> None:
        """End a zoom or fit interaction."""
        results = self.figure.get_selection_results()
        if any(['fit' in key for key in self.cid.keys()]):
            if self.fit_results:
                self.fit_results.add_results(results)
            else:
                self.fit_results = fitting_results.FittingResults(self)
                self.fit_results.add_results(results)
            if not self.fit_results.isVisible():
                self.fit_results.show()
        self.clear_selection()

    def clear_fit(self) -> None:
        """Clear fit overlay artists."""
        targets = self.selected_target_axis()
        self.figure.clear_lines(flags=['fit'], all_panes=targets)
        # update fit results table, so all fit 'show' boxes
        # are unchecked
        if self.fit_results:
            self.fit_results.hide_all_fits()
        self.signals.atrophy.emit()

    def open_fits_results(self) -> None:
        """View the fitting results."""
        if self.fit_results:
            if not self.fit_results.isVisible():
                self.fit_results.show()
        else:
            self.fit_results = fitting_results.FittingResults(self)
            self.fit_results.show()
        self.fit_results.raise_()

    def clear_selection(self) -> None:
        """Reset selection mode."""
        self.clear_guides()
        self.clear_cids('zoom')
        self.clear_cids('fit')
        self.figure.set_cursor_mode('')
        log.debug('Zoom selection cleared')

    def clear_cids(self, target: Optional[str] = None) -> None:
        """
        Disconnect matplotlib event callbacks.

        Parameters
        ----------
        target : str, optional
            Event name associated with the CID to clear. If
            not provided, all CIDs will be cleared.
        """
        cids = list(self.cid.keys())
        if target:
            cids = [i for i in cids if target in i]
        for cid in cids:
            self.figure_widget.canvas.mpl_disconnect(self.cid[cid])
            del self.cid[cid]

    def open_ref_data(self) -> None:
        """View reference line data"""
        if self.reference_window is None:
            self.reference_window = reference_window.ReferenceWindow(self)
            self.reference_window.show()
        else:
            if not self.reference_window.isVisible():
                self.reference_window.show()
        self.reference_window.raise_()

    def update_reference_lines(self) -> None:
        """Update the reference data model."""
        self.figure.update_reference_lines(self.reference_models)

    def unload_reference_model(self) -> None:
        """Unload the reference data model."""
        self.reference_models.unload_data()
        self.figure.unload_reference_model(self.reference_models)

    def toggle_controls(self) -> None:
        """Toggle control panel visibility."""
        if self.control_frame.isVisible():
            log.debug('Hiding controls')
            self.control_frame.hide()
            self.collapse_controls_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_controls_button.setToolTip('Show control panel')
        else:
            log.debug('Showing controls')
            self.control_frame.show()
            self.collapse_controls_button.setArrowType(QtCore.Qt.LeftArrow)
            self.collapse_controls_button.setToolTip('Hide control panel')

    def toggle_cursor(self) -> None:
        """Toggle cursor panel visibility."""
        if self.cursor_frame.isVisible():
            self.cursor_frame.hide()
            self.collapse_cursor_button.setArrowType(QtCore.Qt.UpArrow)
            self.collapse_cursor_button.setToolTip('Show cursor panel')
        else:
            self.cursor_frame.show()
            self.collapse_cursor_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_cursor_button.setToolTip('Hide cursor panel')

    def toggle_file_panel(self) -> None:
        """Toggle file panel visibility."""
        if self.file_choice_panel.isVisible():
            self.file_choice_panel.hide()
            self.collapse_file_choice_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_file_choice_button.setToolTip(
                'Show file choice panel')
        else:
            self.file_choice_panel.show()
            self.collapse_file_choice_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_file_choice_button.setToolTip(
                'Hide file choice panel')

    def toggle_order_panel(self) -> None:
        """Toggle order panel visibility."""
        if self.order_panel.isVisible():
            self.order_panel.hide()
            self.pane_panel.hide()
            self.collapse_order_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_order_button.setToolTip('Show order panel')
        else:
            self.order_panel.show()
            self.pane_panel.show()
            self.collapse_order_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_order_button.setToolTip('Hide order panel')

    def toggle_axis_panel(self) -> None:
        """Toggle axis panel visibility."""
        if self.axis_panel.isVisible():
            self.axis_panel.hide()
            self.collapse_axis_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_axis_button.setToolTip('Show axis panel')
        else:
            self.axis_panel.show()
            self.collapse_axis_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_axis_button.setToolTip('Hide axis panel')

    def toggle_plot_panel(self) -> None:
        """Toggle plot panel visibility."""
        if self.plot_panel.isVisible():
            self.plot_panel.hide()
            self.collapse_plot_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_plot_button.setToolTip('Show plot panel')
        else:
            self.plot_panel.show()
            self.collapse_plot_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_plot_button.setToolTip('Hide plot panel')

    def toggle_analysis_panel(self) -> None:
        """Toggle analysis panel visibility."""
        if self.analysis_panel.isVisible():
            self.analysis_panel.hide()
            self.collapse_analysis_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_analysis_button.setToolTip('Show analysis panel')
        else:
            self.analysis_panel.show()
            self.collapse_analysis_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_analysis_button.setToolTip('Hide analysis panel')

    def print_current_artists(self) -> None:
        """Print current artists."""
        pass

    def _parse_fit_mode(self) -> str:
        """
        Obtain the fitting mode.

        Returns
        -------
        str
            The feature name and baseline.
        """
        feature = str(self.feature_model_selection.currentText()).lower()
        baseline = str(self.background_model_selection.currentText()).lower()
        if feature == 'gaussian':
            feature = 'gauss'
        return f'fit_{feature}_{baseline}'


class ApertureColors(QtWidgets.QWidget):
    """Display aperture colors."""

    def __init__(self, colors: List[str], parent=None):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        seen = list()
        for color in colors:
            if color in seen:
                continue
            else:
                seen.append(color)

            color_item = QtWidgets.QLabel()
            color_item.setText('')
            color_item.setStyleSheet(f"background-color: {color}")

            layout.addWidget(color_item)

        self.setLayout(layout)
        # self.setVisible(False)

    def setTextAlignment(self, *args, **kwargs):  # pragma: no cover
        """Set text alignment."""
        pass
