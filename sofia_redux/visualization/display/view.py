# Licensed under a 3-clause BSD style license - see LICENSE.rst

from contextlib import contextmanager
import os
from typing import (Dict, List, Optional, Sequence,
                    Union, Tuple, TypeVar)

from more_itertools import unique_everseen
import numpy as np
import matplotlib.backend_bases as mbb

from sofia_redux.visualization import log
from sofia_redux.visualization import signals as vs
from sofia_redux.visualization.display import figure, pane, fitting_results
from sofia_redux.visualization.models import high_model
from sofia_redux.visualization.utils.eye_error import EyeError


try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QTreeWidgetItem
    from sofia_redux.visualization.display.ui import simple_spec_viewer as ssv
    from sofia_redux.visualization.display import cursor_location as cl
except ImportError:
    HAS_PYQT5 = False
    QtGui, cl, QTreeWidgetItem = None, None, None
    Event = TypeVar('Error', mbb.MouseEvent, mbb.LocationEvent)

    # duck type parents to allow class definition
    class QtWidgets:
        class QMainWindow:
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
PaneType = TypeVar('PaneType', bound=pane.Pane)


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
        self.model_collection = list()
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
            elif event.key() == QtCore.Qt.Key_Return:
                # enter in the file table displays selected models
                if self.file_table_widget.hasFocus():
                    # send display_model signal
                    self.signals.model_selected.emit()
            elif event.key() == QtCore.Qt.Key_Delete \
                    or event.key() == QtCore.Qt.Key_Backspace:
                if self.file_table_widget.hasFocus():
                    # delete key in the file table removes selected models
                    self.signals.model_removed.emit()
                elif (self.pane_tree_display.hasFocus()
                      or self.figure_widget.hasFocus()):
                    # delete key in the pane table or figure removes
                    # selected pane
                    self.remove_pane()
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
        self.model_collection = list()

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

    def refresh_controls(self) -> None:
        """Refresh all control widgets."""
        self.refresh_file_table()
        self.update_pane_tree()
        self.refresh_order_list()
        self.update_controls()

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
    def toggle_pane_highlight(self) -> None:
        """Toggle the pane border visibility."""
        state = self.hightlight_pane_checkbox.isChecked()
        self.figure.set_pane_highlight_flag(state)
        self.signals.atrophy.emit()

    @QtCore.pyqtSlot()
    def current_pane_changed(self) -> None:
        """Set a new pane as the active pane."""
        self.figure.set_pane_highlight_flag(
            state=self.hightlight_pane_checkbox.isChecked())
        self.clear_selection()
        self.update_controls()
        self.refresh_order_list()
        self.signals.atrophy.emit()

    @QtCore.pyqtSlot()
    def refresh_orders(self) -> None:
        """Refresh the list of displayed orders."""
        self.refresh_order_list()

    @QtCore.pyqtSlot()
    def axis_limits_changed(self) -> None:
        """Change axis limits."""
        try:
            limits = self._pull_limits_from_gui()
        except ValueError:
            log.debug('Illegal limits entered')
        else:
            # if self.all_axes_button.isChecked():
            #     panes = 'all'
            # else:
            #     panes = None
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
        # if self.all_axes_button.isChecked():
        #     panes = 'all'
        # else:
        #     panes = None
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
        # if self.all_axes_button.isChecked():
        #     panes = 'all'
        # else:
        #     panes = None
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
        # if self.all_axes_button.isChecked():
        #     panes = 'all'
        # else:
        #     panes = None
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
        fields = {'x': str(self.x_property_selector.currentText()).lower(),
                  'y': str(self.y_property_selector.currentText()).lower()}
        return fields

    @QtCore.pyqtSlot()
    def current_cursor_location(self, event: Event) -> None:
        """Update the cursor location displays."""
        if self.cursor_checkbox.isChecked():
            if event.inaxes and self.figure.populated():
                data_point = self.figure.data_at_cursor(event)
                idx = self.figure.determine_selected_pane(event.inaxes)
                cursor_position = self.figure.panes[idx].xy_at_cursor(event)
                if self._cursor_popout:
                    self._update_cursor_loc_window(data_point, cursor_position)
                else:
                    self._update_cursor_loc_labels(data_point, cursor_position)
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
            if pane_index is not None:
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

    ####
    # Setup
    ####
    def setup_property_selectors(self, pane_: Optional[PaneType] = None,
                                 target: Optional[Dict] = None) -> None:
        """
        Set up plot field selectors based on loaded data.

        Parameters
        ----------
        pane_ : pane.Pane, optional
            The pane containing loaded data. If not
            provided, apply to the current pane.
        """
        log.debug(f'Updating property selector for pane {pane_}')
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
        if pane_ is None:
            pane_ = self.figure.get_current_pane()
        if pane_ is None:
            # No panes present
            log.debug('No panes present, clearing property selectors')
        else:
            fields = pane_.fields
            if pane_.models:
                if pane_.plot_kind == 'spectrum':
                    order = list(pane_.models.values())[0].retrieve(
                        order=0, level='high')
                    fields = order.data.keys()
                elif pane_.plot_kind == 'image':  # pragma: no cover
                    fields = list(pane_.models.values())[0].images.keys()
            else:
                fields = ['-']

            fields = list(unique_everseen(fields))
            for selector in selectors.values():
                selector.addItems(fields)

            for ax, selector in selectors.items():
                index = selector.findText(pane_.fields[ax])
                if index > 0:
                    selector.setCurrentIndex(index)

    def setup_unit_selectors(self, pane_: Optional[PaneType] = None,
                             target: Optional[Dict] = None) -> None:
        """
        Set up plot unit selectors based on loaded data.

        Parameters
        ----------
        pane_ : pane.Pane, optional
            The pane containing loaded data. If not
            provided, apply to the current pane.
        """
        log.debug(f'Updating unit selectors for {pane_}.')
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass
        unit_selectors = {'x': self.x_unit_selector,
                          y_label: self.y_unit_selector}
        if pane_ is None:
            pane_ = self.figure.get_current_pane()
        if pane_ is None:
            log.debug('No panes currently exist')
        else:
            units = pane_.possible_units()
            current_selection = pane_.current_units()
            for axis, selector in unit_selectors.items():
                selector.clear()
                string_units = [str(i) for i in units[axis]]
                string_units = list(unique_everseen(string_units))
                selector.addItems(string_units)
                if current_selection[axis] in string_units:
                    idx = string_units.index(current_selection[axis])
                else:
                    idx = 0
                selector.setCurrentIndex(idx)
                log.debug(f'Set unit selectors to {string_units}, '
                          f'selected {idx}')

    def setup_axis_limits(self, pane_: Optional[PaneType] = None,
                          target: Optional[Dict] = None) -> None:
        """
        Set up axis limit text boxes based on loaded data.

        Parameters
        ----------
        pane_ : pane.Pane, optional
            The pane containing loaded data. If not provided
            apply to the current pane.
        """
        log.debug(f'Updating axis limits for {pane_}.')
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass
        if pane_ is None:
            pane_ = self.figure.get_current_pane()
        if pane_ is None:
            log.debug('No pane currently present')
        else:
            limits = pane_.get_axis_limits()
            log.debug(f'New limits: {limits}.')
            limit_displays = {'x': [self.x_limit_min, self.x_limit_max],
                              y_label: [self.y_limit_min, self.y_limit_max]}
            for ax, limit_display in limit_displays.items():
                if limits[ax]:
                    lim = [f'{val:.4g}' for val in limits[ax]]
                else:
                    lim = ['', '']
                for display, value in zip(limit_display, lim):
                    display.setText(value)

    def setup_initial_scales(self, pane_: Optional[PaneType] = None,
                             target: Optional[Dict] = None) -> None:
        """
        Set up plot scale selectors based on loaded data.

        Parameters
        ----------
        pane_ : pane.Pane, optional
            The pane containing loaded data. If not provided,
            apply to the current pane.
        """
        y_label = 'y'
        try:
            if target['axis'] == 'alt':
                y_label = 'y_alt'
        except (KeyError, AttributeError, TypeError):
            pass

        if pane_ is None:
            pane_ = self.figure.get_current_pane()
        if pane_ is None:
            log.debug('No pane currently present')
        else:
            self.buttonGroup.setExclusive(False)
            self.buttonGroup_2.setExclusive(False)
            self._block_scale_signals(True)
            scale_displays = {'x': [self.x_scale_linear_button,
                                    self.x_scale_log_button],
                              y_label: [self.y_scale_linear_button,
                                        self.y_scale_log_button]}
            scale = pane_.get_axis_scale()
            for axis, displays in scale_displays.items():
                if scale[axis] == 'linear' or scale[axis] is None:
                    displays[0].setChecked(True)
                    displays[1].setChecked(False)
                elif scale[axis] == 'log':
                    displays[0].setChecked(False)
                    displays[1].setChecked(True)

            self.buttonGroup.setExclusive(True)
            self.buttonGroup_2.setExclusive(True)
            self._block_scale_signals(False)

    def setup_overplot_flag(self, pane_: Optional[PaneType] = None,
                            target: Optional[Dict] = None) -> None:
        if pane_ is None:
            pane_ = self.figure.get_current_pane()
        if pane_ is None:
            log.debug('No pane currently present')
        else:
            state = pane_.overplot_state()
            self.enable_overplot_checkbox.setChecked(state)

    def display_filenames(self, filename: str) -> None:
        """
        Add a file that has been read in to the model collection.

        Logs a warning if file is already in the loaded list.

        Parameters
        ----------
        filename : str
            Name of the file that was added to models.
        """
        if filename not in self.model_collection:
            self.model_collection.append(filename)
        else:
            log.warning(f'Filename {filename} is already in '
                        f'display model list.')

    def remove_filename(self, filename: str) -> None:
        """
        Remove a file from the list.

        Parameters
        ----------
        filename : str
            Name of the model that is no longer being stored.
        """
        try:
            self.model_collection.remove(filename)
        except ValueError:
            # could happen in race conditions
            pass

    def select_color_cycle(self, text: str) -> None:
        """
        Select the color cycle for the plot.

        Parameters
        ----------
        text : ['spectral', 'tableau', 'accessible']
            Color cycle to set.
        """
        self.figure.set_color_cycle(text)

    def select_plot_type(self, text: str) -> None:
        """
        Select the plot type.

        Parameters
        ----------
        text : ['line', 'step', 'scatter']
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
        self.file_table_widget.setRowCount(len(self.model_collection))
        for row_index, filename in enumerate(self.model_collection):
            # display base name
            item = QtWidgets.QTableWidgetItem(os.path.basename(filename))
            # store full name in data and set as tooltip
            item.setData(QtCore.Qt.UserRole, filename)
            item.setToolTip(filename)
            self.file_table_widget.setItem(row_index, col_index, item)
        self.file_table_widget.resizeRowsToContents()
        self.file_table_widget.resizeColumnsToContents()
        self.file_table_widget.show()

    def _clear_file_table(self) -> None:
        """Clear the file table widget."""
        self.file_table_widget.setRowCount(0)
        self.file_table_widget.setColumnCount(1)

    def current_files_selected(self) -> Union[List[str], None]:
        """
        Retrieve all files currently selected in the file list.

        Returns
        -------
        filenames : list of str
            The selected filenames.
        """
        items = self.file_table_widget.selectedItems()
        if not items:
            return
        filenames = [item.data(QtCore.Qt.UserRole) for item in items]
        return filenames

    ####
    # Panes
    ####
    def add_pane(self) -> None:
        """Add a new pane."""
        self.figure.add_panes(n_dims=0, n_panes=1)
        self.signals.atrophy.emit()
        self.update_pane_tree()
        self.signals.current_pane_changed.emit()

    def add_panes(self, n_panes: int, kind: List[str],
                  layout: Optional[str] = 'grid') -> None:
        """
        Add new plot panes.

        Parameters
        ----------
        n_panes : int
            The number of panes to add.
        kind : ['spectrum', 'onedim']
            The kind of pane to add.
        layout : ['grid', 'rows', 'columns']
            The layout method for the new panes.
        """
        n_dims = list()
        # TODO: What if kind is a one element list meant to be a blanket?
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
        """Remove a selected pane."""
        items = self.pane_tree_display.selectedItems()
        if not items:
            return
        pane_number = [item.data(0, QtCore.Qt.UserRole) for item in items]
        self.figure.remove_pane(pane_number)
        self.signals.atrophy.emit()

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

    def update_pane_tree(self) -> None:
        """Update the list of panes and contained models."""
        # get current expansion state and show/hide buttons to restore
        expanded = {}
        buttons = {}
        root = self.pane_tree_display.invisibleRootItem()
        n_panes = root.childCount()
        for i in range(n_panes):
            pane_item = root.child(i)
            expanded[i] = {'pane': pane_item.isExpanded()}
            n_models = pane_item.childCount()
            for j in range(n_models):
                model_item = pane_item.child(j)
                button = self.pane_tree_display.itemWidget(model_item, 0)
                if button is not None:
                    # button row: store current hide/show state
                    row_data = button.property('id')
                    buttons[i] = row_data[1]
                else:
                    try:
                        pane_id, model_id = \
                            model_item.child(1).data(0, QtCore.Qt.UserRole)
                    except (AttributeError, TypeError, IndexError,
                            ValueError):  # pragma: no cover
                        pass
                    else:
                        # model row
                        expanded[i][model_id] = model_item.isExpanded()

        details = self.figure.pane_details()
        self.pane_tree_display.clear()
        root = self.pane_tree_display.invisibleRootItem()
        for label, model_list in details.items():
            # pane number
            pane_id = int(label.split('_')[-1])
            pane_label = f"Pane {pane_id + 1}"
            pane_item = QtWidgets.QTreeWidgetItem([pane_label])
            pane_item.setData(0, QtCore.Qt.UserRole, pane_id)
            root.addChild(pane_item)
            if pane_id in expanded:
                pane_item.setExpanded(expanded[pane_id]['pane'])

            if pane_id == self.figure.current_pane:
                pane_item.setSelected(True)

            for detail in model_list.values():
                # filename
                model_id = detail['model_id']
                model_item = QtWidgets.QTreeWidgetItem([detail['filename']])
                pane_item.addChild(model_item)
                if pane_id in expanded \
                        and model_id in expanded[pane_id]:
                    model_item.setExpanded(expanded[pane_id][model_id])

                # extension name
                item = QtWidgets.QTreeWidgetItem([detail['extension']])
                model_item.addChild(item)

                # enabled check box
                item = QtWidgets.QTreeWidgetItem(['Enabled'])
                if detail['enabled']:
                    item.setCheckState(0, QtCore.Qt.Checked)
                else:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                # store pane and model ids as user data, for retrieval
                # when box is checked/unchecked
                item.setData(0, QtCore.Qt.UserRole, (pane_id, model_id))
                model_item.addChild(item)

                # color, marker for plot
                item = QtWidgets.QTreeWidgetItem(
                    [f"Color, Marker: {detail['marker']}"])
                color = QtGui.QColor(detail['color'])
                pix = QtGui.QPixmap(50, 50)
                pix.fill(color)
                icon = QtGui.QIcon(pix)
                item.setIcon(0, icon)
                model_item.addChild(item)

            # add a button at the end to dis/enable all models
            item = QtWidgets.QTreeWidgetItem()
            pane_item.addChild(item)
            button = QtWidgets.QPushButton()
            if pane_id in buttons:
                state = bool(buttons[pane_id])
            else:
                state = False
            if state:  # pragma: no cover
                button.setText('Show all')
            else:
                button.setText('Hide all')
            button.clicked.connect(self.enable_all_models)
            button.setProperty('id', (pane_id, state))
            self.pane_tree_display.setItemWidget(item, 0, button)

    def remove_data_from_all_panes(self, filename: str) -> None:
        """
        Remove loaded data from the view.

        Parameters
        ----------
        filename : str
            The file to remove.
        """
        self.figure.remove_model_from_pane(filename)
        self.remove_filename(filename)
        # clear any active selection states
        self.clear_selection()

    def display_model(self, model: MT,
                      pane_: Optional[PaneType] = None) -> None:
        """
        Display a model in a plot pane.

        Parameters
        ----------
        model : high_model.HighModel
            The model to display.
        pane_ : pane.Pane
            The pane to display to.  If not provided, the current
            pane will be used.
        """
        self.figure.add_model_to_pane(model, pane_)
        # clear any active selection states
        self.clear_selection()
        self.signals.atrophy_bg_full.emit()
        self.update_controls()

    def assign_models(self, mode: str, models: Dict[str, MT],
                      indices: List[int]) -> None:
        """
        Assign models to panes.

        Parameters
        ----------
        mode : ['split', 'first', 'last', assigned']
            Specifies how to arrange the models on the panes.
            'Split' divides the models as evenly as possible
            across all present panes. 'First' assigns all the
            models to the first pane, while 'last' assigns all
            the models to the last pane. 'Assigned' attaches
            each model to the pane index provided in `indices`.
        models : dict
            Dictionary of models to add. Keys are the model ID,
            with the values being the models themselves.
        indices : list of int, optional
            A list of integers with the same length of `models`.
            Only used for `assigned` mode. Specifies the index
            of the desired pane for the model.

        Raises
        ------
        RuntimeError :
            If an invalid mode is provided.
        """
        self.figure.assign_models(mode=mode, models=models,
                                  indices=indices)

    def models_per_pane(self) -> List[int]:
        """
        Retrieve the number of models in each pane.

        Returns
        -------
        count : list of int
            The number of models in each existing pane.
        """
        return self.figure.models_per_pane()

    def set_current_pane(self, pane_index: int) -> None:
        """
        Set the current pane.

        Parameters
        ----------
        pane_index : int
            The pane to activate.
        """
        pane_change = self.figure.change_current_pane(pane_index)
        if pane_change:
            log.debug(f'Setting current pane to {pane_index}')
            self.signals.atrophy.emit()

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
        Parses the state of the axis selection widget

        Returns
        -------
        target : dict

        """
        text = str(self.axes_selector.currentText()).lower()
        target = dict()
        if text == 'all':
            target['pane'] = 'all'
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
            if 'current' in text:
                target['pane'] = 'current'
            elif 'all' in text:
                target['pane'] = 'all'
            else:
                raise EyeError(f'Unknown target axis selected: {text}')
        return target

    ####
    # Orders
    ####
    def refresh_order_list(self) -> None:
        """Refresh the order list from loaded models."""
        self.order_list_widget.clear()
        if self.figure.pane_count() > 0:
            pane_ = self.figure.get_current_pane()
            orders = pane_.get_orders()
            field = pane_.get_field('y')
            for order in orders:
                details = order.describe()
                item = QtWidgets.QListWidgetItem(details['name'])
                if field not in details['fields']:
                    continue
                if details['fields'][field]:
                    item.setCheckState(QtCore.Qt.Checked)
                else:
                    item.setCheckState(QtCore.Qt.Unchecked)
                self.order_list_widget.addItem(item)

    def add_order(self) -> None:
        """Add a new order."""
        raise NotImplementedError

    def remove_order(self) -> None:
        """Remove an order."""
        raise NotImplementedError

    ####
    # Axis Controls
    ####
    def update_controls(self) -> None:
        """Update control widgets from loaded data."""
        pane_ = self.figure.get_current_pane()
        target = self.selected_target_axis()
        self.setup_unit_selectors(pane_=pane_, target=target)
        self.setup_property_selectors(pane_=pane_, target=target)
        self.setup_axis_limits(pane_=pane_, target=target)
        self.setup_initial_scales(pane_=pane_, target=target)
        self.setup_overplot_flag(pane_=pane_, target=target)

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
        """Set axis limits from the widget contols."""
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
        # all_panes = self.all_axes_button.isChecked()
        targets = self.selected_target_axis()
        self.figure.clear_lines(flags='a',
                                all_panes=targets)
        self.signals.atrophy.emit()

    def reset_zoom(self) -> None:
        """Reset axis limits to full range."""
        self.clear_selection()
        # all_panes = self.all_axes_button.isChecked()
        targets = self.selected_target_axis()
        self.figure.reset_zoom(targets)
        self.signals.atrophy_bg_partial.emit()
        log.info('Zoom reset')

    def start_selection(self, mode: str) -> None:
        """
        Start a user range selection, for zoom or fit.

        Parameters
        ----------
        mode : ['x_zoom', 'y_zoom', 'b_zoom', 'fit']
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
        # all_panes = self.all_axes_button.isChecked()
        targets = self.selected_target_axis()
        self.figure.clear_lines(flags=['fit'],
                                all_panes=targets)
        # update fit results table, so all fit 'show' boxes
        # are unchecked
        if self.fit_results:
            self.fit_results.hide_all_fits()
        self.signals.atrophy.emit()

    def open_fits_results(self) -> None:
        if self.fit_results:
            if not self.fit_results.isVisible():
                self.fit_results.show()
        else:
            self.fit_results = fitting_results.FittingResults(self)
            self.fit_results.show()

    def clear_selection(self) -> None:
        """Reset selection mode."""
        self.clear_guides()
        self.clear_cids('zoom')
        self.clear_cids('fit')
        self.figure.set_cursor_mode('')
        log.info('Zoom selection cleared')

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

    def toggle_pane_panel(self) -> None:
        """Toggle pane panel visibility."""
        if self.pane_panel.isVisible():
            self.pane_panel.hide()
            self.collapse_pane_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_pane_button.setToolTip('Show pane panel')
        else:
            self.pane_panel.show()
            self.collapse_pane_button.setArrowType(QtCore.Qt.DownArrow)
            self.collapse_pane_button.setToolTip('Hide pane panel')

    def toggle_order_panel(self) -> None:
        """Toggle order panel visibility."""
        if self.order_panel.isVisible():
            self.order_panel.hide()
            self.collapse_order_button.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_order_button.setToolTip('Show order panel')
        else:
            self.order_panel.show()
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
        pass

    def _parse_fit_mode(self) -> str:
        feature = str(self.feature_model_selection.currentText()).lower()
        baseline = str(self.background_model_selection.currentText()).lower()
        if feature == 'gaussian':
            feature = 'gauss'
        return f'fit_{feature}_{baseline}'
