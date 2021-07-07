#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import csv
from typing import (Dict, List, Union, Optional,
                    Any, TypeVar)

import matplotlib
import matplotlib.axes as ma
import astropy.units as u
import numpy as np

from sofia_redux.visualization.utils import unit_conversion as uc
from sofia_redux.visualization.utils import model_fit
from sofia_redux.visualization import log
from sofia_redux.visualization.display import pane

try:
    matplotlib.use('QT5Agg')
    matplotlib.rcParams['axes.formatter.useoffset'] = False
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    HAS_PYQT5 = False
    QtCore, QtGui = None, None

    # duck type parents to allow class definition
    class QtWidgets:
        class QDialog:
            pass

    class frw:
        class Ui_Dialog:
            pass
else:
    from PyQt5.QtWidgets import QTableWidgetItem
    from sofia_redux.visualization.display.ui import fit_result_window as frw
    HAS_PYQT5 = True

__all__ = ['FittingResults']

Axes = TypeVar('Axes', ma.Axes, None)
PT = TypeVar('PT', bound=pane.Pane)
UT = TypeVar('UT', u.Quantity, u.Unit)


class FittingResults(QtWidgets.QDialog, frw.Ui_Dialog):
    """
    Fitting results display widget.

    After a curve has been fit to data in the Eye, a
    ``FittingResults`` window is opened to display the
    parameters of the fit and all subsequent fits. The
    window can also be used to remove the fit curves
    from the plots in the Eye.

    Parameters
    ----------
    parent : View
        Parent view widget for the dialog window.

    Attributes
    ----------
    model_fits : list
        Feature fit parameters to display in the widget.
    signals : sofia_redux.visualization.signals.Signals
        Custom signals to pass events to the controller, as needed.
    table_header : list
        Column titles for table.
    canvas :
        Matplotlib canvas for the plot of the most recent fit.
    fig : matplotlib.figure.Figure
        Figure to plot the most recent fit
    ax : matplotlib.axes.Axes
        Axes for the most recent fit

    """

    def __init__(self, parent: Any):
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')

        super(self.__class__, self).__init__(parent)
        self.setupUi(self)
        self.setModal(0)
        self.model_fits = list()
        self.signals = parent.signals
        self.table_header = list()

        self.canvas = self.last_fit_widget.canvas
        self.fig = self.last_fit_widget.canvas.fig
        self.ax = self.fig.add_subplot()
        self.fit_color = '#2848ad'
        self.splitter.setStretchFactor(1, 1)

        self.save_button.clicked.connect(self.save_results)
        self.close_button.clicked.connect(self.close)
        self.clear_button.clicked.connect(self.clear_fit)

    def clear_fit(self) -> None:
        """Clear the table widget display and reset fits stored."""
        # clear table
        self.model_fits = list()
        self._define_table()
        self._clear_figure()
        # will clear the plot in the Eye
        self.signals.clear_fit.emit()

    def add_results(self, fit_params: List[model_fit.ModelFit]) -> None:
        """
        Add fit results to the table display.

        Parameters
        ----------
        fit_params : dict
            Primary keys are model IDs, values are dictionaries
            with order keys.  Values for each order are also dictionaries,
            with keys 'fit', 'x_field', 'y_field', 'x_unit',
            'y_unit', 'lower_limit', 'upper_limit', and 'baseline'.
]        """
        if not isinstance(fit_params, list):
            fit_params = [fit_params]
        for fit in fit_params:
            if fit.get_status() == 'pass':
                self.model_fits.append(fit)
        self._update_table()
        self._update_figure(fit_params)

    def _update_table(self):
        """Reset the contents of the table to reflect all loaded fits"""
        self.table_widget.clearContents()
        self._define_table()
        for row_index, model in enumerate(self.model_fits):
            checkbox = QtWidgets.QCheckBox(parent=self.table_widget)
            checkbox.setChecked(model.get_visibility())
            checkbox.clicked.connect(self.signals.toggle_fit_visibility)
            self.table_widget.setCellWidget(row_index, 0, checkbox)

            details = model.parameters_as_string()
            model_id = details['model_id']
            vhead = QTableWidgetItem(os.path.basename(model_id))
            vhead.setTextAlignment(QtCore.Qt.AlignLeft)
            vhead.setToolTip(model_id)
            self.table_widget.setVerticalHeaderItem(row_index, vhead)

            for col_index, col_name in enumerate(self.table_header[1:]):
                try:
                    text = details[col_name]
                except KeyError:
                    text = 'NA'
                item = QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignHCenter)
                self.table_widget.setItem(row_index, col_index + 1, item)

        # resize rows and columns to contents
        self._resize()
        # scroll to bottom to show latest update
        self.table_widget.scrollToBottom()

    def _resize(self) -> None:
        """Resize the table to current contents."""
        self.table_widget.resizeColumnsToContents()
        self.table_widget.resizeRowsToContents()

    def _define_table(self) -> None:
        """Define the table widget row count."""
        # don't redefine column count, set row count to number
        # of parameter items (filenames/orders within the
        # self.parameters list of dicts)
        columns = {'generic': ['show', 'order', 'x_field', 'y_field', 'type'],
                   'gauss': ['mid_point', 'fwhm', 'amplitude'],
                   'moffat': ['mid_point', 'fwhm', 'amplitude'],
                   'linear': ['mid_point', 'baseline', 'slope'],
                   'constant': ['baseline']}
        fit_types = set()
        for fit in self.model_fits:
            fit_types.update(fit.get_fit_types())
        self.table_header = columns['generic']
        for fit_type in fit_types:
            for col in columns[fit_type]:
                if col not in self.table_header:
                    self.table_header.append(col)
        self.table_widget.setColumnCount(len(self.table_header))
        self.table_widget.setHorizontalHeaderLabels(
            [self._labelize(s) for s in self.table_header])
        self.table_widget.setRowCount(len(self.model_fits))

    @staticmethod
    def _labelize(value: str) -> str:
        """Format a keyword to appropriate style for display."""
        if value == 'fwhm':
            value = 'FWHM'
        else:
            value = value.strip().replace('_', ' ').title()
        return value

    def _clear_figure(self):
        """Remove all information from the last fit"""
        self.ax.clear()
        self.canvas.draw_idle()
        self.last_fit_values.setText('')

    def _update_figure(self, fits: List[model_fit.ModelFit]):
        """
        Plot a model fit on the last fit plot

        Parameters
        ----------
        fit : list of model_fit.ModelFit
            The fits to plot

        """
        # clear old lines
        self.ax.clear()

        html = []
        for fit in fits:
            status = fit.get_status()

            # don't bother with completely empty data sets
            if 'empty' in status.lower():
                continue

            # Plot the data fit was made to
            dataset = fit.get_dataset()
            if all([v is not None for v in dataset.values()]):
                self.ax.step(dataset['x'], dataset['y'], where='mid',
                             color=self.fit_color, alpha=0.9)

            # If the fit was successful, plot the fit
            if status == 'pass':
                # model
                x = np.linspace(fit.get_limits('lower'),
                                fit.get_limits('upper'), 100)
                y = fit.get_fit()(x)
                self.ax.plot(x, y, color='gray', linestyle='dashed',
                             alpha=0.8)
                # mid point
                self.ax.axvline(fit.get_mid_point(), color='gray',
                                linestyle='dotted', alpha=0.6)

            html.append(fit.parameters_as_html())

        # show all the parameters
        self.last_fit_values.setHtml('<br>'.join(html))

        # Configure the plot
        self.ax.set_xlabel(fit.get_fields('x'))
        self.ax.set_ylabel(fit.get_fields('y'))
        self.ax.set_title('Last Fit Feature')
        self.canvas.draw_idle()

    def save_results(self) -> None:
        """Save current table to a CSV file."""
        # Get save filename
        filename = self._get_save_filename()
        if not filename:
            return
        # Get the fit models to save based on selections
        parameters = self._selected_parameters()
        # Write to file
        self._write_parameters(parameters, filename)

    def _selected_parameters(self) -> List[List]:
        """
        Retrieve selected parameters from the table.

        Returns
        -------
        parameters : list of list
            Fit parameters flattened into a single list, where
            each element is a list of row values.
        """
        selected = list()
        flat_params = self.format_parameters(kind='string')
        for index in self.table_widget.selectionModel().selectedRows():
            selected.append(flat_params[index.row()])
        if selected:
            return selected
        else:
            return flat_params

    def gather_models(self) -> List[model_fit.ModelFit]:
        """Return all loaded models with updated visibility"""
        for row_index in range(self.table_widget.rowCount()):
            checkbox = self.table_widget.cellWidget(row_index, 0)
            try:
                fit = self.model_fits[row_index]
            except IndexError:
                continue
            fit.set_visibility(checkbox.isChecked())
        return self.model_fits

    def _get_save_filename(self) -> str:
        """
        Query user for output filename.

        Returns
        -------
        filename : str
            The output path to save to.
        """
        init_directory = os.path.expanduser('~')
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File',
                                                         init_directory)[0]
        return filename

    @staticmethod
    def _write_parameters(parameters: List[List[Union[str, float]]],
                          filename: str) -> None:
        """
        Write fit parameter rows to a CSV file.

        Parameters
        ----------
        parameters : list of list
            Fit parameters flattened into a single list, where
            each element is a list of row values.
        filename : str
            Output file path.
        """
        header = list(parameters[0].keys())
        to_remove = ['axis', 'visible']
        for key in to_remove:
            try:
                header.remove(key)
            except ValueError:  # pragma: no cover
                # this shouldn't be hit, since these keys are always
                # in the parameters as passed
                continue
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for param in parameters:
                row = list()
                for key in header:
                    try:
                        value = param[key]
                    except KeyError:
                        value = 'NA'
                    row.append(value)
                writer.writerow(row)

    def format_parameters(self, kind: str) -> List[Any]:
        """
        Format the fit parameters for all loaded fits.

        Parameters
        ----------
        kind : 'string', 'dict', 'list', 'html'
            Specify the desired structure for the parameters
            to be returned as. String formats all values for
            proper display in the table. Dict compiles
            the parameters for returns to other functions.
            Html returns the string format, additionally
            formatted for HTML display in the text view.

        Returns
        -------
        params : list
            List of formatted fit parameters.

        """
        params = list()
        for fit in self.model_fits:
            if kind == 'string':
                param = fit.parameters_as_string()
            elif kind == 'dict':
                param = fit.parameters_as_dict()
            elif kind == 'list':
                param = fit.parameters_as_list()
            elif kind == 'html':
                param = fit.parameters_as_html()
            else:
                log.debug(f'Unknown format for output '
                          f'parameters: {kind}')
                continue
            params.append(param)
        return params

    def change_units(self, units: Dict[str, str],
                     panes: List[PT] = None,
                     return_new: Optional[bool] = False
                     ) -> Optional[List[model_fit.ModelFit]]:
        """
        Change the units of fits.

        The 'axes' parameter of each model fit loaded in model_fits is
        compared to the 'axes' of each Pane in `panes` to verify the
        model should be updated. The fit along with the current and
        desired units are passed to ``uc.convert_model_fit`` to perform
        the actual conversion. If the conversion fails (more common with
        changing flux units) then the model fit is left unchanged, but is
        hidden so it will not affect the autoscaling of the new Pane.
        The table widget is updated with the new units or visibility
        settings at the end.

        Parameters
        ----------
        units : dict
            The new units to apply. Keys are the axis directions 'x', 'y'.
            Values are the string representations of the new unit,
            like 'um' or 'Jy'.
        panes : list, optional
            List of Pane objects to apply the fit changes to. If
            not provided, all fits will be updated.
        return_new : bool, optional
            If true, return the new fit parameters. Defaults to False.

        Returns
        -------
        parameters: dict
            Details of the new fit parameters.

        """
        for model in self.model_fits:
            # if panes are provided, check for match to fit
            # and take end units from the pane, since actual pane
            # units might not match desired end units (eg. for
            # multiple panes with incompatible flux units)
            if panes is not None:
                matching_pane = False
                for p in panes:
                    if p is not None:
                        if model.axis in p.axes():
                            matching_pane = True
                            ending = {'x': p.units['x'],
                                      'y': p.units['y']}
                            break
                # no matching pane for this fit, skip it
                if not matching_pane:
                    continue
            else:
                ending = units

            starting = model.get_units()
            fit = model.get_fit()

            wave = model.get_mid_point()
            try:
                uc.convert_model_fit(fit, starting, ending, wave)
            except (ValueError, AttributeError) as e:
                log.debug(e)
                self.hide_fit(model)
                continue

            model.set_fit(fit)
            model.set_units(ending)
            lower_limit = uc.convert_wave(model.get_limits('lower'),
                                          starting['x'], ending['x'])
            upper_limit = uc.convert_wave(model.get_limits('upper'),
                                          starting['x'], ending['x'])
            model.set_limits({'lower': lower_limit, 'upper': upper_limit})

        self._update_table()
        if return_new:
            return self.gather_models()

    @staticmethod
    def hide_fit(fit: model_fit.ModelFit) -> None:
        """Set the fit's visibility to False"""
        fit.set_visibility(False)

    def hide_all_fits(self) -> None:
        """Set all fit visibilities to False"""
        for fit in self.model_fits:
            fit.set_visibility(False)
        self._update_table()
