# Licensed under a 3-clause BSD style license - see LICENSE.rst

import colorsys
import copy
import itertools
import os
import re
from typing import (Dict, List, Union, TypeVar,
                    Optional, Tuple, Any, Sequence)
import warnings

from astropy import modeling as am
import numpy as np
import matplotlib
from matplotlib import axes as ma
from matplotlib import backend_bases as mbb
from matplotlib import collections as mcl
from matplotlib import colors as mc
from matplotlib import lines as ml
from matplotlib import patches as mp
import scipy.optimize as sco

from sofia_redux.visualization import log
from sofia_redux.visualization.models import high_model
from sofia_redux.visualization.utils.eye_error import EyeError
from sofia_redux.visualization.utils import model_fit

try:
    matplotlib.use('QT5Agg')
    matplotlib.rcParams['axes.formatter.useoffset'] = False
except ImportError:
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True

__all__ = ['Pane', 'OneDimPane', 'TwoDimPane']

MT = TypeVar('MT', bound=high_model.HighModel)
Num = TypeVar('Num', int, float)
ArrayLike = TypeVar('ArrayLike', List, Tuple, Sequence, np.ndarray)


class Pane(object):
    """
    Plot window management.

    The Pane class is analogous to a matplotlib subplot. It
    contains plot axes and instantiates artists associated with
    them.  This class determines appropriate updates for display
    options, but it does not manage updating artists themselves
    after they are instantiated.  The `Artists` class manages all
    artist modifications.

    The Pane class is abstract.  It should not be instantiated directly:
    it should be subclassed to provide specific display functionality,
    depending on desired plot type.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Plot axes to display in the pane.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        Plot axes to display in the pane.
    models : dict
        Data models displayed in the pane.  Keys are model IDs;
        values are sofia_redux.visualization.models.HighModel instances.
    fields : dict
        Model fields currently displayed in the pane.
    brewer_cycle : list of str
        Color hex values for the 'spectral' color cycle.
    tab10_cycle : list of str
        Color hex values for the 'tableau' color cycle.
    accessible_cycle : list of str
        Color hex values for the 'accessible' color cycle.
    default_colors : list of str
        Default color cycle.  Set to `accessible_cycle` on initialization.
    default_markers : list of str
        Default plot marker cycle.
    fit_linestyles : iterable
        Line style cycle, for use in feature fit overlays.
    border : matplotlib.Rectangle
        Border artist, used to highlight the current pane in the
        view display.
    """

    def __init__(self, ax: Optional[ma.Axes] = None) -> None:
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')

        self.ax = ax
        self.ax_alt = None
        self.models = dict()
        self.fields = dict()
        self.plot_kind = ''
        self.show_overplot = False

        # xvspec default:
        # 9 selections from Brewer 10-color diverging Spectral
        # http://colorbrewer2.org/#type=diverging&scheme=Spectral&n=10
        self.brewer_cycle = [
            '#66c2a5', '#3288bd', '#5e4fa2', '#9e0142',
            '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#abdda4']

        # matplotlib default: plt.cycler("color", plt.cm.tab10.colors)
        self.tab10_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                            '#bcbd22', '#17becf']

        # 7 visually separated colors picked from
        # https://colorcyclepicker.mpetroff.net/
        self.accessible_cycle = ['#2848ad', '#59d4db', '#f37738',
                                 '#c6ea6c', '#b9a496', '#d0196b', '#7b85d4']

        # default is accessible
        self.default_colors = self.accessible_cycle

        self.default_markers = ['x', 'o', '^', '+', 'v', '*']
        self.fit_linestyles = itertools.cycle(['dashed',
                                               'dotted', 'dashdot'])
        self.border = None

    def axes(self):
        return [self.ax, self.ax_alt]

    def set_axis(self, ax: ma.Axes, kind: str = 'primary') -> None:
        """
        Assign an axes instance to the pane.

        Also instantiates the border highlight for the pane.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to assign.
        kind : str, optional
            If given, specifies which axis ('primary' or 'secondary')
            is being given. Defaults to 'primary'.
        """
        if kind == 'primary':
            self.ax = ax
            self.border = self._add_border_highlight()
        else:
            self.ax_alt = ax

    def overplot_state(self) -> bool:
        return self.show_overplot

    def model_count(self) -> int:
        """
        Return number of models attached to the pane.

        Returns
        -------
        count : int
            The model count.
        """
        return len(self.models)

    def add_model(self, model: MT) -> None:
        """
        Add a model to the pane.

        Parameters
        ----------
        model : high_model.HighModel
            The model to add.
        """
        raise NotImplementedError

    def remove_model(self, filename: Optional[str] = None,
                     model: Optional[MT] = None) -> None:
        """
        Remove a model from the pane.

        Either filename or model must be specified.

        Parameters
        ----------
        filename : str, optional
            Model filename to remove.
        model: high_model.HighModel, optional
            Model instance to remove.
        """
        raise NotImplementedError

    def possible_units(self) -> Dict[str, List[Any]]:
        """
        Retrieve possible units for the model.

        Returns
        -------
        units : dict
            Keys are axis names; values are lists of unit names.
        """
        raise NotImplementedError

    def current_units(self) -> Dict[str, str]:
        """
        Retrieve the currently displayed units.

        Returns
        -------
        units : dict
            Keys are axis names; values are unit names.
        """
        raise NotImplementedError

    def set_orders(self, orders: Dict[str, List[Any]]) -> None:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are model IDs; values are lists of orders
            to enable.
        """
        raise NotImplementedError

    def _add_border_highlight(self, color: Optional[str] = None,
                              width: Optional[int] = 2) -> mp.Rectangle:
        """
        Add a border to the pane to identify it.

        Created when an axis is added to the pane.

        Parameters
        ----------
        color : str, optional
            Color of the border.  If not specified, the second color
            in the default color cycle will be used.
        width : int, optional
            Width of the border.

        Returns
        -------
        rect : matplotlib.Rectangle
            Artist object representing the border.
        """
        if color is None:
            color = self.default_colors[1]

        # bottom-left and top-right
        x0, y0 = 0, 0
        x1, y1 = 1, 1

        rect = mp.Rectangle((x0, y0), x1 - x0, y1 - y0,
                            color=color, animated=True,
                            zorder=-1, lw=2 * width + 1, fill=None)
        rect.set_visible(False)
        rect.set_transform(self.ax.transAxes)
        self.ax.add_patch(rect)
        return rect

    def get_border(self) -> mp.Rectangle:
        """
        Retrieve the current border artist.

        Returns
        -------
        border : matplotlib.Rectangle
            The border artist.
        """
        return self.border

    def set_border_visibility(self, state: bool) -> None:
        """
        Set the border visibility.

        Parameters
        ----------
        state : bool
            If True, the border is shown.  If False, the border
            is hidden.
        """
        try:
            self.border.set_visible(state)
        except AttributeError:
            pass

    def get_axis_limits(self) -> Dict[str, Union[Tuple, str]]:
        """
        Get the current axis limits.

        Returns
        -------
        limits : dict
            Keys are axis names; values are current limits.
        """
        raise NotImplementedError

    def get_unit_string(self, axis: str) -> str:
        """
        Get the current unit string for the plot label.

        Parameters
        ----------
        axis : str
            The axis name to retrieve units from.

        Returns
        -------
        unit_name : str
            The unit label.
        """
        raise NotImplementedError

    def get_axis_scale(self) -> Dict[str, str]:
        """
        Get the axis scale setting.

        Returns
        -------
        scale : dict
            Keys are axis names; values are 'log' or 'linear'.
        """
        raise NotImplementedError

    def get_field(self, axis: Optional[str]) -> Union[str, Dict[str, str]]:
        """
        Get the currently displayed field.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the field only for the specified axis.

        Returns
        -------
        field : str or dict
            If axis is specified, only the field name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            field name values is returned.
        """
        raise NotImplementedError

    def get_unit(self, axis: Optional[str]) -> Union[str, Dict[str, str]]:
        """
        Get the currently displayed unit.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the unit only for the specified axis.

        Returns
        -------
        unit : str or dict
            If axis is specified, only the unit name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            unit name values is returned.
        """
        raise NotImplementedError

    def get_scale(self, axis: Optional[str]) -> Union[str, Dict[str, str]]:
        """
        Get the currently displayed scale.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the scale only for the specified axis.

        Returns
        -------
        scale : str or dict
            If axis is specified, only the scale name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            scale name values is returned.
        """
        raise NotImplementedError

    def get_orders(self, axis: Optional[str],
                   **kwargs) -> Union[str, Dict[str, str]]:
        """
        Get the currently displayed orders.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the orders only for the specified axis.

        Returns
        -------
        orders : str or dict
            If axis is specified, only the order name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            order name values is returned.
        """
        raise NotImplementedError

    def set_limits(self, limits: Dict[str, float]) -> None:
        """
        Set the plot limits.

        Parameters
        ----------
        limits : dict
            Keys are axis names; values are limits to set.
        """
        raise NotImplementedError

    def set_scales(self, scale: Dict[str, str]) -> None:
        """
        Set the plot scale.

        Parameters
        ----------
        scale : dict
            Keys are axis names; values are scales to set
            ('log' or 'linear').
        """
        raise NotImplementedError

    def set_units(self, units: Dict[str, str], axes: str) -> None:
        """
        Set the plot units.

        Parameters
        ----------
        units : dict
            Keys are axis names; values are units to set.
        axes : str
            Which axes to pull data from.
        """
        raise NotImplementedError

    def set_fields(self, fields: Dict[str, str]) -> None:
        """
        Set the plot fields.

        Parameters
        ----------
        fields : dict
            Keys are axis names; values are fields to set.
        """
        raise NotImplementedError

    def set_color_cycle_by_name(self, cycle_name: str) -> None:
        """
        Set the color cycle to be used in displayed plots.

        Parameters
        ----------
        cycle_name : ['spectral', 'tableau', 'accessible']
            The color cycle to set.
        """
        # TODO -- allow general matplotlib cmap names
        name = cycle_name.lower()
        if 'spectral' in name:
            self.default_colors = self.brewer_cycle
        elif 'tableau' in name:
            self.default_colors = self.tab10_cycle
        else:
            self.default_colors = self.accessible_cycle

    def update_colors(self) -> None:
        """Update colors in currently displayed plots."""
        raise NotImplementedError

    def perform_zoom(self, zoom_points: List, direction: str) -> None:
        """
        Perform a zoom action.

        Parameters
        ----------
        zoom_points : list
            User selected cursor positions, defining the zoom
            limits.
        direction :  str
            Direction of the zoom.
        """
        raise NotImplementedError

    def set_plot_type(self, plot_type: str) -> None:
        """
        Set the plot line type.

        Parameters
        ----------
        plot_type : str
            The plot type to set.
        """
        raise NotImplementedError

    def set_markers(self, state: bool) -> List:
        """Set the plot marker symbols."""
        raise NotImplementedError

    def set_grid(self, state: bool) -> None:
        """Set the plot grid visibility."""
        raise NotImplementedError

    def set_error(self, state: bool) -> None:
        """Set the plot error range visibility."""
        raise NotImplementedError

    def create_artists_from_current_models(self) -> None:
        """Create new artists from all current models."""
        raise NotImplementedError


class OneDimPane(Pane):
    """
    Single axis pane, for one-dimensional plots.

    This class is primarily intended to support spectral flux plots,
    but may be used for any one-dimensional line or scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Plot axes to display in the pane.

    Attributes
    ----------
    orders : dict
        Enabled spectral orders.
    plot_kind : str
        Kind of plot displayed (e.g. 'spectrum').
    fields : dict
        Keys are axis names; values are currently displayed fields.
    units : dict
        Keys are axis names; values are currently displayed units.
    scale : dict
        Keys are axis names; values are currently displayed plot scales.
    limits : dict
        Keys are axis names; values are currently displayed plot limits.
    plot_type : ['step', 'line', 'scatter']
        The currently displayed plot line type.
    show_markers : bool
        Flag for marker visibility.
    show_grid : bool
        Flag for grid visibility.
    show_error : bool
        Flag for error range visibility.
    colors : dict
        Current plot colors.  Keys are model IDs; values are color
        hex values.
    markers : dict
        Current plot colors.  Keys are model IDs; values are marker
        symbol names.
    guide_line_style : dict
        Style parameters for guide line overlays.
    data_changed : bool
        Flag to indicate that data has changed recently.
    """

    def __init__(self, ax: Optional[ma.Axes] = None) -> None:
        super().__init__(ax)
        self.orders = dict()
        self.plot_kind = 'spectrum'
        self.fields = {'x': 'wavepos',
                       'y': 'spectral_flux',
                       'y_alt': None}
        self.units = dict.fromkeys(['x', 'y', 'y_alt'], '')
        self.scale = {'x': 'linear', 'y': 'linear', 'y_alt': None}
        self.limits = {'x': [0, 1], 'y': [0, 1], 'y_alt': None}
        self.plot_type = 'step'
        self.show_markers = False
        self.show_grid = False
        self.show_error = True
        self.show_overplot = False

        self.colors = dict()
        self.markers = dict()

        self.guide_line_style = {'linestyle': ':',
                                 'color': 'darkgray',
                                 'linewidth': 1,
                                 'animated': True}

        self.data_changed = False

    ####
    # Models/Data
    ####
    def model_summaries(self) -> Dict[str, Dict[str, Union[str, bool]]]:
        """
        Summarize the models contained in this pane.

        Returns
        -------
        details : dict
            Keys and value types are 'filename': str, 'extension': str,
            'model_id': str, 'enabled': bool, 'color': str, 'marker': str.
        """
        details = dict()
        for model_id, model in self.models.items():
            details[model_id] = {'filename': os.path.basename(model.filename),
                                 'extension': self.fields['y'],
                                 'alt_extension': self.fields['y_alt'],
                                 'model_id': model_id,
                                 'enabled': model.enabled,
                                 'color': mc.to_hex(self.colors[model_id]),
                                 'marker': self.markers[model_id],
                                 }
        return details

    def add_model(self, model: MT) -> Dict[str, Any]:
        """
        Copy a model to the pane.

        The model is copied so the data can be manipulated
        without changing the root model.

        Parameters
        ----------
        model : high_model.HighModel
            Model object to add.

        Returns
        -------
        new_lines : dict
            Keys are model IDs; values are dicts with order number
            keys and dict values, containing new artists added to
            the Pane axes.
        """
        new_lines = dict()
        if model.id not in self.models.keys():
            self.models[model.id] = copy.copy(model)
            self.models[model.id].extension = self.fields['y']

            color_index = model.index % len(self.default_colors)
            log.debug(f'Model index: {model.index}; '
                      f'color index: {color_index}')
            self.colors[model.id] = self.default_colors[color_index]

            marker_index = model.index % len(self.default_markers)
            self.markers[model.id] = self.default_markers[marker_index]

            self.orders[model.id] = [order.number for order in model.orders]
            self.data_changed = True

            new_order_lines = self._plot_model(model)
            if new_order_lines:
                new_lines[model.id] = new_order_lines

        return new_lines

    def remove_model(self, filename: Optional[str] = None,
                     model: Optional[MT] = None) -> None:
        """
        Remove a model from the pane.

        The model can be specified by either its filename or
        the model directly.

        Parameters
        ----------
        filename : str, optional
            Name of the file corresponding to the model to remove.
        model : high_model.HighModel, optional
            Model object to remove.
        """
        new_models = dict()
        for k, m in self.models.items():
            if filename is not None:
                if filename in m.filename:
                    log.debug(f'Removing model {m.filename}')
                    self.data_changed = True
                    try:
                        del self.orders[m.id]
                    except KeyError:
                        pass
                    try:
                        del self.colors[m.id]
                    except KeyError:
                        pass
                    try:
                        del self.markers[m.id]
                    except KeyError:
                        pass
                else:
                    new_models[k] = m
            else:
                if m.id == model.id:
                    log.debug(f'Removing model {m.filename}')
                    self.data_changed = True
                    try:
                        del self.orders[m.id]
                    except KeyError:
                        pass
                    try:
                        del self.colors[m.id]
                    except KeyError:
                        pass
                    try:
                        del self.markers[m.id]
                    except KeyError:
                        pass
                else:
                    new_models[k] = m
        self.models = new_models
        if self.model_count() == 0:
            self.units = dict()

    def contains_model(self, model_id: str, order: Optional[int] = None
                       ) -> bool:
        if model_id in self.models.keys():
            if order is not None:
                if order in self.orders[model_id]:
                    return True
                else:
                    return False
            else:
                return True
        return False

    def possible_units(self) -> Dict[str, List[str]]:
        """
        Determine the possible units for the current fields.

        Returns
        -------
        units : dict
            Keys are axis names; values are lists of unit names.
        """
        available = dict()
        for axis in ['x', 'y', 'y_alt']:
            available[axis] = list()
            for model in self.models.values():
                low_model = model.retrieve(order=0, field=self.fields[axis],
                                           level='low')
                if low_model is None:
                    continue
                available[axis].extend(
                    low_model.available_units[low_model.kind])
        return available

    def current_units(self) -> Dict[str, str]:
        """
        Determine the current units for the current fields.

        Returns
        -------
        units : dict
            Keys are axis names; values are unit names.
        """
        current = {'x': '', 'y': '', 'y_alt': ''}
        if not self.models:
            return current
        for axis in ['x', 'y', 'y_alt']:
            # take unit from first model only -- they better match
            models = list(self.models.values())
            low_model = models[0].retrieve(
                order=0, field=self.fields[axis], level='low')
            if low_model is None:
                continue
            current[axis] = low_model.unit_key
        return current

    def set_orders(self, orders: Dict[str, List[int]]) -> None:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are model IDs; values are lists of orders
            to enable.
        """
        for model_id, model_orders in orders.items():
            try:
                available_orders = self.orders[model_id]
            except KeyError:
                continue
            else:
                log.debug(f'Enabling {model_orders} in '
                          f'{available_orders}')
                self.models[model_id].enable_orders(model_orders)
                self.data_changed = True

    def set_model_enabled(self, model_id: str, state: bool) -> None:
        """
        Mark a model as enabled (visible) or disabled (hidden).

        Parameters
        ----------
        model_id : str
            The ID for the model to enable.
        state : bool
            If True, enable the model.  If False, disable the model.
        """
        log.debug(f'Model {model_id} enabled: {state}')
        self.models[model_id].enabled = state

    def set_all_models_enabled(self, state: bool) -> None:
        """
        Mark all model as enabled (visible) or disabled (hidden).

        Parameters
        ----------
        state : bool
            If True, enable the model.  If False, disable the model.
        """
        for model_id in self.models:
            self.set_model_enabled(model_id, state)

    ####
    # Plotting
    ####
    def create_artists_from_current_models(self) -> Dict[str, Dict]:
        """Create new artists from all current models."""
        new_artists = dict()
        for model_id, model in self.models.items():
            new_artists[model_id] = (self._plot_model(model))
        return new_artists

    def _plot_model(self, model: MT) -> Dict[int, Dict]:
        """
        Plot a model in the current axes.

        Parameters
        ----------
        model : high_model.HighModel
            The model to plot.

        Returns
        -------
        new_lines : dict
            Keys are order numbers; values are dicts containing new
            artists added to the Pane axes, associated with the model.
        """
        log.debug(f'Starting limits: {self.get_axis_limits()}, '
                  f'{self.ax.get_autoscale_on()}')

        model.extension = self.fields['y']
        log.debug(f'Plotting {len(self.orders[model.id])} orders '
                  f'for {model.id}')
        new_lines = dict()
        n_orders = len(model.orders)
        if n_orders > 1:
            alpha_val = np.linspace(0.7, 1, n_orders)
            log.debug(f'Alpha values for {n_orders} orders: {alpha_val}')
        else:
            alpha_val = [1]
        for order_i, orders in enumerate(model.orders):
            new_line = dict()
            order = orders.number
            spectrum = model.retrieve(order=order, level='low',
                                      field=self.fields['y'])
            # order not available: try the next one
            if spectrum is None:
                continue

            # convert to current units, or remove model and
            # stop trying to load it
            log.debug(f'Current units: {self.units}')
            if any(self.units.values()):
                try:
                    self._convert_low_model_units(model, order,
                                                  'x', self.units['x'])
                    self._convert_low_model_units(model, order,
                                                  'y', self.units['y'])
                except ValueError:
                    log.warning('Incompatible units. '
                                'Try a different pane.')
                    self.remove_model(model=model)
                    break

            x = model.retrieve(order=order, level='raw',
                               field=self.fields['x'])
            y = model.retrieve(order=order, level='raw',
                               field=self.fields['y'])

            if x is None or y is None:
                log.debug(f'Failed to retrieve raw data for primary '
                          f'{model.id}, {order}, {self.fields}')
                continue

            if x.shape != y.shape:
                log.debug(f'Incompatible data shapes: '
                          f'{x.shape} and {y.shape}; skipping.')
                continue

            visible = model.enabled or not spectrum.enabled
            label = f'{model.id}, Order {order + 1}, {self.fields["y"]}'
            line = self._plot_single_line(x, y, model_id=model.id,
                                          visible=visible, label=label,
                                          alpha=alpha_val[order_i])
            new_line['line'] = {'artist': line,
                                'x_field': self.fields['x'],
                                'y_field': self.fields['y']}

            cursor = self._plot_cursor(x, y, model_id=model.id)

            new_line['cursor'] = {'artist': cursor,
                                  'x_field': self.fields['x'],
                                  'y_field': self.fields['y']}

            if 'flux' in self.fields['y'].lower():
                error = model.retrieve(order=order, level='raw',
                                       field='spectral_error')
                line = self._plot_flux_error(x, y, error,
                                             color=self.colors[model.id],
                                             label=(f'{model.id}, '
                                                    f'Order {order + 1}, '
                                                    f'spectral_error'))
                if (not model.enabled or not spectrum.enabled
                        or not self.show_error):
                    line.set_visible(False)
                new_line['error_range'] = {'artist': line,
                                           'x_field': self.fields['x'],
                                           'y_field': self.fields['y']}
            if self.show_overplot:
                y = model.retrieve(order=order, level='raw',
                                   field=self.fields['y_alt'])
                if y is None:
                    log.debug(f'Failed to retrieve raw data for alt y '
                              f'{model.id}, {order}, {self.fields}')
                elif x.shape == y.shape:
                    # only add overplot if x and y shapes match

                    label = (f'{model.id}, Order {order + 1}, '
                             f'{self.fields["y_alt"]}')
                    line = self._plot_single_line(x, y, model_id=model.id,
                                                  visible=visible, label=label,
                                                  axis='alt')
                    new_line['line_alt'] = {'artist': line,
                                            'x_field': self.fields['x'],
                                            'y_field': self.fields['y_alt']}

                    cursor = self._plot_cursor(x, y, model_id=model.id,
                                               axis='alt')

                    new_line['cursor_alt'] = {'artist': cursor,
                                              'x_field': self.fields['x'],
                                              'y_field': self.fields['y_alt']}

            # turn on/off grid lines as desired
            self.ax.grid(self.show_grid)

            new_lines[order] = new_line

        # set initial units from first loaded model
        if not any(self.units.values()):
            self.units = self.current_units()

        # reset units if no models are loaded
        if self.model_count() == 0:
            self.units = dict.fromkeys(self.units, '')

        log.debug(f'Ending limits: {self.get_axis_limits()}, '
                  f'{self.ax.get_autoscale_on()}')
        self.apply_configuration()
        return new_lines

    def _plot_single_line(self, x, y, label, model_id, visible,
                          axis='primary', alpha=1):
        if axis == 'primary':
            ax = self.ax
            color = self.colors[model_id]
            width = 1.5
            style = '-'
        else:
            ax = self.ax_alt
            alpha = 0.5
            color = self.colors[model_id]
            width = 1
            style = ':'

        style_kwargs = {'color': color,
                        'alpha': alpha,
                        'linewidth': width,
                        'linestyle': style}
        if self.show_markers or self.plot_type == 'scatter':
            style_kwargs['marker'] = self.markers[model_id]
        if self.plot_type == 'scatter':
            style_kwargs['linestyle'] = ''

        if self.plot_type == 'step':
            (line,) = ax.step(x, y, where='mid', animated=True, label=label,
                              **style_kwargs)
        else:
            (line,) = ax.plot(x, y, animated=True, label=label,
                              **style_kwargs)
        if not visible:
            line.set_visible(False)
        return line

    def _plot_cursor(self, x, y, model_id, axis='primary'):
        if axis == 'primary':
            ax = self.ax
            alpha = 1
            marker = 'x'
            color = self.colors[model_id]
        else:
            ax = self.ax_alt
            alpha = 0.5
            marker = '^'
            color = self.colors[model_id]
        style_kwargs = {'marker': marker,
                        'color': color,
                        'alpha': alpha}
        cursor = ax.scatter(x[0], y[0], animated=True, **style_kwargs)
        cursor.set_visible(False)
        return cursor

    def _plot_flux_error(self, x: np.ndarray, y: np.ndarray,
                         error: np.ndarray, label: str,
                         color: str) -> mcl.PolyCollection:
        """
        Plot the error range on the flux.

        Parameters
        ----------
        x : np.ndarray
            Array of x-data.
        y : np.ndarray
            Array of y-data.
        error : np.ndarray
            Array of error on `y`. Assumed to be symmetric.
        label : str
            Label to give to the line.
        color : str
            Color for the error range artist.

        Returns
        -------
        poly : matplotlib.collections.PolyCollection
            The plotted error lines.
        """
        options = {'alpha': 0.2, 'animated': True, 'label': label,
                   'color': color}

        if self.plot_type == 'step' or self.plot_type == 'scatter':
            options['step'] = 'mid'
        poly = self.ax.fill_between(x, y - error, y + error,
                                    **options)
        return poly

    def apply_configuration(self) -> None:
        """Update limits, scales, and labels for the plot."""
        self._apply_limits()
        self._apply_scales()
        self._apply_labels()

    def _apply_limits(self) -> None:
        """Update plot limits for current data."""
        if self.data_changed:
            self.reset_zoom()
            self.data_changed = False
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.ax.set_xlim(self.limits['x'])
                self.ax.set_ylim(self.limits['y'])
                if self.show_overplot:
                    self.ax_alt.set_ylim(self.limits['y_alt'])

    def _apply_scales(self) -> None:
        """Update plot scales for current preferences."""
        self.ax.set_xscale(self.scale['x'])
        self.ax.set_yscale(self.scale['y'])
        if self.show_overplot:
            self.ax_alt.set_yscale(self.scale['y_alt'])

    def _apply_labels(self) -> None:
        """Update plot labels for current fields and units."""
        x_label = self._generate_axes_label(self.fields['x'],
                                            self.units.get('x', ''))
        self.ax.set_xlabel(x_label)

        y_label = self._generate_axes_label(self.fields['y'],
                                            self.units.get('y', ''))
        self.ax.set_ylabel(y_label)

        if self.show_overplot:
            y_label = self._generate_axes_label(self.fields['y_alt'],
                                                self.units.get('y_alt', ''))
            self.ax_alt.set_ylabel(y_label)

        for ax in [self.ax, self.ax_alt]:
            try:
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
            except AttributeError:
                pass

    @staticmethod
    def _generate_axes_label(name: str, unit: str) -> str:
        """
        Generate a formatted axis label.

        Parameters
        ----------
        name : str
            The field name for the axis.
        unit : str
            Unit name for the axis.

        Returns
        -------
        label : str
            The formatted label.
        """
        name_string = name.capitalize()
        unit_string = unit.__str__()
        if len(unit_string) == 0:
            label = f'{name_string}'
        elif len(unit_string) > 10:
            label = f'{name_string}\n[{unit_string}]'
        else:
            label = f'{name_string} [{unit_string}]'
        return label

    def update_colors(self) -> List[Dict[str, str]]:
        """Update plot colors for current loaded data."""
        updates = list()
        for model_id, model in self.models.items():
            color_index = model.index % len(self.default_colors)
            log.debug(f'Model index: {model.index}; '
                      f'color index: {color_index}')
            self.colors[model.id] = self.default_colors[color_index]
            for order_number in self.orders[model_id]:
                update = dict()
                update['model_id'] = model_id
                update['order'] = order_number
                update['new_color'] = self.colors[model_id]
                updates.append(update)
                # also update any fit overlays
                update = dict()
                update['model_id'] = model_id
                update['order'] = order_number
                update['new_color'] = self.grayscale(self.colors[model_id])
                update['data_id'] = 'model_fit'
                updates.append(update)
        # also update border color
        updates.append({'model_id': 'border',
                        'new_color': self.default_colors[1]})
        return updates

    def update_visibility(self) -> List[Dict[str, Any]]:
        """Update plot visibility for current loaded data."""
        updates = list()
        for model_id, model in self.models.items():
            for order_number in self.orders[model_id]:
                spectrum = model.retrieve(order=order_number, level='low',
                                          field=self.fields['y'])
                visible = model.enabled & spectrum.enabled
                update = dict()
                update['model_id'] = model_id
                update['order'] = order_number
                update['new_visibility'] = visible
                update['new_error_visibility'] = self.show_error & visible
                updates.append(update)
        return updates

    ####
    # Getters
    ####
    def get_axis_limits(self, axis: Optional[str] = None
                        ) -> Union[Dict[str, List], List]:
        """
        Get the current axis limits.

        Parameters
        ----------
        axis : str, optional
            The axis name to retrieve limits from. If not provided,
            both axis limits for the axes are returned.

        Returns
        -------
        limits : list or dict
            If axis is specified, only the limits for that axis are
            returned.  Otherwise, a dictionary with axis name keys and
            limit values is returned.  Limit values are [low, high].
        """
        if axis is None:
            return self.limits
        else:
            return self.limits[axis]

    def get_unit_string(self, axis: str) -> str:
        """
        Get the current unit string for the plot label.

        Parameters
        ----------
        axis : str
            The axis name to retrieve units from.

        Returns
        -------
        unit_name : str
            The unit label.
        """
        if axis == 'x':
            label = self.ax.get_xlabel()
        elif axis == 'y':
            label = self.ax.get_ylabel()
        elif axis == 'z':
            label = self.ax.get_zlabel()
        else:
            raise SyntaxError(f'Invalid axis selection {axis}')
        try:
            unit = re.split(r'[\[\]]', label)[1]
        except IndexError:
            unit = '-'
        return unit

    def get_axis_scale(self) -> Dict[str, str]:
        """
        Get the axis scale setting.

        Returns
        -------
        scale : dict
            Keys are axis names; values are 'log' or 'linear'.
        """
        return self.scale

    def get_orders(self, enabled_only: bool = False,
                   by_model: bool = False) -> Union[Dict[str, List], List]:
        """
        Get the orders available for the models in this pane.

        Parameters
        ----------
        enabled_only : bool, optional
            If set, only return the enabled orders. Otherwise,
            return all orders. Default is False.
        by_model : bool, optional.
            If set, return a dictionary with the keys are model names
            and the values are the orders for that model. Otherwise,
            return a list of all model orders combined.

        Returns
        -------
        orders : list, dict
            Format and details depend on arguments.
        """
        if by_model:
            orders = dict()
        else:
            orders = list()
        for model_id, model in self.models.items():
            if enabled_only:
                model_orders = model.list_enabled()['orders']
            else:
                model_orders = model.orders
            if by_model:
                orders[model_id] = model_orders
            else:
                orders.extend(model_orders)
        return orders

    def get_field(self, axis: Optional[str] = None
                  ) -> Union[Dict[str, str], str]:
        """
        Get the currently displayed field.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the field only for the specified axis.

        Returns
        -------
        field : str or dict
            If axis is specified, only the field name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            field name values is returned.
        """
        if axis is None or axis == '':
            field = self.fields
        else:
            try:
                if axis == 'alt':
                    field = self.fields['y_alt']
                else:
                    field = self.fields[axis]
            except KeyError:
                field = None
        if field is None:
            raise EyeError(f'Unable to retrieve field for axis {axis}')
        else:
            return field

    def get_unit(self, axis: Optional[str] = None
                 ) -> Union[str, Dict[str, str]]:
        """
        Get the currently displayed unit.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the unit only for the specified axis.

        Returns
        -------
        unit : str or dict
            If axis is specified, only the unit name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            unit name values is returned.
        """
        if axis is None or axis == '':
            if not self.units:
                units = {'x': '', 'y': '', 'y_alt': ''}
            else:
                units = self.units
        else:
            try:
                if axis == 'alt':
                    units = self.units['y_alt']
                else:
                    units = self.units[axis]
            except KeyError:
                units = None
        if units is None:
            raise EyeError(f'Unable to retrieve units for axis {axis}')
        else:
            return units

    def get_scale(self, axis: Optional[str] = None
                  ) -> Union[str, Dict[str, str]]:
        """
        Get the currently displayed scale.

        Parameters
        ----------
        axis : str, optional
            If provided, retrieves the scale only for the specified axis.

        Returns
        -------
        scale : str or dict
            If axis is specified, only the scale name for that axis is
            returned.  Otherwise, a dictionary with axis name keys and
            scale name values is returned.
        """
        if axis is None or axis == '':
            scale = self.scale
        else:
            try:
                if axis == 'alt':
                    scale = self.scale['y_alt']
                else:
                    scale = self.scale[axis]
            except KeyError:
                scale = None
        if scale is None:
            raise EyeError(f'Unable to retrieve scale for axis {axis}')
        else:
            return scale

    ####
    # Setters
    ####
    # TODO: Add legend support
    def set_legend(self):
        """
        Set a legend for the plot.

        Not yet implemented.
        """
        raise NotImplementedError

    def set_limits(self, limits: Dict[str, Sequence[Num]]) -> None:
        """
        Set the plot limits.

        Parameters
        ----------
        limits : dict
            Keys are axis names; values are limits to set.
        """
        log.debug(f'Setting axis limits to {limits}')
        for axis in self.limits.keys():
            try:
                self.limits[axis] = sorted(limits[axis])
            except (KeyError, TypeError):
                continue
        try:
            alt = self.ax_alt.get_ylim()
        except AttributeError:
            alt = None
        log.debug(f'Verify: x={self.ax.get_xlim()}, '
                  f'y={self.ax.get_ylim()}, y_alt={alt}')

    def set_scales(self, scales: Dict[str, str]) -> None:
        """
        Set the plot scale.

        Parameters
        ----------
        scales : dict
            Keys are axis names; values are scales to set
            ('log' or 'linear').
        """
        log.debug(f'Setting axis scales to {scales}')
        for axis in self.scale.keys():
            try:
                if scales[axis] in ['linear', 'log']:
                    self.scale[axis] = scales[axis]
            except KeyError:
                continue

    def _convert_low_model_units(self, model: MT, order_number: int,
                                 axis: str, target_unit: str) -> None:
        """
        Convert data to new units.

        Parameters
        ----------
        model : high_model.HighModel
            The model to modify.
        order_number : int
            The spectral order to modify.
        axis : str
            The axis name to modify ('x' or 'y').
        target_unit : str
            The unit to convert to.

        Raises
        ------
        ValueError
            If data cannot be converted to the specified units.
        """
        spectrum = model.retrieve(order=order_number,
                                  level='low',
                                  field=self.fields[axis])
        if spectrum is None:
            raise EyeError(f'Retrieved None from {model} (order '
                           f'{order_number}, field {self.fields[axis]}')
        if spectrum.kind in ['flux', 'wavelength']:
            wave_spectrum = model.retrieve(order=order_number,
                                           level='low', field='wavepos')
            wavelength_data = wave_spectrum.data
            wavelength_unit = wave_spectrum.unit_key
        else:
            wavelength_data = None
            wavelength_unit = None

        # raises ValueError if not possible
        spectrum.convert(target_unit, wavelength_data, wavelength_unit)

        if 'flux' in self.fields[axis]:
            error_spectrum = model.retrieve(
                order=order_number, level='low', field='spectral_error')
            error_spectrum.convert(target_unit, wavelength_data,
                                   wavelength_unit)

    def set_units(self, units: Dict[str, str], axes: str,
                  ) -> Tuple[List[Any], List[Any]]:
        """
        Set the plot units.

        Parameters
        ----------
        units : dict
            Keys are axis names; values are units to set.
        axes: 'primary', 'alt', 'both', 'all'
            Which Axes object to pull data from.
        """
        updates = list()
        for axis, current_unit in self.units.items():
            try:
                target_unit = units[axis]
            except KeyError:
                continue
            if target_unit == current_unit:
                continue
            updated = False
            for model_id, model in self.models.items():
                for order_number in self.orders[model_id]:
                    try:
                        self._convert_low_model_units(model, order_number,
                                                      axis, target_unit)
                    except ValueError:
                        log.debug(f'Cannot convert units to '
                                  f'{target_unit} for {model_id}; ignoring')
                        break
                    else:
                        updated = True
                        update = dict()
                        update['model_id'] = model_id
                        update['order'] = order_number
                        update['field'] = self.fields[axis]
                        data = model.retrieve(order=order_number,
                                              level='raw',
                                              field=self.fields[axis])
                        update[f'new_{axis}_data'] = data
                        updates.append(update)
            if updated:
                self.units[axis] = units[axis]
                self.data_changed = True
        if 'flux' in self.fields['y'] and len(updates) > 0:
            error_updates = self._update_error_artists()
        else:
            error_updates = list()
        return updates, error_updates

    def _update_error_artists(self) -> List[Dict[str, Any]]:
        """Update error range artists to new data."""
        updates = list()
        for model_id, model in self.models.items():
            for order in self.orders[model_id]:
                error = model.retrieve(order=order, level='raw',
                                       field='spectral_error')
                x = model.retrieve(order=order, level='raw',
                                   field=self.fields['x'])
                y = model.retrieve(order=order, level='raw',
                                   field=self.fields['y'])
                poly = self._plot_flux_error(x, y, error,
                                             color=self.colors[model.id],
                                             label=(f'{model.id}, '
                                                    f'Order {order + 1}, '
                                                    f'spectral_error'))
                if not model.enabled or not self.show_error:
                    poly.set_visible(False)

                update = {'model_id': model_id, 'order': order,
                          'new_artist': poly}
                updates.append(update)
        return updates

    def set_fields(self, fields: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Set the plot fields.

        Parameters
        ----------
        fields : dict
            Keys are axis names; values are fields to set.

        Returns
        -------
        updates : list
            List of dictionaries that each describe the field
            change to a single model.

        """
        previous_fields = self.fields.copy()
        updates = list()
        for axis in self.fields.keys():
            try:
                new_field = fields[axis]
            except KeyError:
                continue
            if self.fields[axis] != new_field:
                valid = all([m.valid_field(new_field)
                             for m in self.models.values()])
                if valid:
                    self.fields[axis] = new_field
                    # reset units so that conversion is not triggered
                    # self.units = dict()
                    self.set_default_units_for_fields()
                    self.data_changed = True
                    for model_id, model_ in self.models.items():
                        for order_number in self.orders[model_id]:
                            update = dict()
                            update['model_id'] = model_id
                            update['order'] = order_number
                            update['old_field'] = previous_fields[axis]
                            update['new_field'] = new_field
                            data = model_.retrieve(order=order_number,
                                                   level='raw',
                                                   field=self.fields[axis])
                            update[f'new_{axis}_data'] = data
                            updates.append(update)
                else:
                    # ignore: this is likely from setting an
                    # all primary/all overplots from another pane's value
                    log.debug(f'Invalid field provided for axis {axis}: '
                              f'{fields[axis]}')
        return updates

    def set_default_units_for_fields(self) -> None:
        """Set default unit values for current fields."""
        for model_id, model in self.models.items():
            for order_number in self.orders[model_id]:
                for axis in self.fields.keys():
                    low_model = model.retrieve(order=order_number,
                                               level='low',
                                               field=self.fields[axis])
                    if low_model is not None:
                        self.units[axis] = low_model.get_unit()

    def set_plot_type(self, plot_type: str) -> List[Dict[str, Any]]:
        """
        Set the plot line type.

        Parameters
        ----------
        plot_type : ['step', 'line', 'scatter']
            The plot type to set.

        Returns
        -------
        updates : list
            A list of dictionaries that each describe the change
            to a single model.

        """
        self.plot_type = plot_type.lower()
        updates = list()
        for model_id in self.models.keys():
            for order_number in self.orders[model_id]:
                update = dict()
                update['model_id'] = model_id
                update['order'] = order_number
                update['new_type'] = self.plot_type
                if self.plot_type == 'scatter' or self.show_markers:
                    update['new_marker'] = self.markers[model_id]
                updates.append(update)
        return updates

    def set_markers(self, state: bool) -> List[Dict[str, Any]]:
        """
        Set plot marker symbols.

        Only applies to scatter plots. Non-scatter plots
        will accept the new state but not update any
        artists.

        Parameters
        ----------
        state : bool
            Defines the visibility of the makers. True
            will make the markers visible, False will
            make the markers invisible.

        Returns
        -------
        updates : list
            If updates can be made (ie, scatter plot) then
            is is a list of dictionaries describing the
            change for each model. Otherwise it is an
            empty list.
        """
        self.show_markers = bool(state)
        updates = list()
        # no-op for scatter plots
        if self.plot_type == 'scatter':
            return updates
        for model_id in self.models.keys():
            for order_number in self.orders[model_id]:
                update = dict()
                update['model_id'] = model_id
                update['order'] = order_number
                if self.show_markers:
                    update['new_marker'] = self.markers[model_id]
                else:
                    update['new_marker'] = None
                updates.append(update)
        return updates

    def set_grid(self, state: bool) -> None:
        """Set the plot grid visibility."""
        self.show_grid = bool(state)
        if self.ax:
            self.ax.grid(self.show_grid)

    def set_error(self, state: bool) -> None:
        """Set the plot error range visibility."""
        self.show_error = bool(state)

    def set_overplot(self, state: bool) -> None:
        if bool(state) is bool(self.show_overplot):
            return
        self.show_overplot = bool(state)
        if self.show_overplot:
            if self.ax:
                self.ax_alt = self.ax.twinx()
                self.ax_alt.autoscale(enable=True)
                self.fields['y_alt'] = 'transmission'
                self.scale['y_alt'] = 'linear'
                self.limits['y_alt'] = [0, 1]
        else:
            self.fields['y_alt'] = ''
            self.scale['y_alt'] = ''
            self.limits['y_alt'] = list()

    def reset_alt_axes(self, remove=False):
        if remove:
            try:
                self.ax_alt.remove()
            except AttributeError:
                log.debug(f'Failed to remove alt ax {self.ax_alt}')
            else:
                log.debug('Successfully removed alt ax')
        self.ax_alt = None
        self.fields['y_alt'] = ''
        self.scale['y_alt'] = ''
        self.limits['y_alt'] = list()

    ####
    # Mouse events
    ####
    def data_at_cursor(self, event: mbb.MouseEvent) -> Dict[str, List[Dict]]:
        """
        Retrieve the model data at the cursor location.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Mouse motion event.

        Returns
        -------
        data_coords : dict
            Keys are model IDs; values are lists of dicts
            containing 'order', 'bin', 'bin_x', 'bin_y',
            'x_field', 'y_field', 'color', and 'visible'
            values to display.
        """
        data = dict()
        cursor_x = event.xdata
        for model_id, model in self.models.items():
            data[model_id] = list()
            for order_number in self.orders[model_id]:
                visible = model.enabled
                x_data = model.retrieve(order=order_number, level='raw',
                                        field=self.fields['x'])
                y_data = model.retrieve(order=order_number, level='raw',
                                        field=self.fields['y'])
                # skip order if cursor is out of range
                if (cursor_x < np.nanmin(x_data)
                        or cursor_x > np.nanmax(x_data)):
                    visible = False

                index = int(np.nanargmin(np.abs(x_data - cursor_x)))
                x = x_data[index]
                y = y_data[index]
                if all(np.isnan([x, y])):
                    visible = False

                data[model_id].append({'order': order_number,
                                       'bin': index,
                                       'bin_x': x,
                                       'bin_y': y,
                                       'x_field': self.fields['x'],
                                       'y_field': self.fields['y'],
                                       'color': self.colors[model_id],
                                       'visible': visible,
                                       'alt': False
                                       })
                if self.show_overplot:
                    y_data = model.retrieve(order=order_number,
                                            level='raw',
                                            field=self.fields['y_alt'])
                    y = y_data[index]
                    if all(np.isnan([x, y])):
                        visible = False
                    data[model_id].append({'order': order_number,
                                           'bin': index,
                                           'bin_x': x,
                                           'bin_y': y,
                                           'x_field': self.fields['x'],
                                           'y_field': self.fields['y_alt'],
                                           'color': self.colors[model_id],
                                           'visible': visible,
                                           'alt': True
                                           })
        return data

    def xy_at_cursor(self, event: mbb.MouseEvent) -> Tuple[float, float]:
        """
        Retrieve the x and y data at the cursor location.

        Ignores the secondary y-axis, if present.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Mouse motion event.

        Returns
        -------
        data_coords : tuple of float
            (x, y) values for the primary axis.
        """
        cursor_x = event.xdata
        cursor_y = event.ydata

        # When overplots are enabled, only the top axis is returned as
        # the event.x/ydata.
        # Transform the cursor position to get the primary y data/
        if self.ax != event.inaxes:
            inv = self.ax.transData.inverted()
            _, cursor_y = inv.transform(
                np.array((event.x, event.y)).reshape(1, 2)).ravel()

        return cursor_x, cursor_y

    def plot_crosshair(self, cursor_pos: Optional[Union[Tuple, List]] = None
                       ) -> Dict[str, ml.Line2D]:
        """
        Create crosshair artists to show the cursor location.

        Parameters
        ----------
        cursor_pos : tuple or list, optional
            Current cursor position [x, y].  If not specified, the
            cursor is initially set near the center of the plot.

        Returns
        -------
        crosshair : dict
            Keys are 'v' and 'h'; values are vertical and horizontal line
            artists, respectively.
        """
        if cursor_pos is None:
            # set cursor near center of plot
            cursor_pos = [np.mean(self.ax.get_xlim()),
                          np.mean(self.ax.get_ylim())]
        cross = dict()
        cross['v'] = self.ax.axvline(cursor_pos[0], **self.guide_line_style,
                                     visible=False)
        cross['h'] = self.ax.axhline(cursor_pos[1], **self.guide_line_style,
                                     visible=False)
        return cross

    def plot_guides(self, cursor_pos: Union[Tuple, List],
                    kind: str) -> Dict[str, Dict[str, ml.Line2D]]:
        """
        Create guide artists.

        Guides are lines that stretch the full width or height
        of the axes, depending on the direction. They are used
        to denote the edges of ranges of interest, such as
        new zoom limits or the limits for data to fit a curve to.

        Parameters
        ----------
        cursor_pos : tuple or list
            Current cursor position [x, y].
        kind : ['horizontal', 'vertical', 'cross', 'x', 'y', 'b']
            For 'horizontal', 'y', 'cross', or 'b', a horizontal
            guide will be created.  For 'vertical', 'x', 'cross',
            or 'b', a vertical guide will be created.

        Returns
        -------
        guides : dict
            Keys are 'v' and/or 'h'; values are vertical and horizontal
            line artists, respectively.
        """
        guides = dict()
        if kind in ['vertical', 'cross', 'x', 'b']:
            guides['v'] = {
                'artist': self.ax.axvline(cursor_pos[0],
                                          **self.guide_line_style)}
        if kind in ['horizontal', 'cross', 'y', 'b']:
            guides['h'] = {
                'artist': self.ax.axhline(cursor_pos[1],
                                          **self.guide_line_style)}
        return guides

    def perform_zoom(self, zoom_points: Sequence[Sequence[Num]],
                     direction: str) -> Dict[str, List[float]]:
        """
        Perform a zoom action.

        Parameters
        ----------
        zoom_points : list
            User selected cursor positions, defining the zoom
            limits.
        direction :  ['x', 'y', 'b']
            Direction of the zoom.

        Returns
        -------
        limits : dict
            Keys are axis names ('x' and 'y'); values are the
            new axis limits.
        """
        log.debug('Perform zoom')
        limits = self.get_axis_limits()
        log.debug('Changing axis limits')
        if direction in ['x', 'b']:
            x_limits = [zp[0] for zp in zoom_points]
            log.debug(f'Updating x limits to {x_limits}')
            limits['x'] = [min(x_limits), max(x_limits)]
        if direction in ['y', 'b']:
            y_limits = [zp[1] for zp in zoom_points]
            log.debug(f'Updating y limits to {y_limits}')
            limits['y'] = [min(y_limits), max(y_limits)]

        self.set_limits(limits)
        self.ax.autoscale(enable=False)
        if self.show_overplot:
            self.ax_alt.autoscale(enable=False)
        return limits

    def reset_zoom(self) -> None:
        """Reset plot limits to the full range for current data."""
        # note: relim works for lines, but not for true scatter plots
        self.ax.relim(visible_only=True)
        self.ax.autoscale(enable=True)

        new_limits = {'x': self.ax.get_xlim(),
                      'y': self.ax.get_ylim()}
        if self.show_overplot:
            self.ax_alt.relim(visible_only=True)
            self.ax_alt.autoscale(enable=True)

            # the xlim and ylim calls need to be repeated here to
            # avoid a bug in matplotlib (currently v3.3.4)
            # relating to twinx and relim
            # https://github.com/matplotlib/matplotlib/issues/16723
            new_limits = {'x': self.ax.get_xlim(),
                          'y': self.ax.get_ylim(),
                          'y_alt': self.ax_alt.get_ylim()}
        self.set_limits(limits=new_limits)

    def perform_fit(self, kind: str, limits: Sequence[Sequence[Num]]
                    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Perform a fit to a plot feature.

        Parameters
        ----------
        kind : str
            Kind is formatted as 'fit_{feature model}_{baseline mode}'
            Current features available are Gaussian1D and Moffat1D.
            Current baselines available are Const1D and Linear1D. If
            a baseline or feature is not desired, the corresponding field
            is set to 'none'.
        limits : list of list
            Low and high limits for data to fit, as
            [[low_x, low_y], [high_x, high_y]].  Currently
            only the x values are used.

        Returns
        -------
        fit_artists, fit_params : dict, dict
            The fit_artists dict contains overlay artists showing
            the center line and model shape for the fit to data.
            Keys are {model.id}_fit_line and {model.id}_fit_center.
            The fit_params dict contains the fit parameters, for
            display in a separate table widget.  Keys are model IDs.
        """
        try:
            feature = kind.split('_')[1]
            background = kind.split('_')[2]
        except IndexError:
            log.error(f'Improperly formatted fit kind {kind}; skipping')
            fit_artists, fit_params = None, None
        else:
            if feature == 'none' and background == 'none':
                log.error(f'Feature and/or background must be provided'
                          f' {kind}; skipping')
                fit_artists, fit_params = None, None
            else:
                fit_artists, fit_params = self.feature_fit(feature, background,
                                                           limits)
        return fit_artists, fit_params

    def feature_fit(self, feature: str, background: str,
                    limits: Sequence[Sequence[Num]]) -> Tuple[Dict, Dict]:
        fit_params = list()
        successes = list()
        for model_id, model_ in self.models.items():
            # model not enabled: skip
            if not model_.enabled:
                log.debug(f'Model {model_id} is not '
                          f'enabled; skipping.')
                continue

            for orders in model_.orders:
                order = orders.number
                spectrum = model_.retrieve(order=order, level='low',
                                           field=self.fields['y'])
                # order not available: try the next one
                if spectrum is None:
                    log.debug(f'Order {order} is not '
                              f'available; skipping.')
                    continue
                if not spectrum.enabled:
                    log.debug(f'Order {order} is not '
                              f'enabled; skipping.')
                    continue

                x = model_.retrieve(order=order, level='raw',
                                    field=self.fields['x'])
                y = model_.retrieve(order=order, level='raw',
                                    field=self.fields['y'])

                # scale data to account for units with very small increments
                xs = np.nanmean(x)
                ys = np.nanmean(y)
                norm_limits = [[limits[0][0] / xs, limits[0][1] / ys],
                               [limits[1][0] / xs, limits[1][1] / ys]]
                xnorm = x / xs
                ynorm = y / ys

                blank_fit = self._initialize_model_fit(feature, background,
                                                       norm_limits, model_id,
                                                       order)
                try:
                    xnorm, ynorm, fit_init, bounds = self.initialize_models(
                        feature, background, xnorm, ynorm, norm_limits)
                except EyeError as e:
                    if str(e) == 'empty_order':
                        log.debug(f'Order {order} has no valid data '
                                  f'in range; skipping.')
                    fit_params.append(blank_fit)
                    blank_fit.set_status(str(e))
                    continue
                else:
                    blank_fit.set_dataset(x=xnorm, y=ynorm)

                try:
                    fit_result = self.calculate_fit(xnorm, ynorm,
                                                    fit_init, bounds)
                except EyeError as e:
                    successes.append(False)
                    if str(e) == 'fit_failed':
                        log.debug(f'Fit failed; skipping {model_id} '
                                  f'order {order}')
                    blank_fit.set_status(str(e))
                    continue
                else:
                    blank_fit.set_status('pass')
                    blank_fit.set_fit(fit_result)
                finally:
                    # unscale parameters and data
                    blank_fit.scale_parameters(xs, ys)
                    blank_fit.set_dataset(x=(xnorm * xs), y=(ynorm * ys))
                    blank_fit.set_limits(limits)
                    fit_params.append(blank_fit)

        fit_artists = self.generate_fit_artists(fit_params)
        return fit_artists, fit_params

    def _initialize_model_fit(self, feature: str, background: str,
                              limits: List[List[float]], model_id: str,
                              order: int) -> model_fit.ModelFit:
        fit = model_fit.ModelFit()

        # catch '-' from dropdown
        ftypes = []
        for ft in [feature, background]:
            if ft != '-':
                ftypes.append(ft)
        if len(ftypes) == 0:
            ftypes.append('constant')
        fit.set_fit_type(ftypes)

        fit.set_fields(self.fields)
        fit.set_limits(limits)
        fit.set_units({'x': self.units['x'], 'y': self.units['y']})
        fit.set_axis(self.ax)
        fit.set_model_id(model_id)
        fit.set_order(order)
        return fit

    @staticmethod
    def calculate_fit(x_data, y_data, fit, bounds):
        def fit_func(x, *params):
            fit.parameters = params
            return fit(x)

        try:
            coeffs, _ = sco.curve_fit(fit_func, x_data, y_data,
                                      p0=fit.parameters,
                                      bounds=bounds)
        except ValueError as e:
            if 'bound' in str(e):
                # specified bounds are infeasible, return initial model
                # This should be hit only when the user has selected
                # no feature and no baseline to fit -- ie. constant zero.
                return fit
            else:
                raise EyeError('fit_failed') from None
        except RuntimeError:
            # fit failed
            raise EyeError('fit_failed') from None
        else:
            fit.parameters = coeffs
        return fit

    def initialize_models(self, feature, background,
                          x_data, y_data, limits):

        xval, yval = self._subselect_data(x_data, y_data, limits)
        guess = self._generate_initial_guess(xval, yval, limits)

        if feature == 'moffat':
            feature_init = am.models.Moffat1D(guess['amplitude'],
                                              guess['peak_location'],
                                              guess['width'],
                                              guess['power_index'])
            # Bounds: amplitude, max location, gamma, alpha
            lower_bounds = [-np.inf, limits[0][0], 0, 0]
            upper_bounds = [np.inf, limits[1][0], guess['width'] * 3, np.inf]
        elif feature in ['gaussian', 'gauss']:
            feature_init = am.models.Gaussian1D(guess['amplitude'],
                                                guess['peak_location'],
                                                guess['width'])
            # Bounds: amplitude, max location, gamma
            lower_bounds = [-np.inf, limits[0][0], 0]
            upper_bounds = [np.inf, limits[1][0], guess['width'] * 3]
        else:
            feature_init = None
            lower_bounds = []
            upper_bounds = []

        if background == 'linear':
            background_init = am.models.Linear1D(0, guess['background'])
            # Bounds: slope, intercept
            lower_bounds.extend([-np.inf, -np.inf])
            upper_bounds.extend([np.inf, np.inf])
        elif background == 'constant':
            background_init = am.models.Const1D(guess['background'])
            # Bounds: constant amplitude
            lower_bounds.extend([-np.inf])
            upper_bounds.extend([np.inf])
        elif feature_init is None:
            # special case: no fit selected for either feature or
            # background.  Just return 0.
            background_init = am.models.Const1D(0)
            lower_bounds.extend([0])
            upper_bounds.extend([0])
        else:
            background_init = None

        if feature_init is None:
            fit_init = background_init
        elif background_init is not None:
            fit_init = feature_init + background_init
        else:
            fit_init = feature_init
        bounds = (lower_bounds, upper_bounds)
        return xval, yval, fit_init, bounds

    @staticmethod
    def _subselect_data(x_data, y_data, limits):
        xrange = [limits[0][0], limits[1][0]]
        valid = ~(np.isnan(x_data) | np.isnan(y_data))
        in_range = (x_data >= xrange[0]) & (x_data <= xrange[1])

        # no valid data: skip
        if not np.any(in_range & valid):
            raise EyeError('empty_order')

        xval = x_data[in_range & valid]
        yval = y_data[in_range & valid]
        return xval, yval

    @staticmethod
    def _generate_initial_guess(x, y, limits):
        guess = dict()
        mid_index = len(x) // 2
        guess['peak_location'] = x[mid_index]
        guess['background'] = float(np.nanmedian(y))
        guess['amplitude'] = y[mid_index] - guess['background']
        guess['width'] = (limits[1][0] - limits[0][0]) / 3
        guess['power_index'] = 2
        return guess

    def generate_fit_artists(self, model_fits: Union[model_fit.ModelFit,
                                                     List[model_fit.ModelFit]],
                             x_data: Optional[ArrayLike] = None) -> Dict:
        fit_artists = dict()
        if not isinstance(model_fits, list):
            model_fits = [model_fits]
        for obj in model_fits:
            fit = obj.get_fit()
            if fit is None:
                continue

            model_id = obj.get_model_id()
            order = obj.get_order()
            id_tag = obj.get_id()
            if x_data is None:
                fields = obj.get_fields()
                limits = obj.get_limits()
                x = self.models[model_id].retrieve(order=order,
                                                   level='raw',
                                                   field=fields['x'])
                y = self.models[model_id].retrieve(order=order,
                                                   level='raw',
                                                   field=fields['y'])

                xrange = [limits['lower'], limits['upper']]
                valid = ~(np.isnan(x) | np.isnan(y))
                in_range = (x >= xrange[0]) & (x <= xrange[1])
                x_data = x[in_range & valid]

            linestyle = next(self.fit_linestyles)
            fit_line, fit_center = self.plot_fit(fit_obj=obj, x=x_data,
                                                 style=linestyle)
            # todo - this overwrites any previous assignment to fit_artists,
            #  should be fixed for multiple fits passed
            fit_artists = self._add_new_artist(fit_artists, model_id, order,
                                               fit_line, fit_center, id_tag)

        return fit_artists

    @staticmethod
    def _add_new_artist(fit_artists, model_id, order, fit_line,
                        fit_center, id_tag):
        line_key = f'{model_id}_fit_line'
        center_key = f'{model_id}_fit_center'
        if line_key not in fit_artists:
            fit_artists[line_key] = dict()
            fit_artists[center_key] = dict()
        fit_artists[line_key][order] = {'fit': {'artist': fit_line,
                                                'data_id': id_tag}}
        fit_artists[center_key][order] = {'fit': {'artist': fit_center,
                                                  'data_id': id_tag}}
        return fit_artists

    def plot_fit(self, x: ArrayLike, style: str,
                 fit_obj: Optional[model_fit.ModelFit] = None,
                 fit: Optional[am.Model] = None,
                 model_id: Optional[str] = '', order: Optional[int] = int,
                 feature: Optional[str] = '',
                 ) -> Tuple[ml.Line2D, ml.Line2D]:
        """
        Create overlay artists representing a fit to a plot feature.

        Overlay colors are grayscale representations of the displayed
        model colors.

        Parameters
        ----------
        x : array-like
            Independent coordinates for the fit overlay.
        fit : astropy.modeling.Model
            The callable model, fit to the data.
        model_id : str
            Model ID associated with the fit.
        order : int
            Order number associated with the fit.
        style : str
            Line style for the overlay.

        Returns
        -------
        model, centroid : tuple of Line2D
            The model artist, plotted over the input x data, and
            a vertical line artist representing the centroid position.
        """
        if fit_obj is not None:
            model_id = fit_obj.get_model_id()
            order = fit_obj.get_order()
            feature = fit_obj.get_feature()
            fit = fit_obj.get_fit()
            fit_obj.axis = self.ax

        # convert model color to grayscale, to distinguish
        # from plot, but keep some separation between different models
        gray = self.grayscale(self.colors[model_id])

        label = f'{model_id}, Order {order + 1}, {feature.capitalize()}'
        (line,) = self.ax.plot(x, fit(x), animated=True,
                               label=f'{label} fit',
                               color=gray, linewidth=2, alpha=0.8,
                               linestyle=style)
        line.set_visible(fit_obj.get_visibility())

        midpoint = fit_obj.get_mid_point()
        vline = self.ax.axvline(midpoint, animated=True,
                                label=f'{label} centroid',
                                color=gray, linewidth=1, alpha=0.6,
                                linestyle=style)
        vline.set_visible(fit_obj.get_visibility())

        return line, vline

    @staticmethod
    def grayscale(color):
        """Translate color hex to equivalent grayscale value."""
        yiq = colorsys.rgb_to_yiq(*mc.to_rgb(color))
        gray = str(yiq[0])
        return gray


class TwoDimPane(Pane):
    """
    Two-axis pane, for displaying images.

    Not yet implemented.
    """

    def __init__(self, ax: Optional[ma.Axes] = None) -> None:
        super().__init__(ax)
