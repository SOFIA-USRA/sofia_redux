# Licensed under a 3-clause BSD style license - see LICENSE.rst

import colorsys
import copy
import itertools
import os
import re
import uuid
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
from matplotlib import text as mt
from matplotlib import image as mi
import scipy.optimize as sco

from sofia_redux.visualization import log
from sofia_redux.visualization.signals import Signals
from sofia_redux.visualization.models import high_model, reference_model
from sofia_redux.visualization.utils.eye_error import EyeError
from sofia_redux.visualization.utils import model_fit
from sofia_redux.visualization.display import drawing

try:
    matplotlib.use('QT5Agg')
    matplotlib.rcParams['axes.formatter.useoffset'] = False
except ImportError:  # pragma: no cover
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True

__all__ = ['Pane', 'OneDimPane', 'TwoDimPane']

MT = TypeVar('MT', bound=high_model.HighModel)
RT = TypeVar('RT', bound=reference_model.ReferenceData)
DT = TypeVar('DT', bound=drawing.Drawing)
Num = TypeVar('Num', int, float)
Art = TypeVar('Art', ml.Line2D, mt.Annotation, mi.AxesImage)
ArrayLike = TypeVar('ArrayLike', List, Tuple, Sequence, np.ndarray)
IDT = TypeVar('IDT', str, uuid.UUID)


class Pane(object):
    """
    Plot window management.

    The Pane class is analogous to a matplotlib subplot. It
    contains plot axes and instantiates artists associated with
    them.  This class determines appropriate updates for display
    options, but it does not manage updating artists themselves
    after they are instantiated.  The `Gallery` class manages all
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
    ax_alt : matplotlib.axes.Axes
        Alternative plot axes to display in the pane.
    models : dict
        Data models displayed in the pane.  Keys are model IDs;
        values are sofia_redux.visualization.models.HighModel instances.
    reference : dict
        Reference model for lines and corresponding labels.
        Values are sofia_redux.visualization.models.reference_model
        .ReferenceData instances.
    fields : dict
        Model fields currently displayed in the pane.
    show_overplot: bool
        To over plot the alternative axis.
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

    def __init__(self, signals: Signals, ax: Optional[ma.Axes] = None) -> None:
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')

        self.ax = ax
        self.ax_alt = None
        self.models = dict()
        self.reference = None
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
        # Removed gray color (#f7f7f7) because gray is its own
        # compliment, so all apertures will have the same color
        self.tab10_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2',
                            '#bcbd22', '#17becf']

        # 7 visually separated colors picked from
        # https://colorcyclepicker.mpetroff.net/
        self.accessible_cycle = ['#2848ad', '#59d4db', '#f37738',
                                 '#c6ea6c', '#b9a496', '#d0196b', '#7b85d4']

        # default is accessible
        self.default_colors = self.accessible_cycle

        self.aperture_cycle = dict()
        self.order_cycle = list()

        self.set_aperture_cycle()

        self.default_markers = ['x', 'o', '^', '+', 'v', '*']
        self.fit_linestyles = itertools.cycle(['dashed',
                                               'dotted', 'dashdot'])
        self.border = None

    def __eq__(self, other):  # pragma: no cover
        return self.ax == other.ax

    def axes(self):
        """
        Get primary and alternate axes.

        Returns
        -------
        axes : list of matplotlib.axes.Axes
            The current axes, as [primary, alternate].
        """
        return [self.ax, self.ax_alt]

    def set_axis(self, ax: ma.Axes, kind: str = 'primary') -> None:
        """
        Assign an axes instance to the pane.

        Also instantiates the border highlight for the pane.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to assign.
        kind : {'primary', 'secondary'}, optional
            If given, specifies which axis is being given.
            Defaults to 'primary'.
        """
        if kind == 'primary':
            self.ax = ax
            self.border = self._add_border_highlight()
        else:
            self.ax_alt = ax

    def overplot_state(self) -> bool:
        """
        Get current overplot state.

        Returns
        -------
        state : bool
            True if overplot is shown; False otherwise.
        """
        return self.show_overplot

    def model_count(self) -> int:
        """
        Get the number of models attached to the pane.

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
                     model_id: Optional[IDT] = None,
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

    def model_extensions(self, model_id: str) -> List[str]:
        """
        Obtain a list of extensions for a model with a given model_id.

        Parameters
        ----------
        model_id : str
            Specific id to a model object (high_model.HighModel).

        Returns
        -------
        model.extension() : list of str
            Model extensions are a list of the hdu names for a given hdul.
            If `model_id` is not found, returns empty list.
        """
        try:
            model = self.models[model_id]
        except (IndexError, KeyError):
            return list()
        else:
            return model.extensions()

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

    def set_orders(self, orders: Dict[str, List[Any]],
                   enable: Optional[bool] = True,
                   aperture: Optional[bool] = False,
                   return_updates:
                   Optional[bool] = False) -> Optional[List[Any]]:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are model IDs; values are lists of orders
            to enable.
        enable : bool, optional
            Set to True to enable; False to disable.
        aperture : bool, optional
            True if apertures are present; False otherwise.
        return_updates : bool, optional
            If set, an update structure is returned.


        """
        raise NotImplementedError

    def ap_order_state(self, model_ids: Union[List[IDT], IDT]
                       ) -> Tuple[Dict[IDT, int], Dict[IDT, int]]:
        """
        Determine number of orders and apertures for each model_id.

        Parameters
        ----------
        model_ids: uuid.UUID, list(uuid.UUID)
            Single or list of model_ids.

        Returns
        -------
        apertures, orders: dict
            Dictionaries with model_ids as keys and values are
            the total number of apertures and orders respectively
            that model contains.

        """
        if not isinstance(model_ids, list):
            model_ids = [model_ids]
        apertures = dict.fromkeys(model_ids)
        orders = dict.fromkeys(model_ids)
        for model_id in model_ids:
            try:
                n_ap, n_ord = self.models[model_id].ap_order_count()
            except (IndexError, KeyError):
                n_ap = 0
                n_ord = 0
            apertures[model_id] = n_ap
            orders[model_id] = n_ord
        return apertures, orders

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

    def set_units(self, units: Dict[str, str], axes: str) -> Any:
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
        self.set_aperture_cycle()

    @staticmethod
    def grayscale(color: str, scheme: Optional[str] = 'yiq'):
        """
        Translate color hex to equivalent grayscale value.

        Parameters
        ----------
        color : str
            Color hex value.
        scheme : {'yiq', 'hex', 'rgb'}
            Color scheme to use for formatting output grayscale.

        Returns
        -------
        gray : str
            Grayscale value equivalent to the input color.
        """
        if isinstance(color, list):
            color = color[0]
        yiq = colorsys.rgb_to_yiq(*mc.to_rgb(color))
        if scheme == 'yiq':
            gray = str(yiq[0])
        elif scheme == 'hex':
            gray = mc.to_hex(colorsys.yiq_to_rgb(yiq[0], 0, 0))
        elif scheme == 'rgb':
            gray = colorsys.yiq_to_rgb(yiq[0], 0, 0)
        else:
            raise EyeError(f'Unknown color scheme {scheme}')
        return gray

    def set_aperture_cycle(self):
        """Set the color cycle for apertures."""
        self.aperture_cycle = dict()
        for color in self.default_colors:
            self.aperture_cycle[color] = self.analogous(color)

    @staticmethod
    def split_complementary(color):
        """
        Produce split complementary colors to the input color.

        Parameters
        ----------
        color : str
            Color hex value.

        Returns
        -------
        complements : list of str
            Two-element list of complements to the input color.
        """
        r, g, b = tuple(mc.to_rgb(color))
        # value has to be 0<x<1 in order to convert to hls
        # hls provides color in radial scale
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # get hue changes at 150 and 210 degrees
        deg_150_hue = h + (150.0 / 360.0)
        deg_210_hue = h + (210.0 / 360.0)
        # convert to rgb
        color_150_rgb = list(map(lambda x: round(x * 255),
                                 colorsys.hls_to_rgb(deg_150_hue, l, s)))
        color_210_rgb = list(map(lambda x: round(x * 255),
                                 colorsys.hls_to_rgb(deg_210_hue, l, s)))

        color_150 = mc.rgb2hex([x / 255 for x in color_150_rgb])
        color_210 = mc.rgb2hex([x / 255 for x in color_210_rgb])

        return [color_150, color_210]

    @staticmethod
    def analogous(color, degree=130.0):
        """
        Produce list of analogous colors to the input color.

        Parameters
        ----------
        color : str
            Color hex value.
        degree : float
            Color wheel angle to set as the distance above and below
            the input color.

        Returns
        -------
        analogous : list of str
            Two-element list of colors analogous to the input color.
        """
        r, g, b = tuple(mc.to_rgb(color))
        # set color wheel angle
        degree /= 360.0
        # hls provides color in radial scale
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # rotate hue by d
        h = [(h + d) % 1 for d in (-degree, degree)]
        analogous_list = list()
        for nh in h:
            new_rgb = list(map(lambda x: round(x * 255),
                               colorsys.hls_to_rgb(nh, l, s)))
            analogous_list.append(mc.rgb2hex([x / 255 for x in new_rgb]))
        return analogous_list

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
        Current plot markers.  Keys are model IDs; values are marker
        symbol names.
    ref_color: str
        Color value used for all reference line overlays.
    guide_line_style : dict
        Style parameters for guide line overlays.
    data_changed : bool
        Flag to indicate that data has changed recently.
    signals : sofia_redux.visualization.signals.Signals
        Custom signals, used to pass information to the controlling Figure.
    """

    def __init__(self, signals: Signals, ax: Optional[ma.Axes] = None) -> None:
        super().__init__(signals, ax)
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
        self.ref_color = 'dimgray'

        self.guide_line_style = {'linestyle': ':',
                                 'color': 'darkgray',
                                 'linewidth': 1,
                                 'animated': True}

        self.data_changed = True
        self.signals = signals

    def __eq__(self, other):
        if isinstance(other, OneDimPane):
            return self.ax == other.ax
        else:
            return False

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
                                 'color': mc.to_hex(self.colors[model_id][0]),
                                 'marker': self.markers[model_id],
                                 }
        return details

    def add_model(self, model: MT) -> List[DT]:
        """
        Add a model to the pane.

        The model is copied before adding so the data can be manipulated
        without changing the root model.

        If the model already exists in the pane, no action is taken.

        Parameters
        ----------
        model : high_model.HighModel
            Model object to add.

        Returns
        -------
        new_lines : list
            Keys are model IDs; values are dicts with order number
            keys and dict values, containing new artists added to
            the Pane axes.
        """
        new_lines = list()

        if model.id not in self.models.keys():
            self.models[model.id] = copy.deepcopy(model)
            self.models[model.id].extension = self.fields['y']

            marker_index = model.index % len(self.default_markers)
            self.markers[model.id] = self.default_markers[marker_index]

            self.orders[model.id] = list()
            self.colors[model.id] = list()
            for order in model.orders:
                label = f'{order.number}.{order.aperture}'
                self.orders[model.id].append(label)

                base_color_index = model.index % len(self.default_colors)
                base_color = self.default_colors[base_color_index]
                if order.aperture > 0:
                    aperture_colors = [base_color]
                    aperture_colors.extend(self.aperture_cycle[base_color])
                    color_index = order.aperture % len(aperture_colors)
                    color = aperture_colors[color_index]
                else:
                    color = base_color
                self.colors[model.id].append(color)
                log.debug(f'Model index: {model.index}; '
                          f'color index: {base_color_index}')

            self.data_changed = True

            new_order_lines = self._plot_model(self.models[model.id])
            if new_order_lines:
                new_lines.extend(new_order_lines)
            else:
                raise EyeError('Unable to plot model')

        return new_lines

    def update_model(self, models: MT):
        """
        Update a pre-existing model with a copy of the original model.

        The model is copied so that the data can be manipulated
        without changing the root model.

        Parameters
        ----------
        models : high_model.HighModel
            Model object to add.
        """

        for model_id, backup_model in models.items():
            if model_id in self.models.keys():
                # Need to get the current enabled status of the model
                # and apply it to the backup
                state = self.models[model_id].enabled_state()
                self.models[model_id] = copy.deepcopy(backup_model)
                self.models[model_id].set_enabled_state(state)

    def remove_model(self, filename: Optional[str] = None,
                     model_id: Optional[IDT] = None,
                     model: Optional[MT] = None) -> None:
        """
        Remove a model from the pane.

        The model can be specified by either its filename or
        the model directly.

        Parameters
        ----------
        filename : str, optional
            Name of the file corresponding to the model to remove.
        model_id : str, optional
            Specific id to a model object (high_model.HighModel).
        model : high_model.HighModel, optional
            Model object to remove.
        """
        target_mid = None
        if model_id:
            target_mid = model_id
        else:
            for k, m in self.models.items():
                if model:
                    if k == model.id:
                        target_mid = k
                elif filename:
                    if filename == m.filename:
                        target_mid = k
        if target_mid:
            log.debug(f'Removing {target_mid}')
            for collection in [self.models, self.colors, self.markers,
                               self.orders]:
                try:
                    del collection[target_mid]
                except KeyError:
                    pass
            if self.model_count() == 0:
                self.units = dict()
        else:
            log.debug(f'Unable to remove {model_id} ({model}, {filename})')

    def contains_model(self, model_id: str, order: Optional[int] = None
                       ) -> bool:
        """
        Check if a specified model exists in the pane.

        Parameters
        ----------
        model_id : str
            Model to be inspected.
        order : int, optional
            Order number to match. If provided, the order's presence
            in the pane is verified. If not, only the model's presence
            is verified.

        Returns
        -------
        bool
            True if the model/order are present; False otherwise.

        """
        if model_id in self.models.keys():
            if order is not None:
                if order in self.orders[model_id]:
                    return True
                else:
                    return False
            else:
                return True
        return False

    def update_reference_data(self, reference_models: Optional[RT] = None,
                              plot: Optional[bool] = True
                              ) -> Optional[List[Union[DT, Dict]]]:
        """
        Update the reference data.

        Parameters
        ----------
        reference_models : reference_model.ReferenceData, optional
            Reference models to update.
        plot : bool
            If set, reference data is (re)plotted. Otherwise, current
            reference options are returned.

        Returns
        -------
        reference_artists : list
            A list of drawing.Drawing objects when `plot` is True.
            A list of dictionaries when `plot` is False. If the
            inputs are invalid then None is returned.

        """
        if reference_models is None and self.reference is None:
            return None
        if self.fields['x'] != 'wavepos':
            return None
        if reference_models is not None:
            # replace any existing reference with new one
            self.reference = reference_models
        if plot:
            reference_artists = self._plot_reference_lines()
        else:
            reference_artists = self._current_reference_options()

        return reference_artists

    def unload_ref_model(self):
        """Unload the reference data."""
        if self.reference:
            self.reference.unload_data()

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
        else:
            if any(self.units.values()):
                return self.units
            else:
                for axis in current.keys():
                    # take unit from first model only -- they better match
                    if self.fields[axis] is None:
                        continue
                    models = list(self.models.values())
                    low_model = models[0].retrieve(
                        order=0, field=self.fields[axis], level='low')
                    if low_model is None:
                        continue
                    current[axis] = low_model.unit_key
                return current

    def set_orders(self, orders: Dict[IDT, List[int]],
                   enable: Optional[bool] = True,
                   aperture: Optional[bool] = False,
                   return_updates: Optional[bool] = False
                   ) -> Optional[List[DT]]:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are model IDs; values are lists of orders
            to enable.
        enable : bool, optional
            Set to True to enable; False to disable.
        aperture : bool, optional
            True if apertures are present; False otherwise.
        return_updates : bool, optional
            If set, an update structure is returned.

        Returns
        -------
        updates : list, optional
            Order visibility updates applied.
        """
        updates = list()
        for model_id, model_orders in orders.items():
            try:
                available_orders = self.orders[model_id]
            except KeyError:
                continue
            else:
                log.debug(f'{"Enabling" if enable else "Disabling"} '
                          f' {model_orders} in '
                          f'{available_orders}')
                self.models[model_id].enable_orders(model_orders, enable,
                                                    aperture)
                self.data_changed = True

                if return_updates:
                    updates.extend(self._order_visibility_updates(model_id))
        return updates

    def _order_visibility_updates(self, model_id, error=False):
        model = self.models[model_id]
        updates = list()
        for order_number in self.orders[model_id]:
            try:
                o_num, a_num = order_number.split('.')
            except ValueError:  # pragma: no cover
                o_num = order_number
                a_num = 0
            order = model.retrieve(order=o_num, aperture=a_num,
                                   level='high')
            spectrum = model.retrieve(order=order_number, aperture=a_num,
                                      level='low', field=self.fields['y'])
            if order is None or spectrum is None:  # pragma: no cover
                continue
            visible = model.enabled & order.enabled & spectrum.enabled
            error_visible = visible & self.show_error
            mid_model = f'{o_num}.{a_num}'

            args = {'high_model': model.filename, 'mid_model': mid_model,
                    'model_id': model.id, 'axes': 'any'}
            if error:
                line = drawing.Drawing(updates={'visible': error_visible},
                                       kind='error', **args)

                updates.append(line)

            else:
                line = drawing.Drawing(updates={'visible': visible},
                                       kind='line', **args)

                updates.append(line)
                line = drawing.Drawing(updates={'visible': error_visible},
                                       kind='error', **args)

                updates.append(line)
        return updates

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
            If True, enable all models.  If False, disable all models.
        """
        for model_id in self.models:
            self.set_model_enabled(model_id, state)

    ####
    # Plotting
    ####
    def create_artists_from_current_models(self) -> List[DT]:
        """
        Create new artists from all current models.

        Returns
        -------
        new_artists : list
            A list of reference_model.ReferenceData objects associated with
            the current model.
        """
        # TODO add a clause for y-alt
        if self.fields['y'] == self.fields['x']:
            self.signals.obtain_raw_model.emit()

        new_artists = list()
        to_remove = set()
        for model_id, model in self.models.items():
            new_artist = self._plot_model(model)
            if new_artist is None:
                to_remove.add(model)
            else:
                new_artists.extend(new_artist)

        if to_remove:
            names = list()
            for model in list(to_remove):
                self.remove_model(model=model)
                names.append(os.path.basename(model.filename))
            log.warning(f'Files {", ".join(names)} are not compatible with '
                        f'the pane\'s current settings.')

        return new_artists

    def _plot_model(self, model: MT) -> Optional[List[DT]]:
        """
        Plot a model in the current axes.

        Parameters
        ----------
        model : high_model.HighModel
            The model to plot.

        Returns
        -------
        new_lines : list of Drawing
            New lines plotted. None is returned if the model is
            incompatible with the pane, such as irreconcilable units.
        """
        log.debug(f'Starting limits: {self.get_axis_limits()}, '
                  f'{self.ax.get_autoscale_on()}')

        model.extension = self.fields['y']

        log.debug(f'Plotting {len(self.orders[model.id])} orders '
                  f'for {model.id}')
        new_lines = list()
        n_orders = len(model.orders)
        n_apertures = max([model.num_aperture, 1])
        if n_orders > 1:
            alpha_val = np.linspace(0.7, 1, n_orders)
            log.debug(f'Alpha values for {n_orders} orders: {alpha_val}')
        else:
            alpha_val = [1]
        for ap_i in range(n_apertures):
            for order_i, orders in enumerate(model.orders):
                order = orders.number
                aperture = orders.aperture
                spectrum = model.retrieve(order=order, level='low',
                                          field=self.fields['y'],
                                          aperture=aperture)

                if spectrum is None:
                    continue
                # convert to current units, or remove model and
                # stop trying to load it
                # log.debug(f'Current units: {self.units}')
                if any(self.units.values()):
                    try:
                        if self.fields['y'] == self.fields['x']:
                            log.debug(f'Testing same fields: {self.fields}')
                            x_model, y_model = self.get_xy_data(model, order,
                                                                aperture)
                            x = x_model.retrieve(order=order, level='raw',
                                                 field=self.fields['x'],
                                                 aperture=aperture)
                            y = y_model.retrieve(order=order, level='raw',
                                                 field=self.fields['y'],
                                                 aperture=aperture)
                        else:
                            self._convert_low_model_units(model, order, 'y',
                                                          self.units['y'],
                                                          aperture)
                            self._convert_low_model_units(model, order, 'x',
                                                          self.units['x'],
                                                          aperture)
                            x = model.retrieve(order=order, level='raw',
                                               field=self.fields['x'],
                                               aperture=aperture)
                            y = model.retrieve(order=order, level='raw',
                                               field=self.fields['y'],
                                               aperture=aperture)
                    except ValueError:
                        return None
                else:
                    x = model.retrieve(order=order, level='raw',
                                       field=self.fields['x'],
                                       aperture=aperture)
                    y = model.retrieve(order=order, level='raw',
                                       field=self.fields['y'],
                                       aperture=aperture)

                if x is None or y is None:
                    log.debug(f'Failed to retrieve raw data for primary '
                              f'{model.id}, {order}, {self.fields}')
                    continue

                if x.shape != y.shape:
                    log.debug(f'Incompatible data shapes: '
                              f'{x.shape} and {y.shape}; skipping.')
                    continue

                visible = model.enabled and spectrum.enabled and orders.enabled

                label_fields = [f'{model.filename}', f'Order {order + 1}',
                                f'{self.fields["y"]}']
                if n_apertures > 1:
                    label_fields.insert(2, f'Aperture {aperture + 1}')
                label = ', '.join(label_fields)

                # add line artists
                line = self._plot_single_line(x, y, model_id=model.id,
                                              visible=visible, label=label,
                                              alpha=alpha_val[order_i],
                                              aperture=aperture)
                new_line = {'artist': line, 'fields': self.fields,
                            'kind': 'line', 'high_model': model.filename,
                            'mid_model': f'{order}.{aperture}',
                            'label': label, 'axes': 'primary',
                            'model_id': model.id}

                new_lines.append(drawing.Drawing(**new_line))

                # add cursor artists
                cursor = self._plot_cursor(x, y, model_id=model.id,
                                           aperture=aperture)
                new_line['artist'] = cursor
                new_line['kind'] = 'cursor'

                new_lines.append(drawing.Drawing(**new_line))

                fields_with_errors = {'flux': 'spectral_error',
                                      'response': 'response_error'}
                for field, error_field in fields_with_errors.items():
                    if field not in self.fields['y'].lower():
                        continue
                    elif error_field in self.fields['y'].lower():
                        continue
                    error = model.retrieve(order=order, level='raw',
                                           field=error_field,
                                           aperture=aperture)
                    label_fields[-1] = f'{error_field}'
                    label = ', '.join(label_fields)
                    color = self.colors[model.id][aperture]
                    line = self._plot_flux_error(x, y, error,
                                                 color=color,
                                                 label=label)
                    error_visible = visible and self.show_error
                    if not error_visible:
                        line.set_visible(False)
                    new_line['artist'] = line
                    new_line['kind'] = 'error_range'
                    new_line['label'] = label
                    new_lines.append(drawing.Drawing(**new_line))
                if self.show_overplot:
                    y = model.retrieve(order=order, level='raw',
                                       field=self.fields['y_alt'])
                    if y is None:
                        log.debug(f'Failed to retrieve raw data for alt y '
                                  f'{model.id}, {order}, {self.fields}')
                    elif x.shape == y.shape:
                        # only add overplot if x and y shapes match
                        label_fields[-1] = f'{self.fields["y_alt"]}'
                        label = ', '.join(label_fields)
                        line = self._plot_single_line(x, y, model_id=model.id,
                                                      visible=visible,
                                                      label=label, axis='alt')
                        new_line['axes'] = 'alt'
                        new_line['artist'] = line
                        new_line['kind'] = 'line'
                        new_line['label'] = label
                        new_lines.append(drawing.Drawing(**new_line))

                        cursor = self._plot_cursor(x, y, model_id=model.id,
                                                   axis='alt',
                                                   aperture=aperture)
                        new_line['artist'] = cursor
                        new_line['kind'] = 'cursor'
                        new_lines.append(drawing.Drawing(**new_line))

            # turn on/off grid lines as desired
            else:
                # This is only reached if the order loop did
                # not break. Breaks occur if the data is incompatible
                # with the current pane, and will be true for all
                # apertures in the model. Without this and the following
                # break statement the user would get a separate pop-up
                # warning for each aperture.
                continue
            break  # pragma: no cover
        self.ax.grid(self.show_grid)

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
                          axis='primary', alpha=1, aperture=None):
        if axis == 'primary':
            ax = self.ax
            width = 1.5
            scalex = True
            if self.plot_type == 'scatter':
                style = ''
                marker = self.markers[model_id]
            else:
                style = '-'
                if self.show_markers:
                    marker = self.markers[model_id]
                else:
                    marker = None
        else:
            ax = self.ax_alt
            alpha = 0.5
            width = 1
            style = ':'
            scalex = False
            marker = None
        if aperture is None:
            aper_index = 0
        else:
            aper_index = aperture % len(self.aperture_cycle)
        color = self.colors[model_id][aper_index]
        style_kwargs = {'color': color,
                        'alpha': alpha,
                        'linewidth': width,
                        'linestyle': style,
                        'marker': marker,
                        'scalex': scalex}

        if self.plot_type == 'step':
            (line,) = ax.step(x, y, where='mid', animated=True, label=label,
                              **style_kwargs)
        else:
            (line,) = ax.plot(x, y, animated=True, label=label,
                              **style_kwargs)
        if not visible:
            line.set_visible(False)
        return line

    def _plot_cursor(self, x, y, model_id, axis='primary', aperture=0):
        """
        Create a cursor marker artist.

        Parameters
        ----------
        x : np.ndarray
            Array of x-axis data points
        y : np.ndarray
            Array of y-axis data points
        model_id : uuid.UUID4
            Unique id (uuid4) associated with each hdul.
        axis : {'primary', 'alt'}
            It denotes which axis is being considered - primary or the
            alternative axis. Default is set to `Primary`.

        Returns
        -------
        Cursor : matplotlib.collections.PathCollection
            Cursor artist for the given axis.
        """
        if axis == 'primary':
            ax = self.ax
            alpha = 1
            marker = 'x'
            color = self.colors[model_id][aperture]
        else:
            ax = self.ax_alt
            alpha = 0.5
            marker = '^'
            color = self.colors[model_id][aperture]
        style_kwargs = {'marker': marker,
                        'color': color,
                        'alpha': alpha,
                        'linestyle': None}
        # cursor = ax.scatter(x[0], y[0], animated=True, **style_kwargs)
        cursor = ax.plot(x[0], y[0], animated=True, **style_kwargs)[0]
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
        if error is None:
            error = np.full(len(x), np.nan)

        poly = self.ax.fill_between(x, y - error, y + error,
                                    **options)
        return poly

    def _plot_reference_lines(self) -> List[Optional[DT]]:
        """
        Generate a list of reference lines.

        Returns
        -------
        lines : list
            List of drawing.Drawing objects for reference lines to be plotted.
        """
        lines = list()
        if self.reference is None or len(self.models) == 0:
            return lines
        label_style = {'color': self.ref_color, 'fontfamily': 'monospace',
                       'rotation': 'vertical', 'horizontalalignment': 'right',
                       'zorder': 0, 'alpha': 0.6, 'fontsize': 'small'}
        line_style = {'color': self.ref_color, 'alpha': 0.6, 'linestyle': '-',
                      'zorder': 0}

        label_ypos = 0.05  # in axes fraction
        to_plot = self._window_line_list()
        for name, wavelengths in to_plot.items():
            for wavelength in wavelengths:
                if self.reference.get_visibility('ref_line'):
                    line = self.ax.axvline(wavelength, label=name,
                                           **line_style)
                    if self.reference.get_visibility('ref_label'):
                        label = self.ax.annotate(
                            name, (wavelength, label_ypos),
                            xycoords=('data', 'axes fraction'), **label_style)
                    else:
                        label = None
                else:
                    line = None
                    label = None

                new = {'artist': line, 'kind': 'ref_line',
                       'fields': self.fields,
                       'high_model': 'reference', 'mid_model': 'line',
                       'axis': 'primary', 'pane': self}
                lines.append(drawing.Drawing(**new))
                new = {'artist': label, 'kind': 'ref_label',
                       'fields': self.fields,
                       'high_model': 'reference', 'mid_model': 'label',
                       'axis': 'primary', 'data_id': f'{wavelength:.5f}',
                       'pane': self}
                lines.append(drawing.Drawing(**new))
        return lines

    def _window_line_list(self, xlim: Optional[Union[Tuple, List[Num]]] = None
                          ) -> Dict[str, List[float]]:
        if xlim is None:
            xlim = self.get_axis_limits('x')
        names_in_limits = dict()
        try:
            converted_lines = self.reference.convert_line_list_unit(
                target_unit=self.units['x'])
        except KeyError:
            return names_in_limits

        for n, w in converted_lines.items():
            for x in w:
                if xlim[0] < x < xlim[1]:
                    if n in names_in_limits:
                        names_in_limits[n].append(x)
                    else:
                        names_in_limits[n] = [x]
        if self.units['x'] == 'pixel':
            names_in_limits = dict()

        return names_in_limits

    def _current_reference_options(self):
        kinds = ['ref_line', 'ref_label']
        artist_kinds = ['ref_lines', 'ref_labels']
        options = list()
        if self.reference is None:
            return options

        to_plot = self._window_line_list()
        for name, wavelength in to_plot.items():
            for a_kind, kind in zip(artist_kinds, kinds):
                option = dict()
                option['model_id'] = a_kind
                option['order'] = name
                option['color'] = self.colors.get(kind, self.ref_color)
                option['visible'] = self.reference.get_visibility(kind)
                options.append(option)
        return options

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
                    self.ax_alt.set_xlim(self.limits['x'])
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

    def update_colors(self) -> List[DT]:
        """
        Update plot colors for current loaded data.

        Returns
        -------
        updates : list of drawing.Drawing
            Drawing objects containing all color updates.
        """
        updates = list()
        for model_id, model in self.models.items():
            self.colors[model.id] = list()
            for order_number in self.orders[model_id]:
                base_color_index = model.index % len(self.default_colors)
                base_color = self.default_colors[base_color_index]
                a = int(order_number.split('.')[1])
                if a > 0:
                    aperture_colors = self.aperture_cycle[base_color]
                    index = a % len(aperture_colors)
                    line_color = aperture_colors[index]
                else:
                    line_color = base_color
                self.colors[model_id].append(line_color)
                fit_color = self.grayscale(line_color)
                args = {'high_model': model.filename,
                        'mid_model': order_number,
                        'model_id': model.id}
                line = drawing.Drawing(kind='line', axes='primary',
                                       updates={'color': line_color}, **args)
                line_alt = drawing.Drawing(kind='line', axes='alt',
                                           updates={'color': line_color},
                                           **args)
                error = drawing.Drawing(kind='error_range',
                                        updates={'color': line_color}, **args)
                fit = drawing.Drawing(kind='fit', updates={'color': fit_color},
                                      **args)
                cursor = drawing.Drawing(kind='cursor',
                                         updates={'color': line_color}, **args)
                updates.append(line)
                updates.append(line_alt)
                updates.append(error)
                updates.append(fit)
                updates.append(cursor)

        # also update border color
        border = drawing.Drawing(high_model='border', kind='border',
                                 updates={'color': self.default_colors[1]})
        updates.append(border)

        return updates

    def update_visibility(self, error: Optional = False) -> List[DT]:
        """
        Update plot visibility for current loaded data.

        Parameters
        ----------
        error : bool, optional
            If set, visibility update is applied to error range plots
            only.

        Returns
        -------
        updates : list of drawing.Drawing
            Drawing objects containing all visibility updates.
        """
        updates = list()
        for model_id, model in self.models.items():
            updates.extend(self._order_visibility_updates(model_id, error))
        return updates

    ####
    # Getters
    ####
    def get_axis_limits(self, axis: Optional[str] = None
                        ) -> Union[Dict[str, List[Num]], List[Num]]:
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
                   by_model: bool = False,
                   target_model: Optional[MT] = None,
                   filename: Optional[str] = None,
                   model_id: Optional[IDT] = None,
                   kind: Optional[str] = 'order',
                   ) -> Union[Dict[str, List], List]:
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
        target_model : high_model.HighModel, optional
            Target model
        filename : str, optional
            Name of file
        model_id : uuid.UUID, optional
            Unique UUID for an HDUL.
        kind : str, optional
            If 'aperture', only apertures are returned. If 'order',
            only orders are returned. Otherwise, both are returned.

        Returns
        -------
        orders : list, dict
            Format and details depend on arguments.
        """
        target = None
        identifiers = [model_id, target_model, filename]
        if all([i is None for i in identifiers]):
            targets = list(self.models.values())
        else:
            try:
                target = self.models[model_id]
            except KeyError:
                for mid, model in self.models.items():
                    if target_model:  # pragma: no cover
                        if model == target_model:
                            target = model
                        else:
                            continue
                    elif filename:
                        if model.filename == filename:
                            target = model
                        else:  # pragma: no cover
                            continue
            if target is None:
                return list()
            else:
                targets = [target]
        orders = dict()
        for target in targets:
            if enabled_only:
                enabled = target.list_enabled()
                if kind == 'aperture':
                    model_orders = enabled['apertures']
                elif kind == 'order':
                    model_orders = enabled['orders']
                else:
                    model_orders = list(set(enabled['apertures']
                                            + enabled['orders']))
            else:
                if kind == 'aperture':
                    model_orders = [o.aperture for o in target.orders]
                elif kind == 'order':
                    model_orders = [o.number for o in target.orders]
                else:
                    model_orders = list(set(
                        [o.aperture for o in target.orders]
                        + [o.number for o in target.orders]))
            orders[target.id] = model_orders
        if by_model:
            return orders
        else:
            full_orders = list()
            for v in orders.values():
                full_orders.extend(list(v))
            full_orders = list(set(full_orders))
            return full_orders

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
                                 axis: str, target_unit: str, aperture: int
                                 ) -> None:
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
        spectrum = model.retrieve(order=order_number, aperture=aperture,
                                  level='low', field=self.fields[axis])
        if spectrum is None:
            raise EyeError(f'Retrieved None from {model} (order '
                           f'{order_number}, field {self.fields[axis]}')
        if spectrum.kind in ['flux', 'wavelength']:
            wave_spectrum = model.retrieve(order=order_number,
                                           aperture=aperture,
                                           level='low', field='wavepos')
            wavelength_data = wave_spectrum.data
            wavelength_unit = wave_spectrum.unit_key
        else:
            wavelength_data = None
            wavelength_unit = None
        spectrum.convert(target_unit, wavelength_data, wavelength_unit)

        if 'flux' in self.fields[axis]:
            error_spectrum = model.retrieve(aperture=aperture,
                                            order=order_number,
                                            level='low',
                                            field='spectral_error')
            if error_spectrum is not None:
                error_spectrum.convert(target_unit, wavelength_data,
                                       wavelength_unit)

    def get_xy_data(self, model, order, aperture):
        """
        Get copies of the low model data for x and y axes.

        Parameters
        ----------
        model : high_model.HighModel
            The model to be copied.
        order : int
            The spectral order to retrieve.
        aperture : int
            The aperture to retrieve.

        Returns
        -------
        x_model : models.high_model.Grism
            A copy of model along the x-axis
        y_model : models.high_model.Grism
            A copy of model along the y-axis
        """
        x_model = copy.deepcopy(model)
        y_model = copy.deepcopy(model)
        self._convert_low_model_units(y_model, order,
                                      'y', self.units['y'], aperture=aperture)
        self._convert_low_model_units(x_model, order,
                                      'x', self.units['x'], aperture=aperture)
        return x_model, y_model

    def set_units(self, units: Dict[str, str], axes: str,
                  ) -> List[DT]:
        """
        Set the plot units.

        Parameters
        ----------
        units : dict
            Keys are axis names; values are units to set.
        axes: 'primary', 'alt', 'both', 'all'
            Which Axes object to pull data from.

        Returns
        -------
        updates : list
            A list of all artists for plotting the changes in units.
            May not return anything when unable to convert units.
        """
        updates = list()
        if 'y_alt' not in units and axes in ['alt', 'all']:
            units['y_alt'] = units['y']
            if axes == 'alt':
                del units['y']
        for axis, current_unit in self.units.items():
            try:
                target_unit = units[axis]
            except KeyError:
                continue
            if target_unit == current_unit:
                continue
            updated = False
            min_lim = np.inf
            max_lim = -np.inf

            # pixel conversion
            if current_unit == 'pixel':
                self.signals.obtain_raw_model.emit()

            if self.units['x'] == 'pixel' and axis == 'y':
                self.signals.obtain_raw_model.emit()

            for model_id, model in self.models.items():
                for order_number in self.orders[model_id]:
                    o_num, a_num = (int(i) for i in order_number.split('.'))
                    try:
                        self._convert_low_model_units(model, o_num,
                                                      axis, target_unit,
                                                      a_num)
                    except ValueError:
                        log.debug(f'Cannot convert units to '
                                  f'{target_unit} for {model_id}; '
                                  f'ignoring')
                        break
                    else:
                        updated = True
                        data = model.retrieve(order=o_num,
                                              level='raw', aperture=a_num,
                                              field=self.fields[axis])
                        details = {f'{axis}_data': data}
                        update = drawing.Drawing(high_model=model.filename,
                                                 mid_model=order_number,
                                                 model_id=model.id,
                                                 fields=self.fields,
                                                 kind='line',
                                                 updates=details)
                        updates.append(update)

                        # track limit changes for reference data windowing
                        data_min, data_max = np.nanpercentile(data, [0, 100])
                        if data_min < min_lim:
                            min_lim = data_min
                        if data_max > max_lim:
                            max_lim = data_max

            if updated:
                self.units[axis] = units[axis]
                self.data_changed = True
                if np.all(np.isfinite([min_lim, max_lim])):
                    log.debug(f'Min/max data limits: {min_lim} -> {max_lim}')
                    self.limits[axis] = [min_lim, max_lim]

        if len(updates) > 0:
            if 'flux' in self.fields['y']:
                updates.extend(self._update_error_artists())

            if self.reference:
                ref_updates = self._update_reference_artists()
                updates.extend(ref_updates)
        return updates

    def _update_error_artists(self) -> List[DT]:
        """
        Update error range artists to new data.

        Returns
        -------
        updates : list
            A list of all artists for plotting the changes
            in units in error bars.
        """
        updates = list()

        for model_id, model in self.models.items():
            for order in self.orders[model_id]:
                o_num, a_num = order.split('.')
                args = {'order': o_num, 'aperture': a_num, 'level': 'raw'}
                error = model.retrieve(field='spectral_error', **args)
                x = model.retrieve(field=self.fields['x'], **args)
                y = model.retrieve(field=self.fields['y'], **args)
                label = (f'{model.id}, Order {int(o_num) + 1}, '
                         f'Aperture {int(a_num) + 1} spectral_error')
                color = self.colors[model.id][int(a_num)]
                poly = self._plot_flux_error(x, y, error, label=label,
                                             color=color)
                if not model.enabled or not self.show_error:
                    poly.set_visible(False)
                update = drawing.Drawing(high_model=model.filename,
                                         mid_model=order, axis='primary',
                                         kind='error', label=label,
                                         updates={'artist': poly},
                                         model_id=model.id)
                updates.append(update)
        return updates

    def _update_reference_artists(self) -> List[DT]:
        updates = self._plot_reference_lines()
        return updates

    def set_fields(self, fields: Dict[str, str]) -> None:
        """
        Set the plot fields.

        Parameters
        ----------
        fields : dict
            Keys are axis names; values are fields to set.
        """
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
                    self.set_default_units_for_fields()
                    self.data_changed = True
                else:  # pragma: no cover
                    # ignore: this is likely from setting an
                    # all primary/all overplots from another pane's value
                    log.debug(f'Invalid field provided for axis {axis}: '
                              f'{fields[axis]}')

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

    def set_plot_type(self, plot_type: str) -> List[DT]:
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
        for model_id, model in self.models.items():
            for order_number in self.orders[model_id]:
                details = {'type': self.plot_type}
                if self.plot_type == 'scatter' or self.show_markers:
                    details['marker'] = self.markers[model_id]
                update = drawing.Drawing(high_model=model.filename,
                                         mid_model=order_number,
                                         kind='line', axes='primary',
                                         updates=details, model_id=model.id)
                updates.append(update)
        return updates

    def set_markers(self, state: bool) -> List[DT]:
        """
        Set plot marker symbols.

        Only applies to non-scatter plots. Scatter plots
        will accept the new state but not update any
        artists.

        Parameters
        ----------
        state : bool
            Defines the visibility of the makers. True
            will make the markers visible, False will
            hide the markers.

        Returns
        -------
        updates : list
            List of drawings describing the changes for each model.
        """
        self.show_markers = bool(state)
        updates = list()
        # no-op for scatter plots
        if self.plot_type == 'scatter':
            return updates
        for model_id, model in self.models.items():
            for order_number in self.orders[model_id]:
                if self.show_markers:
                    marker = {'marker': self.markers[model_id]}
                else:
                    marker = {'marker': None}
                args = {'high_model': model.filename,
                        'mid_model': order_number,
                        'kind': 'line', 'axes': 'primary',
                        'model_id': model.id, 'updates': marker}
                updates.append(drawing.Drawing(**args))
        return updates

    def get_marker(self, model_id: Union[List[IDT], IDT]) -> List[str]:
        """
        Get the plot markers for a given model_id.

        Parameters
        ----------
        model_id : uuid.UUID or list of uuid.UUID
          Unique id associated with an HDUL

        Returns
        -------
        markers : list
            A list of markers for the model_id
        """
        markers = list()
        if not isinstance(model_id, list):
            model_id = [model_id]

        for model_name, marker in self.markers.items():
            for name in model_id:
                if model_name == name:
                    markers.append(marker)
        return markers

    def get_color(self, model_id):
        """
        Return colors for a model.

        Parameters
        ----------
        model_id : uuid.UUID
            Unique id of HDUL

        Returns
        -------
        colors : list
            List of colors
        """
        colors = list()
        if not isinstance(model_id, list):
            model_id = [model_id]

        for model_name, color in self.colors.items():
            for name in model_id:
                if model_name == name:
                    colors.append(color)
        return colors

    def set_grid(self, state: bool) -> None:
        """
        Set the plot grid visibility.

        Parameters
        ----------
        state : bool
            True for enabling grid, otherwise False.
        """
        self.show_grid = bool(state)
        if self.ax:
            self.ax.grid(self.show_grid)

    def set_error(self, state: bool) -> None:
        """
        Set the plot error range visibility.

        Parameters
        ----------
        state : bool
            True for enabling error-bars, otherwise False.
        """
        self.show_error = bool(state)

    def set_overplot(self, state: bool) -> None:
        """
        Enable/disable overplot, dependent axes, field, and limits.

        Parameters
        ----------
        state : bool
            A boolean which is assigned to the state of show_overplot.
        """
        if bool(state) is bool(self.show_overplot):  # pragma: no cover
            return
        self.show_overplot = bool(state)
        if self.show_overplot:
            if self.ax:
                self.ax_alt = self.ax.twinx()
                self.ax_alt.autoscale(enable=True, axis='y')
                self.fields['y_alt'] = 'transmission'
                self.scale['y_alt'] = 'linear'
                self.limits['y_alt'] = [0, 1]
        else:
            self.fields['y_alt'] = ''
            self.scale['y_alt'] = ''
            self.limits['y_alt'] = list()

    def reset_alt_axes(self, remove=False):
        """
        Remove or reset the alternate axis.

        Replaces the `ax_alt` attribute with None and resets fields,
        scale, and limit for the 'y_alt' axis to empty values.

        Parameters
        ----------
        remove : bool, optional
            If set, an attempt is made to remove the alternate axis
            from the plot before resetting it.
        """
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
                try:
                    o, a = order_number.split('.')
                except ValueError:  # pragma: no cover
                    o_num = int(order_number)
                    a_num = 0
                else:
                    o_num = int(o)
                    a_num = int(a)
                # o_num, a_num are zero-based indexing
                order = model.retrieve(order=o_num, level='high',
                                       aperture=a_num)
                spectrum = model.retrieve(order=o_num, level='low',
                                          field=self.fields['y'],
                                          aperture=a_num)
                if self.fields['y'] == self.fields['x']:
                    x_model, y_model = self.get_xy_data(model, o_num, a_num)
                else:
                    x_model, y_model = model, model

                x_data = x_model.retrieve(order=o_num, level='raw',
                                          field=self.fields['x'],
                                          aperture=a_num)
                y_data = y_model.retrieve(order=o_num, level='raw',
                                          field=self.fields['y'],
                                          aperture=a_num)

                # skip entirely if order has no data
                data_list = [x_data, y_data, model, order, spectrum]
                if any([x is None for x in data_list]):  # pragma: no cover
                    continue

                visible = model.enabled & order.enabled & spectrum.enabled
                # skip order if cursor is out of range
                if (cursor_x < np.nanmin(x_data)
                        or cursor_x > np.nanmax(x_data)):
                    visible = False

                index = int(np.nanargmin(np.abs(x_data - cursor_x)))
                x = x_data[index]
                y = y_data[index]
                if all(np.isnan([x, y])):
                    visible = False

                data[model_id].append({'filename': model.filename,
                                       'order': o_num, 'aperture': a_num,
                                       'bin': index, 'bin_x': x, 'bin_y': y,
                                       'x_field': self.fields['x'],
                                       'y_field': self.fields['y'],
                                       'color': self.colors[model_id][a_num],
                                       'visible': visible,
                                       'alt': False
                                       })
                if self.show_overplot:
                    y_data = model.retrieve(order=o_num,
                                            level='raw',
                                            field=self.fields['y_alt'],
                                            aperture=a_num)
                    if y_data is not None:
                        y = y_data[index]
                        if all(np.isnan([x, y])):
                            visible = False
                        # Add a 65% alpha to alt-axis color
                        color = '#A6' + self.colors[model_id][a_num].strip('#')
                        data[model_id].append(
                            {'filename': model.filename,
                             'order': o_num, 'aperture': a_num,
                             'bin': index, 'bin_x': x, 'bin_y': y,
                             'x_field': self.fields['x'],
                             'y_field': self.fields['y_alt'],
                             'color': color, 'visible': visible, 'alt': True
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
                       ) -> List[DT]:
        """
        Create crosshair artists to show the cursor location.

        Parameters
        ----------
        cursor_pos : tuple or list, optional
            Current cursor position [x, y].  If not specified, the
            cursor is initially set near the center of the plot.

        Returns
        -------
        crosshair : list
            Two element list containing the vertical and horizontal
            crosshair drawings.
        """
        if cursor_pos is None:
            # set cursor near center of plot
            cursor_pos = [np.mean(self.ax.get_xlim()),
                          np.mean(self.ax.get_ylim())]
        vertical = self.ax.axvline(cursor_pos[0], **self.guide_line_style,
                                   visible=False)
        horizontal = self.ax.axhline(cursor_pos[1], **self.guide_line_style,
                                     visible=False)
        vert = drawing.Drawing(high_model='crosshair', mid_model='vertical',
                               kind='crosshair', artist=vertical,
                               fields=self.fields, pane=self)
        horiz = drawing.Drawing(high_model='crosshair', mid_model='horizontal',
                                kind='crosshair', artist=horizontal,
                                fields=self.fields, pane=self)
        return [vert, horiz]

    def plot_guides(self, cursor_pos: Union[Tuple, List],
                    kind: str) -> List[DT]:
        """
        Create guide gallery.

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
        guides : list
            Guide drawings corresponding to plotted lines.
        """
        guides = list()
        if kind in ['vertical', 'cross', 'x', 'b']:
            vertical = self.ax.axvline(cursor_pos[0],
                                       **self.guide_line_style)
            vert = drawing.Drawing(high_model='guide', mid_model='vertical',
                                   kind='guide', artist=vertical,
                                   pane=self, fields=self.fields)
            guides.append(vert)
        if kind in ['horizontal', 'cross', 'y', 'b']:
            horizontal = self.ax.axhline(cursor_pos[1],
                                         **self.guide_line_style)
            horiz = drawing.Drawing(high_model='guide', mid_model='horizontal',
                                    kind='guide', artist=horizontal,
                                    pane=self, fields=self.fields)
            guides.append(horiz)
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

        # Don't want to include reference data in the relim
        # note: relim works for lines, but not for true scatter plots
        children = list()
        if self.reference:
            visibility = self.reference.get_visibility('ref_line')
            for child in self.ax.get_children():
                if (isinstance(child, ml.Line2D)
                        or isinstance(child, mt.Text)):
                    if child.get_color() == self.ref_color:
                        child.set_visible(False)
                        children.append(child)

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
        if self.reference:
            names_in_limits = self._window_line_list(new_limits['x'])
            for child in children:
                try:
                    label = str(child.get_text())
                except AttributeError:
                    label = str(child.get_label())
                if label in names_in_limits:
                    child.set_visible(visibility)

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
            The fit_artists dict contains overlay gallery showing
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
        """
        Fit a spectral feature.

        Parameters
        ----------
        feature : {'moffat', 'gaussian', 'gauss'}
            Feature model to fit.
        background : {'linear', 'constant'}
            Baseline model to fit (Const1D or Linear1D).
        limits : list of list
            Low and high limits for data to fit, as
            [[low_x, low_y], [high_x, high_y]].  Currently
            only the x values are used.

        Returns
        -------
        fit_artists, fit_params : Tuple[list, list]
             fit_artists - a list of drawing.Drawing objects for fits
             fit_params - a list of fitting parameters objects for fits
        """
        fit_params = list()
        successes = list()
        for model_id, model_ in self.models.items():
            # model not enabled: skip
            if not model_.enabled:
                log.debug(f'Model {model_.filename} is not enabled; skipping.')
                continue

            filename = model_.filename
            for order in model_.orders:
                order_num = order.number
                aper_num = order.aperture
                order_model = model_.retrieve(order=order_num, level='high',
                                              aperture=aper_num)
                if order_model is None or not order_model.enabled:
                    log.debug(f'Order {order_num} of model {model_.filename} '
                              f'is not enabled; skipping')
                    continue
                spectrum = model_.retrieve(order=order_num, level='low',
                                           aperture=aper_num,
                                           field=self.fields['y'])
                if spectrum is None:  # pragma: no cover
                    log.debug(f'Order {order_num + 1} is not '
                              f'available; skipping.')
                    continue
                if not spectrum.enabled:
                    log.debug(f'Spectrum {self.fields["y"]} of order '
                              f'{order_num}, model {model_.filename} is not '
                              f'enabled; skipping')
                    continue

                x = model_.retrieve(order=order_num, level='raw',
                                    aperture=aper_num, field=self.fields['x'])
                y = model_.retrieve(order=order_num, level='raw',
                                    aperture=aper_num, field=self.fields['y'])

                # scale data to account for units with very small increments
                xs = np.nanmean(x)
                ys = np.nanmean(y)
                norm_limits = [[limits[0][0] / xs, limits[0][1] / ys],
                               [limits[1][0] / xs, limits[1][1] / ys]]
                xnorm = x / xs
                ynorm = y / ys

                color = self.colors[model_id][aper_num]
                blank_fit = self._initialize_model_fit(feature, background,
                                                       norm_limits, model_id,
                                                       filename, order_num,
                                                       aper_num, x, color)
                try:
                    xnorm, ynorm, fit_init, bounds = self.initialize_models(
                        feature, background, xnorm, ynorm, norm_limits)
                except EyeError as e:
                    if str(e) == 'empty_order':
                        log.debug(f'Order {order_num + 1}, aperture '
                                  f'{aper_num + 1}  has no valid data '
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
                                  f'order {order_num + 1}')
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
                              limits: List[List[float]], model_id: uuid.UUID,
                              filename: str, order: int, aperture: int,
                              columns: ArrayLike, color: str
                              ) -> model_fit.ModelFit:
        fit = model_fit.ModelFit()

        # catch '-' from dropdown
        ftypes = list()
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
        fit.set_filename(filename)
        fit.set_order(order)
        fit.set_aperture(aperture)
        fit.set_columns(columns)
        fit.set_color(color)
        return fit

    @staticmethod
    def calculate_fit(x_data, y_data, fit, bounds):
        """
        Calculate the fit to a dataset.

        Parameters
        ----------
        x_data : np.ndarray
            x-axis data
        y_data : np.ndarray
            y-axis data
        fit : astropy.modeling.Model
            Initial fitted model.
        bounds : tuple[list,list]
            List of allowed upper bounds and lowers bounds to the fitting
            parameters. Empty when feature is not available but
            [-np.inf,np.inf] for just the background.

        Returns
        -------
        fit : astropy.modeling.Model
            The fit model.
        """

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

        """
        Initialize fitting models.

        Parameters
        ----------
        feature : {'moffat', 'gaussian', 'gauss'}
            Feature model to fit.
        background : {'linear', 'constant'}
            Baseline model to fit (Const1D or Linear1D).
        x_data : np.ndarray
            x-axis data
        y_data : np.ndarray
            y-axis data
        limits : list of lists
            Low and high limits for data to fit, as
            [[low_x, low_y], [high_x, high_y]].  Currently
            only the x values are used.

        Returns
        -------
        xval : np.nparray
            X-array within the given limits.
        yval : np.nparray
            Y-array within the given limits.
        fit_init : astropy.modeling.Model
            The combined fit to the data and background.
        bounds : tuple[list,list]
            List of allowed upper bounds and lowers bounds to the fitting
            parameters. Empty when feature is not available but
            [-np.inf,np.inf] for just the background.

        """
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
        """
        Generate an initial fitting guess for a given dataset.

        Parameters
        ----------
        x : array
            x-axis data
        y : array
            y-axis data
        limits : list of lists
            Low and high limits for data to fit, as
            [[low_x, low_y], [high_x, high_y]].  Currently
            only the x values are used.

        Returns
        -------
        guess : Dict
            A dictionary with keys ('peak_location', 'background',
            'amplitude', 'width', 'power_index') and their corresponding
            values.
        """
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
                             x_data: Optional[ArrayLike] = None) -> List[DT]:
        """
        Generate fit artists for a pane

        Parameters
        ----------
        model_fits : model_fit.ModelFit or list of model_fit.ModelFit
            Models to plot.
        x_data : array-like, optional
            Data to plot fit y-values on. If not provided, data is
            retrieved from the low model.
        """
        fit_artists = list()
        if not isinstance(model_fits, list):
            model_fits = [model_fits]
        for obj in model_fits:
            fit = obj.get_fit()
            if fit is None:
                continue

            model_id = obj.get_model_id()
            filename = obj.get_filename()
            order = obj.get_order()
            aperture = obj.get_aperture()
            id_tag = obj.get_id()
            if x_data is None:
                fields = obj.get_fields()
                limits = obj.get_limits()
                x = self.models[model_id].retrieve(order=order,
                                                   level='raw',
                                                   aperture=aperture,
                                                   field=fields['x'])
                y = self.models[model_id].retrieve(order=order,
                                                   level='raw',
                                                   aperture=aperture,
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
            mid_model = f'{order}.{aperture}'
            fl = drawing.Drawing(high_model=filename, mid_model=mid_model,
                                 kind='fit_line', data_id=id_tag,
                                 artist=fit_line, model_id=model_id)
            fc = drawing.Drawing(high_model=filename, mid_model=mid_model,
                                 kind='fit_center', data_id=id_tag,
                                 artist=fit_center, model_id=model_id)
            fit_artists.append(fl)
            fit_artists.append(fc)

        return fit_artists

    def plot_fit(self, x: ArrayLike, style: str,
                 fit_obj: Optional[model_fit.ModelFit] = None,
                 fit: Optional[am.Model] = None,
                 model_id: Optional[str] = '', order: Optional[int] = 0,
                 aperture: Optional[int] = 0, feature: Optional[str] = '',
                 ) -> Tuple[ml.Line2D, ml.Line2D]:
        """
        Create overlay artists representing a fit to a plot feature.

        Overlay colors are grayscale representations of the displayed
        model colors.

        Parameters
        ----------
        x : array-like
            Independent coordinates for the fit overlay.
        style : str
            Line style for the overlay.
        fit_obj : model_fit.ModelFit, optional
            If provided, fit, model, orderm aperture, and feature
            are retrieved from the provided object instead of the input
            parameters.
        fit : astropy.modeling.Model, optional
            The callable model, fit to the data.
        model_id : str, optional
            Model ID associated with the fit.
        order : int, optional
            Order number associated with the fit.
        aperture : int, optional
            Aperture associated with the fit.
        feature : str, optional
            Feature type for the fit.

        Returns
        -------
        model, centroid : tuple of Line2D
            The model artist, plotted over the input x data, and
            a vertical line artist representing the centroid position.
        """
        if fit_obj is not None:
            model_id = fit_obj.get_model_id()
            order = fit_obj.get_order()
            aperture = fit_obj.get_aperture()
            feature = fit_obj.get_feature()
            fit = fit_obj.get_fit()
            fit_obj.axis = self.ax

        # convert model color to grayscale, to distinguish
        # from plot, but keep some separation between different models
        gray = self.grayscale(self.colors[model_id])

        label = (f'{model_id}, Order {order + 1}, Aperture {aperture + 1},'
                 f'{feature.capitalize()}')
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


class TwoDimPane(Pane):
    """
    Two-axis pane, for displaying images.

    Not yet implemented.
    """

    def __init__(self, ax: Optional[ma.Axes] = None) -> None:
        super().__init__(ax)
