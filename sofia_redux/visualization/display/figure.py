# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import (List, Any, Dict, Union,
                    Optional, TypeVar, Tuple)

import matplotlib.axes as ma
from matplotlib import gridspec
from matplotlib import style as ms
from matplotlib import backend_bases as mbb
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.signals import Signals
from sofia_redux.visualization.models import high_model
from sofia_redux.visualization.display import pane, blitting, artists
from sofia_redux.visualization.utils.eye_error import EyeError
from sofia_redux.visualization.utils.model_fit import ModelFit

__all__ = ['Figure']

MT = TypeVar('MT', bound=high_model.HighModel)
PT = TypeVar('PT', bound=pane.Pane)
PID = TypeVar('PID', int, str)


class Figure(object):
    """
    Oversee the plot.

    The Figure class is analogous to matplotlib.figure.Figure. It
    handles all plots generated. There is only one `Figure` instance
    created for the Eye. Each plot resides in a `Pane` object.
    The `Figure` receives commands from the `View` object
    and decides which `Pane` it applies to, as well as how it should
    be formatted. The `Figure` does not deal with any Qt objects.

    Parameters
    ----------
    figure_widget : QtWidgets.QWidget
        The widget in the Qt window containing the matplotlib
        canvas to display to.
    signals : sofia_redux.visualization.signals.Signals
        Custom signals recognized by the Eye interface, used
        to trigger callbacks outside of the Figure.

    Attributes
    ----------
    widget : QtWidgets.QWidget
        The widget in the Qt window that `Figure` connects to.
    fig : matplotlib.figure.Figure
        The figure object typically thought of inside of the Qt
        widget.
    panes : list
        A list of `Pane` objects.
    gs : gridspec.GridSpec
        The gridspec that details how the panes are aligned and
        spaced out.
    signals : sofia_redux.visualization.signals.Signals
        Collection of PyQt signals for passing on information to
        other parts of the Eye.
    _current_pane : int
        Index of the currently selected pane in `panes`.
    highlight_pane : bool
        Flag to specify if the current pane should be highlighted.
    """

    def __init__(self, figure_widget, signals: Signals) -> None:
        self.widget = figure_widget
        self.fig = figure_widget.canvas.fig
        self.panes = list()
        self.gs = None
        self.signals = signals
        self._current_pane = 0
        self.block_current_pane_signal = False
        self.highlight_pane = True
        self.layout = 'grid'
        self.color_cycle = 'Accessible'
        self.plot_type = 'Step'
        self.show_markers = False
        self.show_grid = False
        self.show_error = True
        self.dark_mode = False
        self._cursor_locations = list()
        self._cursor_mode = None
        self._cursor_pane = None
        self._fit_params = list()

        self.artists = artists.Artists()
        self.blitter = blitting.BlitManager(canvas=figure_widget.canvas,
                                            artists=self.artists)

    @property
    def current_pane(self) -> int:
        """Pane: Currently active pane."""
        return self._current_pane

    @current_pane.setter
    def current_pane(self, value: int) -> None:
        """Set the current pane index to `value`."""
        self._current_pane = value
        if not self.block_current_pane_signal:
            self.signals.current_pane_changed.emit()

    def set_pane_highlight_flag(self, state: bool) -> None:
        """
        Set the visibility for a pane highlight border.

        Parameters
        ----------
        state : bool
            True to show; False to hide.
        """
        self.highlight_pane = state
        self.artists.set_pane_highlight_flag(pane_number=self.current_pane,
                                             state=state)

    def set_block_current_pane_signal(self, value: bool = True) -> None:
        """
        Set the flag to block the current_pane signal.

        Parameters
        ----------
        value : bool, optional
            True to block the current_pane signal; False to allow it
            to propagate.
        """
        self.block_current_pane_signal = value

    def set_layout_style(self, value: str = 'grid') -> None:
        """
        Set the layout style.

        Parameters
        ----------
        value : ['grid', 'rows', 'columns'], optional
            The layout to set.
        """
        self.layout = value

    ####
    # Panes
    ####
    def populated(self) -> bool:
        """
        Check for pane existence.

        Returns
        -------
        bool
            True if any panes exist, else False.
        """
        return len(self.panes) > 0

    def pane_count(self) -> int:
        """
        Retrieve the number of panes.

        Returns
        -------
        int
            The pane count.
        """
        return len(self.panes)

    def pane_layout(self) -> Union[None, Tuple[int, int]]:
        """
        Retrieve the current pane layout.

        Returns
        -------
        geometry : tuple of int, or None
            If there is an active layout, (nrow, ncol) is returned.
            Otherwise, None.
        """
        if self.gs is None:
            return None
        else:
            return self.gs.get_geometry()

    def add_panes(self, n_dims: Union[int, Tuple, List],
                  n_panes: int = 1) -> None:
        """
        Add new panes to the figure.

        Parameters
        ----------
        n_dims : int, list-like
            Specifies the number of dimensions for the new panes,
            which determines if they are for spectra or images. If
            multiple panes are being added, this can be a single
            value that will apply to all new panes, or a list-like
            object that specifies the dimensions for each new pane.
        n_panes : int
            Number of panes to be added.

        Raises
        ------
        RuntimeError :
            If inconsistent or invalid options are given.
        """
        self.set_block_current_pane_signal(True)
        self.fig.clear()
        if n_panes == 0:
            return

        if not isinstance(n_dims, (tuple, list)):
            n_dims = [n_dims] * n_panes
        else:
            if len(n_dims) != n_panes:
                raise RuntimeError(f'Length of pane dimensions '
                                   f'does not match number of panes'
                                   f'requested: {len(n_dims)} != '
                                   f'{n_panes}')
        for i, dimension in zip(range(n_panes), n_dims):
            if dimension == 0:
                new_pane = pane.OneDimPane()
            elif dimension == 1:
                new_pane = pane.OneDimPane()
            elif dimension == 2:
                new_pane = pane.TwoDimPane()
            else:
                raise RuntimeError(f'Invalid number of dimensions for '
                                   f'pane: {dimension}')

            # set user defaults
            new_pane.set_color_cycle_by_name(self.color_cycle)
            new_pane.set_plot_type(self.plot_type)
            new_pane.set_markers(self.show_markers)
            new_pane.set_grid(self.show_grid)
            new_pane.set_error(self.show_error)

            self.panes.append(new_pane)

        self._assign_axes()
        self.reset_artists()
        self.current_pane = len(self.panes) - 1
        self.set_block_current_pane_signal(False)
        self.signals.atrophy_bg_full.emit()

    def _assign_axes(self) -> None:
        """Assign axes to all current panes."""
        if not self.populated():
            return
        self._derive_pane_grid()
        for i, pane_ in enumerate(self.panes):
            ax = self.fig.add_subplot(self.gs[i])
            pane_.set_axis(ax)
            if pane_.show_overplot:
                ax_alt = ax.twinx()
                ax_alt.autoscale(enable=True)
                pane_.set_axis(ax_alt, kind='alt')

    def _derive_pane_grid(self) -> None:
        """
        Determine the gridspec to populate with panes.

        Uses self.layout to determine the style (may be
        'grid', 'rows', or 'columns').
        """
        n_tot = len(self.panes)
        if self.layout == 'rows':
            n_rows = n_tot
            n_cols = 1
        elif self.layout == 'columns':
            n_rows = 1
            n_cols = n_tot
        else:
            n_rows = int(np.ceil(np.sqrt(n_tot)))
            n_cols = int(np.ceil(n_tot / n_rows))
        self.gs = gridspec.GridSpec(n_rows, n_cols, figure=self.fig)

    def remove_artists(self) -> None:
        """Remove all artists."""
        self.artists.reset_artists('all')

    def reset_artists(self) -> None:
        """Recreate all artists from models."""
        self.artists.reset_artists('all')
        for pane_ in self.panes:
            new_artists = pane_.create_artists_from_current_models()
            successes = 0
            successes += self.artists.add_artists(new_artists)
            if successes != len(new_artists):
                log.debug('Error encountered while creating artists')

        # add borders back in
        self._add_pane_artists()

        # add fits back in; managed by view
        self.signals.toggle_fit_visibility.emit()

    def _add_pane_artists(self) -> None:
        """Track existing border artists."""
        borders = dict()
        for pane_number, pane_ in enumerate(self.panes):
            if self.highlight_pane:
                if pane_number == self._current_pane:
                    visible = True
                elif self._current_pane is None and pane_number == 0:
                    # catch for case where border is added before
                    # current pane is set, on initialization
                    visible = True
                else:
                    visible = False
            else:
                visible = False
            borders[f'pane_{pane_number}'] = {
                'kind': 'border',
                'artist': pane_.get_border(),
                'visible': visible}
        self.artists.add_patches(borders)

    def _add_crosshair(self) -> None:
        """Track existing crosshair artists."""
        crosshairs = dict()
        for pane_number, pane_ in enumerate(self.panes):
            pane_lines = pane_.plot_crosshair()
            for direction, artist in pane_lines.items():
                crosshairs[f'pane_{pane_number}_{direction}'] = {
                    'kind': 'crosshair',
                    'artist': artist,
                    'visible': False,
                    'direction': direction}
        self.artists.add_crosshairs(crosshairs)

    def model_matches_pane(self, pane_: PID, model_: MT) -> bool:
        """
        Check if a model is displayable in a specified pane.

        Currently always returns True.

        Parameters
        ----------
        pane_ : Pane
            The pane to test against.
        model_ : Model
            The model to test.

        Returns
        -------
        bool
            True if `model` can be added to `pane_`
        """
        # TODO: make this work
        return True

    def remove_all_panes(self) -> None:
        """Remove all panes."""
        self.set_block_current_pane_signal(True)
        self.fig.clear()
        self.panes = list()
        self.current_pane = None
        self.set_block_current_pane_signal(False)

    def remove_pane(self, pane_id: List[int]) -> None:
        """
        Remove a specified pane.

        Parameters
        ----------
        pane_id : list of int
            List of pane indices to remove.
        """
        self.set_block_current_pane_signal(True)
        self.fig.clear()

        # delete specified pane
        keep_panes = list()
        for idx in range(len(self.panes)):
            if idx not in pane_id:
                keep_panes.append(self.panes[idx])
        self.panes = keep_panes

        # reset current pane
        if self.populated():
            self._assign_axes()
            self.current_pane = len(self.panes) - 1
        else:
            # no panes left - leave figure blank
            self.current_pane = None

        self.reset_artists()
        self.set_block_current_pane_signal(False)
        self.signals.atrophy_bg_full.emit()

    def pane_details(self) -> Dict[str, Dict[str, Any]]:
        """
        Compile summaries of all panes in the figure.

        Returns
        -------
        dict
            Keys are 'pane_{i}', where i is the pane index.
            Values are the model summary dicts for the pane.
        """
        details = dict()
        for i, _pane in enumerate(self.panes):
            details[f'pane_{i}'] = _pane.model_summaries()
        return details

    def change_current_pane(self, pane_id: int) -> bool:
        """
        Set the current pane to the index provided.

        Parameters
        ----------
        pane_id : int
            Index of the new current pane.

        Returns
        -------
        changed : bool
            True if current pane has changed; False if the new
            current pane is the same as the old current pane.
        """
        if self.current_pane == pane_id:
            return False
        elif len(self.panes) > pane_id >= 0:
            self.current_pane = pane_id
            return True
        else:
            return False

    def determine_selected_pane(self, ax: ma.Axes) -> Optional[int]:
        """
        Determine what pane corresponds to a provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to find.

        Returns
        -------
        index : int, None
            The pane index of the correct pane, or None if the
            axes were not found.
        """
        for i, _pane in enumerate(self.panes):
            if ax in _pane.axes():
                return i
        return None

    def determine_pane_from_model(self, model_id: str,
                                  order: Optional[int] = None) -> List[PT]:
        found = list()
        for pane_ in self.panes:
            if pane_.contains_model(model_id, order=order):
                found.append(pane_)
        return found

    def get_current_pane(self) -> Optional[PT]:
        """
        Return the current pane.

        Returns
        -------
        pane : sofia_redux.visualization.display.pane.Pane, None
            The current pane if a current pane exists;
            None otherwise.
        """
        if self.populated():
            return self.panes[self.current_pane]
        else:
            return None

    def get_fields(self, target: Optional[Any] = None) -> List:
        """
        Get the fields associated with a given pane selection.

        Parameters
        ----------
        target : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        fields : list
            List of fields for corresponding to pane selection
            provided.
        """
        panes, axes = self.parse_pane_flag(target)
        fields = list()
        for _pane in panes:
            fields.append(_pane.get_field(axis=axes))
        return fields

    def get_units(self, target: Optional[Any] = None) -> List:
        """
        Get the units associated with a given pane selection.

        Parameters
        ----------
        target : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        units : list
            List of units for corresponding to pane selection
            provided.
        """
        panes, axes = self.parse_pane_flag(target)
        units = list()
        for _pane in panes:
            units.append(_pane.get_unit(axis=axes))
        return units

    def get_orders(self, target: Optional[Any] = None) -> Dict:
        """
        Get the orders associated with a given pane selection.

        Parameters
        ----------
        target : str, None, dict, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        orders : dict
            Dictionary of orders for corresponding to pane selection
            provided. Keys are the indices of the panes.
        """
        panes, axes = self.parse_pane_flag(target)
        orders = dict()
        for i, _pane in enumerate(panes):
            orders[i] = _pane.get_orders(enabled_only=True,
                                         by_model=True)
        return orders

    def get_scales(self, target: Optional[Any] = None) -> List:
        """
        Get the axes scales associated with a given pane selection.

        Parameters
        ----------
        target : str, None, dict, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.

        Returns
        -------
        scales : list
            List of axes scales for corresponding to pane selection
            provided.
        """
        log.debug(f'Getting scale for {target}.')
        panes, axes = self.parse_pane_flag(target)
        scales = list()
        for _pane in panes:
            scales.append(_pane.get_scale())
        return scales

    ####
    # Handling models
    ####
    def assign_models(self, mode: str, models: Dict[str, MT],
                      indices: Optional[List[int]] = None) -> None:
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
        if mode == 'split':
            pane_count = self.pane_count()
            model_keys = list(models.keys())
            models_per_pane = self._assign_models_per_pane(
                model_count=len(models), pane_count=pane_count)
            for i, pane_ in enumerate(self.panes):
                model_count = models_per_pane[i]
                for j in range(model_count):
                    model_ = models[model_keys.pop(0)]
                    self.add_model_to_pane(model_=model_, pane_=pane_)
        elif mode == 'first':
            for model_ in models.values():
                self.add_model_to_pane(model_=model_, pane_=self.panes[0])
        elif mode == 'last':
            for model_ in models.values():
                self.add_model_to_pane(model_=model_, pane_=self.panes[-1])
        elif mode == 'assigned':
            # TODO: This method assumes the dictionary remains ordered,
            #  which is the default behavior of dictionaries. However,
            #  it is not reliable. Change this so `indices` are also
            #  a dict with the same keys as models.
            for model_, pane_index in zip(models.values(), indices):
                self.add_model_to_pane(model_=model_,
                                       pane_=self.panes[pane_index])
        else:
            raise RuntimeError('Invalid mode')

    @staticmethod
    def _assign_models_per_pane(model_count: int,
                                pane_count: int) -> List[int]:
        """
        Calculate the number of models per pane.

        Divides the number of models evenly between the existing
        panes.  Used with the 'split' assignment mode.

        Parameters
        ----------
        model_count : int
            The total number of models to assign.
        pane_count : int
            The number of panes available.

        Returns
        -------
        models_per_pane : list of int
            List with `pane_count` elements, containing the
            number of models to assign to each pane.
        """
        models_per_pane = [model_count // pane_count] * pane_count
        remainder = model_count % pane_count
        models_per_pane = [mp + 1
                           if i < remainder else mp
                           for i, mp in enumerate(models_per_pane)]
        return models_per_pane

    def models_per_pane(self) -> List[int]:
        """
        Retrieve the number of models in each pane.

        Returns
        -------
        count : list of int
            The number of models in each existing pane.
        """
        count = [p.model_count() for p in self.panes]
        return count

    def add_model_to_pane(self, model_: MT,
                          pane_: Optional[PT] = None) -> None:
        """
        Add model to current pane.

        If there are currently no panes, create one.
        If there are panes but the model is not
        compatible with them, create a new one and
        add the model there.

        Parameters
        ----------
        model_ : sofia_redux.visualization.models.high_model.HighModel
            Model to add.
        pane_ : sofia_redux.visualization.display.pane.Pane
            Pane to add model to. If not provided, add
            to current pane
        """
        if pane_ is None:
            if not self.populated():
                self.add_panes(model_.default_ndims, n_panes=1)
            pane_ = self.panes[self.current_pane]
        if self.model_matches_pane(pane_, model_):
            additions = pane_.add_model(model_)
        else:
            self.add_panes(n_dims=model_.default_ndims, n_panes=1)
            additions = self.panes[self.current_pane].add_model(model_)
        successes = self.artists.add_artists(artists=additions)
        if successes:
            log.info(f'Added {successes} models to panes')

    def remove_model_from_pane(
            self, filename: Optional[str] = None,
            model_: Optional[MT] = None,
            panes: Optional[Union[PT, List[PT]]] = None) -> None:
        """
        Remove a model from one or more panes.

        The model to remove can be specified by either
        its filename or the model itself.

        Parameters
        ----------
        filename : str, optional
            Name of the file to remove
        model_ : sofia_redux.visualization.spectrum.model.Model, optional
            Model object to remove
        panes : sofia_redux.visualization.spectrum.panes.Pane, optional
            A list of pane objects to remove the
            model from. If not provided, the model will
            be removed from all panes.

        Raises
        ------
        RuntimeError
            If neither ``filename`` nor ``model`` are
            provided.
        """
        if filename is None and model_ is None:
            raise RuntimeError('Must specify which model to remove '
                               'with either its filename or the '
                               'model itself.')
        if panes is None:
            panes = self.panes
        elif not isinstance(panes, list):
            panes = [panes]
        for _pane in panes:
            _pane.remove_model(filename=filename, model=model_)

        # trigger full artist and background regeneration
        self.clear_all()
        self.signals.atrophy_bg_full.emit()

    ####
    # Plotting
    ####
    def refresh(self, bg_full: bool, bg_partial: bool) -> None:
        """
        Refresh the figure canvas.

        Parameters
        ----------
        bg_full : bool
            If True, the full background will be redrawn, including
            the plot axes.
        bg_partial : bool
            If True and bg_full is False, the background will
            be redrawn, but the axes will remain the same.
        """
        if bg_full or bg_partial:
            for _pane in self.panes:
                if bg_full:
                    _pane.data_changed = True
                else:
                    _pane.data_changed = False
                _pane.apply_configuration()
            self.blitter.update_all()
        else:
            self.blitter.update_animated()

    def clear_all(self) -> None:
        """Clear all artists and redraw panes."""
        self.set_block_current_pane_signal(True)
        self.fig.clear()
        self._assign_axes()
        self.reset_artists()
        self.set_block_current_pane_signal(False)
        self.signals.atrophy_bg_partial.emit()

    def change_axis_limits(self, limits: Dict[str, float],
                           target: Optional[Any] = None) -> None:
        """
        Change the axis limits for specified panes.

        Parameters
        ----------
        limits : dict
            Keys are 'x', 'y'.  Values are [low, high] limits for
            the axis.
        target : str, None, dict, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane objects
        """
        log.debug(f'Update axis limits for {target} panes to '
                  f'{limits}')
        panes, axes = self.parse_pane_flag(target)
        for _pane in panes:
            if _pane is not None:
                _pane.set_limits(limits)

    def change_axis_unit(self, units: Dict[str, str],
                         target: Optional[Any] = None) -> None:
        """
        Change the axis unit for specified panes.

        If incompatible units are specified, the current units
        are left unchanged.

        Parameters
        ----------
        units : dict
            Keys are 'x', 'y'.  Values are the units to convert
            to.
        target : str, None, dict, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
        """
        panes, axes = self.parse_pane_flag(target)

        for _pane in panes:
            if _pane is not None:
                line_updates, error_updates = _pane.set_units(units, axes)
                self.artists.update_line_data(pane=_pane,
                                              updates=line_updates,
                                              axes=axes)
                if error_updates:
                    self.artists.update_error_ranges(pane=_pane,
                                                     updates=error_updates)

    def change_axis_field(self, fields: Dict[str, str],
                          target: Optional[Any] = None) -> None:
        """
        Change the axis field for specified panes.

        Parameters
        ----------
        fields : dict
            Keys are 'x', 'y'.  Values are the field names to
            change to.
        target : str, None, list of int, or list of Pane objects
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            ints or Pane objects.
        """
        panes, axes = self.parse_pane_flag(target)
        if panes is None:
            log.info(f'No valid panes found for (target, fields) = '
                     f'({target}, {fields})')
        else:
            if axes == 'both':
                fields['y_alt'] = fields['y']
            elif axes == 'alt':
                fields['y_alt'] = fields.pop('y')
            else:
                fields.pop('y_alt', None)
            for _pane in panes:
                if _pane is not None:
                    _pane.set_fields(fields)

            # trigger full artist regeneration
            self.clear_all()

    def set_orders(self, orders: Dict[int, Dict[str, List[int]]]) -> None:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are indices for the panes to update. Values
            are dicts, with model ID keys, order list values.
        """
        for pane_id, pane_orders in orders.items():
            _pane = self.parse_pane_flag([pane_id])[0]
            if _pane is None:
                continue
            elif isinstance(_pane, list):
                _pane = _pane[0]
            _pane.set_orders(pane_orders)

    def set_scales(self, scales: Dict[str, str],
                   target: Optional[Any] = None) -> None:
        """
        Set the axis scale for specified panes.

        Parameters
        ----------
        scales : dict
            Keys are 'x', 'y'.  Values are 'linear' or 'log'.
        target : str, None, list of int, or list of Pane objects
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            ints or Pane objects.
        """
        panes, axes = self.parse_pane_flag(target)
        if panes is None:
            return
        if axes in ['both', 'all']:
            scales['y_alt'] = scales['y']
        elif axes == 'alt':
            scales['y_alt'] = scales.pop('y')
        else:
            scales.pop('y_alt', None)
        for _pane in panes:
            if _pane is not None:
                _pane.set_scales(scales)

    def set_overplot_state(self, state: bool, target: Optional[Any] = None
                           ) -> None:
        """Set the pane overplot flag"""
        panes, axes = self.parse_pane_flag(target)
        if state:  # Turning on
            for _pane in panes:
                if _pane is not None:
                    _pane.set_overplot(state)
            self.reset_artists()
        else:  # Turning off
            self.artists.reset_artists(selection='alt', panes=panes)
            for _pane in panes:
                if _pane is not None:
                    _pane.reset_alt_axes(remove=True)
                    _pane.set_overplot(state)

        # trigger full regeneration: some things get orphaned
        # when axes change (eg. border artist)
        self.clear_all()

    def parse_pane_flag(
            self, flags: Optional[Union[List, Dict, List[Union[int, PID]]]]) \
            -> Tuple[List[PT], str]:
        """
        Parse the specified panes from an input flag.

        Parameters
        ----------
        flags : str, None, list of int, or list of Panes
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            ints or Pane objects.

        Returns
        -------
        list of Pane
            List of panes corresponding to input flag.
        """
        log.debug(f'Parsing {flags} ({type(flags)})')
        axis = ''
        panes = None
        if flags is None:
            panes = [self.get_current_pane()]
        elif flags == 'all':
            panes = self.panes
        elif isinstance(flags, list):
            if all([isinstance(p, int) for p in flags]):
                panes = list(map(self.panes.__getitem__, flags))
            elif not all([isinstance(p, pane.Pane) for p in flags]):
                raise TypeError('List of panes can only contain '
                                'integers or Pane objects.')
            else:
                panes = flags
        elif isinstance(flags, dict):
            try:
                flag = flags['pane']
            except KeyError:
                raise EyeError(f'Unable to parse pane flag {flags}')
            else:
                if flag == 'all':
                    panes = self.panes
                elif flag == 'current':
                    panes = [self.get_current_pane()]
                else:
                    raise EyeError(f'Unable to parse pane flag {flags}')
                axis = flags.get('axis', axis)
        log.debug(f'Parsed {panes} ({type(panes)})')
        return panes, axis

    def set_color_cycle(self, cycle_name: str) -> None:
        """
        Set the color cycle in all panes.

        Parameters
        ----------
        cycle_name: ['spectral', 'tableau', 'accessible']
            Color cycle to set.
        """
        self.color_cycle = cycle_name
        if self.populated():
            for pane_ in self.panes:
                pane_.set_color_cycle_by_name(cycle_name)
                updates = pane_.update_colors()
                self.artists.update_artist_options(pane_=pane_,
                                                   options=updates)
        self.signals.atrophy.emit()

    def set_plot_type(self, plot_type: str) -> None:
        """
        Set the plot type in all panes.

        Parameters
        ----------
        plot_type: ['line', 'step', 'scatter']
            Plot type to set.
        """
        self.plot_type = plot_type
        if self.populated():
            for pane_ in self.panes:
                line_updates = pane_.set_plot_type(plot_type)
                self.artists.update_line_type(pane_=pane_,
                                              updates=line_updates)
        self.signals.atrophy.emit()

    def set_markers(self, state: bool = True) -> None:
        """
        Set the marker visibility in all panes.

        Parameters
        ----------
        state: bool, optional
            If True, markers will be shown.  If False, they
            will be hidden.
        """
        self.show_markers = state
        if self.populated():
            for pane_ in self.panes:
                marker_updates = pane_.set_markers(state)
                self.artists.update_artist_options(pane_=pane_,
                                                   options=marker_updates)
        self.signals.atrophy.emit()

    def set_grid(self, state: bool = True) -> None:
        """
        Set the grid visibility in all panes.

        Parameters
        ----------
        state: bool, optional
            If True, gridlines will be shown.  If False, they
            will be hidden.
        """
        self.show_grid = state
        if self.populated():
            for pane_ in self.panes:
                pane_.set_grid(state)
        self.signals.atrophy_bg_partial.emit()

    def set_error(self, state: bool = True) -> None:
        """
        Set the error range visibility in all panes.

        Parameters
        ----------
        state: bool, optional
            If True, error ranges will be shown.  If False, they
            will be hidden.
        """
        self.show_error = state
        if self.populated():
            for pane_ in self.panes:
                pane_.set_error(state)
                updates = pane_.update_visibility()
                self.artists.update_artist_options(pane_=pane_,
                                                   options=updates)
        self.signals.atrophy.emit()

    def set_dark_mode(self, state: bool = True) -> None:
        """
        Set a dark background in all panes.

        Parameters
        ----------
        state: bool, optional
            If True, dark mode will be enabled.  If False, it
            will be disabled.
        """
        self.dark_mode = state
        if self.dark_mode:
            ms.use('dark_background')
            self.fig.set_facecolor('black')
        else:
            ms.use('default')
            self.fig.set_facecolor('white')
        self.clear_all()

    def set_enabled(self, pane_id: int, model_id: str,
                    state: bool) -> None:
        """
        Enable or disable a specified model.

        Parameters
        ----------
        pane_id : int
            Pane ID to update.
        model_id : str
            Model ID to modify.
        state : bool
            If True, model will be enabled (shown).  If False,
            model will be disabled (hidden).
        """
        pane_ = self.panes[pane_id]
        pane_.set_model_enabled(model_id, state)
        updates = pane_.update_visibility()
        self.artists.update_artist_options(pane_=pane_, options=updates)
        self.signals.atrophy.emit()

    def set_all_enabled(self, pane_id: int, state: bool) -> None:
        """
        Enable or disable all models in a pane.

        Parameters
        ----------
        pane_id : int
            Pane ID to update.
        state : bool
            If True, models will be enabled (shown).  If False,
            models will be disabled (hidden).
        """
        pane_ = self.panes[pane_id]
        pane_.set_all_models_enabled(state)
        updates = pane_.update_visibility()
        self.artists.update_artist_options(pane_=pane_, options=updates)
        self.signals.atrophy.emit()

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
        initial_params = self._font_sizes()
        fontsize = kwargs.get('fontsize', 40)
        self._font_sizes(fontsize)
        self.fig.savefig(filename, **kwargs)
        self._font_sizes(initial_params)

    def _font_sizes(self, sizes: Optional[Union[int, float, Dict]] = None
                    ) -> Union[None, Dict]:
        """
        Set font sizes for plot image.

        New font sizes are set directly in the pane axes.

        Parameters
        ----------
        sizes : int, float, or dict, optional
            Base size(s) to start with.

        Returns
        -------
        initial_params : dict or None
            Starting sizes for axis_label and tick_label, before
            update.
        """
        if sizes is None:
            initial_params = {'axis_label': list(), 'tick_label': list()}
        else:
            initial_params = None
        for pane_ in self.panes:
            labels = [pane_.ax.xaxis.label, pane_.ax.yaxis.label]
            tick_labels = [pane_.ax.get_xticklabels(),
                           pane_.ax.get_yticklabels()]
            for i, label in enumerate(labels):
                if sizes is None:
                    initial_params['axis_label'].append(label.get_fontsize())
                else:
                    if isinstance(sizes, (int, float)):
                        size = sizes
                    else:
                        size = sizes['axis_label'][i]
                    label.set_fontsize(size)

            for i, tick in enumerate(tick_labels):
                if sizes is None:
                    initial_params['tick_label'].append(tick[0].get_fontsize())
                else:
                    if isinstance(sizes, (int, float)):
                        size = sizes
                    else:
                        size = sizes['tick_label'][i]
                    for t in tick:
                        t.set_fontsize(size)

        return initial_params

    ####
    # Mouse events
    ####
    def data_at_cursor(self, event: mbb.MouseEvent) -> Dict:
        """
        Retrieve the plot data at the cursor position.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse motion event.

        Returns
        -------
        data_point : dict
            Keys are filenames; values are lists of dicts
            containing 'order', 'bin', 'bin_x', 'bin_y',
            'x_field', 'y_field', 'color', and 'visible'
            values for the displayed models.
        """
        pane_index = self.determine_selected_pane(event.inaxes)
        data_point = self.panes[pane_index].data_at_cursor(event)
        self.artists.update_marker(data_point)
        return data_point

    def crosshair(self, event: mbb.MouseEvent) -> None:
        """
        Display a crosshair at the cursor position.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse motion event.
        """
        pane_index = self.determine_selected_pane(event.inaxes)
        if pane_index is not None and pane_index == self._current_pane:
            data_point = self.panes[pane_index].xy_at_cursor(event)
            direction = self._parse_cursor_direction(mode='crosshair')
            self.artists.update_crosshair(pane_index, data_point=data_point,
                                          direction=direction)
            self.signals.atrophy.emit()

    def clear_crosshair(self) -> None:
        """Clear any displayed crosshairs."""
        self.artists.reset_artists(selection='crosshair')

    def reset_data_points(self) -> None:
        """Reset any displayed cursor markers."""
        if self.populated():
            self.artists.hide_cursor_markers()

    def reset_zoom(self, all_panes: Optional[bool] = False) -> None:
        """
        Reset axis limits to defaults.

        Parameters
        ----------
        all_panes : bool, optional
            If True, all axes will be reset. Otherwise, only
            the current pane will be reset.
        """
        if not self.populated():
            return
        if all_panes:
            panes = self.panes
        else:
            panes = [self.panes[self.current_pane]]
        for _pane in panes:
            _pane.reset_zoom()
        self.signals.atrophy_bg_partial.emit()

    def set_cursor_mode(self, mode: str) -> None:
        """
        Set the cursor mode.

        Cursor modes are used to manage zoom and feature fit
        interactions.

        Parameters
        ----------
        mode : ['x_zoom', 'y_zoom', 'b_zoom', 'fit', '']
            The mode to set.
        """
        self._fit_params = list()
        self._cursor_mode = mode
        self._cursor_locations = list()
        self.clear_crosshair()
        self._cursor_pane = None
        if mode != '':
            self._cursor_pane = self._current_pane
            self._add_crosshair()
        self.signals.atrophy_bg_partial.emit()

    def record_cursor_location(self, event: mbb.MouseEvent) -> None:
        """
        Store the current cursor location.

        Depending on the current user interaction in zoom or fit mode,
        either a guide is displayed at the location (first click), or
        all guides are cleared and the stored cursor locations
        are passed to the `end_cursor_records` method (second click).

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse motion event.
        """
        pane_index = self.determine_selected_pane(event.inaxes)
        if pane_index is not None and pane_index == self._cursor_pane:
            location = self.panes[pane_index].xy_at_cursor(event)
            if None in location:  # pragma: no cover
                # could happen if overplot data is present and
                # doesn't exactly match plot data
                return
            self._cursor_locations.append(location)

            if len(self._cursor_locations) == 2:
                self.end_cursor_records(pane_index)
            else:
                guide_artists = self.panes[pane_index].plot_guides(
                    location, kind=self._parse_cursor_direction())
                art_dict = dict()
                for direction, line in guide_artists.items():
                    art_dict[f'{direction}_guide'] = {0: {'guide': line}}
                self.artists.add_artists(art_dict)
                self.signals.atrophy.emit()

    def end_cursor_records(self, pane_index: int) -> None:
        """
        Complete zoom or fit interactions.

        User specified locations are used to either set axis
        limits or else display a new fit to a plot feature.

        Parameters
        ----------
        pane_index : int
            Index of the pane to update.
        """
        if 'zoom' in self._cursor_mode:
            self._end_zoom(pane_index)
        elif 'fit' in self._cursor_mode:
            self._end_fit(pane_index)
        # reset cursor, but not fit params -- is needed for the
        # cursor recording
        self._cursor_mode = ''
        self._cursor_locations = list()

        self.signals.atrophy.emit()
        # this signal will clear guides
        self.signals.end_cursor_recording.emit()
        # this will trigger clear_selection
        self.signals.end_zoom_mode.emit()

    def _parse_cursor_direction(self, mode: str = 'zoom') -> str:
        """
        Parse the cursor direction from the cursor mode.

        Parameters
        ----------
        mode : ['zoom', 'crosshair'], optional
            If crosshair, possible directions are 'v', 'h', or 'vh'.
            If zoom, possible directions are 'x', 'y', or 'b'.
            Defaults to 'zoom'

        Returns
        -------
        direction : str
            The crosshair or guide direction corresponding to
            the current cursor mode.
        """
        if 'zoom' in self._cursor_mode:
            direction = self._cursor_mode.split('_')[0]
        elif 'fit' in self._cursor_mode:
            direction = 'x'
        else:
            direction = 'b'

        if mode == 'crosshair':
            if direction == 'x':
                direction = 'v'
            elif direction == 'y':
                direction = 'h'
            else:
                direction = 'hv'
        return direction

    def _end_zoom(self, pane_index: int,
                  direction: Optional[str] = None) -> None:
        """
        Finish zoom interaction.

        Parameters
        ----------
        pane_index : int
            Index of the pane to update.
        direction : ['x', 'y', 'b']
            If not provided, will be determined from the current
            cursor mode.
        """
        if direction is None:
            direction = self._parse_cursor_direction()

        # perform zoom
        self.panes[pane_index].perform_zoom(
            zoom_points=self._cursor_locations, direction=direction)

        # clear all h and v guides
        self.artists.reset_artists(selection='h_guide',
                                   panes=self.panes[pane_index])
        self.artists.reset_artists(selection='v_guide',
                                   panes=self.panes[pane_index])

        # partial background: reset limits, data has not changed
        self.signals.atrophy_bg_partial.emit()

    def _end_fit(self, pane_index: int) -> None:
        """
        Finish the feature fit interaction.

        Parameters
        ----------
        pane_index : int
            Index of the pane to update.
        """
        if len(self._cursor_locations) != 2:
            return

        # sort limits by x before performing fit
        if self._cursor_locations[0][0] <= self._cursor_locations[1][0]:
            limits = self._cursor_locations
        else:
            limits = [self._cursor_locations[1], self._cursor_locations[0]]

        fit_artists, fit_params = self.panes[pane_index].perform_fit(
            self._cursor_mode, limits)

        self.artists.add_artists(fit_artists)
        self._fit_params.extend(fit_params)

    def get_selection_results(self) -> List[ModelFit]:
        """
        Retrieve feature fit parameters.

        Returns
        -------
        fit_params : list of ModelFit
            Models fit to spectral selections.
        """
        return self._fit_params

    def clear_lines(self, flags: Union[str, List[str]],
                    all_panes: Optional[bool] = False) -> None:
        """
        Clear all displayed guides.

        Parameters
        ----------
        flags : str
            Type of guides to clear.
        all_panes : bool, optional
            If True, all panes will be updated. Otherwise,
            only the current pane will be updated.
        """
        if not self.populated():
            return

        if all_panes:
            panes = self.panes
        else:
            panes = [self.panes[self.current_pane]]

        if not isinstance(flags, list):
            flags = [flags]
        for flag in flags:
            if flag == 'fit':
                self.artists.reset_artists(flag, panes=panes)
            else:
                self.artists.reset_artists(f'{flag}_guide', panes=panes)

    def toggle_fits_visibility(self, fits: List[ModelFit]) -> None:
        # Tell artist to update fit artist
        # If failure, loop over panes. Ask each one to make new
        #   artist for fit. If model, order, fields don't match,
        #   pane does nothing. Else remake fit artists and return them.
        #   Then add new artists to Artists
        for fit in fits:
            options = [{'model_id': fit.get_model_id(),
                        'order': fit.get_order(), 'data_id': fit.get_id(),
                        'new_visibility': fit.get_visibility()}]
            for pane_ in self.panes:
                if fit.get_axis() in pane_.axes():
                    result = self.artists.update_artist_options(
                        pane_, kinds='fit', options=options)
                    if not result:
                        self._regenerate_fit_artists(pane_, fit, options)

                elif (fit.get_fields('x') == pane_.get_field('x')
                      and fit.get_fields('y') == pane_.get_field('y')):
                    self._regenerate_fit_artists(pane_, fit, options)

    def stale_fit_artists(self, fits: List[Dict]):
        matching_panes = self._panes_matching_model_fits(fits)
        for pane_idx, fits in matching_panes.items():
            pane_ = self.panes[pane_idx]
            self.artists.reset_artists(selection='fit',
                                       panes=[pane_])
            for fit in fits:
                self._regenerate_fit_artists(pane_, fit)

    def _regenerate_fit_artists(self, pane_, fit, options=None):
        # fit_params = {fit['model_id']: {fit['order']: fit}}
        # fit_artists = pane_.generate_gauss_fit_artists(fit_params)
        try:
            fit_artists = pane_.generate_fit_artists(fit)
        except (KeyError, IndexError):
            # can happen if model id no longer exists
            return
        self.artists.add_artists(fit_artists)
        if options:
            self.artists.update_artist_options(pane_, kinds='fit',
                                               options=options)

    def _panes_matching_model_fits(self, fits: List[ModelFit]
                                   ) -> Dict[int, List[ModelFit]]:
        matching_panes = dict()
        for fit in fits:
            for idx, pane_ in enumerate(self.panes):
                if fit.get_axis() in pane_.axes():
                    if idx in matching_panes:
                        matching_panes[idx].append(fit)
                    else:
                        matching_panes[idx] = [fit]
        return matching_panes
