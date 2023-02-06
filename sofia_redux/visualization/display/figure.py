# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import (List, Any, Dict, Union,
                    Optional, TypeVar, Tuple)

import uuid
import matplotlib.axes as ma
from matplotlib import gridspec
from matplotlib import style as ms
from matplotlib import backend_bases as mbb
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.signals import Signals
from sofia_redux.visualization.models import high_model, reference_model
from sofia_redux.visualization.display import pane, blitting, gallery, drawing
from sofia_redux.visualization.utils.eye_error import EyeError
from sofia_redux.visualization.utils.model_fit import ModelFit

__all__ = ['Figure']

MT = TypeVar('MT', bound=high_model.HighModel)
RT = TypeVar('RT', bound=reference_model.ReferenceData)
PT = TypeVar('PT', bound=pane.Pane)
PID = TypeVar('PID', int, str)
IDT = TypeVar('IDT', uuid.UUID, str)


class Figure(object):
    """
    Oversee the plot.

    The Figure class is analogous to matplotlib.figure.Figure. It
    handles all plots generated. There is only one `Figure` instance
    created for the Eye. Each plot resides in a `Pane` object.
    The `Figure` receives commands from the `View` object
    and decides which `Pane` it applies to, as well as how it should
    be formatted. The `Figure` does not manage any Qt objects.

    Parameters
    ----------
    figure_widget : QtWidgets.QWidget
        The widget in the Qt window containing the matplotlib
        canvas to display to.
    signals : sofia_redux.visualization.signals.Signals
        Custom signals recognized by the Eye interface, used
        to trigger callbacks outside the Figure.

    Attributes
    ----------
    widget : QtWidgets.QWidget
        The widget in the Qt window that `Figure` connects to.
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object contained in the Qt widget.
    panes : list
        A list of `Pane` objects.
    gs : gridspec.GridSpec
        The gridspec that details how the panes are aligned and
        spaced out.
    signals : sofia_redux.visualization.signals.Signals
        Collection of PyQt signals for passing on information to
        other parts of the Eye.
    _current_pane : list
        List of indices of the currently selected panes in `panes`.
    highlight_pane : bool
        Flag to specify if the current pane should be highlighted.
    layout : {'grid', 'rows', 'columns'}
        Determines the layout in a plot.
    color_cycle : {'spectral', 'tableau', 'accessible'}
        Color cycle to set.
    plot_type : {'line', 'step', 'scatter'}
        Plot type to set.
    show_markers : bool
        If set, markers will be shown.
    show_grid : bool
        If set, a background grid is shown.
    dark_mode : bool
        If set, dark mode is enabled.
    recording : bool
        Status flag, indicating whether a cursor mode is active.
    gallery : gallery.Gallery
        Gallery object tracking plot artists in the figure.
    blitter : blitting.Blitmanager
        Blitting manager for the figure.
    """

    def __init__(self, figure_widget, signals: Signals) -> None:
        self.widget = figure_widget
        self.fig = figure_widget.canvas.fig
        self.panes = list()
        self.gs = None
        self.signals = signals
        self._current_pane = list()
        self.block_current_pane_signal = False
        self.highlight_pane = True
        self.layout = 'grid'
        self.color_cycle = 'Accessible'
        self.plot_type = 'Step'
        self.show_markers = False
        self.show_grid = False
        self.show_error = True
        self.dark_mode = False
        self.recording = False
        self._cursor_locations = list()
        self._cursor_mode = None
        self._cursor_pane = None
        self._fit_params = list()

        self.gallery = gallery.Gallery()
        self.blitter = blitting.BlitManager(canvas=figure_widget.canvas,
                                            gallery=self.gallery,
                                            signals=signals)

    @property
    def current_pane(self) -> List[int]:
        """list of int : Currently active panes."""
        return self._current_pane

    @current_pane.setter
    def current_pane(self, value: List[int]) -> None:
        if not isinstance(value, list):
            value = [value]
        value = filter(lambda i: self.valid_pane(i, len(self.panes)), value)
        self._current_pane = list(value)
        if not self.block_current_pane_signal:
            self.signals.current_pane_changed.emit()

    @staticmethod
    def valid_pane(index: Any, pane_count: int) -> bool:
        """
        Check if a pane index is valid.

        Parameters
        ----------
        index : int
            Index of a pane
        pane_count : int
            Total number of panes

        Returns
        -------
        bool
            If the index is an int and less than pane_count, returns True
            otherwise False.
        """
        try:
            index = int(index)
        except (ValueError, TypeError):
            return False
        else:
            if 0 <= index <= pane_count:
                return True
            else:
                return False

    def set_pane_highlight_flag(self, state: bool) -> None:
        """
        Set the visibility for a pane highlight border.

        Parameters
        ----------
        state : bool
            True to show; False to hide.
        """
        self.highlight_pane = state
        self.gallery.set_pane_highlight_flag(pane_numbers=self.current_pane,
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
                raise RuntimeError(f'Length of pane dimensions does not match '
                                   f'number of panes requested: {len(n_dims)} '
                                   f'!= {n_panes}')

        for i, dimension in zip(range(n_panes), n_dims):
            if dimension == 0:
                new_pane = pane.OneDimPane(self.signals)
            elif dimension == 1:
                new_pane = pane.OneDimPane(self.signals)
            elif dimension == 2:
                new_pane = pane.TwoDimPane(self.signals)
            else:
                raise RuntimeError(f'Invalid number of dimensions for '
                                   f'pane: {dimension}')

            # set user defaults
            new_pane.set_color_cycle_by_name(self.color_cycle)
            new_pane.set_plot_type(self.plot_type)
            new_pane.set_markers(self.show_markers)
            new_pane.set_grid(self.show_grid)
            new_pane.set_error(self.show_error)
            new_pane.data_changed = True

            self.panes.append(new_pane)

        self._assign_axes()
        self.reset_artists()
        self.current_pane = [len(self.panes) - 1]
        self.set_block_current_pane_signal(False)
        self.signals.atrophy_bg_partial.emit()

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
        self.gallery.reset_artists('all')

    def reset_artists(self) -> None:
        """Recreate all artists from models."""
        self.gallery.reset_artists('all')
        for pane_ in self.panes:
            new_drawings = pane_.create_artists_from_current_models()
            successes = 0
            successes += self.gallery.add_drawings(new_drawings)
            if successes != len(new_drawings):
                log.debug('Error encountered while creating artists')

        # add borders back in
        self._add_pane_artists()

        # add fits back in; managed by view
        self.signals.toggle_fit_visibility.emit()

        # add reference models if available
        self.signals.update_reference_lines.emit()

    def _add_pane_artists(self) -> None:
        """Track existing border artists."""
        borders = dict()
        for pane_number, pane_ in enumerate(self.panes):
            if self.highlight_pane:
                if (self.current_pane is not None
                        and pane_number in self.current_pane):
                    visible = True
                elif self._current_pane is None and pane_number == 0:
                    # catch for case where border is added before
                    # current pane is set, on initialization
                    visible = True
                else:
                    visible = False
            else:
                visible = False
            borders[f'pane_{pane_number}'] = {'kind': 'border',
                                              'artist': pane_.get_border(),
                                              'visible': visible}
        self.gallery.add_patches(borders)

    def _add_crosshair(self) -> None:
        """Track existing crosshair artists."""
        crosshairs = list()
        for pane_number, pane_ in enumerate(self.panes):
            pane_lines = pane_.plot_crosshair()
            for pane_line in pane_lines:
                model_name = f'crosshair_pane_{pane_number}'
                pane_line.set_high_model(model_name)
                pane_line.set_visible(False)
                crosshairs.append(pane_line)

        self.gallery.add_crosshairs(crosshairs)

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

    def remove_pane(self, pane_id: Optional[List[int]] = None) -> None:
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
            if pane_id is not None:
                if idx not in pane_id:
                    keep_panes.append(self.panes[idx])
            else:
                if idx not in self.current_pane:
                    keep_panes.append(self.panes[idx])

        self.panes = keep_panes

        # reset current pane
        if self.populated():
            self._assign_axes()
            self.current_pane = [len(self.panes) - 1]
        else:
            # no panes left - leave figure blank
            self.current_pane = None

        self.reset_artists()
        self.set_block_current_pane_signal(False)

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

    def change_current_pane(self, pane_id: List[int]) -> bool:
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
        elif len(pane_id) == 0:
            self.current_pane = list()
            return True
        elif all([len(self.panes) > p >= 0 for p in pane_id]):
            self.current_pane = pane_id
            return True
        else:
            return False

    def determine_selected_pane(self, ax: Optional[ma.Axes] = None,
                                all_ax: Optional[bool] = False
                                ) -> List[int]:
        """
        Determine what pane corresponds to a provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to find.
        all_ax : bool
            True if all axis are to be selected.

        Returns
        -------
        index : int, None
            The pane index of the correct pane, or None if the
            axes were not found.
        """
        if all_ax:
            panes = list(range(len(self.panes)))
        else:
            panes = list()
            for i, _pane in enumerate(self.panes):
                if ax in _pane.axes():
                    panes.append(i)
        return panes

    def determine_pane_from_model(self, model_id: str,
                                  order: Optional[int] = None) -> List[PT]:
        """
        Determine pane containing specified model.

        Parameters
        ----------
        model_id : str
            Specific model_id associated with the model to be found.
        order : int, optional
            Specific order (aperture number) to be found.

        Returns
        -------
        found : list[(int,sofia_redux.visualization.display.pane.Pane)]
            List of panes containing given model_id and order number.

        """
        found = list()
        for i, pane_ in enumerate(self.panes):
            if pane_.contains_model(model_id, order=order):
                found.append((i, pane_))
        return found

    def get_current_pane(self) -> Optional[List[PT]]:
        """
        Return the current pane.

        Returns
        -------
        pane : sofia_redux.visualization.display.pane.Pane, None
            The current pane if a current pane exists;
            None otherwise.
        """
        if self.populated():
            panes = [self.panes[i] for i in self.current_pane]
            return panes
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

    def get_orders(self, target: Optional[Any] = None,
                   enabled_only: Optional[bool] = True,
                   model: Optional[MT] = None,
                   filename: Optional[str] = None,
                   model_id: Optional[IDT] = None,
                   group_by_model: Optional[bool] = True,
                   kind: Optional[str] = 'order') -> Dict:
        """
        Get the orders associated with a given pane selection.

        Parameters
        ----------
        target : str, None, dict, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.
        enabled_only : bool, optional
            Determines if an order is going to be visible or not.
        model : high_model.HighModel, optional
            Target model
        filename : str
            Name of file
        model_id : uuid.UUID
            Unique UUID for an HDUL.
        group_by_model : bool, optional.
            If set, return a dictionary with the keys are model names
            and the values are the orders for that model. Otherwise,
            return a list of all model orders combined.

        Returns
        -------
        orders : dict
            Dictionary of orders for corresponding to pane selection
            provided. Keys are the indices of the panes.
        """
        panes, axes = self.parse_pane_flag(target)
        orders = dict()

        if panes:
            for i, _pane in enumerate(panes):
                pane_index = self.panes.index(_pane)
                orders[pane_index] = _pane.get_orders(
                    enabled_only=enabled_only, by_model=group_by_model,
                    target_model=model, filename=filename,
                    model_id=model_id, kind=kind)
        return orders

    def ap_order_state(self, target, model_ids):
        """
        Get the current aperture and order configuration.

        Parameters
        ----------
        target : str, None, list of int, or list of Panes
            Panes to examine.
        model_ids : list of UUID
            Model IDs to examine.

        Returns
        -------
        apertures, orders : dict, dict
            Keys are pane index for which the specified model was
            found. Values are numbers of apertures and orders displayed,
            respectively, for that pane.
        """
        panes, axes = self.parse_pane_flag(target)
        apertures = dict()
        orders = dict()
        if panes:
            for i, pane_ in enumerate(panes):
                pane_index = self.panes.index(pane_)
                ap_count, ord_count = pane_.ap_order_state(model_ids)
                orders[pane_index] = ord_count
                apertures[pane_index] = ap_count
        return apertures, orders

    def get_scales(self, target: Optional[Any] = None) -> List:
        """
        Get the axis scales associated with a given pane selection.

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

    def model_backup(self, models: MT, target):
        """
        Obtain the backup models and assign them to panes.

        Parameters
        ----------
        models : high_model.HighModel
            HighModels to be loaded into pane
        target : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.
        """
        panes, axes = self.parse_pane_flag(target)
        for _pane in panes:
            if _pane is not None:
                _pane.update_model(models)

    def assign_models(self, mode: str, models: Dict[str, MT],
                      indices: Optional[List[int]] = None) -> int:
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
        errors = 0
        if mode == 'split':
            pane_count = self.pane_count()
            model_keys = list(models.keys())
            models_per_pane = self._assign_models_per_pane(
                model_count=len(models), pane_count=pane_count)
            for i, pane_ in enumerate(self.panes):
                model_count = models_per_pane[i]
                for j in range(model_count):
                    model_ = models[model_keys.pop(0)]
                    try:
                        self.add_model_to_pane(model_=model_, panes=pane_)
                    except EyeError:
                        errors += 1
        elif mode == 'first':
            for model_ in models.values():
                try:
                    self.add_model_to_pane(model_=model_, panes=self.panes[0])
                except EyeError:
                    errors += 1
        elif mode == 'last':
            for model_ in models.values():
                try:
                    self.add_model_to_pane(model_=model_, panes=self.panes[-1])
                except EyeError:
                    errors += 1
        elif mode == 'assigned':
            # TODO: This method assumes the dictionary remains ordered,
            #  which is the default behavior of dictionaries. However,
            #  it is not reliable. Change this so `indices` are also
            #  a dict with the same keys as models.
            for model_, pane_index in zip(models.values(), indices):
                try:
                    self.add_model_to_pane(model_=model_,
                                           panes=self.panes[pane_index])
                except EyeError:
                    errors += 1
        else:
            raise RuntimeError('Invalid mode')
        return errors

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
                          panes: Optional[Union[PT, List[PT]]] = None) -> None:
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
        panes : sofia_redux.visualization.display.pane.Pane, Optional
            A list of panes to which we add model. If not provided, add
            to current pane.
        """
        if panes is None:
            if not self.populated():
                self.add_panes(model_.default_ndims, n_panes=1)
            panes = [self.panes[p] for p in self.current_pane]
        elif not isinstance(panes, list):
            panes = [panes]

        successes = list()
        for pane_ in panes:
            additions = list()
            if self.model_matches_pane(pane_, model_):
                try:
                    addition = pane_.add_model(model_)
                except EyeError:
                    pane_.remove_model(model=model_)
                    raise
                    # index = self.panes.index(pane_)
                    # log.warning(f'{os.path.basename(model_.filename)}  '
                    #             f'incompatible with Pane {index + 1:d}')
                else:
                    additions.extend(addition)
            else:
                self.add_panes(n_dims=model_.default_ndims, n_panes=1)
                for p in self.current_pane:
                    try:
                        addition = self.panes[p].add_model(model_)
                    except EyeError:
                        self.panes[p].remove_model(model=model_)
                        raise
                    else:
                        additions.extend(addition)
            if additions:
                successes.append(self.gallery.add_drawings(additions))

        if successes:
            log.debug('Added model to panes')
            self.signals.update_reference_lines.emit()

    def remove_model_from_pane(
            self, filename: Optional[str] = None,
            model_id: Optional[IDT] = None,
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
        model_id : uuid.UUID
            Unique Id associated with an HDUL
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
        if filename is None and model_ is None and model_id is None:
            raise RuntimeError('Must specify which model to remove '
                               'with either its filename or the '
                               'model itself.')
        if filename is not None and not isinstance(filename, list):
            filename = [filename]
        if model_id is not None and not isinstance(model_id, list):
            model_id = [model_id]

        if panes is None:
            panes = self.panes
        elif not isinstance(panes, list):
            panes = [panes]

        parsed = list()
        for i, p in enumerate(panes):
            if isinstance(p, int):
                try:
                    parsed.append(self.panes[p])
                except IndexError:
                    pass
            elif not isinstance(p, pane.Pane):
                pass
            else:
                parsed.append(p)

        for _pane in parsed:
            if filename:
                for name in filename:
                    _pane.remove_model(filename=name, model=model_)
            elif model_id:
                for mid in model_id:
                    _pane.remove_model(model_id=mid, model=model_)

        # trigger full artist and background regeneration
        self.clear_all()
        if self.recording:
            self.end_cursor_records()
        self.signals.atrophy_bg_full.emit()

    def update_reference_lines(self, models: RT):
        """
        Remove and replot reference lines.

        Parameters
        ----------
        models : reference_model.ReferenceData
             reference_model.ReferenceData objects for the reference lines.
        """

        self.gallery.reset_artists(selection='reference')
        if models.get_visibility('ref_line'):
            for pane_ in self.panes:
                additions = pane_.update_reference_data(models, plot=True)
                if additions:
                    success = self.gallery.add_drawings(additions)
                    if success:
                        log.debug('Updated reference data')
        self.signals.atrophy_bg_partial.emit()

    def model_extensions(self, model_id, pane_=None,
                         pane_index: Optional[int] = None) -> List[str]:
        """
        Obtain an extension list of a model in a pane.

        Parameters
        ----------
        model_id : uuid.UUID
            Unique model id associated with an HDUL
        pane_ : sofia_redux.visualization.spectrum.panes.Pane, optional
            Pane object containing the model. If not provided, all panes
            will be checked.
        pane_index : int, optional
            Index of the pane from which model extensions are desired when
            no pane has been specified.

        Returns
        -------
        ext : list
            List of extensions.
        """
        if pane_ is None:
            if pane_index is None:
                return list()
            else:
                pane_ = self.panes[pane_index]
        ext = pane_.model_extensions(model_id)
        return ext

    ####
    # Plotting
    ####
    def unload_reference_model(self, models):
        """Unload the reference model."""
        # TODO: argument is unnecessary here
        for pane_ in self.panes:
            pane_.unload_ref_model()

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

                # update reference data for new limits
                ref_updates = _pane.update_reference_data()
                if ref_updates is not None:
                    self.gallery.update_reference_data(pane_=_pane,
                                                       updates=ref_updates)

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
        changed = False
        for _pane in panes:
            if _pane is not None:
                _pane.set_units(units, axes)
                changed = True
        # trigger full artist regeneration
        if changed:
            self.clear_all()

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
            log.debug(f'No valid panes found for (target, fields) = '
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

    def set_orders(self, orders: Dict[int, Dict[IDT, List[int]]],
                   enable: Optional[bool] = True,
                   aperture: Optional[bool] = False) -> None:
        """
        Enable specified orders.

        Parameters
        ----------
        orders : dict
            Keys are indices for the panes to update. Values
            are dicts, with model ID keys, order list values.
        enable : bool, optional
            If set enable the orders, otherwise disable orders.
            Defaults to True.
        aperture : bool, optional
            If set the order numbers in `orders` are actually
            aperture numbers. Defaults to False.
        """
        for pane_id, pane_orders in orders.items():
            pane_ = self.parse_pane_flag([pane_id])
            pane_ = pane_[0]
            if pane_ is None:
                continue
            elif isinstance(pane_, list):
                pane_ = pane_[0]
            updates = pane_.set_orders(pane_orders, enable, aperture,
                                       return_updates=True)
            self.gallery.update_artist_options(pane_=pane_, options=updates)

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
        """
        Set the pane overplot flag.

        Parameters
        ----------
        state : bool
            True to show overplot; False to hide
        target : str, None, or list of int
            May be set to 'all' to apply to all panes, None
            to apply only to the current pane, or a list of
            pane indices to modify.
        """
        panes, axes = self.parse_pane_flag(target)
        if state:  # Turning on
            for _pane in panes:
                if _pane is not None:
                    _pane.set_overplot(state)
            self.reset_artists()
        else:  # Turning off
            self.gallery.reset_artists(selection='alt', panes=panes)
            for _pane in panes:
                if _pane is not None:
                    _pane.reset_alt_axes(remove=True)
                    _pane.set_overplot(state)

        # trigger full regeneration: some things get orphaned
        # when axes change (eg. border artist)
        self.clear_all()

    def parse_pane_flag(self, flags: Optional[Union[List, Dict,
                                                    List[Union[int, PID]]]]
                        ) -> Tuple[List[PT], str]:
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
        panes, axis : list, str
            List of panes corresponding to input flag and corresponding axis
        """
        log.debug(f'Parsing {flags} ({type(flags)})')
        axis = ''
        panes = None
        if flags is None:
            panes = self.get_current_pane()
        elif flags == 'all':
            panes = self.panes
        elif isinstance(flags, int):
            try:
                panes = [self.panes[flags]]
            except IndexError:
                log.debug(f'Unable to parse pane flag {flags}: '
                          f'Invalid index')
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
                pane_flags = flags['pane']
            except KeyError:
                raise EyeError(f'Unable to parse pane flag {flags}')
            else:
                if pane_flags == 'all':
                    panes = self.panes
                elif pane_flags == 'current':
                    panes = [self.get_current_pane()]
                elif isinstance(pane_flags, list):
                    panes = list()
                    for flag in pane_flags:
                        if isinstance(flag, pane.Pane):
                            panes.append(flag)
                        elif isinstance(flag, int):
                            try:
                                panes.append(self.panes[flag])
                            except IndexError:
                                log.debug(f'Invalid pane flag: {flag}')
                                continue
                        else:
                            log.debug(f'Invalid pane flag: {flag}')
                            continue
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
        full_updates = dict()
        if self.populated():
            for i, pane_ in enumerate(self.panes):
                pane_.set_color_cycle_by_name(cycle_name)
                updates = pane_.update_colors()
                self.gallery.update_artist_options(pane_=pane_,
                                                   options=updates)
                full_updates[i] = updates
        self.signals.atrophy.emit()
        return full_updates

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
                self.gallery.update_line_type(pane_=pane_,
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
                self.gallery.update_artist_options(pane_=pane_,
                                                   options=marker_updates)
        self.signals.atrophy.emit()

    def get_markers(self, model_id, pane_):
        """
        Get the markers in a pane.

        Parameters
        ----------
        model_id : uuid.UUID
            Unique id associated with an HDUL
        pane_ : sofia_redux.visualization.display.pane.Pane
            The pane object from which markers are to be obtained.

        Returns
        -------
        markers : list
            A list of markers in a pane for a model_id
        """
        panes, _ = self.parse_pane_flag(pane_)
        markers = list()
        for pane_ in panes:
            markers.extend(pane_.get_marker(model_id))
        return markers

    def get_colors(self, model_id, pane_):
        """
        Get the colors in a pane.

        Parameters
        ----------
        model_id : uuid.UUID
            Unique id associated with an HDUL
        pane_ : sofia_redux.visualization.display.pane.Pane
            The pane object from which colors are to be obtained.

        Returns
        -------
        colors : list
            A list of colors in a pane for a model_id
        """
        panes, _ = self.parse_pane_flag(pane_)
        colors = list()
        for pane_ in panes:
            colors.extend(pane_.get_color(model_id))
        return colors

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

                updates = pane_.update_visibility(error=True)
                self.gallery.update_artist_options(pane_=pane_,
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
        self.gallery.update_artist_options(pane_=pane_, options=updates)
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
        self.gallery.update_artist_options(pane_=pane_, options=updates)
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
        pane_indexes = self.determine_selected_pane(event.inaxes)
        data_points = dict()
        for pane_index in pane_indexes:
            data_points.update(self.panes[pane_index].data_at_cursor(event))
        self.gallery.update_marker(data_points)

        return data_points

    def crosshair(self, event: mbb.MouseEvent) -> None:
        """
        Display a crosshair at the cursor position.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse motion event.
        """
        pane_index = self.determine_selected_pane(event.inaxes)
        if isinstance(pane_index, list):
            if len(pane_index) > 0:
                pane_index = pane_index[0]
            else:
                pane_index = None
        if (pane_index is not None
                and self._cursor_pane is not None
                and pane_index in self._cursor_pane):
            data_point = self.panes[pane_index].xy_at_cursor(event)
            direction = self._parse_cursor_direction(mode='crosshair')
            self.gallery.update_crosshair(pane_index, data_point=data_point,
                                          direction=direction)
            self.signals.atrophy.emit()

    def clear_crosshair(self) -> None:
        """Clear any displayed crosshairs."""
        self.gallery.reset_artists(selection='crosshair')

    def reset_data_points(self) -> None:
        """Reset any displayed cursor markers."""
        if self.populated():
            self.gallery.hide_cursor_markers()

    def reset_zoom(self, all_panes: Optional[bool] = False,
                   targets: Optional[Dict] = None) -> None:
        """
        Reset axis limits to defaults.

        Parameters
        ----------
        all_panes : bool, optional
            If True, all axes will be reset. Otherwise, only
            the current pane will be reset.
        targets : dict, optional
            Specific panes to reset, specified as a list of int
            under the 'pane' key in the input dictionary.
        """
        if not self.populated():
            return
        if all_panes:
            panes = self.panes
        elif targets:
            try:
                pane_numbers = targets['pane']
            except KeyError:  # pragma: no cover
                # missing 'pane' in targets - shouldn't happen under
                # normal circumstances
                panes = [self.panes[p] for p in self.current_pane]
            else:
                panes = [self.panes[p] for p in pane_numbers]
        else:
            panes = [self.panes[p] for p in self.current_pane]
        # Don't want to include reference data in relim
        for _pane in panes:
            self.gallery.reset_artists(selection='reference', panes=[_pane])
            _pane.reset_zoom()
            ref_artists = _pane.update_reference_data()
            if ref_artists is not None:
                self.gallery.update_reference_data(pane_=_pane,
                                                   updates=ref_artists)
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
            self.recording = True
            self._cursor_pane = self._current_pane
            self._add_crosshair()
        else:
            self.recording = False
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
        pane_index = self.determine_selected_pane(event.inaxes)[0]
        if pane_index in self._cursor_pane:
            location = self.panes[pane_index].xy_at_cursor(event)
            if None in location:  # pragma: no cover
                # could happen if overplot data is present and
                # doesn't exactly match plot data
                return
            self._cursor_locations.append(location)

            if len(self._cursor_locations) == 2:
                self.end_cursor_records()
            else:
                # after the first click, make sure the next click is
                # in the same pane
                # otherwise, weirdness happens with 'All' pane button
                # and mismatched pane displays
                self._cursor_pane = [pane_index]

                guide_drawings = self.panes[pane_index].plot_guides(
                    location, kind=self._parse_cursor_direction())
                self.gallery.add_drawings(guide_drawings)
                self.signals.atrophy.emit()

    def end_cursor_records(self, pane_index: Optional[int] = None) -> None:
        """
        Complete zoom or fit interactions.

        User specified locations are used to either set axis
        limits or else display a new fit to a plot feature.

        Parameters
        ----------
        pane_index : int
            Index of the pane to update.
        """
        if pane_index is None:
            pane_index = self.current_pane
        if not isinstance(pane_index, list):  # pragma: no cover
            # this should not be reachable
            pane_index = [pane_index]
        if len(self._cursor_locations) == 2:
            for index in pane_index:
                if 'zoom' in self._cursor_mode:
                    self._end_zoom(index)
                elif 'fit' in self._cursor_mode:
                    self._end_fit(index)
        # reset cursor, but not fit params -- is needed for the
        # cursor recording
        self._cursor_mode = ''
        self._cursor_locations = list()
        self.recording = False

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
        if len(self._cursor_locations) == 2:
            self.panes[pane_index].perform_zoom(
                zoom_points=self._cursor_locations, direction=direction)
        else:
            log.debug('Cancelling zoom')
            self._cursor_locations = list()

        # clear all h and v guides
        self.gallery.reset_artists(selection='h_guide',
                                   panes=self.panes[pane_index])
        self.gallery.reset_artists(selection='v_guide',
                                   panes=self.panes[pane_index])

        # update reference data for new limits
        ref_updates = self.panes[pane_index].update_reference_data()
        if ref_updates is not None:
            self.gallery.update_reference_data(pane_=self.panes[pane_index],
                                               updates=ref_updates)

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

        fit_drawings, fit_params = self.panes[pane_index].perform_fit(
            self._cursor_mode, limits)

        self.gallery.add_drawings(fit_drawings)
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
            panes = [self.panes[p] for p in self.current_pane]

        if not isinstance(flags, list):
            flags = [flags]
        for flag in flags:
            if flag == 'fit':
                self.gallery.reset_artists(flag, panes=panes)
            else:
                self.gallery.reset_artists(f'{flag}_guide', panes=panes)

    def toggle_fits_visibility(self, fits: List[ModelFit]) -> None:
        """
        Update fit artist visibility.

        If failure, loop over panes. Ask each one to make new
        drawing for fit. If model, order, fields don't match,
        pane does nothing. Else remake fit drawing and return them.
        Then add new drawing to Gallery.

        Parameters
        ----------
        fits : list of ModelFit
            Models fit to spectral selections.
        """
        for fit in fits:
            options = {'high_model': fit.get_filename(), 'kind': 'fit',
                       'model_id': fit.get_model_id(),
                       'mid_model': f'{fit.get_order()}.{fit.get_aperture()}',
                       'data_id': fit.get_id(),
                       'updates': {'visible': fit.get_visibility()}}
            options = [drawing.Drawing(**options)]
            for pane_ in self.panes:
                if fit.get_axis() in pane_.axes():
                    result = self.gallery.update_artist_options(
                        pane_, kinds='fit', options=options)
                    if not result:
                        self._regenerate_fit_artists(pane_, fit, options)

                elif (fit.get_fields('x') == pane_.get_field('x')
                      and fit.get_fields('y') == pane_.get_field('y')):
                    self._regenerate_fit_artists(pane_, fit, options)

    def stale_fit_artists(self, fits: List[ModelFit]):
        """
        Recreate fit artists for a list of model fits.

        Removes all Drawing instances for fits on a given pane and makes
        them anew. It accommodates any changes (unit, scale, etc) that doesn't
        change the data at all but does invalidate the current artists
        (which are now marked "stale").

        Parameters
        ----------
        fits : list of ModelFit
            Models fit to spectral selections.
        """
        matching_panes = self._panes_matching_model_fits(fits)
        for pane_idx, fits in matching_panes.items():
            pane_ = self.panes[pane_idx]
            self.gallery.reset_artists(selection='fit', panes=[pane_])
            for fit in fits:
                self._regenerate_fit_artists(pane_, fit)

    def _regenerate_fit_artists(self, pane_, fit, options=None):
        """
        Assign fit artists to a pane.

        Parameters
        ----------
        fit : ModelFit
            Models fit to spectral selections.
        pane_ : pane.Pane
            Pane object to add fit artists to.
        """
        try:
            fit_drawings = pane_.generate_fit_artists(fit)
        except (KeyError, IndexError):
            # can happen if model id no longer exists
            return
        self.gallery.add_drawings(fit_drawings)
        if options:
            self.gallery.update_artist_options(pane_, kinds='fit',
                                               options=options)

    def _panes_matching_model_fits(self, fits: List[ModelFit]
                                   ) -> Dict[int, List[ModelFit]]:
        """
        Get matching panes for a given list of model fits.

        Parameters
        ----------
        fits : list of ModelFit
            Models fit to spectral selections.

        Returns
        -------
        matching_panes : dict
            The key is the index of the pane and values are  list of all the
            Modelfits in that pane.
        """
        matching_panes = dict()
        for fit in fits:
            for idx, pane_ in enumerate(self.panes):
                if fit.get_axis() in pane_.axes():
                    if idx in matching_panes:
                        matching_panes[idx].append(fit)
                    else:
                        matching_panes[idx] = [fit]
        return matching_panes
