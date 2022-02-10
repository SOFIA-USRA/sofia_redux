# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import (List, Dict, Optional, Tuple,
                    Union, TypeVar, Any)
import numpy as np
from numpy import ma as mask

import matplotlib.artist as ma
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PatchCollection

from sofia_redux.visualization import log
from sofia_redux.visualization.display import pane
from sofia_redux.visualization.display.pane import Pane

__all__ = ['Artists']

AT = TypeVar('AT', Line2D, PathCollection, PatchCollection, ma.Artist)
PT = TypeVar('PT', pane.Pane, pane.OneDimPane, pane.TwoDimPane)


class Artists(object):
    """
    Track display artists for viewer plots.

    Artists held by this class are associated with matplotlib
    axes, instantiated and controlled by the
    `sofia_redux.visualization.pane` interface.  This class implements
    updates and modifications to the existing artists.  It does not
    create them, or explicitly track their associated axes.

    Attributes
    ----------
    arts : dict
        Keys are artist types, with allowed values 'line', 'cursor',
        'error_range', 'crosshair', 'guide', 'patch', and 'fit'.
        Values are lists of dicts, containing the artist information
        (e.g. 'model_id', 'artist', 'state').
    """
    def __init__(self):
        self.arts = {'line': list(),
                     'line_alt': list(),
                     'cursor': list(),
                     'cursor_alt': list(),
                     'error_range': list(),
                     'crosshair': list(),
                     'guide': list(),
                     'patch': list(),
                     'fit': list()}

    def add_patches(self, patches: Dict) -> None:
        """
        Add patch artists for tracking.

        Parameters
        ----------
        patches : dict
            Patches to add.  Must have keys 'kind', 'artist', 'visible'.
        """
        for pane_label, patch in patches.items():
            kind = patch['kind']
            artist = patch['artist']
            artist.set_visible(patch['visible'])
            self.arts['patch'].append({'model_id': f'{kind}_{pane_label}',
                                       'artist': artist,
                                       'state': 'new'})

    def set_pane_highlight_flag(self, pane_number: int, state: bool) -> None:
        """
        Set visibility for border pane patches.

        Border pane patches must have been previously added to tracking
        with the `add_patches` method.

        Parameters
        ----------
        pane_number : int
            Pane index associated with the border to update.
        state : bool
            Visibility state to set.  If True, the patch will be visible;
            if False, it will be hidden.
        """
        for patch in self.arts['patch']:
            if patch['model_id'] == f'border_pane_{pane_number}':
                patch['artist'].set_visible(state)
            elif 'border' in patch['model_id']:
                patch['artist'].set_visible(False)

    def add_crosshairs(self, crosshairs: Dict) -> None:
        """
        Add crosshair artists for tracking.

        Parameters
        ----------
        crosshairs : dict
            Must contain keys 'kind', 'artist', 'visible',
            'direction'.
        """
        for pane_label, crosshair in crosshairs.items():
            kind = crosshair['kind']
            artist = crosshair['artist']
            artist.set_visible(crosshair['visible'])
            self.arts['crosshair'].append(
                {'model_id': f'{kind}_{pane_label}',
                 'artist': artist,
                 'state': 'new',
                 'direction': crosshair['direction']})

    def update_crosshair(self, pane_number: int,
                         data_point: Optional[Tuple] = None,
                         direction: Optional[str] = 'hv') -> None:
        """
        Update crosshair position and visibility.

        Any specified directions ('h' or 'v', for horizontal
        and vertical) are updated and made visible.  Any unspecified
        are hidden.

        Parameters
        ----------
        pane_number : int
            Pane index associated with the crosshair.
        data_point : tuple of float
            (x, y) location to update the cursor to.
        direction : ['h', 'v', 'hv'], optional
            Direction of crosshair to update.
        """
        for crosshair in self.arts['crosshair']:
            model_name = f'crosshair_pane_{pane_number}'
            if crosshair['model_id'].startswith(model_name):
                dir = crosshair['direction']
                if dir in direction:
                    if data_point is not None:
                        if dir == 'v':
                            crosshair['artist'].set_xdata(data_point[0])
                        else:
                            crosshair['artist'].set_ydata(data_point[1])
                    crosshair['artist'].set_visible(True)
                else:
                    crosshair['artist'].set_visible(False)
            else:
                # this is hit when zooming in another pane
                crosshair['artist'].set_visible(False)

    def hide_crosshair(self) -> None:
        """Hide all crosshair artists."""
        for crosshair in self.arts['crosshair']:
            crosshair['artist'].set_visible(False)

    def update_line_data(self, pane: Pane, updates: List[Dict],
                         axes: Optional[str] = 'primary') -> None:
        """
        Update data associated with line artists in a single pane.

        Parameters
        ----------
        pane : Pane
            Pane object holding the line artists to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'field'. Keys are from 'new_x_data',
            'new_y_data', and 'new_y_alt_data'.
        axes: 'primary', 'alt', 'both', 'all'

        """
        pri_lines = self.artists_in_pane(pane, 'line')
        alt_lines = self.artists_in_pane(pane, 'line_alt')
        for update in updates:
            lines = list()
            if (axes in ['both', 'all', 'alt']
                    or 'new_x_data' in update.keys()
                    or 'new_y_alt_data' in update.keys()):
                lines.extend(alt_lines)
            if (axes in ['both', 'all', 'primary']
                    or 'new_x_data' in update.keys()
                    or 'new_y_data' in update.keys()):
                lines.extend(pri_lines)
            for line in lines:
                if (line['model_id'] == update['model_id']
                        and line['order'] == update['order']
                        and update['field'] in line['fields']):
                    if isinstance(line['artist'], Line2D):
                        self._set_line_data(line, update)
                    elif isinstance(line['artist'], PathCollection):
                        self._set_scatter_data(line, update)
                    else:
                        log.debug(f'Unable to process artist type '
                                  f'{type(line["artist"])}')

    @staticmethod
    def _set_line_data(line: Dict, update: Dict) -> None:
        """
        Update data for Line2D artists.

        Parameters
        ----------
        line : dict
            The artist to update. Must contain the 'artist'
            key, with a Line2D value.
        update : dict
            The data to update to.  Must contain either the
            'new_x_data' or the 'new_y_data' key.
        """
        try:
            line['artist'].set_xdata(update['new_x_data'])
        except KeyError:
            line['artist'].set_ydata(update['new_y_data'])

    @staticmethod
    def _set_scatter_data(line: Dict, update: Dict) -> None:
        """
        Update data for PathCollection artists (scatter plots).

        Parameters
        ----------
        line : dict
            The artist to update. Must contain the 'artist'
            key, with a PathCollection value.
        update : dict
            The data to update to.  Must contain either the
            'new_x_data' or the 'new_y_data' key.
        """
        current_data = line['artist'].get_offsets()
        try:
            x_data = update['new_x_data']
        except KeyError:
            x_data = current_data[:, 0]
        try:
            y_data = update['new_y_data']
        except KeyError:
            y_data = current_data[:, 1]
        new_data = mask.array(np.vstack((x_data, y_data)).T)
        line['artist'].set_offsets(new_data)

    def update_line_fields(self, pane_: PT, updates: List[Dict]) -> None:
        """
        Update data and labels for a new plot field in a single pane.

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the line artists to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'new_field' keys, as well as either
            'new_x_data' or 'new_y_data' (but not both).
        """
        lines = self.artists_in_pane(pane_=pane_, kind='line')
        for line in lines:
            for update in updates:
                if (line['model_id'] == update['model_id']
                        and line['order'] == update['order']):
                    current_label = line['artist'].get_label()
                    old_field = current_label.split()[-1]
                    new_label = current_label.replace(old_field,
                                                      update['new_field'])
                    line['artist'].set_label(new_label)
                    try:
                        line['artist'].set_xdata(update['new_x_data'])
                    except KeyError:
                        line['artist'].set_ydata(update['new_y_data'])
                    line['state'] = 'new'

    def update_line_type(self, pane_: PT, updates: List[Dict]) -> None:
        """
        Update plot line type.

        Line2D artists are updated in place; PathCollection artists
        are replaced with an equivalent new artist.

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the line artists to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'new_type' keys.  The 'new_type' should
            be one of 'step', 'line', or 'scatter'.  The dict may
            also contain a 'new_marker' key.
        """
        lines = self.artists_in_pane(pane_=pane_, kind='line')
        for line in lines:
            for update in updates:
                if (line['model_id'] == update['model_id']
                        and line['order'] == update['order']):
                    marker = update.get('new_marker')
                    if isinstance(line['artist'], Line2D):
                        if update['new_type'] == 'step':
                            line['artist'].set_drawstyle('steps-mid')
                            line['artist'].set_linestyle('-')
                            line['artist'].set_marker(marker)
                        elif update['new_type'] == 'line':
                            line['artist'].set_drawstyle('default')
                            line['artist'].set_linestyle('-')
                            line['artist'].set_marker(marker)
                        elif update['new_type'] == 'scatter':
                            line['artist'].set_drawstyle('default')
                            line['artist'].set_linestyle('')
                            line['artist'].set_marker(marker)

                    elif isinstance(line['artist'], PathCollection):
                        new_artist = self.convert_to_line(line['artist'],
                                                          update['new_type'],
                                                          marker)
                        line['artist'].remove()
                        self._replace_artist(kind='line',
                                             model=update['model_id'],
                                             order=update['order'],
                                             new_artist=new_artist)

    @staticmethod
    def convert_to_scatter(line_artist: Line2D,
                           marker: str) -> PathCollection:
        """
        Convert a line plot to a scatter plot.

        Parameters
        ----------
        line_artist : Line2D
            The line artist to replace.
        marker : str
            The marker symbol to use in the plot.  If None,
            the default 'o' symbol is used.

        Returns
        -------
        scatter_artist : PathCollection
            The new scatter plot artist.
        """
        x, y = line_artist.get_data()
        color = line_artist.get_color()
        ax = line_artist.axes
        label = line_artist.get_label()
        if marker is None:
            marker = 'o'
        scatter_artist = ax.scatter(x, y, color=color, label=label,
                                    animated=True, marker=marker)
        return scatter_artist

    @staticmethod
    def convert_to_line(scatter_artist: PathCollection,
                        drawstyle: str,
                        marker: str) -> Line2D:
        """
        Convert a scatter plot to a line plot.

        Parameters
        ----------
        scatter_artist : PathCollection
            The scatter plot artist to replace.
        drawstyle : str
            The line drawing style for the new line plot.
            Should be 'line' or 'step'.
        marker : str
            The marker symbol to use in the plot.  If None,
            the default 'o' symbol is used.

        Returns
        -------
        line_artist : Line2D
            The new line plot artist.
        """
        data = scatter_artist.get_offsets().data
        color = scatter_artist.get_facecolor()[0]
        ax = scatter_artist.axes
        label = scatter_artist.get_label()
        line_artist = ax.plot(data[:, 0], data[:, 1], c=color,
                              label=label, animated=True,
                              marker=marker)[0]
        if drawstyle == 'line':
            line_artist.set_drawstyle('default')
        else:
            line_artist.set_drawstyle('steps-mid')
        return line_artist

    def update_artist_options(self, pane_: Union[PT, int],
                              kinds: Optional[Union[List[str], str]] = None,
                              options: Optional[List[Dict[str, Any]]] = None
                              ) -> bool:
        """
        Update artist display options.

        Currently supported options for each artist type are:
           - 'line' artists: 'new_color', 'new_visibility', 'new_marker'
           - 'error_range' artists: 'new_color', 'new_visibility'
           - 'patch' artists: 'new_color'
           - 'cursor' artists: 'new_color'

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the artists to update.
        kinds : str, list
            Kinds of artists to update. If all artists are to be
            updated, set to 'all'.
        options : list of dict
            Options to apply.  The dicts must contain 'model_id' and
            'order' keys.  Other keys allowed are 'new_color',
            'new_visibility', and 'new_marker'.
        """

        if kinds is None:
            kinds = ['all']
        elif not isinstance(kinds, list):
            kinds = [kinds]

        results = list()
        for kind in kinds:
            if kind in ['line', 'line_alt', 'all']:
                out = self._update_line_artist_options(pane_, options)
                results.append(out)
            if kind in ['error', 'error_range', 'all']:
                out = self._update_error_artist_options(pane_, options)
                results.append(out)
            if kind in ['border', 'all']:
                out = self._update_border_artist_options(pane_, options)
                results.append(out)
            if kind in ['cursor', 'cursor_alt', 'all']:
                out = self._update_cursor_artist_options(pane_, options)
                results.append(out)
            if kind in ['fit', 'all']:
                out = self._update_fit_artist_options(pane_, options)
                results.append(out)
        return any(results)

    def _update_line_artist_options(self, pane_: PT,
                                    options: List[Dict[str, Any]]) -> bool:
        """
        Update options for Line objects.

        Parameters
        ----------
        pane_ : Pane
            Pane associated with artist
        options : list
            List of dictionaries describing the updates to make

        Returns
        -------
        results : bool
            True if any of the updates are successful, False if
            they all fail.

        """
        # plot lines: markers, colors, visibility
        pri_lines = self.artists_in_pane(pane_=pane_, kind='line')
        alt_lines = self.artists_in_pane(pane_=pane_, kind='line_alt')
        results = list()
        for line in pri_lines + alt_lines:
            for option in options:
                if 'data_id' in option:
                    continue
                if (line['model_id'] == option['model_id']
                        and line['order'] == option['order']):
                    if (isinstance(line['artist'], Line2D)
                            and line in pri_lines):
                        self._update_marker_style(line, option)
                    results.append(self._update_visibility(line, option))
                    results.append(self._update_color(line, option))
        return any(results)

    def _update_error_artist_options(self, pane_: PT,
                                     options: List[Dict[str, Any]]) -> bool:
        """
        Update options for Line objects.

        Parameters
        ----------
        pane_ : Pane
            Pane associated with artist
        options : list
            List of dictionaries describing the updates to make

        Returns
        -------
        results : bool
            True if any of the updates are successful, False if
            they all fail.

        """

        # error shading: color and error visibility
        lines = self.artists_in_pane(pane_=pane_, kind='error_range')
        results = list()
        for line in lines:
            for option in options:
                if 'data_id' in option:
                    continue
                if (line['model_id'] == option['model_id']
                        and line['order'] == option['order']):
                    results.append(self._update_error_visibility(line, option))
                    results.append(self._update_color(line, option))
        return any(results)

    def _update_border_artist_options(self, pane_: PT,
                                      options: List[Dict[str, Any]]) -> bool:
        """
        Update options for Line objects.

        Parameters
        ----------
        pane_ : Pane
            Pane associated with artist
        options : list
            List of dictionaries describing the updates to make

        Returns
        -------
        results : bool
            True if any of the updates are successful, False if
            they all fail.

        """
        # borders: color only
        patches = self.artists_in_pane(pane_=pane_, kind='patch')
        results = list()
        for patch in patches:
            for option in options:
                if 'data_id' in option:
                    continue
                if 'border' in option['model_id']:
                    results.append(self._update_color(patch, option))
        return any(results)

    def _update_cursor_artist_options(self, pane_: PT,
                                      options: List[Dict[str, Any]]) -> bool:
        """
        Update options for cursor objects.

        Parameters
        ----------
        pane_ : Pane
            Pane associated with artist
        options : list
            List of dictionaries describing the updates to make

        Returns
        -------
        results : bool
            True if any of the updates are successful, False if
            they all fail.

        """
        points = self.artists_in_pane(pane_=pane_, kind='cursor')
        points_alt = self.artists_in_pane(pane_=pane_, kind='cursor_alt')
        results = list()
        for point in points + points_alt:
            for option in options:
                if 'data_id' in option:
                    continue
                if (point['model_id'] == option['model_id']
                        and point['order'] == option['order']):
                    results.append(self._update_color(point, option))
        return any(results)

    def _update_fit_artist_options(self, pane_: PT,
                                   options: List[Dict[str, Any]]) -> bool:
        """
        Update options for artists of curve fits.

        Parameters
        ----------
        pane_ : Pane
            Pane associated with artist
        options : list
            List of dictionaries describing the updates to make

        Returns
        -------
        results : bool
            True if any of the updates are successful, False if
            they all fail.

        """
        lines = self.artists_in_pane(pane_=pane_, kind='fit')
        results = list()
        for line in lines:
            for option in options:
                if 'data_id' not in option or 'data_id' not in line:
                    continue
                if (option['model_id'] in line['model_id']
                        and line['order'] == option['order']
                        and option['data_id'] in line['data_id']):
                    results.append(self._update_color(line, option))
                    results.append(self._update_visibility(line, option))
        return any(results)

    @staticmethod
    def _update_marker_style(line: Dict, option: Dict) -> bool:
        """
        Update marker style for Line2D artists.

        Parameters
        ----------
        line : dict
            The artist to update. Must contain the 'artist' key.
        option : dict
            The option to update to.  Must contain the 'new_marker'
            key.
        """
        try:
            line['artist'].set_marker(option['new_marker'])
        except (KeyError, ValueError):
            return False
        else:
            return True

    @staticmethod
    def _update_color(line: Dict, option: Dict) -> bool:
        """
        Update color for existing artists.

        Parameters
        ----------
        line : dict
            The artist to update. Must contain the 'artist'
            key.
        option : dict
            The option to update to.  Must contain the 'new_color'
            key.
        """
        try:
            line['artist'].set_color(option['new_color'])
        except (KeyError, ValueError):
            return False
        else:
            return True

    @staticmethod
    def _update_visibility(line: Dict, option: Dict) -> bool:
        """
        Update visibility for existing artists.

        Parameters
        ----------
        line : dict
            The artist to update. Must contain the 'artist'
            key.
        option : dict
            The option to update to.  Must contain the 'new_visibility'
            key.
        """
        try:
            line['artist'].set_visible(option['new_visibility'])
        except (KeyError, ValueError):
            return False
        else:
            return True

    @staticmethod
    def _update_error_visibility(line: Dict, option: Dict) -> bool:
        """
        Update visibility for existing error_range artists.

        Parameters
        ----------
        line : dict
            The artist to update. Must contain the 'artist'
            key.
        option : dict
            The option to update to.  Must contain the 'new_visibility'
            key.
        """
        try:
            line['artist'].set_visible(option['new_error_visibility'])
        except (KeyError, ValueError):
            return False
        else:
            return True

    def _replace_artist(self, kind: str, model: str, order: int,
                        new_artist: AT) -> None:
        """
        Replace an existing artist with a new one.

        Parameters
        ----------
        kind : str
            The artist kind.
        model : str
            The model ID associated with the artist.
        order : int
            The order associated with the artist.
        new_artist : matplotlib.artist.Artist
            The new artist.
        """
        for artist in self.arts[kind]:
            if artist['model_id'] == model and artist['order'] == order:
                artist['artist'] = new_artist
                break

    def update_error_ranges(self, pane: Pane, updates: List[Dict]) -> None:
        """
        Update data associated with error range artists.

        Typically called with the `update_line_data` method, which
        updates data associated with a line plot.  However,
        error range artists are always replaced with a new artist,
        rather than updated in place.

        Parameters
        ----------
        pane : Pane
            Pane object holding the line artists to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'new_artist' keys.
        """
        lines = self.artists_in_pane(pane_=pane, kind='error_range')
        for line in lines:
            for update in updates:
                if (line['model_id'] == update['model_id']
                        and line['order'] == update['order']):
                    label = line['artist'].get_label()
                    line['artist'] = update['new_artist']
                    line['artist'].set_label(label)

    def remove_line(self, pane, model_id: Optional[str] = None,
                    order: Optional[int] = None,
                    label: Optional[str] = None) -> None:
        """Remove a data line from being tracked in the artists."""
        # todo - implement or delete?
        raise NotImplementedError

    def reset_artists(self, selection: str,
                      panes: Optional[List] = None) -> None:
        """
        Reset and remove all artists for a given selection.

        Parameters
        ----------
        selection : str
            Type of artists to reset. Acceptable values are 'lines'
            which resets the data lines, 'cursor' which resets the
            marker for the cursor location, 'collections' which
            resets scatter plots, 'v_guides', 'h_guides', 'f_guides',
            'a_guides' which resets vertical, horizontal, fit,
            and all guides, respectively. The 'all' flag clears everything.
        panes : list, optional
            The pane to clear the selected artists from. If not provided,
            the selected artists will be cleared from all panes.
        """
        log.debug(f'Resetting {selection} artists')
        if selection == 'all':
            for kind, arts in self.arts.items():
                self._clear_artists(kind=kind, panes=panes)
        elif selection == 'alt':
            for kind in ['line_alt', 'cursor_alt']:
                self._clear_artists(kind=kind, panes=panes)
        elif selection in ['line', 'line_alt', 'cursor', 'cursor_alt',
                           'collections', 'crosshair', 'patch', 'fit']:
            self._clear_artists(kind=selection, panes=panes)
        elif 'guide' in selection:
            self._clear_guides(flag=selection[0], panes=panes)
        else:
            log.debug(f'Invalid artist selection {selection}. '
                      f'No reset performed')

    def _clear_guides(self, flag: str, panes: List[PT]) -> None:
        """
        Remove guide artists.

        Parameters
        ----------
        flag : ['v', 'h', 'f', 'a']
            Denotes what kind of guide to clear
        panes : list
            List of panes to clear from.
        """
        others = list()
        to_clear = list()
        if panes is None:
            to_clear = self.arts['guide']
        else:
            if not np.iterable(panes):
                panes = [panes]
            for pane_ in panes:
                to_clear.extend(self.artists_in_pane(pane_, kind='guide'))
        for artist in to_clear:
            if artist['model_id'].startswith(flag) or flag == 'a':
                artist['artist'].remove()
            else:
                others.append(artist)
        self.arts['guide'] = others

    def _clear_artists(self, kind: str, panes: List) -> None:
        """
        Clear artists.

        Parameters
        ----------
        kind : str
            Denotes the kind of artist to clear.
        panes : list
            List of panes to clear artists from.
        """
        if panes is None:
            to_clear = self.artists_in_pane(panes, kind=kind)
        else:
            to_clear = list()
            for pane_ in panes:
                to_clear.extend(self.artists_in_pane(pane_, kind=kind))
        new_arts = list()
        for art in self.arts[kind]:
            if art not in to_clear:
                new_arts.append(art)
            else:
                try:
                    art['artist'].remove()
                except ValueError:
                    continue
        self.arts[kind] = new_arts

    def add_artists(self, artists: Dict) -> int:
        """
        Add multiple artists.

        Parameters
        ----------
        artists : dict
            Collection of artists to add. Keys are the model_id
            for the artist. Values are dictionaries of the artist
            details.

        Returns
        -------
        count : int
            Number of artists successfully added.
        """
        log.info(f'Adding {len(artists)} artists')
        results = list()
        for name, artist in artists.items():
            results.append(self.add_artist(artist=artist, name=name))
        return sum(results)

    def add_artist(self, artist: Dict, name: str) -> bool:
        """
        Add a single artist.

        Parameters
        ----------
        artist : dict
            Dictionary describing the artist. Keys are the
            order number the artist is associated with.
            Values are a dictionary of values.
        name : str
            Model_id associated with the artists being added.

        Returns
        -------
        success : bool
            True if the addition was successful, False otherwise.
        """
        success = False
        for order_number, lines in artist.items():
            for line_type, line_field in self.arts.items():
                try:
                    obj = lines[line_type]
                except KeyError:
                    continue
                else:
                    artist_object = obj['artist']
                    if artist_object is None:  # pragma: no cover
                        continue
                    artist_object.set_animated(True)
                    x_field = obj.get('x_field', None)
                    y_field = obj.get('y_field', None)
                    id_number = obj.get('data_id', None)
                    result = {'model_id': name, 'order': order_number,
                              'artist': artist_object, 'state': 'new',
                              'fields': [x_field, y_field],
                              'data_id': id_number}
                    line_field.append(result)
                success = True
        return success

    def artists_in_pane(self, pane_: Pane, kind: Optional[str] = None) -> List:
        """
        Find artists in a given pane.

        Parameters
        ----------
        pane_ : Pane
            Pane to query.
        kind : str, optional
            Type of artist to search for, such as line. If not
            provided, return artists of all kind.

        Returns
        -------
        targets : list
            List of artists in the pane
        """
        try:
            arts = self.arts[kind]
        except KeyError:
            arts = list()
            for k, v in self.arts.items():
                arts.extend(v)
        if pane_ is None:
            return arts
        else:
            targets = list()
            for art in arts:
                if art['artist'] in pane_.ax.get_children():
                    targets.append(art)
                elif pane_.show_overplot:
                    if art['artist'] in pane_.ax_alt.get_children():
                        targets.append(art)
            return targets

    def gather_artists(self, mode: Optional[str] = 'all',
                       preserve: Optional[bool] = False) -> List:
        """
        Gather up all artists of a mode.

        Parameters
        ----------
        mode : str, optional
            Mode of artists to grab. If not provided, grab all
            artists.
        preserve : bool, optional
            If set, do not update the state of the artist.
            Otherwise, mark all stats as `fresh`.

        Returns
        -------
        gathered : list
            List of all requested artists.
        """
        gathered = list()
        for kind, artists in self.arts.items():
            for artist in artists:
                if self._artist_fits_mode(artist, mode):
                    gathered.append(artist['artist'])
                if not preserve:
                    artist['state'] = 'fresh'
        return gathered

    def artists_at_event(self, event) -> List:  # pragma: no cover
        """
        Gather the artists located at an event.

        Parameters
        ----------
        event :
            Event in question.

        Returns
        -------
        selected : list
            All artists that occur at the event.
        """
        selected = list()
        artists = self.gather_artists(mode='all', preserve=True)
        for artist in artists:
            if artist.contains(event):
                selected.append(artist)
        return selected

    def print_artists(self) -> None:
        """Print all artists to screen."""
        artists = self.gather_artists(mode='all')
        for artist in artists:
            print(artist)

    @staticmethod
    def _artist_fits_mode(artist: Dict, mode: str) -> bool:
        """
        Check if an artist fits the mode.

        Parameters
        ----------
        artist : dict
            Artist properties dictionary.
        mode : str
            Mode being queries. Options are 'new', 'all', or
            'viable'. 'Viable' does not include stale artists.

        Returns
        -------
        result : bool
            True if the artist and mode are compatible, False otherwise.
        """
        if mode == 'new' and artist['state'] == 'new':
            return True
        elif mode == 'all':
            return True
        elif mode == 'viable' and artist['state'] in ['new', 'fresh']:
            return True
        else:
            return False

    @staticmethod
    def _flatten(mixed: List[Union[int, Tuple]]) -> List:
        """
        Flatten a list of tuples into a single list.

        Parameters
        ----------
        mixed : list
            Ints or tuples to flatten.

        Returns
        -------
        list
            The flattened list.
        """
        f = [(m,) if not isinstance(m, tuple) else m
             for m in mixed]
        return list(sum(f, ()))

    def age_artists(self, mode: Optional[str] = 'all') -> None:
        """
        Mark artists as stale.

        Parameters
        ----------
        mode : str, optional
            The mode to age.
        """
        artists = self.gather_artists(mode=mode, preserve=True)
        for artist in artists:
            artist['state'] = 'stale'

    def update_marker(self, data_point: Dict[str, List[Dict]]) -> None:
        """
        Update markers to new data points.

        Parameters
        ----------
        data_point : dict
            New artist dict.
        """
        cursor_arts = self.arts['cursor'] + self.arts['cursor_alt']
        for model_id, data in data_point.items():
            if not data:
                continue
            for datum in data:
                fields = [datum['x_field'], datum['y_field']]
                for marker in cursor_arts:
                    if (marker['model_id'] == model_id
                            and marker['order'] == datum['order']
                            and marker['fields'] == fields):
                        if not datum['visible']:
                            # just hide if data is out of range
                            marker['artist'].set_visible(False)
                        else:
                            marker['artist'].set_offsets([datum['bin_x'],
                                                          datum['bin_y']])
                            marker['artist'].set_visible(True)
                        marker['state'] = 'new'

    def hide_cursor_markers(self) -> None:
        """Hide all cursor markers."""
        cursor_arts = self.arts['cursor'] + self.arts['cursor_alt']
        for marker in cursor_arts:
            marker['artist'].set_visible(False)
