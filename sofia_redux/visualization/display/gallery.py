# Licensed under a 3-clause BSD style license - see LICENSE.rst
import itertools
from typing import (List, Dict, Optional, Tuple,
                    Union, TypeVar)
import numpy as np
import matplotlib.artist as ma
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PatchCollection
from matplotlib.backend_bases import RendererBase

from sofia_redux.visualization import log
from sofia_redux.visualization.display import pane, drawing
from sofia_redux.visualization.display.pane import Pane

__all__ = ['Gallery']

AT = TypeVar('AT', Line2D, PathCollection, PatchCollection, ma.Artist)
PT = TypeVar('PT', pane.Pane, pane.OneDimPane, pane.TwoDimPane)
DT = TypeVar('DT', bound=drawing.Drawing)


class Gallery(object):
    """
    Track display gallery for viewer plots.

    Gallery held by this class are associated with matplotlib
    axes, instantiated and controlled by the
    `sofia_redux.visualization.pane` interface.  This class implements
    updates and modifications to the existing gallery.  It does not
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
                     'fit': list(),
                     'text': list(),
                     'ref_line': list(),
                     'ref_label': list()}

    def __str__(self):
        s = f'Gallery at {hex(id(self))}:\n'
        for key, value in self.arts.items():
            s += f'{key.capitalize()}: {len(value)}\n'
        return s

    def add_patches(self, patches: Dict) -> None:
        """
        Add patch gallery for tracking.

        Parameters
        ----------
        patches : dict
            Patches to add.  Must have keys 'kind', 'artist', 'visible'.
        """
        for pane_label, patch in patches.items():
            kind = patch['kind']
            artist = patch['artist']
            artist.set_visible(patch['visible'])
            name = f'{kind}_{pane_label}'
            draw = drawing.Drawing(artist=artist, kind=kind,
                                   high_model=name)
            self.arts['patch'].append(draw)

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
            if patch.match_high_model(f'border_pane_{pane_number}'):
                patch.set_visible(state)
            elif patch.match_high_model('border'):
                patch.set_visible(False)

    def add_crosshairs(self, crosshairs: List[DT]) -> None:
        """
        Add crosshair gallery for tracking.

        Parameters
        ----------
        crosshairs : dict
            Must contain keys 'kind', 'artist', 'visible',
            'direction'.
        """
        for crosshair in crosshairs:
            if crosshair.get_kind() == 'crosshair':
                if crosshair not in self.arts['crosshair']:
                    self.arts['crosshair'].append(crosshair)
                else:
                    log.debug('Crosshair already present')

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
            if crosshair.match_high_model(model_name):
                d = crosshair.get_mid_model()[0]
                if d in direction:
                    if data_point is not None:
                        if d == 'v':
                            crosshair.set_data(data_point[0], 'x')
                        else:
                            crosshair.set_data(data_point[1], 'y')
                    crosshair.set_visible(True)
                else:
                    crosshair.set_visible(False)
            else:
                # this is hit when zooming in another pane
                crosshair.set_visible(False)

    def hide_crosshair(self) -> None:
        """Hide all crosshair gallery."""
        for crosshair in self.arts['crosshair']:
            crosshair.set_visible(False)

    def update_line_data(self, pane_: Pane, updates: List[DT],
                         axes: Optional[str] = 'primary') -> None:
        """
        Update data associated with line gallery in a single pane.

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the line gallery to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'field'. Keys are from 'new_x_data',
            'new_y_data', and 'new_y_alt_data'.
        axes: 'primary', 'alt', 'both', 'all'

        """
        pri_lines = self.artists_in_pane(pane_, 'line')
        alt_lines = self.artists_in_pane(pane_, 'line_alt')
        for draw in updates:
            lines = list()
            update = draw.get_updates()
            if (axes in ['both', 'all', 'alt']
                    or 'new_x_data' in update.keys()
                    or 'new_y_alt_data' in update.keys()):
                lines.extend(alt_lines)
            if (axes in ['both', 'all', 'primary']
                    or 'new_x_data' in update.keys()
                    or 'new_y_data' in update.keys()):
                lines.extend(pri_lines)
            for line in lines:
                if line.matches(draw):
                    line.set_data(update=draw)

    def update_line_fields(self, pane_: PT, updates: List[DT]) -> None:
        """
        Update data and labels for a new plot field in a single pane.

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the line gallery to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'new_field' keys, as well as either
            'new_x_data' or 'new_y_data' (but not both).
        """
        lines = self.artists_in_pane(pane_=pane_, kinds='line')
        for line in lines:
            for update in updates:
                if line.matches(update):
                    line.update_line_fields(update)

                    current_label = line.get_artist().get_label()
                    old_field = current_label.split()[-1]
                    new_field = update.get_updates()['new_field']
                    new_label = current_label.replace(old_field, new_field)
                    line.get_artist().set_label(new_label)
                    try:
                        line.get_artist().set_xdata(
                            update.get_updates()['new_x_data'])
                    except KeyError:
                        line.get_artist().set_ydata(
                            update.get_updates()['new_y_data'])
                    line.set_state('new')

    def update_line_type(self, pane_: PT, updates: List[DT]) -> None:
        """
        Update plot line type.

        Line2D gallery are updated in place; PathCollection gallery
        are replaced with an equivalent new artist.

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the line gallery to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'type' keys.  The 'type' should
            be one of 'step', 'line', or 'scatter'.  The dict may
            also contain a 'marker' key.
        """
        lines = self.artists_in_pane(pane_=pane_, kinds='line')
        for line in lines:
            for update in updates:
                if line.matches(update) and line.match_axes(update.axes):
                    details = update.get_updates()
                    marker = details.get('marker')
                    if isinstance(line.get_artist(), Line2D):
                        props = {'drawstyle': 'default', 'linestyle': '-',
                                 'marker': marker}
                        if details['type'] == 'step':
                            props['drawstyle'] = 'steps-mid'
                        elif details['type'] == 'line':
                            pass
                        elif details['type'] == 'scatter':
                            props['linestyle'] = ''
                        line.get_artist().update(props)

                    elif isinstance(line.get_artist(), PathCollection):
                        line.convert_to_line(drawstyle=details['type'],
                                             marker=marker)

    def update_artist_options(self, pane_: Union[PT, int],
                              kinds: Optional[Union[List[str], str]] = None,
                              options: Optional[List[DT]] = None
                              ) -> bool:
        """
        Update artist display options.

        Currently supported options for each artist type are:
           - 'line' gallery: 'color', 'visibility', 'marker'
           - 'error_range' gallery: 'color', 'visibility'
           - 'fit' gallery: 'color', 'visibility'
           - 'patch' gallery: 'color'
           - 'cursor' gallery: 'color'

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the gallery to update.
        kinds : str, list
            Kinds of gallery to update. If all gallery are to be
            updated, set to 'all'.
        options : list of dict
            Options to apply.  The dicts must contain 'model_id' and
            'order' keys.  Other keys allowed are 'color',
            'visibility', and 'marker'.
        """
        if kinds is None:
            if options is not None:
                if not isinstance(options, list):
                    options = [options]
                kinds = [o.get_kind() for o in options]
                if all([not bool(k) for k in kinds]):
                    kinds = ['all']
            else:
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
            if kind in ['fit', 'fit_line', 'fit_center', 'all']:
                out = self._update_fit_artist_options(pane_, options)
                results.append(out)
            if kind in ['ref_line', 'ref_label', 'all']:
                out = self._update_reference_artist_options(pane_, options)
                results.append(out)
        return any(results)

    def _update_line_artist_options(self, pane_: PT,
                                    options: List[DT]) -> bool:
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
        pri_lines = self.artists_in_pane(pane_=pane_, kinds='line')
        alt_lines = self.artists_in_pane(pane_=pane_, kinds='line_alt')
        results = list()
        for line in pri_lines + alt_lines:
            for option in options:
                if option.get_data_id():  # pragma: no cover
                    continue
                if line.matches(option) and line.match_axes(option.axes):
                    results.append(line.update_options(option, kind='line'))
                    break
        return any(results)

    def _update_error_artist_options(self, pane_: PT,
                                     options: List[DT]) -> bool:
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
        lines = self.artists_in_pane(pane_=pane_, kinds='error_range')
        results = list()
        for line in lines:
            for option in options:
                if option.get_data_id():  # pragma: no cover
                    continue
                if line.matches(option):
                    results.append(line.update_options(option,
                                                       kind='error_range'))
                    break
        return any(results)

    def _update_border_artist_options(self, pane_: PT,
                                      options: List[DT]) -> bool:
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
        patches = self.artists_in_pane(pane_=pane_, kinds='patch')
        results = list()
        for patch in patches:
            for option in options:
                if option.get_data_id():  # pragma: no cover
                    continue
                if option.match_high_model('border'):
                    results.append(patch.update_options(option,
                                                        kind='patch'))
                    break
        return any(results)

    def _update_cursor_artist_options(self, pane_: PT,
                                      options: List[DT]) -> bool:
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
        points = self.artists_in_pane(pane_=pane_, kinds='cursor')
        points_alt = self.artists_in_pane(pane_=pane_, kinds='cursor_alt')
        results = list()
        for point in points + points_alt:
            for option in options:
                if option.get_data_id():  # pragma: no cover
                    continue
                if point.matches(option):
                    results.append(point.update_options(option, kind='cursor'))
                    break
        return any(results)

    def _update_fit_artist_options(self, pane_: PT,
                                   options: List[DT]) -> bool:
        """
        Update options for gallery of curve fits.

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
        lines = self.artists_in_pane(pane_=pane_, kinds='fit')
        results = list()
        for line in lines:
            for option in options:
                if (not option.get_data_id()
                        or not line.get_data_id()):  # pragma: no cover
                    continue
                if line.matches(option):
                    results.append(line.update_options(option, kind='fit'))
                    break
        return any(results)

    def _update_reference_artist_options(self, pane_: PT,
                                         options: List[DT]) -> bool:
        lines = self.artists_in_pane(pane_=pane_, kinds='ref_lines')
        labels = self.artists_in_pane(pane_=pane_, kinds='ref_labels')
        results = list()
        for draw in lines + labels:
            for option in options:
                if option.get_data_id():  # pragma: no cover
                    continue
                if option.matches(draw):
                    results.append(draw.update_options(option, kind='ref'))
                    break
        return any(results)

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

    def update_error_ranges(self, pane_: Pane, updates: List[DT]) -> None:
        """
        Update data associated with error range gallery.

        Typically called with the `update_line_data` method, which
        updates data associated with a line plot.  However,
        error range gallery are always replaced with a new artist,
        rather than updated in place.

        Parameters
        ----------
        pane_ : Pane
            Pane object holding the line gallery to update.
        updates : list of dict
            Updates to apply.  The dicts must contain 'model_id',
            'order', and 'new_artist' keys.
        """
        lines = self.artists_in_pane(pane_=pane_, kinds='error_range')
        for line in lines:
            for draw in updates:
                if line.matches(draw):
                    label = line.get_artist().get_label()
                    new_artist = draw.get_updates()['artist']
                    new_artist.set_label(label)
                    line.set_artist(new_artist)

    def update_reference_data(self, pane_: Pane, updates: List[DT]) -> None:
        """

        Parameters
        ----------
        pane_
        updates

        Returns
        -------

        """
        # Remove original ref lines as the number of lines might
        # have changed and the labels almost definitely have
        self._clear_artists(kind=['ref_line', 'ref_label'],
                            panes=[pane_])
        self.add_drawings(updates)

    def catch_label_overlaps(self, renderer: RendererBase) -> List:
        to_remove = set()
        updated = list()
        labels = self.arts['ref_label']
        labels.sort(key=lambda x: float(x.data_id))
        label_idx = list(range(len(labels)))
        for id1, id2 in itertools.combinations(label_idx, 2):
            if id1 in to_remove:
                continue
            label1 = labels[id1]
            label2 = labels[id2]

            if not label1.match_pane(label2.pane):
                continue

            artist1 = label1.artist
            artist2 = label2.artist

            if not (artist1.get_visible()
                    and artist2.get_visible()):  # pragma: no cover
                continue

            b1 = artist1.get_tightbbox(renderer)
            b2 = artist2.get_tightbbox(renderer)
            overlap = b1.overlaps(b2)

            if overlap:
                text1 = artist1.get_text()
                if len(text1) < 30:
                    artist1.set_text(f'{artist1.get_text()},'
                                     f'{artist2.get_text()}')
                elif not text1.endswith('...'):
                    # truncate over-long labels
                    artist1.set_text(f'{artist1.get_text()},'
                                     f'...')

                to_remove.add(id2)
                updated.append(label1)
            else:  # pragma: no cover
                pass

        for index in sorted(to_remove, reverse=True):
            draw = self.arts['ref_label'].pop(index)
            draw.remove()
        return updated

    def reset_artists(self, selection: str,
                      panes: Optional[List] = None) -> None:
        """
        Reset and remove all gallery for a given selection.

        Parameters
        ----------
        selection : str
            Type of gallery to reset. Acceptable values are 'lines'
            which resets the data lines, 'cursor' which resets the
            marker for the cursor location, 'collections' which
            resets scatter plots, 'v_guides', 'h_guides', 'f_guides',
            'a_guides' which resets vertical, horizontal, fit,
            and all guides, respectively. The 'all' flag clears everything.
        panes : list, optional
            The pane to clear the selected gallery from. If not provided,
            the selected gallery will be cleared from all panes.
        """
        log.debug(f'Resetting {selection} gallery')
        if selection == 'all':
            for kind, arts in self.arts.items():
                self._clear_artists(kind=kind, panes=panes)
        elif selection == 'alt':
            for kind in ['line_alt', 'cursor_alt']:
                self._clear_artists(kind=kind, panes=panes)
        elif selection in ['line', 'line_alt', 'cursor', 'cursor_alt',
                           'collections', 'crosshair', 'patch', 'fit']:
            self._clear_artists(kind=selection, panes=panes)
        elif selection in ['ref_line', 'ref_label', 'reference']:
            self._clear_artists(kind=['ref_line', 'ref_label'], panes=panes)
        elif 'guide' in selection:
            self._clear_guides(flag=selection[0], panes=panes)
        else:
            log.debug(f'Invalid artist selection {selection}. '
                      f'No reset performed')

    def _clear_guides(self, flag: str, panes: List[PT]) -> None:
        """
        Remove guide gallery.

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
                to_clear.extend(self.artists_in_pane(pane_, kinds='guide'))
        for draw in to_clear:
            if draw.get_mid_model().startswith(flag) or flag == 'a':
                draw.get_artist().remove()
            else:
                others.append(draw)
        self.arts['guide'] = others

    def _clear_artists(self, kind: Union[List[str], str],
                       panes: List) -> None:
        """
        Clear gallery.

        Parameters
        ----------
        kind : str
            Denotes the kind of artist to clear.
        panes : list
            List of panes to clear gallery from.
        """
        if panes is None:
            to_clear = self.artists_in_pane(panes, kinds=kind)
        else:
            to_clear = list()
            for pane_ in panes:
                to_clear.extend(self.artists_in_pane(pane_, kinds=kind))
        if not isinstance(kind, list):
            kind = [kind]
        for k in kind:
            new_draws = list()
            for draw in self.arts[k]:
                if draw not in to_clear:
                    new_draws.append(draw)
                else:
                    try:
                        draw.get_artist().remove()
                    except ValueError:
                        continue
            self.arts[k] = new_draws

    def add_drawings(self, drawings: List[DT]) -> int:
        results = list()
        for d in drawings:
            results.append(self.add_drawing(d))
        return sum(results)

    def add_drawing(self, drawing_: DT) -> bool:
        if not isinstance(drawing_, drawing.Drawing):
            raise TypeError(f'Drawing {drawing_} is not of a valid type')
        success = False
        artist = drawing_.get_artist()
        if artist is not None:
            drawing_.set_animated(True)
            kind = drawing_.get_kind()
            if 'fit' in kind:
                kind = 'fit'
            try:
                self.arts[kind].append(drawing_)
            except KeyError:
                self.arts[kind] = [drawing_]
            success = True
        return success

    def artists_in_pane(self, pane_: Pane,
                        kinds: Optional[Union[List[str], str]] = None
                        ) -> List[DT]:
        """
        Find gallery in a given pane.

        Parameters
        ----------
        pane_ : Pane
            Pane to query.
        kinds : str, optional
            Type of artist to search for, such as line. If not
            provided, return gallery of all kind.

        Returns
        -------
        targets : list
            List of gallery in the pane
        """
        targets = list()
        # gallery = list()
        if not isinstance(kinds, list):
            kinds = [kinds]
        for kind in kinds:
            try:
                arts = self.arts[kind]
            except KeyError:
                arts = list()
                for k, v in self.arts.items():
                    arts.extend(v)
            if pane_ is None:
                targets.extend(arts)
            else:
                for art in arts:
                    if art.in_pane(pane_, alt=pane_.show_overplot):
                        targets.append(art)

        return targets

    def gather_artists(self, mode: Optional[str] = 'all',
                       preserve: Optional[bool] = False,
                       return_drawing: Optional[bool] = False) -> List[DT]:
        """
        Gather up all gallery of a mode.

        Parameters
        ----------
        mode : str, optional
            Mode of gallery to grab. If not provided, grab all
            gallery.
        preserve : bool, optional
            If set, do not update the state of the artist.
            Otherwise, mark all stats as `fresh`.

        Returns
        -------
        gathered : list
            List of all requested gallery.
        """
        gathered = list()
        for kind, artists in self.arts.items():
            for drawing_ in artists:
                if self._drawing_fits_mode(drawing_, mode):
                    if return_drawing:
                        gathered.append(drawing_)
                    else:
                        gathered.append(drawing_.get_artist())
                if not preserve:
                    drawing_.set_state('fresh')
        return gathered

    def artists_at_event(self, event) -> List:  # pragma: no cover
        """
        Gather the gallery located at an event.

        Parameters
        ----------
        event :
            Event in question.

        Returns
        -------
        selected : list
            All gallery that occur at the event.
        """
        selected = list()
        artists = self.gather_artists(mode='all', preserve=True)
        for artist in artists:
            if artist.contains(event):
                selected.append(artist)
        return selected

    def print_artists(self) -> None:
        """Print all gallery to screen."""
        artists = self.gather_artists(mode='all')
        for artist in artists:
            print(artist)

    @staticmethod
    def _drawing_fits_mode(drawing_: DT, mode: str) -> bool:
        """
        Check if an artist fits the mode.

        Parameters
        ----------
        artist : dict
            Artist properties dictionary.
        mode : str
            Mode being queries. Options are 'new', 'all', or
            'viable'. 'Viable' does not include stale gallery.

        Returns
        -------
        result : bool
            True if the artist and mode are compatible, False otherwise.
        """
        if mode == 'new' and drawing_.get_state() == 'new':
            return True
        elif mode == 'all':
            return True
        elif mode == 'viable' and drawing_.get_state() in ['new', 'fresh']:
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
        Mark gallery as stale.

        Parameters
        ----------
        mode : str, optional
            The mode to age.
        """
        artists = self.gather_artists(mode=mode, preserve=True,
                                      return_drawing=True)
        for drawing_ in artists:
            drawing_.set_state('stale')

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
                if datum.get('alt', False):
                    axes = 'alt'
                    fields = {'x': datum['x_field'],
                              'y_alt': datum['y_field']}
                else:
                    axes = 'primary'
                    fields = {'x': datum['x_field'],
                              'y': datum['y_field']}

                for marker in cursor_arts:
                    checks = [marker.match_high_model(model_id),
                              marker.match_mid_model(datum['order']),
                              marker.match_fields(fields),
                              marker.match_axes(axes)]
                    if all(checks):
                        if not datum['visible']:
                            # just hide if data is out of range
                            marker.set_visible(False)
                        else:
                            marker.set_data(axis='all', data=[datum['bin_x'],
                                                              datum['bin_y']])
                            marker.set_visible(True)
                        marker.set_state('new')

    def hide_cursor_markers(self) -> None:
        """Hide all cursor markers."""
        cursor_arts = self.arts['cursor'] + self.arts['cursor_alt']
        for marker in cursor_arts:
            marker.visible = False
