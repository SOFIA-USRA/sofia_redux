from typing import Dict, Optional

import numpy as np
from matplotlib import artist as mart
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from numpy import ma as mask

from sofia_redux.visualization import log

__all__ = ['Drawing']


class Drawing(object):
    """
    Class to hold an individual matplotlib artist

    Attributes
    ----------
    kind : str
        Available kinds:
            - line
            - border
            - error
            - crosshair
            - guide
            - cursor
            - fit_line
            - fit_center
            - text
            - reference
            - patch
    """

    def __init__(self, **kwargs):

        self._artist = kwargs.get('artist', None)
        self._high_model = str(kwargs.get('high_model', ''))
        self._mid_model = str(kwargs.get('mid_model', ''))
        self._data_id = str(kwargs.get('data_id', ''))
        self._kind = kwargs.get('kind', '')
        self._pane = kwargs.get('pane', None)
        self._axes = kwargs.get('axes', 'primary')
        self._fields = self._parse_fields(kwargs.get('fields'))
        self._label = kwargs.get('label', '')
        self._updates = kwargs.get('updates', dict())
        self._state = 'new'
        self._update = False
        if self.artist:
            self.populate_properties(kwargs)

    def __eq__(self, other):
        if isinstance(other, Drawing):
            checks = [self.match_high_model(other.get_high_model()),
                      self.match_kind(other.get_kind()),
                      self.match_mid_model(other.get_mid_model()),
                      self.match_data_id(other.get_data_id()),
                      self.match_pane(other.get_pane()),
                      self.match_fields(other.get_fields()),
                      self.match_axes(other.get_axes())
                      ]
            return all(checks)
        return False

    @staticmethod
    def _type_error(type_, field):
        raise TypeError(f'Improper type {type_} for {field}.')

    @staticmethod
    def _value_error(value, field):
        raise ValueError(f'Invalid value {value} for {field}.')

    @property
    def artist(self):
        return self._artist

    @artist.setter
    def artist(self, artist):
        if isinstance(artist, mart.Artist):
            self._artist = artist
        else:
            self._type_error(type(artist), 'artist')

    @property
    def high_model(self):
        return self._high_model

    @high_model.setter
    def high_model(self, model):
        self._high_model = str(model)

    @property
    def mid_model(self):
        return self._mid_model

    @mid_model.setter
    def mid_model(self, model):
        self._mid_model = str(model)

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = str(kind)

    @property
    def data_id(self):
        return self._data_id

    @data_id.setter
    def data_id(self, data_id):
        self._data_id = str(data_id)

    @property
    def color(self):
        if self.artist is None:
            return None
        try:
            return self.artist.get_color()
        except AttributeError:  # pragma: no cover
            # doesn't seem reachable with current artists
            return self.artist.get_facecolor()

    @color.setter
    def color(self, color):
        if self.artist is None:
            return
        try:
            self.artist.set_color(color)
        except AttributeError:  # pragma: no cover
            # doesn't seem reachable with current artists
            self.artist.set_facecolor(color)

    @property
    def visible(self):
        if self.artist is None:
            return None
        else:
            return self.artist.get_visible()

    @visible.setter
    def visible(self, visible: bool):
        if self.artist is not None:
            self.artist.set_visible(visible)

    @property
    def marker(self):
        if self.artist is None:
            return None
        else:
            try:
                return self.artist.get_marker()
            except AttributeError:
                return None

    @marker.setter
    def marker(self, marker: str):
        if self.artist is not None:
            try:
                self.artist.set_marker(marker)
            except AttributeError:
                pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: str):
        self._state = str(state)

    @property
    def pane(self):
        return self._pane

    @pane.setter
    def pane(self, pane):
        self._pane = pane

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, axes):
        if isinstance(axes, str):
            axes = str(axes).strip().lower()
            if axes in ['pri', 'p', 'primary']:
                self._axes = 'primary'
            elif axes in ['sec', 's', 'secondary', 'alt', 'alternate']:
                self._axes = 'alt'
            else:
                self._value_error(axes, 'axes')
        else:
            self._type_error(type(axes), 'axes')

    @property
    def fields(self):
        return self._fields

    @fields.setter
    def fields(self, fields):
        new = self._parse_fields(fields)
        self._fields.update(new)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = str(label)

    @property
    def updates(self):
        return self._updates

    @updates.setter
    def updates(self, updates):
        if isinstance(updates, dict):
            self._updates = updates
        else:
            self._type_error(type(updates), 'updates')

    @property
    def update(self):
        return self._update

    @update.setter
    def update(self, update):
        self._update = bool(update)

    def update_options(self, other, kind='default') -> bool:
        if not isinstance(other, Drawing):
            raise RuntimeError('Can only update with another Drawing')
        props = dict()
        updates = other.get_updates()
        if kind in ['default', 'cursor', 'fit', 'ref', 'border',
                    'error_range', 'line', 'patch']:
            props['color'] = updates.get('color', self.color)
        if kind in ['default', 'fit', 'ref', 'line', 'patch', 'border',
                    'error_range']:
            props['visible'] = bool(updates.get('visible', self.visible))
        if kind in ['default', 'line']:
            props['marker'] = updates.get('marker', self.marker)
        if props:
            self._artist.update(props)
            return True
        return False

    def clear_updates(self) -> None:
        self.updates = dict()

    def populate_properties(self, kwargs: Dict):
        props = dict()
        artist_props = ['visible', 'color', 'marker', 'alpha', 'linestyle']
        for key in artist_props:
            try:
                props[key] = kwargs[key]
            except KeyError:
                continue
        self.artist.update(props)

    def _parse_fields(self, fields):
        if fields is None:
            return dict()
        if isinstance(fields, dict):
            return fields
        elif isinstance(fields, list):
            parsed = dict()
            for ax, field in zip(['x', 'y', 'y_alt'], fields):
                parsed[ax] = field
            return parsed
        else:
            self._type_error(type(fields), 'fields')

    def matches(self, other, strict=False):
        checks = [self.match_high_model(other.high_model),
                  self.match_kind(other.kind),
                  self.match_mid_model(other.mid_model),
                  self.match_data_id(other.data_id)]
        if strict:
            checks.extend([self.match_pane(other.pane),
                           self.match_fields(other.fields),
                           self.match_axes(other.axes)
                           ])

        return all(checks)

    def match_high_model(self, name):
        match = str(name).lower() in self.high_model.lower()
        return match

    def match_kind(self, kind):
        if 'fit' in kind and 'fit' in self.kind:
            return True
        elif 'error' in kind and 'error' in self.kind:
            return True
        else:
            return kind == self.kind

    def match_mid_model(self, name):
        match = str(name).lower() in self.mid_model.lower()
        return match

    def match_data_id(self, data_id):
        return str(data_id).lower() == self.data_id.lower()

    def match_pane(self, pane):
        return self.pane == pane

    def match_fields(self, fields):
        checks = list()
        if isinstance(fields, dict):
            for ax, field in fields.items():
                try:
                    checks.append(field == self.fields[ax])
                except KeyError:
                    checks.append(False)
        elif isinstance(fields, list):
            for field, ax in zip(fields, ['x', 'y', 'y_alt']):
                checks.append(field == self.fields[ax])
        return all(checks)

    def match_axes(self, axes):
        return self.axes == axes

    def match_text(self, artist):
        if isinstance(self.artist, type(artist)):
            try:
                return str(self.artist.get_text()) == str(artist.get_text())
            except AttributeError:
                return False
        else:
            return False

    def apply_updates(self, updates):
        self.set_data(updates)

    def set_data(self, data=None, axis: Optional[str] = None,
                 update: Optional = None):
        artist = None
        if update is not None:
            try:
                data = update.updates['x_data']
            except KeyError:
                try:
                    data = update.updates['y_data']
                except KeyError:
                    artist = update.updates['artist']
                else:
                    axis = 'y'
            else:
                axis = 'x'
        if isinstance(self.artist, Line2D):
            self._set_line_data(data=data, axis=axis, artist=artist)
        elif isinstance(self.artist, PathCollection):
            self._set_scatter_data(data=data, axis=axis, artist=artist)
        else:
            log.debug(f'Unable to process artist type '
                      f'{type(self.artist)}')

    def _set_line_data(self, data=None, axis=None, artist=None):
        try:
            if data is not None and axis is not None:
                props = {f'{axis}data': data}
            else:
                props = {'xdata': artist.get_xdata(),
                         'ydata': artist.get_ydata()}
            self.artist.update(props)

        except AttributeError:
            pass

    def _set_scatter_data(self, data=None, axis=None, artist=None):
        current_data = self.get_artist().get_offsets()
        if data is not None and axis is not None:
            if axis == 'all':
                new_data = mask.array(data)
            else:
                if axis == 'x':
                    x_data = data
                    y_data = current_data[:, 1]
                elif axis == 'y':
                    x_data = current_data[:, 0]
                    y_data = data
                else:
                    x_data = current_data[:, 0]
                    y_data = current_data[:, 1]

                new_data = mask.array(np.vstack((x_data, y_data)).T)
        elif artist is not None:
            new_data = artist.get_offsets()
        self.artist.set_offsets(new_data)

    def update_line_fields(self, update):
        pass

    def in_pane(self, pane, alt=False) -> bool:
        if (self.artist in pane.ax.get_children()
                or (alt and self.artist in pane.ax_alt.get_children())):
            return True
        else:
            return False

    def set_artist(self, artist):
        self.artist = artist

    def set_high_model(self, high_model):
        self.high_model = str(high_model)

    def set_mid_model(self, mid_model):
        self.mid_model = str(mid_model)

    def set_data_id(self, data_id):
        self.data_id = str(data_id)

    def set_kind(self, kind):
        self.kind = kind

    def set_pane(self, pane):
        self.pane = pane

    def set_axes(self, axes):
        self.axes = axes

    def set_label(self, label):
        self.label = label

    def set_state(self, state):
        self.state = state

    def set_update(self, update):
        self.update = update

    def set_updates(self, updates):
        self.updates.update(updates)

    def set_fields(self, fields):
        self.fields = fields

    def set_visible(self, visible):
        self.visible = visible

    def set_color(self, color):
        self.color = color

    def set_marker(self, marker):
        self.marker = marker

    def get_artist(self):
        return self.artist

    def get_high_model(self):
        return self.high_model

    def get_mid_model(self):
        return self.mid_model

    def get_data_id(self):
        return self.data_id

    def get_kind(self):
        return self.kind

    def get_pane(self):
        return self.pane

    def get_axes(self):
        return self.axes

    def get_label(self):
        return self.label

    def get_state(self):
        return self.state

    def get_update(self):
        return self.update

    def get_updates(self):
        return self.updates

    def get_fields(self):
        return self.fields

    def get_visible(self):
        return self.visible

    def get_color(self):
        return self.color

    def get_marker(self):
        return self.marker

    def set_animated(self, state):
        self._artist.set_animated(state)

    def get_animated(self):
        if self._artist:
            return self._artist.get_animated()
        else:
            return None

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

    def convert_to_line(self, drawstyle: str, marker: str) -> None:
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
        data = self.artist.get_offsets()
        color = self.artist.get_facecolor()[0]
        ax = self.artist.axes
        label = self.artist.get_label()
        line_artist = ax.plot(data[:, 0], data[:, 1], c=color,
                              label=label, animated=True,
                              marker=marker)[0]
        if drawstyle == 'line':
            line_artist.set_drawstyle('default')
        else:
            line_artist.set_drawstyle('steps-mid')
        self.artist.remove()
        self.artist = line_artist

    def remove(self):
        self._artist.remove()
        self._artist = None
