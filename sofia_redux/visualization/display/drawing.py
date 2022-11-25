import uuid
import re
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
    Class to hold an individual matplotlib artist.
    """

    def __init__(self, **kwargs):
        self._artist = kwargs.get('artist', None)
        self._high_model = str(kwargs.get('high_model', ''))
        self._mid_model = str(kwargs.get('mid_model', ''))
        self._data_id = str(kwargs.get('data_id', ''))  # Wavelength of ref
        self._model_id = kwargs.get('model_id', None)  # UUID
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
        """
        matplotlib.collections.PolyCollection : Matplotlib artist.

        Matplotlib artist appropriate to the data type, such as
        a rectangle for the border or a line for the data plot.
        """
        return self._artist

    @artist.setter
    def artist(self, artist):
        if isinstance(artist, mart.Artist):
            self._artist = artist
        else:
            self._type_error(type(artist), 'artist')

    @property
    def high_model(self):
        """str : High-level model name for the drawing."""
        return self._high_model

    @high_model.setter
    def high_model(self, model):
        self._high_model = str(model)

    @property
    def mid_model(self):
        """str : Mid-level model name for the drawing."""
        return self._mid_model

    @mid_model.setter
    def mid_model(self, model):
        self._mid_model = str(model)

    @property
    def model_id(self) -> uuid.UUID:
        """
        uuid.UUID : Model ID for the drawing.

        Unique id associated with a single input file.
        """
        return self._model_id

    @model_id.setter
    def model_id(self, model_id: uuid.UUID) -> None:
        self._model_id = model_id

    @property
    def kind(self):
        """
        str : Classification category for the drawing.

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
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = str(kind)

    @property
    def data_id(self):
        """
        str : Reference data ID.

        Labels reference data lines, e.g. with wavelength values.
        """
        return self._data_id

    @data_id.setter
    def data_id(self, data_id):
        self._data_id = str(data_id)

    @property
    def color(self):
        """str : Matplotlib color for artist."""
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
        """bool : Visibility state for the artist"""
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
        """str : Marker associated with the artist."""
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
        """
        str : Current state of the drawing.

        Set to 'new' on initialization.
        """
        return self._state

    @state.setter
    def state(self, state: str):
        self._state = str(state)

    @property
    def pane(self):
        """Pane : Display pane containing the drawing."""
        return self._pane

    @pane.setter
    def pane(self, pane):
        self._pane = pane

    @property
    def axes(self):
        """
        str : Axes containing the drawing artist.

        May be 'primary' or 'alternate'.
        """
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
        """
        dict : Plot fields associated with drawing axes.

        Typical values are: 'wavepos', 'spectral_flux', 'spectral_error',
        'transmission', 'response'.
        """
        return self._fields

    @fields.setter
    def fields(self, fields):
        new = self._parse_fields(fields)
        self._fields.update(new)

    @property
    def label(self):
        """str : Label for the drawing."""
        return self._label

    @label.setter
    def label(self, label):
        self._label = str(label)

    @property
    def updates(self):
        """
        dict : New updates to apply to the drawing.

        Keys represent the axis being changed and the values are lists of
        values to update.
        """
        return self._updates

    @updates.setter
    def updates(self, updates):
        if isinstance(updates, dict):
            self._updates = updates
        else:
            self._type_error(type(updates), 'updates')

    @property
    def update(self):
        """bool : Flag to indicate an update is required."""
        return self._update

    @update.setter
    def update(self, update):
        self._update = bool(update)

    def update_options(self, other, kind='default') -> bool:
        """
        Update plot options from another drawing.

        Parameters
        ----------
        other : Drawing
            Drawing to copy updates from.
        kind : str
            Kind of drawing to apply updates to. If default, all available
            updates are applied.

        Returns
        -------
        success : bool
            True if an update is found and applied; False otherwise.
        """
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
        """Clear any existing updates."""
        self.updates = dict()

    def populate_properties(self, kwargs: Dict):
        """
        Set standard properties for the artist

        Parameters
        ----------
        kwargs : dict
           Keys may be: 'visible', 'color', 'marker', 'alpha', 'linestyle'.
           Values must be appropriate for the Matplotlib artist for the
           associated property.
        """
        props = dict()
        artist_props = ['visible', 'color', 'marker', 'alpha', 'linestyle']
        for key in artist_props:
            try:
                props[key] = kwargs[key]
            except KeyError:
                continue
        self.artist.update(props)

    def _parse_fields(self, fields):
        """Parse plot fields by axis from input list or dict."""
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
        """
        Check if this drawing matches another.

        Required attributes for matching are model_id, kind,
        mid_model, and data_id. If `strict` is set, pane,
        fields, and axes must additionally match.

        Parameters
        ----------
        other : Drawing
            The other drawing to compare against.
        strict : bool
            If True, drawings must match exactly.

        Returns
        -------
        match : bool
            True if drawings match; False otherwise.
        """
        # Change to compare UUID instead of high_model/mid_model
        checks = [self.match_id(other.model_id),
                  self.match_kind(other.kind),
                  self.match_mid_model(other.mid_model),
                  self.match_data_id(other.data_id)]
        if strict:
            checks.extend([self.match_pane(other.pane),
                           self.match_fields(other.fields),
                           self.match_axes(other.axes)])
        return all(checks)

    def match_id(self, model_id: uuid.UUID) -> bool:
        """
        Match model IDs.

        Parameters
        ----------
        model_id : uuid.UUID
            Model ID to compare to this drawing's model_id attribute.

        Returns
        -------
        success : bool
            True if model IDs match; False otherwise.
        """
        match = model_id == self._model_id
        return match

    def match_high_model(self, name):
        """
        Match high models.

        Parameters
        ----------
        name : str
            Model name to compare to this drawing's high_model attribute.

        Returns
        -------
        success : bool
            True if model IDs match; False otherwise.
        """
        match = str(name).lower() in self.high_model.lower()
        return match

    def match_kind(self, kind):
        """
        Match drawing kinds.

        Parameters
        ----------
        kind : str
            Kind to compare to this drawing's kind attribute.

        Returns
        -------
        success : bool
            True if kinds match; False otherwise.
        """
        if 'fit' in kind and 'fit' in self.kind:
            return True
        elif 'error' in kind and 'error' in self.kind:
            return True
        else:
            return kind == self.kind

    def match_mid_model(self, name):
        """
        Match mid models.

        Parameters
        ----------
        name : str
            Model name to compare to this drawing's mid_model attribute.

        For multi-order spectra, mid-model is formatted as
        <order>.<aperture>. Both must match.

        Returns
        -------
        success : bool
            True if model IDs match; False otherwise.
        """

        try:
            re.match(r'\d+\.\d+', self.mid_model)[0]
        except TypeError:
            try:
                other = int(name)
                this = int(self.mid_model)
            except ValueError:
                match = str(name).lower() in self.mid_model.lower()
            else:
                match = this == other
        else:
            match = name == self.mid_model
        return match

    def match_data_id(self, data_id):
        """
        Match data IDs.

        Parameters
        ----------
        data_id : str
            Data ID to compare to this drawing's data_id attribute.

        Returns
        -------
        success : bool
            True if data IDs match; False otherwise.
        """
        match = str(data_id).lower() == self.data_id.lower()
        return match

    def match_pane(self, pane):
        """
        Match panes.

        Parameters
        ----------
        pane : Pane
            Pane  to compare to this drawing's pane attribute.

        Returns
        -------
        success : bool
            True if panes match; False otherwise.
        """
        return self.pane == pane

    def match_fields(self, fields):
        """
        Match fields.

        Parameters
        ----------
        fields : list or dict
            Fields to compare to this drawing's fields attribute.

        Returns
        -------
        success : bool
            True if fields match; False otherwise.
        """
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
        """
        Match axes.

        If either value is 'any', True is always returned.

        Parameters
        ----------
        axes : str
            Axes to compare to this drawing's axes attribute.

        Returns
        -------
        success : bool
            True if fields match; False otherwise.
        """
        if axes == 'any' or self.axes == 'any':
            return True
        else:
            return self.axes == axes

    def match_text(self, artist):
        """
        Match text.

        Parameters
        ----------
        artist : matplotlib.Artist
            Artist containing text values, retrievable via get_text().

        Returns
        -------
        success : bool
            True if text values match; False otherwise.
        """
        if isinstance(self.artist, type(artist)):
            try:
                return str(self.artist.get_text()) == str(artist.get_text())
            except AttributeError:
                return False
        else:
            return False

    def apply_updates(self, updates):
        """
        Apply updates to the current drawing.

        Parameters
        ----------
        updates : dict
           Updates to apply.
        """
        self.set_data(updates)

    def set_data(self, data=None, axis: Optional[str] = None,
                 update: Optional = None):
        """
        Set data for the current artist.

        Parameters
        ----------
        data : array-like, optional
            If provided, may be used to directly set the data for the artist.
        axis : {'x', 'y'}, optional
            Specifies the axis to set data for.
        update : dict, optional
            Keys may be 'x_data', 'y_data', 'artist'. If 'x_data' or
            'y_data' are provided, they override the `data` and `axis`
            inputs.
        """
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
            except AttributeError:
                try:
                    data = update['x_data']
                except KeyError:
                    try:
                        data = update['y_data']
                    except KeyError:
                        artist = update['artist']
                    else:
                        axis = 'y'
                else:
                    axis = 'x'
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

    def update_line_fields(self, update):  # pragma: no cover
        """
        Update line fields.

        Currently has no effect.
        """
        # todo - implement or remove placeholder
        pass

    def in_pane(self, pane, alt=False) -> bool:
        """
        Check if current drawing is in specified pane.

        Parameters
        ----------
        pane : Pane
           Pane instance to check.
        alt : bool, optional
           If set, alternate axes are checked as well as the primary.

        Returns
        -------
        bool
            True if artist is in specified pane; False otherwise.
        """
        if (self.artist in pane.ax.get_children()
                or (alt and self.artist in pane.ax_alt.get_children())):
            return True
        else:
            return False

    def set_artist(self, artist):
        """Set the `artist` attribute."""
        self.artist = artist

    def set_high_model(self, high_model):
        """Set the `high_model` attribute."""
        self.high_model = str(high_model)

    def set_mid_model(self, mid_model):
        """Set the `mid_model` attribute."""
        self.mid_model = str(mid_model)

    def set_data_id(self, data_id):
        """Set the `data_id` attribute."""
        self.data_id = str(data_id)

    def set_model_id(self, model_id):
        """Set the `model_id` attribute."""
        self.model_id = str(model_id)

    def set_kind(self, kind):
        """Set the `kind` attribute."""
        self.kind = kind

    def set_pane(self, pane):
        """Set the `pane` attribute."""
        self.pane = pane

    def set_axes(self, axes):
        """Set the `axes` attribute."""
        self.axes = axes

    def set_label(self, label):
        """Set the `label` attribute."""
        self.label = label

    def set_state(self, state):
        """Set the `state` attribute."""
        self.state = state

    def set_update(self, update):
        """Set the `update` attribute."""
        self.update = update

    def set_updates(self, updates):
        """Update the `updates` attribute."""
        self.updates.update(updates)

    def set_fields(self, fields):
        """Set the `fields` attribute."""
        self.fields = fields

    def set_visible(self, visible):
        """Set the `visible` attribute."""
        self.visible = visible

    def set_color(self, color):
        """Set the `color` attribute."""
        self.color = color

    def set_marker(self, marker):
        """Set the `marker` attribute."""
        self.marker = marker

    def get_artist(self):
        """Get the `artist` attribute."""
        return self.artist

    def get_high_model(self):
        """Get the `high_model` attribute."""
        return self.high_model

    def get_mid_model(self):
        """Get the `mid_model` attribute."""
        return self.mid_model

    def get_data_id(self):
        """Get the `data_id` attribute."""
        return self.data_id

    def get_model_id(self):
        """Get the `model_id` attribute."""
        return self.model_id

    def get_kind(self):
        """Get the `kind` attribute."""
        return self.kind

    def get_pane(self):
        """Get the `pane` attribute."""
        return self.pane

    def get_axes(self):
        """Get the `axes` attribute."""
        return self.axes

    def get_label(self):
        """Get the `label` attribute."""
        return self.label

    def get_state(self):
        """Get the `state` attribute."""
        return self.state

    def get_update(self):
        """Get the `update` attribute."""
        return self.update

    def get_updates(self):
        """Get the `updates` attribute."""
        return self.updates

    def get_fields(self):
        """Get the `fields` attribute."""
        return self.fields

    def get_visible(self):
        """Get the `visible` attribute."""
        return self.visible

    def get_color(self):
        """Get the `color` attribute."""
        return self.color

    def get_marker(self):
        """Get the `marker` attribute."""
        return self.marker

    def get_linestyle(self):
        """Get the linestyle associated with the artist."""
        if self.artist:
            return self.artist.get_linestyle()
        else:
            return None

    def set_animated(self, state):
        """Set the animated state for the artist."""
        self._artist.set_animated(state)

    def get_animated(self):
        """Get the animated state for the artist."""
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
        """Remove an artist from the plot and from this drawing."""
        self._artist.remove()
        self._artist = None
