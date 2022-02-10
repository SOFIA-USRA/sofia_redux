# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units
import numpy as np

from sofia_redux.scan.flags.frame_flags import FrameFlags
from sofia_redux.scan.flags.flagged_data import FlaggedData
from sofia_redux.scan.utilities.class_provider import \
    frames_instance_for
from sofia_redux.scan.frames import frames_numba_functions
from sofia_redux.scan.coordinate_systems.coordinate import Coordinate
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.flags.mounts import Mount
from sofia_redux.scan.coordinate_systems.index_2d import Index2D

__all__ = ['Frames']


class Frames(FlaggedData):

    flagspace = FrameFlags

    def __init__(self):
        """
        The Frames class contains all time-dependent data in an integration.
        """
        super().__init__()
        self.mjd = None
        self.lst = None
        self.sin_a = None
        self.cos_a = None
        self.dof = None
        self.dependents = None
        self.relative_weight = None
        self.sign = None
        self.transmission = None
        self.temp_c = None
        self.temp_wc = None
        self.temp_wc2 = None
        self.has_telescope_info = None
        self.valid = None
        self.validated = None

        # Special 2d data
        self.data = None
        self.sample_flag = None
        self.source_index = None
        self.map_index = None
        self.sample_equatorial = None

        # Vectors
        self.equatorial = None
        self.chopper_position = None

        # Special reference
        self.integration = None
        self.equatorial_system = None

        # channel indices
        self.frame_fixed_channels = None

    @property
    def referenced_attributes(self):
        """Referenced attributes are those that are referenced during copy."""
        refs = super().referenced_attributes
        refs.add('integration')
        return refs

    @property
    def readout_attributes(self):
        """Attributes that will be operated on by the `shift` method."""
        return {'data'}

    @property
    def default_field_types(self):
        """
        Used to define the default values for data arrays.

        Returns a dictionary of structure {field: default_value}.  The default
        values have the following effects:

        type - empty numpy array of the given type.
        value - full numpy array of the given value.
        astropy.units.Unit - empty numpy array (float) in the given unit.
        astropy.units.Quantity - full numpy array of the given quantity.

        If a tuple is provided, the array will have additional axes appended
        such that the first element gives the type as above, and any additional
        integers give additional axes dimensions,  e.g. (0.0, 2, 3) would
        result in a numpy array filled with zeros of shape (self.size, 2, 3).

        Returns
        -------
        fields : dict
        """
        fields = super().default_field_types
        fields.update({
            'mjd': np.nan,
            'lst': np.nan * units.Unit('hourangle'),
            'sin_a': np.nan,
            'cos_a': np.nan,
            'dof': 1.0,
            'dependents': 0.0,
            'relative_weight': 1.0,
            'sign': 1,
            'transmission': 1.0,
            'temp_c': float,
            'temp_wc': float,
            'temp_wc2': float,
            'has_telescope_info': True,
            'valid': True,
            'validated': False,
            'equatorial': (EquatorialCoordinates, 'degree'),
            'chopper_position': (Coordinate2D, 'arcsec'),
            'frame_fixed_channels': -1,
        })
        return fields

    @property
    def default_channel_fields(self):
        """
        Returns default frame/channel type default values.

        This framework is similar to `default_field_types`, but is used to
        populate frame/channel data of shape (n_frames, n_channels).

        Returns
        -------
        fields : dict
        """
        return {'data': float,
                'sample_flag': 0,
                'source_index': -1,
                'map_index': (Index2D, -1),
                'sample_equatorial': units.Unit('deg')}

    @property
    def internal_attributes(self):
        """
        Returns attribute names that are internal to the data for get actions.

        These attributes should always be returned as-is regardless
        of indexing.

        Returns
        -------
        set (str)
        """
        attributes = super().internal_attributes
        attributes.add('frame_fixed_channels')
        return attributes

    @property
    def channel_size(self):
        """
        The number of channels in the frame data.

        Returns
        -------
        int
        """
        if self.data is None:
            return 0
        return self.data.shape[1]

    @property
    def scan(self):
        if self.integration is None:
            return None
        else:
            return self.integration.scan

    @property
    def info(self):
        """
        Return the scan info object.

        Returns
        -------
        Info
        """
        if self.scan is None:
            return None
        return self.scan.info

    @property
    def astrometry(self):
        """
        Returns the astrometry information from the parent scan.

        Returns
        -------
        AstrometryInfo
        """
        return self.info.astrometry

    @property
    def scan_equatorial(self):
        """
        Return the scan equatorial coordinates.

        Returns
        -------
        equatorial : EquatorialCoordinates
            The equatorial coordinates (single RA/DEC) of the scan.
        """
        return self.astrometry.equatorial

    @property
    def channels(self):
        if self.integration is None:
            return None
        else:
            return self.integration.channels

    @property
    def configuration(self):
        if self.info is None:
            return None
        else:
            return self.info.configuration

    @property
    def channel_fixed_index(self):
        if self.channels is None:
            return None
        else:
            return self.channels.data.fixed_index

    @property
    def special_fields(self):
        """
        Return fields that do not comply with the shape of other data.

        This is of particular importance for `delete_indices`.  Although all
        arrays must have shape[0] = self.size, special handling may be required
        in certain cases.

        Returns
        -------
        fields : set (str)
        """
        fields = super().special_fields
        fields.add('integration')
        return fields

    @classmethod
    def instance_from_instrument_name(cls, name):
        """
        Return a Frames instance for a given instrument.

        Parameters
        ----------
        name : str
            The name of the instrument.

        Returns
        -------
        Frames
        """
        return frames_instance_for(name)

    def insert_blanks(self, insert_indices):
        """
        Inserts blank frame data.

        Actual indices should be passed in.  To delete based on fixed index
        values, please convert first using `find_fixed_indices`.

        Blank data are set to 0 in whatever unit is applicable.

        Parameters
        ----------
        insert_indices : numpy.ndarray of (int)
            The index locations to insert.

        Returns
        -------
        None
        """
        super().insert_blanks(insert_indices)
        new_invalid = insert_indices + np.arange(insert_indices.size)
        self.valid[new_invalid] = False

    def initialize(self, integration, size):
        """
        Initializes a frame object with default values.

        Parameters
        ----------
        integration : Integration
        size : int
            The total number of frames.

        Returns
        -------
        None
        """
        self.set_frame_size(size)
        self.integration = integration
        self.set_channels(self.channels)  # self.channels are from integration
        self.equatorial.set_epoch(self.integration.info.telescope.epoch)

    def set_default_channel_values(self, channel_size):
        """
        Populate channel data fields with default values.

        The default values are loaded from the `default_channel_fields`
        property which returns a dictionary of the form {field_name: value}.
        If the value is a type, the default values will be empty numpy arrays.
        Other valid values can be astropy quantities or standard python types
        such as int, float, str, etc.  All fields will be set to numpy arrays
        of the same type and filled with the same value.

        All arrays will be of the shape (self.size, channel_size).

        Parameters
        ----------
        channel_size : int
            The number of channels

        Returns
        -------
        None
        """
        shape = self.size, channel_size

        for key, value in self.default_channel_fields.items():
            if isinstance(value, type):
                setattr(self, key, np.empty(shape, dtype=value))
            elif isinstance(value, units.Quantity):
                setattr(self, key, np.full(shape, value.value) * value.unit)
            elif isinstance(value, units.Unit):
                setattr(self, key, np.empty(shape, dtype=float) * value)

            elif isinstance(value, tuple):
                if issubclass(value[0], Coordinate2D):
                    coordinate_class, value = value[0], value[1]
                    coordinate_shape = (2,) + shape
                    if isinstance(value, units.Quantity):
                        unit = value.unit
                        fill_values = np.full(
                            coordinate_shape, value.value) * unit
                    elif isinstance(value, units.Unit):
                        unit = value
                        fill_values = np.empty(
                            coordinate_shape, dtype=float * unit)
                    elif isinstance(value, str):
                        unit = units.Unit(value)
                        fill_values = np.empty(
                            coordinate_shape, dtype=float) * unit
                    else:
                        fill_values = np.full(coordinate_shape, value)
                    setattr(
                        self, key, coordinate_class(fill_values, copy=False))

                elif isinstance(value, units.Quantity):
                    setattr(self, key,
                            np.full(shape, value.value) * value.unit)

                elif isinstance(value, units.Unit):
                    setattr(self, key, np.empty(shape, dtype=float) * value)

                elif isinstance(value, type):
                    setattr(self, key, np.empty(shape, dtype=value))

                else:
                    setattr(self, key, np.full(shape, value))

            else:
                setattr(self, key, np.full(shape, value))

    def set_channels(self, channels=None):
        """
        Predominantly updates fixed channel indices.

        Will set default values if the channel type data has not already been
        defined.  Otherwise, the channels provided must be a subset of the
        previously defined channels.  In this case, channel type data is
        slimmed down to those that are still found in the new channels.

        Parameters
        ----------
        channels : Channels, optional
            The channels from which to base channel type data.  Defaults to
            integration channels.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.channels

        if channels is None or self.channels is None:
            raise ValueError("No channels supplied or available.")

        if self.data is None:
            self.set_default_channel_values(channels.size)
            self.frame_fixed_channels = self.channel_fixed_index.copy()
            return

        self.validate_channel_indices()

    def find_channel_fixed_indices(self, fixed_indices, cull=True):
        """
        Returns the actual indices given channel fixed indices.

        The fixed indices are those that are initially loaded.  Returned
        indices are their locations in the data arrays.

        This operates on the channel fixed indices, important for some fields
        of the frame data (data, sample_flags, etc.).

        Parameters
        ----------
        fixed_indices : int or np.ndarray (int)
            The fixed indices of the channels.
        cull : bool, optional
            If `True`, do not include fixed indices not found in the result.
            If `False`, missing indices will be replaced by -1.

        Returns
        -------
        indices : ndarray of int
            The indices of `fixed_indices` in the data arrays.
        """
        if self.channels is None:
            return np.empty(0, dtype=int)

        return self.channels.data.find_fixed_indices(fixed_indices, cull=cull)

    def set_frame_size(self, n_frames):
        """
        Set the number of frames in an integration and default values.

        Parameters
        ----------
        n_frames : int

        Returns
        -------
        None
        """
        self.fixed_index = np.arange(n_frames)
        self.set_default_values()

    def get_frame_count(self, keep_flag=None, discard_flag=None,
                        match_flag=None):
        """
        Return the number of frames in an integration.

        A number of flags may also be supplied to return the number of a
        certain type of frame.

        Parameters
        ----------
        keep_flag : int or ChannelFlagTypes, optional
            Flag values to keep in the calculation.
        discard_flag : int or ChannelFlagTypes, optional
            Flag values to discard_flag from the calculation.
        match_flag : int or ChannelFlagTypes, optional
            Only matching flag values will be used in the calculation.

        Returns
        -------
        n_frames : int
            The number of matching frames in the integration.
        """
        if self.size == 0:
            return 0
        elif keep_flag is None and discard_flag is None and match_flag is None:
            return self.valid.sum()
        else:
            valid = self.get_flagged_indices(keep_flag=keep_flag,
                                             discard_flag=discard_flag,
                                             match_flag=match_flag,
                                             indices=False)
            valid &= self.valid
            return valid.sum()

    def set_transmission(self, transmission, indices=None):
        """
        Set the frame transmission.

        Parameters
        ----------
        transmission : float or numpy.ndarray (float)
            The transmission values to set.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)
        if self.is_singular:
            self.transmission = transmission
        else:
            self.transmission[indices] = transmission

    def slim(self, channels=None):
        """
        Slim channel type data to those channels still present.

        Parameters
        ----------
        channels : Channels, optional
            The channels object.  If not supplied, defaults to the integration
            channels.

        Returns
        -------
        None
        """
        if channels is None:
            self.validate_channel_indices()
        else:
            self.set_channels(channels)
        self.source_index = None
        self.map_index = None

    def validate_channel_indices(self):
        """
        Check that the frame channel indices are consistent.

        Returns
        -------
        None
        """
        for indices in [self.channel_fixed_index, self.frame_fixed_channels]:
            if not isinstance(indices, np.ndarray):
                raise ValueError("Channel fixed indices for frames not set.")

        if self.channel_fixed_index.shape == self.frame_fixed_channels.shape:
            if np.allclose(self.channel_fixed_index,
                           self.frame_fixed_channels):
                return

        mask = self.channel_fixed_index[:, None] == (
            self.frame_fixed_channels)
        present = np.any(mask, axis=0)
        indices = np.nonzero(present)[0]

        for channel_attribute in self.default_channel_fields.keys():
            value = getattr(self, channel_attribute, None)
            if value is None:
                continue
            setattr(self, channel_attribute, value[:, indices])

        self.frame_fixed_channels = self.channel_fixed_index.copy()

    def jackknife(self):
        """
        Randomly set signs for roughly half of the frames.

        Returns
        -------
        None
        """
        random = np.random.random(self.size)
        self.sign[random < 0.5] *= -1

    def get_source_gain(self, mode_flag):
        """
        Return the source gain.

        The basic frame class will only return a result for the TOTAL_POWER
        flag.  All others will raise an error.  The source gain here is
        defined as gain = transmission * sign.

        Parameters
        ----------
        mode_flag : FrameFlagTypes or str or int
            The gain mode flag type.

        Returns
        -------
        gain : numpy.ndarray (float)
            The source gains.
        """
        mode_flag = self.flagspace.convert_flag(mode_flag)
        if mode_flag != self.flagspace.flags.TOTAL_POWER:
            raise ValueError(f"{self.__class__} does not define "
                             f"{mode_flag} signal mode.")
        return self.sign * self.transmission

    def validate(self):
        """
        Validate frame data after read.

        Should set the `validated` (checked) attribute if necessary.

        Returns
        -------
        None
        """
        native_coordinates = self.get_absolute_native_coordinates()
        frames_numba_functions.validate_frames(
            valid=self.valid,
            cos_a=self.cos_a,
            sin_a=self.sin_a,
            native_sin_lat=native_coordinates.sin_lat,
            native_cos_lat=native_coordinates.cos_lat,
            validated=self.validated,
            has_telescope_info=self.has_telescope_info,
            mount=self.info.instrument.mount.value,
            left_nasmyth=Mount.LEFT_NASMYTH.value,
            right_nasmyth=Mount.RIGHT_NASMYTH.value)

    def get_equatorial(self, offsets, indices=None, equatorial=None):
        """
        Return equatorial coordinates given offsets from the base equatorial.

        The return result (lon, lat) is:

        lon = base_lon + (position.lon / cos(scan.lat))
        lat = base_lat + position.lat

        Parameters
        ----------
        offsets : Coordinate2D
            The (x, y) equatorial offsets of shape () or (shape,)
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.
        equatorial : EquatorialCoordinates, optional
            The equatorial output frame.  The default is the same as the frame
            equatorial frame.

        Returns
        -------
        equatorial : EquatorialCoordinates
        """
        indices, size = self.get_index_size(indices)
        if equatorial is None:
            equatorial = self.equatorial.empty_copy()
            if equatorial.epoch.singular:
                equatorial.epoch = self.equatorial.epoch.copy()
            else:
                equatorial.epoch = self.equatorial.epoch[indices].copy()

        native_offsets = self.get_native_xy(offsets, indices=indices)
        x = (self.equatorial.x[indices]
             / self.info.astrometry.equatorial.cos_lat)
        y = self.equatorial.y[indices]

        shaped = len(native_offsets.shape) > 1
        if shaped:
            x, y = x[..., None], y[..., None]

        equatorial.set_native_longitude(x + native_offsets.x)
        equatorial.set_native_latitude(y + native_offsets.y)

        return equatorial

    def get_equatorial_native_offset(self, position, indices=None,
                                     offset=None):
        """
        Return the horizontal offsets of a position relative to scan center.

        The final return position is the sum of the position offsets, and the
        equatorial coordinates of the frame relative to the scan center.

        Parameters
        ----------
        position : Coordinate2D
            The (x, y) horizontal offsets of shape () or (m,) giving the
            (lon, lat) offset positions.  If not set, the default is to return
            the offsets relative to the scan equatorial position.
        indices : int or numpy.ndarray (int), optional
            The frame indices that apply of shape (n,) (if an array was used).
            The default is all indices.
        offset : Coordinate2D, optional
            An optional output array to store and return the coordinates.

        Returns
        -------
        native_offsets : Coordinate2D
            An array containing the sum of the equatorial offsets of the frame
            data and the supplied positions.  If multiple frame indices and
            multiple positions are supplied, the resulting coordinate shape
            will be (n, m).  Otherwise, the result will be shaped as either
            (n,) or (m,) or () depending on if indices/position are singular.
        """
        offset = self.get_base_equatorial_native_offset(
            indices=indices, offset=offset)
        native_position = self.get_native_xy(position, indices=indices)

        shaped = len(native_position.shape) > 1
        x, y = offset.x, offset.y
        if shaped:
            x, y = x[..., None], y[..., None]

        offset.set_x(x + position.x, copy=False)
        offset.set_y(y + position.y, copy=False)
        return offset

    def get_base_equatorial_native_offset(self, indices=None, offset=None):
        """
        Return equatorial native offsets of the frames from the scan reference.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to extract the offsets.
        offset : Coordinate2D, optional
            An optional coordinate object to hold the results.

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        offset = self.equatorial.get_native_offset_from(
            self.scan_equatorial, indices=indices, offset=offset)
        return offset

    def get_base_native_offset(self, indices=None, offset=None):
        """
        Return equatorial native offsets of the frames from the scan reference.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to extract the offsets.
        offset : Coordinate2D, optional
            An optional coordinate object to hold the results.

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        return self.get_base_equatorial_native_offset(
            indices=indices, offset=offset)

    def get_native_offset(self, position, indices=None, offset=None):
        """
        Get the equatorial offsets for the given position.

        Parameters
        ----------
        position : Coordinate2D
            The (x, y) offsets.
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.
        offset : Coordinate2D, optional
            The coordinate object on which to store the offsets (returned).

        Returns
        -------
        native_offsets : Coordinate2D
        """
        return self.get_equatorial_native_offset(
            position, indices=indices, offset=offset)

    def get_focal_plane_offset(self, position, indices=None, offset=None):
        """
        Return the offsets on the focal plane from given positional offsets.

        The positions are the rotated from telescope coordinates to pixel
        coordinates.

        Parameters
        ----------
        position : Coordinate2D
            The native offsets to convert to a focal plane offset.
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.
        offset : Coordinate2D, optional
            An optional container to store the offsets (returned).

        Returns
        -------
        offsets : Coordinate2D
            The (x, y) offset coordinates on the focal plane.
        """
        if indices is None:
            indices = slice(None)
        offset = self.get_base_native_offset(indices=indices, offset=offset)
        shaped = offset.shape != () and position.shape != ()

        px, py = position.x, position.y
        if self.is_singular:
            cos_a, sin_a = self.cos_a, self.sin_a
        else:
            cos_a, sin_a = self.cos_a[indices], self.sin_a[indices]
        rx = (offset.x * cos_a) + (offset.y * sin_a)
        ry = (offset.y * cos_a) - (offset.x * sin_a)
        if shaped:
            px, py = px[None], py[None]
            rx, ry = rx[..., None], ry[..., None]
        offset.set_x(px + rx, copy=False)
        offset.set_y(py + ry, copy=False)
        return offset

    def get_apparent_equatorial(self, indices=None, apparent=None):
        """
        Precess equatorial coordinates to the Scan MJD.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to get the apparent equatorial
            coordinates.  The default is all indices.
        apparent : EquatorialCoordinates, optional
            If supplied, will be updated with the equatorial coordinates and
            returned.  Otherwise, a fresh coordinate frame will be created.

        Returns
        -------
        apparent_equatorial : EquatorialCoordinates
        """
        if indices is None:
            indices = slice(None)

        if self.is_singular:
            equatorial = self.equatorial
        else:
            equatorial = self.equatorial[indices]

        if apparent is None:
            apparent = equatorial.copy()
        else:
            apparent.copy_coordinates(equatorial)
        return apparent.transform_to(self.astrometry.apparent_epoch)

    @abstractmethod
    def get_absolute_native_coordinates(self):
        """
        Get absolute spherical (including chopper) coords in telescope frame.

        This is named getNativeCoords() in CRUSH

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        pass

    @abstractmethod
    def get_absolute_native_offsets(self):
        """
        Return absolute spherical offsets in telescope frame.

        This is named GetNativeOffset() in CRUSH

        Returns
        -------
        offsets : Coordinate2D
            The (x, y) native offsets.
        """
        pass

    def pointing_at(self, offset, indices=None):
        """
        Applies pointing correction to coordinates via subtraction.

        Parameters
        ----------
        offset : astropy.units.Quantity (numpy.ndarray)
            An array of
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.

        Returns
        -------
        None
        """
        native_offset = self.get_absolute_native_offsets()
        if native_offset is not None:
            if indices is not None:
                if self.is_singular:
                    sub_offset = native_offset
                else:
                    sub_offset = native_offset[indices]
                sub_offset.subtract(offset)
                if self.is_singular:
                    native_offset = sub_offset
                else:
                    native_offset[indices] = sub_offset
            else:
                native_offset.subtract(offset)

        coordinates = self.get_absolute_native_coordinates()
        if coordinates is not None:
            if indices is not None:
                if self.is_singular:
                    sub_coord = coordinates
                else:
                    sub_coord = coordinates[indices]
                sub_coord.subtract_offset(offset)
                if self.is_singular:
                    coordinates = sub_coord
                else:
                    coordinates[indices] = sub_coord
            else:
                coordinates.subtract_offset(offset)

    def scale(self, factor, indices=None):
        """
        Scale all data (in `data`) by `factor`.

        Parameters
        ----------
        factor : int or float
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to scale.  The default is all.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)

        if factor == 0:
            if self.is_singular:
                self.data = 0.0
            else:
                self.data[indices] = 0.0
        else:
            if self.is_singular:
                self.data *= factor
            else:
                self.data[indices] *= factor

    def invert(self, indices=None):
        """
        Multiply all data (in `data`) by -1.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to scale.  The default is all.

        Returns
        -------
        None
        """
        self.scale(-1.0, indices=indices)

    def get_rotation(self, indices=None):
        """
        Returns the tan of sin(angle), cos(angle).

        Parameters
        ----------
        indices : numpy.ndarray (int), optional
            The frame indices.  The default is all indices.

        Returns
        -------
        angle : astropy.units.Quantity (numpy.ndarray)
            An array of angles of size (N,) or (indices.size,).
        """
        rad = units.Unit('radian')
        if indices is None:
            indices = slice(None)

        if self.is_singular:
            return np.arctan2(self.sin_a, self.cos_a) * rad
        else:
            return np.arctan2(self.sin_a[indices], self.cos_a[indices]) * rad

    def set_rotation(self, angle, indices=None):
        """
        Set the `sin_a` and `cos_a` attributes from an angle.

        The `sin_a` and `cos_a` attributes define the rotation from the pixel
        coordinates to the telescope coordinates.

        Parameters
        ----------
        angle : int or float or numpy.ndarray or Quantity
            The angle to set.  If an array is supplied, must be the same size
            as `indices`.
        indices : numpy.ndarray (int), optional
            The frame indices to set.  The default is all indices.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)

        if self.is_singular:
            self.sin_a = np.sin(angle)
            self.cos_a = np.cos(angle)
        else:
            self.sin_a[indices] = np.sin(angle)
            self.cos_a[indices] = np.cos(angle)

    def get_native_x(self, focal_plane_position, indices=None):
        """
        Return the native x coordinates from a given focal plane position.

        Parameters
        ----------
        focal_plane_position : Coordinates2D
            The (x, y) focal plane offsets.
        indices : int or numpy.ndarray or int, optional
            The frame indices for which to calculate x.  The default is all
            frames of shape.  If a slice or array is provided, will be of
            length n.

        Returns
        -------
        native_x : astropy.units.Quantity (numpy.ndarray)
            An array of shape () or (n,) or (n, shape).
        """
        indices, n = self.get_index_size(indices)
        single_index = n == 0
        single_position = focal_plane_position.shape == ()

        if self.is_singular:
            cos_a = self.cos_a
            sin_a = self.sin_a
        else:
            cos_a = self.cos_a[indices]
            sin_a = self.sin_a[indices]

        xp = focal_plane_position.x
        yp = focal_plane_position.y

        if single_index or single_position:
            x = cos_a * xp
            x -= sin_a * yp
        else:
            x = cos_a[..., None] * xp[None]
            x -= sin_a[..., None] * yp[None]
        return x

    def get_native_y(self, focal_plane_position, indices=None):
        """
        Return the native y coordinates from a given focal plane position.

        Parameters
        ----------
        focal_plane_position : Coordinate2D
            The (x, y) focal plane offsets as a scalar or shape (shape,)
        indices : int or numpy.ndarray or int, optional
            The frame indices for which to calculate x.  The default is all
            frames of shape.  If a slice or array is provided, will be of
            length n.

        Returns
        -------
        native_xy : astropy.units.Quantity (numpy.ndarray)
            An array of shape () or (n,) or (n, shape).
        """
        indices, n = self.get_index_size(indices)
        single_index = n == 0
        single_position = focal_plane_position.shape == ()
        if self.is_singular:
            cos_a = self.cos_a
            sin_a = self.sin_a
        else:
            cos_a = self.cos_a[indices]
            sin_a = self.sin_a[indices]

        xp = focal_plane_position.x
        yp = focal_plane_position.y

        if single_index or single_position:
            y = sin_a * xp
            y += cos_a * yp
        else:
            y = sin_a[..., None] * xp[None]
            y += cos_a[..., None] * yp[None]
        return y

    def get_native_xy(self, focal_plane_position, indices=None,
                      coordinates=None):
        """
        Return the native y coordinates from a given focal plane position.

        Rotates the focal plane positions by the stored cos(a) and sin(a)
        frame positions.

        Parameters
        ----------
        focal_plane_position : Coordinates2D
            The (x, y) focal plane offsets as a scalar of shape (shape,).
        indices : int or numpy.ndarray or int, optional
            The frame indices for which to calculate x.  The default is all
            frames of shape.  If a slice or array is provided, will be of
            length n.
        coordinates : Coordinate2D, optional
            An optional output coordinate system to hold the result (returned).

        Returns
        -------
        native_xy : Coordinates2D
            The native (x, y) coordinates.
        """
        indices, n = self.get_index_size(indices)
        single_index = n == 0
        single_position = focal_plane_position.shape == ()
        if self.is_singular:
            cos_a = self.cos_a
            sin_a = self.sin_a
        else:
            cos_a = self.cos_a[indices]
            sin_a = self.sin_a[indices]

        xp = focal_plane_position.x
        yp = focal_plane_position.y

        if single_index or single_position:
            x = cos_a * xp
            x -= sin_a * yp
            y = sin_a * xp
            y += cos_a * yp
        else:
            x = cos_a[..., None] * xp[None]
            x -= sin_a[..., None] * yp[None]
            y = sin_a[..., None] * xp[None]
            y += cos_a[..., None] * yp[None]

        if coordinates is None:
            coordinates = Coordinate2D(unit=self.equatorial.offset_unit)
        coordinates.set_x(x, copy=False)
        coordinates.set_y(y, copy=False)
        return coordinates

    def add_data_from(self, other_frames, scaling=1.0, indices=None):
        """
        Add data from other frames to these.

        Parameters
        ----------
        other_frames : Frames
        scaling : float, optional
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to add to.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)
        if scaling != 1.0:
            if self.is_singular:
                self.data = self.data + (scaling * other_frames.data)
            else:
                self.data[indices] += scaling * other_frames.data
        else:
            if self.is_singular:
                self.data = self.data + other_frames.data
            else:
                self.data[indices] += other_frames.data

        if self.is_singular:
            self.sample_flag = self.sample_flag | other_frames.sample_flag
        else:
            self.sample_flag[indices] |= other_frames.sample_flag

    def project(self, position, projector, indices=None):
        """
        Project focal plane offsets.

        Parameters
        ----------
        position : Coordinate2D
            The (x, y) position to project to offset.
        projector : AstroProjector
            The projector to store and determine the projected offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to project.

        Returns
        -------
        offsets : Coordinate2D
            The projector offsets.  These will also be stored in the projector
            offsets attribute
        """
        if projector.is_focal_plane():
            projector.set_reference_coordinates()
            # Deproject SFL focal plane offsets
            focal_plane_offset = self.get_focal_plane_offset(
                position, indices=indices)
            projector.coordinates.add_native_offset(focal_plane_offset)
            projector.project()

        elif self.info.astrometry.is_nonsidereal:
            projector.set_reference_coordinates()
            # Deproject SFL native offsets
            equatorial_offset = self.get_equatorial_native_offset(
                position, indices=indices)
            projector.equatorial.add_native_offset(equatorial_offset)
            projector.project_from_equatorial()

        else:
            self.get_equatorial(
                position, indices=indices, equatorial=projector.equatorial)
            projector.project_from_equatorial()

        return projector.offset

    def native_to_native_equatorial_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert native offsets to native equatorial offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The native (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate offsets.  The default is
            all frames (not used by equatorial frames).
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        native_equatorial_offsets : Coordinate2D
        """
        return offset if in_place else offset.copy()

    def native_to_equatorial_offset(self, offset, indices=None, in_place=True):
        """
        Convert native offsets to equatorial offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The native (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate offsets.  The default is
            all frames (not used by equatorial frames).
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        if not in_place:
            offset = offset.copy()
        offset = self.native_to_native_equatorial_offset(offset, in_place=True)
        offset.scale_x(-1.0)
        return offset

    def native_equatorial_to_native_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert native equatorial offsets to native offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The native (x, y) equatorial offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate offsets.  The default is
            all frames (not used by equatorial frames).
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        native_offsets : Coordinate2D
        """
        return offset if in_place else offset.copy()

    def equatorial_to_native_offset(self, offset, indices=None, in_place=True):
        """
        Convert equatorial offsets to native offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The equatorial (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate offsets.  The default is
            all frames (not used by equatorial frames).
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        native_offsets : Coordinate2D
        """
        if not in_place:
            offset = offset.copy()
        offset.scale_x(-1.0)
        self.native_equatorial_to_native_offset(offset)
        return offset

    def native_to_equatorial(self, native, indices=None, equatorial=None):
        """
        Convert native coordinates to equatorial coordinates.

        Parameters
        ----------
        native : EquatorialCoordinates
            The coordinates to convert.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate offsets.  The default is
            all frames (not used by equatorial frames).
        equatorial : EquatorialCoordinates, optional
            If not supplied, the returned coordinates will have a J2000 epoch.
            Otherwise, the equatorial coordinates provided will be populated.

        Returns
        -------
        equatorial_coordinates : EquatorialCoordinates
        """
        if equatorial is None:
            equatorial = EquatorialCoordinates(epoch=J2000)
        equatorial.copy_coordinates(native)

    def equatorial_to_native(self, equatorial, indices=None, native=None):
        """
        Convert equatorial coordinates to native coordinates.

        Parameters
        ----------
        equatorial : EquatorialCoordinates
            The equatorial coordinates to convert.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate offsets.  The default is
            all frames (not used by equatorial frames).
        native : SphericalCoordinates, optional
            The native coordinates to populate.  Will default to spherical
            coordinates if not provided.

        Returns
        -------
        native_coordinates : EquatorialCoordinates
        """
        if native is None:
            native = EquatorialCoordinates(epoch=J2000)
        native.copy_coordinates(equatorial)
        return native

    def get_native_offset_from(self, reference, indices=None):
        """
        Return the native offset from a reference coordinate.

        Parameters
        ----------
        reference : EquatorialCoordinates
            The reference position from which to derive the offsets of the
            frame positions.
        indices : numpy.ndarray (int), optional
            The frame indices to use.  The default is all indices.

        Returns
        -------
        Coordinate2D
        """
        return self.get_equatorial_native_offset_from(
            reference, indices=indices)

    def get_equatorial_native_offset_from(self, reference, indices=None):
        """
        Find the native equatorial offset from a reference position.

        The result will be:

        dx = (x - ref(x)) / cos(ref(y))
        dy = (y - ref(y))

        Parameters
        ----------
        reference : EquatorialCoordinates
            The reference equatorial coordinate(s).
        indices : int or numpy.ndarray (int), optional
            The frame indices for which to derive offsets.  If an array is
            provided is should be of shape (n,).  The default is all indices.

        Returns
        -------
        offset : Coordinate2D
            The native equatorial offsets between the frame equatorial
            positions and a reference position.
        """
        if indices is None:
            indices = slice(None)

        if self.is_singular:
            equatorial = self.equatorial
        else:
            equatorial = self.equatorial[indices]
        return equatorial.get_native_offset_from(reference)

    def get_first_frame_index_from(self, index):
        """
        Return the first valid frame index after and including a given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        first_frame : int
        """
        if index < 0:
            index = self.size + index
        return np.nonzero(self.valid[index:])[0][0] + index

    def get_first_frame_index(self, reference=0):
        """
        Return the first valid frame index of the integration.

        Returns
        -------
        first_frame : int
        reference : int, optional
            If supplied, finds the first frame from `reference`, rather than
            the first index (0).  May take negative values to indicate an
            index relative to the last.
        """
        return self.get_first_frame_index_from(reference)

    def get_last_frame_from(self, index):
        """
        Return the last valid frame index before and including a given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        last_frame : int
        """
        if index < 0:
            index = self.size + index
        return np.nonzero(self.valid[:index])[0][-1]

    def get_last_frame_index(self, reference=None):
        """
        Return the last valid frame index of the integration.

        Returns
        -------
        last_frame : int
        reference : int, optional
        reference : int, optional
            If supplied, finds the last frame before `reference`, rather than
            the last index (self.size).  May take negative values to indicate
            an index relative to the last index.
        """
        if reference is None:
            reference = self.size
        return self.get_last_frame_from(reference)

    def get_first_frame_value(self, field):
        """
        Return the first valid frame data for the given field.

        Parameters
        ----------
        field : str
            Name of the frame data field.

        Returns
        -------
        value
        """
        index = self.get_first_frame_index()
        values = getattr(self, field, None)
        if values is None:
            raise ValueError(f"{self} does not contain {field} field.")
        return values[index]

    def get_last_frame_value(self, field):
        """
        Return the last valid frame data for the given field.

        Parameters
        ----------
        field : str
            Name of the frame data field.

        Returns
        -------
        value
        """
        index = self.get_last_frame_index()
        values = getattr(self, field, None)
        if values is None:
            raise ValueError(f"{self} does not contain {field} field.")
        return values[index]

    def get_first_frame_value_from(self, index, field):
        """
        Return the first valid frame value before and including a given index.

        Parameters
        ----------
        index : int
        field : str
           Name of the frame data field.

        Returns
        -------
        value
        """
        index = self.get_first_frame_index_from(index)
        values = getattr(self, field, None)
        if values is None:
            raise ValueError(f"{self} does not contain {field} field.")
        return values[index]

    def get_last_frame_value_from(self, index, field):
        """
        Return the last valid frame value before and including a given index.

        Parameters
        ----------
        index : int
        field : str
           Name of the frame data field.

        Returns
        -------
        value
        """
        index = self.get_last_frame_from(index)
        values = getattr(self, field, None)
        if values is None:
            raise ValueError(f"{self} does not contain {field} field.")
        return values[index]

    def get_first_frame(self, reference=0):
        """
        Return the first valid frame.

        Parameters
        ----------
        reference : int, optional
            The first actual frame index after which to return the first valid
            frame.  The default is the first (0).

        Returns
        -------
        Frames
        """
        return self[self.get_first_frame_index(reference=reference)]

    def get_last_frame(self, reference=None):
        """
        Return the first valid frame.

        Parameters
        ----------
        reference : int, optional
            The last actual frame index before which to return the last valid
            frame.  The default is the last.

        Returns
        -------
        Frames
        """
        return self[self.get_last_frame_index(reference=reference)]

    def add_dependents(self, dependents, start=None, end=None):
        """
        Add dependents from frame data.

        Parameters
        ----------
        dependents : numpy.ndarray (float)
            Must be the same size as self.size.
        start : int, optional
            The starting frame.
        end : int, optional
            The exclusive ending frame.

        Returns
        -------
        None
        """
        frames_numba_functions.add_dependents(
            dependents=self.dependents,
            dp=dependents,
            frame_valid=self.valid,
            start_frame=start,
            end_frame=end,
            subtract=False)

    def remove_dependents(self, dependents, start=None, end=None):
        """
        Add dependents from frame data.

        Parameters
        ----------
        dependents : numpy.ndarray (float)
            Must be the same size as self.size.
        start : int, optional
            The starting frame.
        end : int, optional
            The exclusive ending frame.

        Returns
        -------
        None
        """
        frames_numba_functions.add_dependents(
            dependents=self.dependents,
            dp=dependents,
            frame_valid=self.valid,
            start_frame=start,
            end_frame=end,
            subtract=True)

    def shift_frames(self, n_frames):
        """
        Shift the "readout" data by a number of frames in either direction.

        This will only shift data fields contained in the 'readout_attributes'
        frame property.  For frame `i`, the result of shifting by `n_frames`
        will be:

        new[i + n_frames] = old[i]

        Frames shifted outside of an array will be set to invalid.

        Parameters
        ----------
        n_frames : int

        Returns
        -------
        None
        """
        shift_valid = np.roll(self.valid, n_frames)
        if n_frames > 0:
            shift_valid[:n_frames] = False
        elif n_frames < 0:
            shift_valid[n_frames:] = False
        self.valid &= shift_valid

        invalid = np.logical_not(self.valid)

        for field in self.readout_attributes:
            value = getattr(self, field, None)
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                saved_invalid = value[invalid]
                value = np.roll(value, n_frames, axis=0)
                value[invalid] = saved_invalid
            elif isinstance(value, Coordinate):
                value.shift(n_frames, fill_value=np.nan)

            setattr(self, field, value)

    @staticmethod
    def correct_factor_dimensions(factor, array):
        """
        Corrects the factor dimensionality prior to an array +-/* etc.

        Frame operations are frequently of the form result = factor op array
        where factor is of shape (n_frames,) and array is of shape
        (n_frames, ...).  This procedure updates the factor shape so that
        array operations are possible.  E.g., if factor is of shape (5,) and
        array is of shape (5, 10), then the output factor will be of shape
        (5, 1) and allow the two arrays to operate with each other.

        Parameters
        ----------
        factor : int or float or numpy.ndarray
            The factor to check.
        array : numpy.ndarray
            The array to check against

        Returns
        -------
        working_factor : numpy.ndarray
        """
        return Coordinate2D.correct_factor_dimensions(factor, array)

    def set_from_downsampled(self, frames, start_indices, valid, window):
        """
        Set the data for these frames by downsampling higher-res frames.

        Parameters
        ----------
        frames : Frames
            The frames at a higher resolution.
        start_indices : numpy.ndarray (int)
            The start indices containing the first index of the high resolution
            frame for each convolution with the window function.  Should be
            of shape (self.size,).
        valid : numpy.ndarray (bool)
            A boolean mask indicating whether a downsampled frame could be
            determined from the higher resolution frames.  Should be of shape
            (self.size,).
        window : numpy.ndarray (float)
            The window function used for convolution of shape (n_window,).

        Returns
        -------
        None
        """
        data, sample_flag = frames_numba_functions.downsample_data(
            data=frames.data,
            sample_flag=frames.sample_flag,
            valid=valid,
            window=window,
            start_indices=start_indices
        )
        self.data = data
        self.sample_flag = sample_flag
        self.valid = valid.copy()
