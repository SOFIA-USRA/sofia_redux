# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.frames.frames import Frames
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['HorizontalFrames']


class HorizontalFrames(Frames):

    def __init__(self):
        super().__init__()
        self.horizontal = None
        self.horizontal_offset = None
        self.cos_pa = None
        self.sin_pa = None
        self.zenith_tau = None

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
            'cos_pa': 0.0,
            'sin_pa': 0.0,
            'zenith_tau': 0.0,
            'horizontal': (HorizontalCoordinates, 'degree'),
            'horizontal_offset': (Coordinate2D, 'arcsec')
        })
        return fields

    @property
    def site(self):
        """
        Return the site coordinates of the associated scan.

        Returns
        -------
        GeodeticCoordinates
        """
        return self.astrometry.site

    def validate(self):
        """
        Validate frame data after read.

        Should set the `validated` (checked) attribute if necessary.

        Returns
        -------
        None
        """
        if self.has_telescope_info.any():
            if self.equatorial is None or self.equatorial.size != self.size:
                self.calculate_equatorial()
            elif self.horizontal is None or self.horizontal.size != self.size:
                self.calculate_horizontal()
        super().validate()

    def get_equatorial(self, offsets, indices=None, equatorial=None):
        """
        Return equatorial coordinates given offsets from the base equatorial.

        The return result (lon, lat) is:

        lon = base_lon + (position.lon / cos(scan.lat))
        lat = base_lat + position.lat

        Parameters
        ----------
        offsets : Coordinate2D
            The (x, y) horizontal offsets of shape () or (shape,)
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.
        equatorial : EquatorialCoordinates, optional
            The equatorial output frame.  The default is the same as the frame
            equatorial frame.

        Returns
        -------
        equatorial : EquatorialCoordinates
        """
        if indices is None:
            indices = slice(None)
        position = self.get_native_xy(offsets, indices=indices)
        if equatorial is None:
            equatorial = self.equatorial.empty_copy()
            if equatorial.epoch.singular:
                equatorial.epoch = self.equatorial.epoch.copy()
            else:
                equatorial.epoch = self.equatorial.epoch[indices].copy()

        shaped = len(position.shape) > 1
        if not self.is_singular:
            cos_pa = self.cos_pa[indices]
            sin_pa = self.sin_pa[indices]
            x = self.equatorial.x[indices]
            y = self.equatorial.y[indices]
        else:
            cos_pa = self.cos_pa
            sin_pa = self.sin_pa
            x = self.equatorial.x
            y = self.equatorial.y

        px, py = position.x, position.y
        cos_lat = self.astrometry.equatorial.cos_lat

        if shaped:
            cos_pa, sin_pa = cos_pa[..., None], sin_pa[..., None]
        rx = ((cos_pa * px) - (sin_pa * py)) / cos_lat
        ry = (cos_pa * py) + (sin_pa * px)

        if shaped:
            x, y = x[..., None], y[..., None]

        equatorial.set_native_longitude(x + rx)
        equatorial.set_native_latitude(y + ry)
        return equatorial

    def get_horizontal(self, offsets, indices=None, horizontal=None):
        """
        Return horizontal coordinates given offsets from the base horizontal.

        Parameters
        ----------
        offsets : Coordinate2D
            The (x, y) horizontal offsets of shape () or (shape,)
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.
        horizontal : HorizontalCoordinates, optional
            The horizontal output frame.  The default is a fresh frame.

        Returns
        -------
        horizontal : HorizontalCoordinates
        """
        if indices is None:
            indices = slice(None)
        if horizontal is None:
            horizontal = Coordinate2D.get_instance('horizontal')

        if not horizontal.is_singular:
            x = horizontal.x[indices]
            y = horizontal.y[indices]
        else:
            x = horizontal.x
            y = horizontal.y

        position = self.get_native_xy(offsets, indices=indices)
        shaped = len(position.shape) > 1
        cos_lat = self.astrometry.horizontal.cos_lat
        if shaped:
            x, y = x[..., None], y[..., None]

        horizontal.set_native_longitude(x + (position.x / cos_lat))
        horizontal.set_native_latitude(y + position.y)
        return horizontal

    def get_horizontal_offset(self, position, indices=None, offset=None):
        """
        Return the horizontal offsets of a position relative to scan center.

        Parameters
        ----------
        position : Coordinate2D, optional
            The (x, y) horizontal offsets.
        indices : int or numpy.ndarray (int), optional
            The frame indices that apply of shape (n,) (if an array was used).
            The default is all indices.
        offset : Coordinate2D, optional
            An optional output array to store and return the coordinates.

        Returns
        -------
        horizontal_offsets : Coordinate2D
            An array containing the sum of the horizontal offsets of the frame
            data and the supplied positions.  If multiple frame indices and
            multiple positions are supplied, the resulting coordinate shape
            will be (n, m).  Otherwise, the result will be shaped as either
            (n,) or (m,) or () depending on if indices/position are singular.
        """
        if indices is None:
            indices = slice(None)
        native_position = self.get_native_xy(position, indices=indices)
        if offset is None:
            offset = Coordinate2D(unit='arcsec')
        shaped = len(native_position.shape) > 1
        x = self.horizontal_offset.x
        y = self.horizontal_offset.y
        if shaped:
            x = x[..., None]
            y = y[..., None]
        offset.set_x(x + native_position.x)
        offset.set_y(y + native_position.y)
        return offset

    def get_native_offset(self, position, indices=None, offset=None):
        """
        Get the horizontal offsets for the given position.

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
        return self.get_horizontal_offset(
            position, indices=indices, offset=offset)

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
        equatorial_offsets : Coordinate2D
            An array containing the sum of the equatorial offsets of the frame
            data and the supplied positions.  If multiple frame indices and
            multiple positions are supplied, the resulting coordinate shape
            will be (n, m).  Otherwise, the result will be shaped as either
            (n,) or (m,) or () depending on if indices/position are singular.
        """
        offset = self.get_horizontal_offset(
            position, indices=indices, offset=offset)
        self.horizontal_to_native_equatorial_offset(
            offset, indices=indices, in_place=True)
        return offset

    def get_absolute_native_coordinates(self):
        """
        Get absolute spherical (including chopper) coords in telescope frame.

        This is named getNativeCoords() in CRUSH

        Returns
        -------
        coordinates : HorizontalCoordinates
        """
        return self.horizontal

    def get_absolute_native_offsets(self):
        """
        Return absolute spherical offsets in telescope frame.

        This is named GetNativeOffset() in CRUSH

        Returns
        -------
        offsets : Coordinate2D
            The (x, y) native offsets.
        """
        return self.horizontal_offset

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
        horizontal_offsets : Coordinate2D
        """
        if offset is None:
            offset = Coordinate2D(unit='arcsec')
        offset.copy_coordinates(self.horizontal_offset)
        return offset

    def project(self, position, projector, indices=None):
        """
        Project a position to offsets.

        Parameters
        ----------
        position : Coordinate2D
            The (x, y) position.
        projector : AstroProjector
            The projector to store and determine the projected offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to project.

        Returns
        -------
        offsets : Coordinate2D
            The (x, y) offsets.
        """
        if projector.is_horizontal():
            projector.set_reference_coordinates()
            horizontal_offset = self.get_horizontal_offset(
                position, indices=indices)
            projector.coordinates.add_native_offset(horizontal_offset)
            return projector.project()

        else:
            return super().project(position, projector, indices=indices)

    def calculate_parallactic_angle(self, lst=None, indices=None):
        """
        Calculate the cos(pa) and sin(pa) values.

        Parameters
        ----------
        lst : astropy.units.Quantity
            If provided, the Local Sidereal Time will be used to calculate
            the position angle from the site and equatorial coordinates.
            Otherwise, the parallactic angle will be calculated from the
            horizontal coordinates.  If an array is provided, should be the
            same shape as `indices`.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices for which to calculate the parallactic angle.
            The default is all frames.

        Returns
        -------
        None
        """
        if isinstance(lst, units.Quantity):
            pa = self.equatorial.get_parallactic_angle(self.site, lst)
        else:
            pa = self.horizontal.get_parallactic_angle(self.site)

        self.set_parallactic_angle(pa, indices=indices)

    def set_parallactic_angle(self, angle, indices=None):
        """
        Sets the `sin_pa` and `cos_pa` parallactic angles.

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

        if not self.is_singular:
            self.sin_pa[indices] = np.sin(angle)
            self.cos_pa[indices] = np.cos(angle)
        else:
            self.sin_pa = np.sin(angle)
            self.cos_pa = np.cos(angle)

    def get_parallactic_angle(self, indices=None):
        """
        Returns the tan of sin(position_angle), cos(position_angle).

        Parameters
        ----------
        indices : numpy.ndarray (int), optional
            The frame indices.  The default is all indices.

        Returns
        -------
        angle : astropy.units.Quantity (numpy.ndarray)
            An array of angles of size (N,) or (indices.size,).
        """
        radian = units.Unit('radian')
        if indices is None:
            indices = slice(None)

        if not self.is_singular:
            result = np.arctan2(
                self.sin_pa[indices], self.cos_pa[indices]) * radian
        else:
            result = np.arctan2(self.sin_pa, self.cos_pa) * radian
        return result

    def calculate_horizontal(self):
        """
        Calculate the horizontal coordinates from the equatorial coordinates.

        Returns
        -------
        None
        """
        apparent = self.get_apparent_equatorial()
        self.horizontal = apparent.to_horizontal(self.site, self.lst)

    def calculate_equatorial(self):
        """
        Calculate the equatorial coordinates from the horizontal coordinates.

        This assumes that the object is tracked on sky, and uses scanning
        offsets on top of the tracking coordinates of the scan.

        Returns
        -------
        None
        """
        if self.scan.is_tracking:
            if self.equatorial is None:
                self.equatorial = self.astrometry.equatorial.empty_copy()

            hx, hy = self.horizontal_offset.x, self.horizontal_offset.y
            sin_pa = self.correct_factor_dimensions(self.sin_pa, hx)
            cos_pa = self.correct_factor_dimensions(self.cos_pa, hx)

            ex, ey = self.astrometry.equatorial.x, self.astrometry.equatorial.y
            cos_lat = self.astrometry.equatorial.cos_lat

            x = ex + (cos_pa * hx - sin_pa * hy) / cos_lat
            y = ey + (cos_pa * hy + sin_pa * hx)
            self.equatorial.set_native_longitude(x)
            self.equatorial.set_native_latitude(y)

        else:
            self.equatorial = self.horizontal.to_equatorial(
                self.site, self.lst)
            self.astrometry.from_apparent.precess(self.equatorial)

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
        super().pointing_at(offset, indices=indices)
        self.calculate_equatorial()

    def set_zenith_tau(self, zenith_tau, indices=None):
        """
        Set the zenith tau values of the frames.

        Parameters
        ----------
        zenith_tau : float or numpy.ndarray (float)
            The zenith tau value(s).
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to apply the values to.  The default is all
            frames.

        Returns
        -------
        None
        """
        if indices is None:
            indices = slice(None)

        if self.is_singular:
            self.zenith_tau = zenith_tau
            sin_lat = self.horizontal.sin_lat
        else:
            self.zenith_tau[indices] = zenith_tau
            sin_lat = self.horizontal.sin_lat[indices]

        self.set_transmission(np.exp(-zenith_tau / sin_lat), indices=indices)

    def horizontal_to_native_equatorial_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert horizontal offsets to native equatorial offsets.

        Rotates by the position angle.

        Parameters
        ----------
        offset : Coordinate2D
            The horizontal (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        native_equatorial_offsets : Coordinate2D
        """
        if indices is None:
            indices = slice(None)
        if not in_place:
            x = offset.x
            offset = offset.copy()
        else:
            x = offset.x.copy()
        y = offset.y

        if self.is_singular:
            cos_pa = self.correct_factor_dimensions(self.cos_pa, x)
            sin_pa = self.correct_factor_dimensions(self.sin_pa, x)
        else:
            cos_pa = self.correct_factor_dimensions(self.cos_pa[indices], x)
            sin_pa = self.correct_factor_dimensions(self.sin_pa[indices], x)
        offset.set_x((cos_pa * x) - (sin_pa * y))
        offset.set_y((sin_pa * x) + (cos_pa * y))
        return offset

    def horizontal_to_equatorial_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert a horizontal offset to an equatorial offset.

        Parameters
        ----------
        offset : Coordinate2D
            The horizontal (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        offset = self.horizontal_to_native_equatorial_offset(
            offset, indices=indices, in_place=in_place)
        offset.scale_x(-1.0)
        return offset

    def equatorial_native_to_horizontal_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert native equatorial offsets to horizontal offsets.

        Rotates by -PA (position angle).

        Parameters
        ----------
        offset : Coordinate2D
            The native equatorial (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        horizontal_offsets : Coordinate2D
        """
        if indices is None:
            indices = slice(None)
        if in_place:
            x = offset.x.copy()
        else:
            x = offset.x
            offset = offset.copy()
        y = offset.y

        if not self.is_singular:
            cos_pa = self.correct_factor_dimensions(self.cos_pa[indices], x)
            sin_pa = self.correct_factor_dimensions(self.sin_pa[indices], x)
        else:
            cos_pa = self.cos_pa
            sin_pa = self.sin_pa

        offset.set_x((cos_pa * x) + (sin_pa * y))
        offset.set_y((cos_pa * y) - (sin_pa * x))
        return offset

    def equatorial_to_horizontal_offset(self, offset, indices=None,
                                        in_place=True):
        """
        Convert equatorial offsets to horizontal offsets.

        Parameters
        ----------
        offset : Coordinate2D
            The equatorial (x, y) offsets.
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to update.  The default is all frames.
        in_place : bool, optional
            If `True`, modify the coordinates in place.  Otherwise, return
            a copy of the offsets.

        Returns
        -------
        horizontal_offsets : Coordinate2D
        """
        if not in_place:
            offset = offset.copy()
        offset.scale_x(-1.0)
        offset = self.equatorial_native_to_horizontal_offset(
            offset, indices=indices, in_place=True)
        return offset

    def native_to_native_equatorial_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert native (horizontal) offsets to native equatorial offsets.

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
        return self.horizontal_to_native_equatorial_offset(
            offset, indices=indices, in_place=in_place)

    def native_equatorial_to_native_offset(
            self, offset, indices=None, in_place=True):
        """
        Convert native equatorial offsets to native (horizontal) offsets.

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
        return self.equatorial_native_to_horizontal_offset(
            offset, indices=indices, in_place=in_place)

    def native_to_equatorial(self, native, indices=None, equatorial=None):
        """
        Convert native (horizontal) coordinates to equatorial coordinates.

        Parameters
        ----------
        native : HorizontalCoordinates
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
        if indices is None:
            indices = slice(None)
        lst = self.lst if self.is_singular else self.lst[indices]

        return native.to_equatorial(self.site, lst, equatorial=equatorial)

    def equatorial_to_native(self, equatorial, indices=None, native=None):
        """
        Convert equatorial coordinates to native (horizontal) coordinates.

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
        native_coordinates : HorizontalCoordinates
        """
        if indices is None:
            indices = slice(None)
        lst = self.lst if self.is_singular else self.lst[indices]

        return equatorial.to_horizontal(
            self.site, lst, equatorial=equatorial, horizontal=native)
