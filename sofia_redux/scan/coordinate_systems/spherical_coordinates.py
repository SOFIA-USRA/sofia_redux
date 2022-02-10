# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.coordinates import Angle
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.coordinate_axis import CoordinateAxis
from sofia_redux.scan.coordinate_systems import \
    coordinate_systems_numba_functions as csnf
from sofia_redux.scan.utilities.utils import get_comment_unit

__all__ = ['SphericalCoordinates']


class SphericalCoordinates(Coordinate2D):

    angular_accuracy = 1e-12 * units.Unit('degree')
    right_angle = 90 * units.Unit('degree')
    pi = 180 * units.Unit('degree')
    two_pi = 360 * units.Unit('degree')
    id_lookup = None
    ids = None
    fits_types = None

    def __init__(self, coordinates=None, unit='degree', copy=True):
        """
        Initialize a SphericalCoordinates object.

        Spherical coordinates are designed to represent longitude/latitude
        coordinates in a given frame.

        Parameters
        ----------
        coordinates : list or tuple or array-like or units.Quantity, optional
            The coordinates used to populate the object during initialization.
            The first (0) value or index should represent longitudinal
            coordinates, and the second should represent latitude.
        unit : units.Unit or str, optional
            The angular unit for the spherical coordinates.  The default is
            'degree'.
        copy : bool, optional
            Whether to explicitly perform a copy operation on the input
            coordinates when storing them into these coordinates.  Note that it
            is extremely unlikely for the original coordinates to be passed in
            as a reference due to the significant checks performed on them.
        """
        self.default_coordinate_system = None
        self.default_local_coordinate_system = None
        self.setup_coordinate_system()
        self.cos_lat = None
        self.sin_lat = None
        super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the spherical coordinates.

        Returns
        -------
        SphericalCoordinates
        """
        return super().copy()

    @property
    def empty_copy_skip_attributes(self):
        """
        Return attributes that are set to None on an empty copy.

        Returns
        -------
        attributes : set (str)
        """
        skip = super().empty_copy_skip_attributes
        skip.add('cos_lat')
        skip.add('sin_lat')
        return skip

    @property
    def coordinate_system(self):
        """
        Return the coordinate system.

        Returns
        -------
        CoordinateSystem
        """
        return self.default_coordinate_system

    @property
    def local_coordinate_system(self):
        """
        Return the local coordinate system.

        Returns
        -------
        CoordinateSystem
        """
        return self.default_local_coordinate_system

    @property
    def longitude_axis(self):
        """
        Return the longitude axis.

        Returns
        -------
        CoordinateAxis
        """
        return self.coordinate_system.axes[0]

    @property
    def latitude_axis(self):
        """
        Return the latitude axis.

        Returns
        -------
        CoordinateAxis
        """
        return self.coordinate_system.axes[1]

    @property
    def x_offset_axis(self):
        """
        Return the offset x-axis.

        Returns
        -------
        CoordinateAxis
        """
        return self.local_coordinate_system.axes[0]

    @property
    def y_offset_axis(self):
        """
        Return the offset y-axis.

        Returns
        -------
        CoordinateAxis
        """
        return self.local_coordinate_system.axes[1]

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the coordinate system.

        Returns
        -------
        str
        """
        return 'SP'

    @property
    def fits_latitude_stem(self):
        """
        Return the string prefix for latitude.

        Returns
        -------
        str
        """
        return 'LAT-'

    @property
    def fits_longitude_stem(self):
        """
        Return the string prefix for longitude.

        Returns
        -------
        str
        """
        return 'LON-'

    @property
    def reverse_longitude(self):
        """
        Return `True` if the longitude axis is reversed.

        Returns
        -------
        bool
        """
        return self.longitude_axis.reverse

    @property
    def reverse_latitude(self):
        """
        Return `True` if the latitude axis is reversed.

        Returns
        -------
        bool
        """
        return self.latitude_axis.reverse

    @property
    def native_longitude(self):
        """
        Return the native longitude.

        Returns
        -------
        astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        return self.x

    @native_longitude.setter
    def native_longitude(self, longitude):
        """
        Set the native longitude.

        Parameters
        ----------
        longitude : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_native_longitude(longitude, copy=True)

    @property
    def native_latitude(self):
        """
        Return the native latitude.

        Returns
        -------
        astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        return self.y

    @native_latitude.setter
    def native_latitude(self, latitude):
        """
        Set the native longitude.

        Parameters
        ----------
        latitude : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_native_latitude(latitude, copy=True)

    @property
    def longitude(self):
        """
        Return the longitude.

        Returns
        -------
        longitude : astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        if self.reverse_longitude:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return self.longitude_axis.reverse_from - self.native_longitude
        else:
            return self.native_longitude

    @longitude.setter
    def longitude(self, values):
        """
        Set the longitude.

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_longitude(values, copy=True)

    @property
    def latitude(self):
        """
        Return the latitude.

        Returns
        -------
        latitude : astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        if self.reverse_latitude:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return self.latitude_axis.reverse_from - self.native_latitude
        else:
            return self.native_latitude

    @latitude.setter
    def latitude(self, values):
        """
        Set the latitude.

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_latitude(values, copy=True)

    @property
    def lon(self):
        """
        Return the longitude.

        Returns
        -------
        longitude : astropy.units.Quantity (float or numpy.ndarray)
        """
        return self.longitude

    @lon.setter
    def lon(self, values):
        """
        Set the longitude.

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.longitude = values

    @property
    def lat(self):
        """
        Return the latitude.

        Returns
        -------
        latitude : astropy.units.Quantity (float or numpy.ndarray)
        """
        return self.latitude

    @lat.setter
    def lat(self, values):
        """
        Set the latitude.

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.latitude = values

    @property
    def offset_unit(self):
        """
        Return the units used for the offset frame.

        Returns
        -------
        astropy.units.Unit
        """
        return self.x_offset_axis.unit

    def __eq__(self, other):
        """
        Test if these spherical coordinates are equal to another.

        Spherical coordinates are considered equal if the longitude
        coordinates match when wrapped in the range 0->360 degrees, and
        latitude coordinates match in the range -180->180 degrees.

        Parameters
        ----------
        other : SphericalCoordinates

        Returns
        -------
        bool
        """
        circle = 360 * units.Unit('degree')
        semi_circle = 180 * units.Unit('degree')

        if other is self:
            return True
        if self.__class__ != other.__class__:
            return False

        if self.coordinates is None:
            return other.coordinates is None
        elif other.coordinates is None:
            return self.coordinates is None

        if self.shape != other.shape:
            return False

        try:
            x1 = Angle(self.longitude).wrap_at(circle)
            x2 = Angle(other.longitude).wrap_at(circle)
            if not np.allclose(x1, x2, equal_nan=True):
                return False
        except units.UnitConversionError:  # pragma: no cover
            return False

        try:
            x1 = Angle(self.latitude).wrap_at(semi_circle)
            x2 = Angle(other.latitude).wrap_at(semi_circle)
            return np.allclose(x1, x2, equal_nan=True)
        except units.UnitConversionError:  # pragma: no cover
            return False

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        SphericalCoordinates
        """
        return super().__getitem__(indices)

    def __str__(self):
        """
        Create a string representation of the spherical coordinates.

        Returns
        -------
        str
        """
        if self.coordinates is None:
            if self.unit is None:  # pragma: no cover
                return 'Empty coordinates'
            else:
                return f'Empty coordinates ({self.unit})'

        if self.singular:
            lon_string = Angle(self.longitude).to_string(unit='degree')
            lat_string = Angle(self.latitude).to_string(unit='degree')
            return f'LON={lon_string} LAT={lat_string}'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                min_lon = Angle(np.nanmin(self.longitude)).to_string(
                    unit='degree')
                max_lon = Angle(np.nanmax(self.longitude)).to_string(
                    unit='degree')
                min_lat = Angle(np.nanmin(self.latitude)).to_string(
                    unit='degree')
                max_lat = Angle(np.nanmax(self.latitude)).to_string(
                    unit='degree')
            return f'LON={min_lon}->{max_lon} LAT={min_lat}->{max_lat}'

    @classmethod
    def register_types(cls):
        """
        Register a number of coordinate classes for later extraction.

        Returns
        -------
        None
        """
        cls.id_lookup = {}
        cls.ids = {}
        cls.fits_types = {}
        for coordinate_type in ['spherical', 'horizontal', 'telescope',
                                'focal_plane', 'equatorial', 'ecliptic',
                                'galactic', 'super_galactic']:
            cls.register(cls.get_class(coordinate_type))

    @classmethod
    def register(cls, coordinate_class):
        """
        Register a given coordinate class as an available spherical system.

        Parameters
        ----------
        coordinate_class : class

        Returns
        -------
        None
        """
        instance = coordinate_class()
        cls.ids[instance.two_letter_code.upper()] = coordinate_class
        cls.id_lookup[coordinate_class] = instance.two_letter_code
        cls.fits_types[instance.fits_longitude_stem] = coordinate_class
        cls.fits_types[instance.fits_latitude_stem] = coordinate_class

    @classmethod
    def get_fits_class(cls, ctype):
        """
        Return a Coordinate class for the given ctype.

        Parameters
        ----------
        ctype : str
            The Coordinate system.

        Returns
        -------
        class (SphericalCoordinates)
        """
        if cls.fits_types is None:
            cls.register_types()

        ctype = ctype.ljust(4, '-').upper()[:4]
        coordinate_class = cls.fits_types.get(ctype)
        if coordinate_class is None:
            raise ValueError(f"Unknown coordinate definition: {ctype}")
        return coordinate_class

    @classmethod
    def get_two_letter_class(cls, class_id):
        """
        Return a Coordinate class for the given class ID.

        Parameters
        ----------
        class_id : str
            The two-letter code for the class.

        Returns
        -------
        class (SphericalCoordinates)
        """
        if cls.ids is None:
            cls.register_types()
        coordinate_class = cls.ids.get(class_id.upper())
        if coordinate_class is None:
            raise ValueError(f"Unknown coordinate definition {class_id}.")
        return coordinate_class

    @classmethod
    def get_class_for(cls, spec):
        """
        Return a spherical coordinate class for the given specification.

        Parameters
        ----------
        spec : str
            The name of, the two-letter code, or an axis ctype name for which
            to return the correct coordinate class.

        Returns
        -------
        class (SphericalCoordinates)
        """
        try:
            return cls.get_class(spec)
        except (ImportError, ModuleNotFoundError, ValueError):
            pass

        try:
            return cls.get_fits_class(spec)
        except ValueError:
            pass

        try:
            return cls.get_two_letter_class(spec)
        except ValueError:
            raise ValueError(f"Unknown coordinate definition {spec}.")

    @classmethod
    def get_two_letter_code_for(cls, class_type):
        """
        Return the two-letter code for a given coordinate class.

        Parameters
        ----------
        class_type : class (SphericalCoordinates)

        Returns
        -------
        two_letter_code : str
        """
        if cls.id_lookup is None:
            cls.register_types()
        return cls.id_lookup.get(class_type)

    @classmethod
    def get_default_system(cls):
        """
        Return the default and local default coordinate system.

        Returns
        -------
        system, local_system : (CoordinateSystem, CoordinateSystem)
        """
        default_coordinate_system = CoordinateSystem(
            name='Spherical Coordinates')
        default_local_coordinate_system = CoordinateSystem(
            name='Spherical Offsets')
        longitude_axis = cls.create_axis('Longitude', 'LON')
        latitude_axis = cls.create_axis('Latitude', 'LAT')
        longitude_offset_axis = cls.create_offset_axis(
            'Longitude Offset', 'dLON')
        latitude_offset_axis = cls.create_offset_axis(
            'Latitude Offset', 'dLAT')
        default_coordinate_system.add_axis(longitude_axis)
        default_coordinate_system.add_axis(latitude_axis)
        default_local_coordinate_system.add_axis(longitude_offset_axis)
        default_local_coordinate_system.add_axis(latitude_offset_axis)
        return default_coordinate_system, default_local_coordinate_system

    @staticmethod
    def create_axis(label, short_label, unit='degree'):
        """
        Create an axis.

        Parameters
        ----------
        label : str
            The name of the axis.
        short_label : str
            A shorthand name for the axis.
        unit : astropy.units.Unit or str, optional
            The axis unit.

        Returns
        -------
        CoordinateAxis
        """
        return CoordinateAxis(label=label, short_label=short_label, unit=unit)

    @staticmethod
    def create_offset_axis(label, short_label, unit='arcsec'):
        """
        Create an offset axis.

        Parameters
        ----------
        label : str
            The name of the offset axis.
        short_label : str
            The shorthand name for the axis.
        unit : astropy.units.Unit or str, optional
            The offset axis unit.

        Returns
        -------
        CoordinateAxis
        """
        return CoordinateAxis(label=label, short_label=short_label, unit=unit)

    def setup_coordinate_system(self):
        """
        Setup the system for the coordinates.

        Returns
        -------
        None
        """
        (self.default_coordinate_system,
         self.default_local_coordinate_system) = self.get_default_system()

    def set_shape(self, shape, empty=False):
        """
        Set the shape of the coordinates.

        Parameters
        ----------
        shape : int or tuple (int)
        empty : bool, optional
            If `True`, create an empty array.  Otherwise, create a zeroed
            array.

        Returns
        -------
        None
        """
        super().set_shape(shape, empty=empty)
        if empty:
            self.sin_lat = np.empty(self.shape, dtype=float)
            self.cos_lat = np.empty(self.shape, dtype=float)
        else:
            self.sin_lat = np.zeros(self.shape, dtype=float)
            self.cos_lat = np.ones(self.shape, dtype=float)

    def set_singular(self, empty=False):
        """
        Create a single coordinate.

        Parameters
        ----------
        empty : bool, optional
            If `True`, create an empty coordinate array.  Otherwise, create a
            zeroed array.

        Returns
        -------
        None
        """
        super().set_singular(empty=empty)
        self.sin_lat = np.asarray(0.0)
        self.cos_lat = np.asarray(1.0)

    def set_y(self, coordinates, copy=True):
        """
        Set the y coordinates.

        Parameters
        ----------
        coordinates : astropy.units.Quantity (float or numpy.ndarray)
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if not isinstance(coordinates, units.Quantity):
                coordinates = coordinates * self.unit
            super().set_y(np.fmod(coordinates, self.pi), copy=copy)
        self.sin_lat = np.sin(self.coordinates[1]).value
        self.cos_lat = np.cos(self.coordinates[1]).value

    def add_y(self, y):
        """
        Add y to coordinates.

        Parameters
        ----------
        y : astropy.units.Quantity (float or numpy.ndarray)
            The value(s) to add.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if not isinstance(y, units.Quantity):
                y = y * self.unit
            super().add_y(np.fmod(y, self.pi))
        self.sin_lat = np.sin(self.coordinates[1]).value
        self.cos_lat = np.cos(self.coordinates[1]).value

    def subtract_y(self, y):
        """
        Subtract y from coordinates.

        Parameters
        ----------
        y : astropy.units.Quantity (float or numpy.ndarray)
            The value(s) to subtract.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if not isinstance(y, units.Quantity):
                y = y * self.unit
            super().subtract_y(np.fmod(y, self.pi))
        self.sin_lat = np.sin(self.coordinates[1]).value
        self.cos_lat = np.cos(self.coordinates[1]).value

    def zero(self, indices=None):
        """
        Set all coordinates to zero.

        Parameters
        ----------
        indices : slice or numpy.ndarray (int or bool), optional
            The indices to set to zero.

        Returns
        -------
        None
        """
        super().zero(indices=indices)
        if self.sin_lat is None:
            return
        if indices is None:
            self.sin_lat.fill(0.0)
            self.cos_lat.fill(1.0)
        else:
            self.sin_lat[indices] = 0.0
            self.cos_lat[indices] = 1.0

    def nan(self, indices=None):
        """
        Set all coordinates to NaN.

        Parameters
        ----------
        indices : slice or numpy.ndarray (int or bool), optional
            The indices to set to NaN.

        Returns
        -------
        None
        """
        super().nan(indices=indices)
        if self.sin_lat is None:
            return
        if indices is None:
            self.sin_lat.fill(np.nan)
            self.cos_lat.fill(np.nan)
        else:
            self.sin_lat[indices] = np.nan
            self.cos_lat[indices] = np.nan

    def set(self, coordinates, copy=True):
        """
        Set the (LON, LAT) coordinates.

        Parameters
        ----------
        coordinates : astropy.units.Quantity (float or numpy.ndarray)
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.set_longitude(coordinates[0], copy=copy)
        self.set_latitude(coordinates[1], copy=copy)

    def set_native(self, coordinates, copy=True):
        """
        Set the native (x, y) coordinates.

        Parameters
        ----------
        coordinates : Coordinates2D or astropy.units.Quantity (numpy.ndarray)
            The native (x, y) coordinates to set.  If an array is provided,
            should be of shape (2,) or (2, n).
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise, reference the
            coordinates if possible.

        Returns
        -------
        None
        """
        if isinstance(coordinates, Coordinate2D):
            self.set_native_longitude(coordinates.x, copy=copy)
            self.set_native_latitude(coordinates.y, copy=copy)
        else:
            self.set_native_longitude(coordinates[0], copy=copy)
            self.set_native_latitude(coordinates[1], copy=copy)

    def set_native_longitude(self, longitude, copy=True):
        """
        Set the native longitude coordinates.

        Parameters
        ----------
        longitude : astropy.units.Quantity (numpy.ndarray or float)
            The native longitude coordinates to update.
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.set_x(longitude, copy=copy)

    def set_native_latitude(self, latitude, copy=True):
        """
        Set the native latitude coordinates.

        Parameters
        ----------
        latitude : astropy.units.Quantity (numpy.ndarray or float)
            The native latitude coordinates to update.
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.set_y(latitude, copy=copy)

    def set_longitude(self, longitude, copy=True):
        """
        Set the longitude coordinates.

        Parameters
        ----------
        longitude : astropy.units.Quantity (numpy.ndarray or float)
            The longitude coordinates to update.
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        if self.reverse_longitude:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if not isinstance(longitude, units.Quantity):
                    longitude = longitude * self.unit
                native_longitude = self.longitude_axis.reverse_from - longitude
            copy = False  # no need to copy
        else:
            native_longitude = longitude
        self.set_native_longitude(native_longitude, copy=copy)

    def set_latitude(self, latitude, copy=True):
        """
        Set the latitude coordinates.

        Parameters
        ----------
        latitude : astropy.units.Quantity (numpy.ndarray or float)
            The latitude coordinates to update.
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        if not self.reverse_latitude:
            native_latitude = latitude
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if not isinstance(latitude, units.Quantity):
                    latitude = latitude * self.unit
                native_latitude = self.latitude_axis.reverse_from - latitude
            copy = False  # no need to copy

        self.set_native_latitude(native_latitude, copy=copy)

    def project(self, projection, output_system):
        """
        Project the coordinates onto a new frame.

        Parameters
        ----------
        projection : Projection
        output_system : Coordinate2D
            A 2-dimensional output coordinates system.

        Returns
        -------
        None
        """
        projection.project(self, output_system)

    def set_projected(self, projection, offset_system):
        """
        Project offsets onto this system.

        Parameters
        ----------
        projection : Projection
        offset_system : Coordinate2D
            A 2-dimensional system of offsets.

        Returns
        -------
        None
        """
        projection.deproject(offset_system, self)

    def get_projected(self, projection):
        """
        Get the projection of this system.

        Parameters
        ----------
        projection : Projection

        Returns
        -------
        Coordinate2D
        """
        return projection.get_projected(self)

    def add_native_offset(self, offset):
        """
        Add a native offset to the native coordinates.

        Parameters
        ----------
        offset : Coordinate2D
            The (x, y) offsets to add.

        Returns
        -------
        None
        """
        self.add_x(offset.x / self.cos_lat)
        self.add_y(offset.y)

    def add_offset(self, offset):
        """
        Add spherical offsets to native coordinates.

        Parameters
        ----------
        offset : Coordinate2D
            The spherical (x, y) offsets to add.

        Returns
        -------
        None
        """
        dx = offset.x / self.cos_lat
        dy = offset.y
        self.subtract_x(dx) if self.reverse_longitude else self.add_x(dx)
        self.subtract_y(dy) if self.reverse_latitude else self.add_y(dy)

    def subtract_native_offset(self, offset):
        """
        Subtract native offsets from the native coordinates.

        Parameters
        ----------
        offset : Coordinate2D
            The native (x, y) offsets to subtract.

        Returns
        -------
        None
        """
        self.subtract_x(offset.x / self.cos_lat)
        self.subtract_y(offset.y)

    def subtract_offset(self, offset):
        """
        Subtract spherical offsets from the native coordinates.

        Parameters
        ----------
        offset : Coordinate2D
            The spherical (x, y) offsets to subtract.

        Returns
        -------
        None
        """
        dx = offset.x / self.cos_lat
        dy = offset.y
        if self.reverse_longitude:
            self.add_x(dx)
        else:
            self.subtract_x(dx)

        # TODO: In case things fails, I reversed this.
        if self.reverse_latitude:
            self.add_y(dy)
        else:
            self.subtract_y(dy)

    def get_native_offset_from(self, reference, offset=None):
        """
        Get the native offset from a reference system.

        Parameters
        ----------
        reference : SphericalCoordinates
            The native reference position(s).
        offset : astropy.units.Quantity, optional
            A work array to fill and return as the result.  If not supplied,
            will be an array the same shape as coordinates or reference
            (whichever is larger).

        Returns
        -------
        Coordinate2D
        """
        if offset is None:
            offset = Coordinate2D(unit=self.offset_unit)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            dx = np.fmod(self.x - reference.x, self.two_pi) * reference.cos_lat
            dy = self.y - reference.y

        offset.x = dx
        offset.y = dy
        return offset

    def get_offset_from(self, reference, offset=None):
        """
        Get the spherical offset from a native reference position.

        Parameters
        ----------
        reference : SphericalCoordinates
            The reference position(s).
        offset : astropy.units.Quantity, optional
            A work array to fill and return as the result.  If not supplied,
            will be an array the same shape as coordinates or reference
            (whichever is larger).

        Returns
        -------
        spherical_offset : Coordinate2D
        """
        offset = self.get_native_offset_from(reference, offset=offset)
        if self.reverse_longitude:
            offset.coordinates[0] *= -1  # Note base coordinate scaling.
        if self.reverse_latitude:
            offset.coordinates[1] *= -1
        return offset

    def standardize(self):
        """
        Get all the coordinates within the correct angular range.

        Sets the range of x-coordinates to be in the range
        -360 < x < 360 degrees, and y-coordinates to the range
        -180 < y < 180 degrees.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.set_x(np.fmod(self.x, self.two_pi), copy=False)
            self.set_y(np.fmod(self.y, self.pi), copy=False)

    def distance_to(self, reference):
        r"""
        Return the distance from these coordinates to a given reference.

        Calculates the distance between two spherical sets of coordinates using
        either the law of cosines or Vincenty's formulae.  First we calculate c
        as::

          c = sin(y) * sin(ry) + cos(y) * phi

        where::

          phi = cos(ry) * cos(rx - x)

        and x, rx are the longitudinal coordinates or the coordinates and
        reference coordinates respectively, and (y, ry) are the latitudinal
        coordinates.

        if \|c\| > 0.9 (indicating intermediate distances), the law of
        cosines is used to return an angle (a) of::

          a = acos(c)

        Otherwise, Vincenty's formula is used to return a value of::

          a = atan2(B, c)

        where::

          B = sqrt((cos(ry) * sin(rx - x))^2 +
                   (cos(y) * sin(ry) - sin(y) * phi)^2)

        Parameters
        ----------
        reference : SphericalCoordinates

        Returns
        -------
        astropy.units.Quantity
            The angular separation from `coordinates` to `point` as an array of
            shape (n,) in units of `coordinates`.
        """
        d = csnf.spherical_distance_to(
            x=np.atleast_1d(self.x.to('radian').value),
            rx=np.atleast_1d(reference.x.to('radian').value),
            cos_lat=np.atleast_1d(self.cos_lat),
            sin_lat=np.atleast_1d(self.sin_lat),
            r_cos_lat=np.atleast_1d(reference.cos_lat),
            r_sin_lat=np.atleast_1d(reference.sin_lat))

        if self.singular and reference.singular:
            d = d.flat[0]

        return (d * units.Unit('radian')).to(self.unit)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with spherical coordinate information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to modify.
        key_stem : str
            The name of the header key to update.
        alt : str, optional
            The alternative coordinate system.

        Returns
        -------
        None
        """
        if not self.singular:
            return  # Can't do this for multiple coordinates

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lon = np.fmod(self.longitude, self.two_pi)
        lat = self.latitude
        if lon < 0:
            lon += self.two_pi

        header[f'{key_stem}1{alt}'] = (
            lon.to('degree').value,
            "The reference longitude coordinate (deg).")

        header[f'{key_stem}2{alt}'] = (
            lat.to('degree').value,
            "The reference latitude coordinate (deg).")

        if alt != '':
            header['WCSAXES'] = 2, 'Number of celestial coordinate axes.'

    def parse_header(self, header, key_stem, alt='', default=None):
        """
        Set the coordinate from the header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to read.
        key_stem : str
        alt : str, optional
            The alternate coordinate system.
        default : Coordinate2D or units.Quantity or SphericalCoordinates
            The (x, y) default coordinate.

        Returns
        -------
        None
        """
        if default is None:
            default_lon = 0 * self.unit
            default_lat = 0 * self.unit
        elif isinstance(default, SphericalCoordinates):
            default_lon = default.lon
            default_lat = default.lat
        elif isinstance(default, Coordinate2D):
            default_lon = default.x
            default_lat = default.y
        else:
            default_lon, default_lat = default

        deg = units.Unit('degree')

        longitude_key = f'{key_stem}1{alt}'
        if longitude_key in header:
            lon = header[longitude_key]
            unit = get_comment_unit(header.comments[longitude_key],
                                    default=deg)
            lon = lon * unit
        else:
            lon = default_lon

        latitude_key = f'{key_stem}2{alt}'
        if latitude_key in header:
            lat = header[latitude_key]
            unit = get_comment_unit(header.comments[latitude_key],
                                    default=deg)
            lat = lat * unit
        else:
            lat = default_lat

        self.set_longitude(lon, copy=False)
        self.set_latitude(lat, copy=False)

    def invert_y(self):
        """
        Scale the y-coordinates by -1.

        Returns
        -------
        None
        """
        super().invert_y()
        self.sin_lat *= -1

    @classmethod
    def equal_angles(cls, angle1, angle2):
        """
        Check whether angles are equal.

        Parameters
        ----------
        angle1 : astropy.units.Quantity
        angle2 : astropy.units.Quantity

        Returns
        -------
        bool or numpy.ndarray (bool)
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.abs(np.fmod(angle1 - angle2, cls.two_pi)
                          ) < cls.angular_accuracy

    def transform(self, pole, phi0, from_coordinates=None, to_coordinates=None,
                  reverse=False):
        """
        Transform spherical coordinates to a new pole.

        Parameters
        ----------
        pole : SphericalCoordinates
            The pole to transform to (or from if `reverse`=`True`).
        phi0 : astropy.units.Quantity
            The angle phi0.
        from_coordinates : SphericalCoordinates, optional
            The coordinates to transform from.  If not supplied, will be
            *these* coordinates.
        to_coordinates : SphericalCoordinates, optional
            The output coordinate system to transform to.  The default is
            *this* system.
        reverse : bool, optional
            If `True`, perform the inverse transform (from the pole rather than
            to the pole).

        Returns
        -------
        output : SphericalCoordinates
        """
        if from_coordinates is None:
            from_coordinates = self

        if to_coordinates is None:
            to_coordinates = self.__class__()

        transformed = csnf.spherical_pole_transform(
            x=np.atleast_1d(from_coordinates.x.to('radian').value),
            px=np.atleast_1d(pole.x.to('radian').value),
            cos_lat=np.atleast_1d(from_coordinates.cos_lat),
            sin_lat=np.atleast_1d(from_coordinates.sin_lat),
            p_cos_lat=np.atleast_1d(pole.cos_lat),
            p_sin_lat=np.atleast_1d(pole.sin_lat),
            phi0=phi0.to('radian').value,
            reverse=reverse) * units.Unit('radian')

        # Reconstruct shape
        singular = False
        if from_coordinates.singular and not pole.singular:
            new_shape = pole.shape
        elif not from_coordinates.singular and pole.singular:
            new_shape = from_coordinates.shape
        elif from_coordinates.singular and pole.singular:
            singular = True
            new_shape = ()
        else:
            new_shape = transformed.shape[1:]  # Too complex - leave flat

        if not singular and transformed.shape[1:] != new_shape:
            real_shape = (from_coordinates.ndim,) + new_shape
            new = np.empty(real_shape, dtype=float) * units.Unit('radian')
            for dimension in range(from_coordinates.ndim):
                new[dimension].flat = transformed[dimension]
            transformed = new
        elif singular:
            transformed = transformed[:, 0]

        to_coordinates.set_native(transformed)
        return to_coordinates

    def inverse_transform(self, pole, phi0, from_coordinates=None,
                          to_coordinates=None):
        """
        Inversely transform spherical coordinates from a pole.

        Parameters
        ----------
        pole : SphericalCoordinates
            The pole from which to transform.
        phi0 : astropy.units.Quantity
            The angle phi0.
        from_coordinates : SphericalCoordinates, optional
            The coordinates to transform from.  If not supplied, will be
            *these* coordinates.
        to_coordinates : SphericalCoordinates, optional
            The output coordinate system to transform to on output.
            The default is *this* system.

        Returns
        -------
        output : astropy.units.Quantity (numpy.ndarray)
            The transformed `coordinates` of shape (2, n) or (2,).
        """
        if from_coordinates is None:
            from_coordinates = self

        return self.transform(pole, phi0, from_coordinates=from_coordinates,
                              to_coordinates=to_coordinates, reverse=True)

    @classmethod
    def zero_to_two_pi(cls, values):
        """
        Return angles in the range 0 -> 2pi.

        Parameters
        ----------
        values : astropy.units.Quantity

        Returns
        -------
        astropy.units.Quantity
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return values % cls.two_pi

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        FlaggedData
        """
        new = super().get_indices(indices)
        if new.coordinates is None:
            return new

        if isinstance(indices, np.ndarray) and indices.shape == ():
            indices = int(indices)

        new.cos_lat = self.cos_lat[indices]
        new.sin_lat = self.sin_lat[indices]
        return new

    def insert_blanks(self, insert_indices):
        """
        Insert blank (NaN) values at the requested indices.

        Follows the logic of :func:`numpy.insert`.

        Parameters
        ----------
        insert_indices : numpy.ndarray (int)

        Returns
        -------
        None
        """
        super().insert_blanks(insert_indices)
        self.cos_lat = np.insert(self.cos_lat, insert_indices, np.nan)
        self.sin_lat = np.insert(self.sin_lat, insert_indices, np.nan)

    def merge(self, other):
        """
        Append other coordinates to the end of these.

        Parameters
        ----------
        other : Coordinate2D

        Returns
        -------
        None
        """
        singular = self.singular
        super().merge(other)
        if singular:
            self.cos_lat = np.atleast_1d(self.cos_lat)
            self.sin_lat = np.atleast_1d(self.sin_lat)

        if not isinstance(other, SphericalCoordinates):
            other = SphericalCoordinates(other)
            other_cos_lat = np.atleast_1d(other.cos_lat)
            other_sin_lat = np.atleast_1d(other.sin_lat)
        else:
            other_cos_lat = np.atleast_1d(other.cos_lat)
            other_sin_lat = np.atleast_1d(other.sin_lat)

        self.cos_lat = np.concatenate((self.cos_lat, other_cos_lat))
        self.sin_lat = np.concatenate((self.sin_lat, other_sin_lat))

    def paste(self, coordinates, indices):
        """
        Paste new coordinate values at the given indices.

        Parameters
        ----------
        coordinates : Coordinate2D
        indices : numpy.ndarray (int)

        Returns
        -------
        None
        """
        super().paste(coordinates, indices)
        if isinstance(coordinates, SphericalCoordinates):
            self.cos_lat[indices] = coordinates.cos_lat
            self.sin_lat[indices] = coordinates.sin_lat
        else:
            lat = coordinates.coordinates[1]
            self.cos_lat[indices] = np.cos(lat)
            self.sin_lat[indices] = np.sin(lat)

    def shift(self, n, fill_value=np.nan):
        """
        Shift the coordinates by a given number of elements.

        Parameters
        ----------
        n : int
        fill_value : float or int or units.Quantity, optional

        Returns
        -------
        None
        """
        if self.singular:
            return  # Can't roll for singular coordinates

        super().shift(n, fill_value=fill_value)
        # Reset the cos_lat, sin_lat attributes
        self.set_x(self.coordinates[0], copy=False)
        self.set_y(self.coordinates[1], copy=False)

    def plot(self, *args, **kwargs):  # pragma: no cover
        """
        Plot the coordinates.

        Parameters
        ----------
        args : values
            Optional positional parameters to pass into pyplot.plot.
        kwargs : dict, optional
            Optional keyword arguments.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        plt.ion()
        c_args = self.coordinates[0].ravel(), self.coordinates[1].ravel()
        if args is not None:
            c_args += args

        plt.plot(*c_args, **kwargs)
        plt.xlabel(f'{self.longitude_axis.label} ({self.unit})')
        plt.ylabel(f'{self.latitude_axis.label} ({self.unit})')
