# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate import Coordinate
from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D

__all__ = ['Coordinate2D1']


class Coordinate2D1(ABC):

    default_dimensions = 2  # standard dimensions (not including extra one)

    def __init__(self, xy=None, z=None, xy_unit=None, z_unit=None, copy=True):
        """
        Initialize 2-D coordinates repeated over an orthogonal axis.

        Parameters
        ----------
        xy : iterable or units.Quantity or Coordinate2D, optional
            The (x, y) coordinates.
        z : array-like or units.Quantity or Coordinate1D, optional
            The coordinates along the z-direction
        xy_unit : str or units.Unit or units.Quantity, optional
            The units of the (x, y) coordinates.
        z_unit : str or units.Unit or units.Quantity, optional
            The units of the z-direction coordinates
        copy : bool, optional
            If `True`, populate these coordinates with a copy of `coordinates`
            rather than the actual coordinates.
        """
        self.xy_coordinates = None
        self.z_coordinates = None

        if isinstance(xy, Coordinate2D1) and z is None:
            self.xy_coordinates = xy.xy_coordinates
            self.z_coordinates = xy.z_coordinates
        elif isinstance(xy, Coordinate3D) and z is None:
            self.xy_coordinates = Coordinate2D([xy.x, xy.y], unit=xy_unit)
            self.z_coordinates = Coordinate1D(xy.z, unit=z_unit)
        elif z is None and hasattr(xy, '__len__') and len(xy) == 3:
            self.xy_coordinates = Coordinate2D(xy[:2], unit=xy_unit)
            self.z_coordinates = Coordinate1D(xy[2], unit=z_unit)
        else:
            if isinstance(xy, Coordinate2D):
                self.xy_coordinates = xy
            else:
                self.xy_coordinates = Coordinate2D(
                    coordinates=xy, unit=xy_unit)

            if isinstance(z, Coordinate1D):
                self.z_coordinates = z
            else:
                self.z_coordinates = Coordinate1D(coordinates=z, unit=z_unit)

        if xy_unit is not None:
            self.xy_coordinates.change_unit(xy_unit)
        if z_unit is not None:
            self.z_coordinates.change_unit(z_unit)

        if copy:
            self.xy_coordinates = self.xy_coordinates.copy()
            self.z_coordinates = self.z_coordinates.copy()

    def __eq__(self, other):
        """
        Check if this Coordinate2D1 is equal to another.

        Parameters
        ----------
        other : Coordinate2D1

        Returns
        -------
        equal : bool
        """
        if other.__class__ != self.__class__:
            return False
        if self.z_coordinates != other.z_coordinates:
            return False
        if self.xy_coordinates != other.xy_coordinates:
            return False
        return True

    def empty_copy(self):
        """
        Return an unpopulated instance of the coordinates.

        Returns
        -------
        Coordinate2D1
        """
        new = Coordinate2D1()
        new.xy_coordinates = self.xy_coordinates.empty_copy()
        new.z_coordinates = self.z_coordinates.empty_copy()
        return new

    def copy(self):
        """
        Return a copy of the Coordinate2D.

        Returns
        -------
        Coordinate2D1
        """
        new = Coordinate2D1()
        new.xy_coordinates = self.xy_coordinates.copy()
        new.z_coordinates = self.z_coordinates.copy()
        return new

    @property
    def ndim(self):
        """
        Return the number of standard dimensions in the coordinates.

        Returns
        -------
        dimensions : tuple (int)
        """
        return 2, 1

    @property
    def x(self):
        """
        Return the x coordinates.

        Returns
        -------
        int or float or numpy.ndarray or units.Quantity
        """
        return self.xy_coordinates.x

    @x.setter
    def x(self, value):
        """
        Set the x coordinates.

        Parameters
        ----------
        value : int or float or numpy.ndarray or units.Quantity

        Returns
        -------
        None
        """
        self.set_x(value)

    @property
    def y(self):
        """
        Return the y coordinates.

        Returns
        -------
        int or float or numpy.ndarray or units.Quantity
        """
        return self.xy_coordinates.y

    @y.setter
    def y(self, value):
        """
        Set the y coordinates.

        Parameters
        ----------
        value : int or float or numpy.ndarray or units.Quantity

        Returns
        -------
        None
        """
        self.set_y(value)

    @property
    def z(self):
        """
        Return the z coordinates.

        Returns
        -------
        int or float or numpy.ndarray or units.Quantity
        """
        return self.z_coordinates.x

    @z.setter
    def z(self, value):
        """
        Set the z coordinates.

        Parameters
        ----------
        value : int or float or numpy.ndarray or units.Quantity

        Returns
        -------
        None
        """
        self.set_z(value)

    @property
    def xy_unit(self):
        """
        Return the unit for the x-coordinate.

        Returns
        -------
        units.Unit
        """
        return self.xy_coordinates.unit

    @xy_unit.setter
    def xy_unit(self, value):
        """
        Set the x-coordinate unit.

        Parameters
        ----------
        value : units.Unit

        Returns
        -------
        None
        """
        if self.xy_coordinates.unit != value:
            self.xy_coordinates.change_unit(value)

    @property
    def z_unit(self):
        """
        Return the unit for the z-coordinate.

        Returns
        -------
        units.Unit
        """
        return self.z_coordinates.unit

    @z_unit.setter
    def z_unit(self, value):
        """
        Set the z-coordinate unit.

        Parameters
        ----------
        value : units.Unit

        Returns
        -------
        None
        """
        if self.z_coordinates.unit != value:
            self.z_coordinates.change_unit(value)

    @property
    def max(self):
        """
        Return the (x, y, z) maximum values.

        Returns
        -------
        Coordinate2D1
        """
        new = self.empty_copy()
        max_x = np.nanmax(self.x)
        max_y = np.nanmax(self.y)
        max_z = np.nanmax(self.z)
        new.set([max_x, max_y, max_z])
        return new

    @property
    def min(self):
        """
        Return the (x, y, z) minimum values.

        Returns
        -------
        Coordinate2D1
        """
        new = self.empty_copy()
        min_x = np.nanmin(self.x)
        min_y = np.nanmin(self.y)
        min_z = np.nanmin(self.z)
        new.set([min_x, min_y, min_z])
        return new

    @property
    def span(self):
        """
        Return the range of the coordinates

        Returns
        -------
        Coordinate2D1
        """
        c_range = self.max
        c_range.subtract(self.min)
        return c_range

    @property
    def length(self):
        """
        Return the distance of the xy coordinates from (0, 0).

        Returns
        -------
        distance : float or numpy.ndarray or astropy.units.Quantity
        """
        return np.hypot(self.x, self.y)

    @property
    def singular(self):
        """
        Return if the coordinates are scalar in nature (not an array).

        Returns
        -------
        bool
        """
        return self.xy_coordinates.singular and self.z_coordinates.singular

    def __str__(self):
        """
        Create a string representation of the coordinates.

        Returns
        -------
        str
        """
        if self.xy_coordinates.size == 0:
            s = 'x=Empty, y=Empty'
        else:
            s = str(self.xy_coordinates)

        if self.z_coordinates.size == 0:
            s += ', z=Empty'
        else:
            s += f', z{str(self.z_coordinates)[1:]}'
        return s

    def __repr__(self):
        """
        Return a string representation of the Coordinate.

        Returns
        -------
        str
        """
        return f'{self.__str__()} {object.__repr__(self)}'

    def __len__(self):
        """
        Return the number of represented coordinates.

        Returns
        -------
        int
        """
        return self.xy_coordinates.size * self.z_coordinates.size

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : tuple (slice or list or int or numpy.ndarray (int))
            The indices to extract.

        Returns
        -------
        Coordinate2D1
        """
        xy = self.xy_coordinates.get_indices(indices[0])
        z = self.z_coordinates.get_indices(indices[1])
        return Coordinate2D1(xy=xy, z=z)

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
        self.xy_coordinates.set_singular(empty=empty)
        self.z_coordinates.set_singular(empty=empty)

    def copy_coordinates(self, coordinates):
        """
        Copy the coordinates from another system to this system.

        Parameters
        ----------
        coordinates : Coordinate2D1

        Returns
        -------
        None
        """
        self.xy_coordinates.copy_coordinates(coordinates.xy_coordinates)
        self.z_coordinates.copy_coordinates(coordinates.z_coordinates)

    def set_x(self, coordinates, copy=True):
        """
        Set the x coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or int or float or units.Quantity
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.xy_coordinates.set_x(coordinates, copy=copy)

    def set_y(self, coordinates, copy=True):
        """
        Set the y coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or int or float or units.Quantity
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.xy_coordinates.set_y(coordinates, copy=copy)

    def set_z(self, coordinates, copy=True):
        """
        Set the z coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or int or float or units.Quantity
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.z_coordinates.set_x(coordinates, copy=copy)

    def set(self, coordinates, copy=True):
        """
        Set the (x, y, z) coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or iterable
            Must have a length of 3 for (x, y, z).
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        self.xy_coordinates.set(coordinates[:2], copy=copy)
        self.z_coordinates.set(coordinates[2], copy=copy)

    def add(self, coordinates):
        """
        Add other coordinates to these.

        Parameters
        ----------
        coordinates : Coordinate2D1

        Returns
        -------
        None
        """
        self.add_x(coordinates.x)
        self.add_y(coordinates.y)
        self.add_z(coordinates.z)

    def add_x(self, x):
        """
        Add z to coordinates.

        Parameters
        ----------
        x : float or numpy.ndarray
            The value(s) to add.

        Returns
        -------
        None
        """
        self.xy_coordinates.add_x(x)

    def add_y(self, y):
        """
        Add z to coordinates.

        Parameters
        ----------
        y : float or numpy.ndarray
            The value(s) to add.

        Returns
        -------
        None
        """
        self.xy_coordinates.add_y(y)

    def add_z(self, z):
        """
        Add z to coordinates.

        Parameters
        ----------
        z : float or numpy.ndarray
            The value(s) to add.

        Returns
        -------
        None
        """
        self.z_coordinates.add_x(z)

    def subtract(self, coordinates):
        """
        Subtract other coordinates from these.

        Parameters
        ----------
        coordinates : Coordinate2D1

        Returns
        -------
        None
        """
        self.subtract_x(coordinates.x)
        self.subtract_y(coordinates.y)
        self.subtract_z(coordinates.z)

    def subtract_x(self, x):
        """
        Subtract z from coordinates.

        Parameters
        ----------
        x : float or numpy.ndarray or astropy.units.Quantity
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.xy_coordinates.subtract_x(x)

    def subtract_y(self, y):
        """
        Subtract z from coordinates.

        Parameters
        ----------
        y : float or numpy.ndarray or astropy.units.Quantity
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.xy_coordinates.subtract_y(y)

    def subtract_z(self, z):
        """
        Subtract z from coordinates.

        Parameters
        ----------
        z : float or numpy.ndarray or astropy.units.Quantity
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.z_coordinates.subtract_x(z)

    def scale(self, factor):
        """
        Scale the coordinates by a factor.

        Parameters
        ----------
        factor : int or float or Coordinate2D1 or Coordinate3D or iterable

        Returns
        -------
        None
        """
        if isinstance(factor, (Coordinate2D1, Coordinate3D)):
            self.scale_x(factor.x)
            self.scale_y(factor.y)
            self.scale_z(factor.z)
        elif hasattr(factor, '__len__') and len(factor) == 3:
            self.scale_x(factor[0])
            self.scale_y(factor[1])
            self.scale_z(factor[2])
        else:
            self.scale_x(factor)
            self.scale_y(factor)
            self.scale_z(factor)

    def scale_x(self, factor):
        """
        Scale the x coordinates by a factor.

        Parameters
        ----------
        factor : float or units.Quantity
            The factor by which to scale the x-coordinates.

        Returns
        -------
        None
        """
        self.xy_coordinates.scale_x(factor)

    def scale_y(self, factor):
        """
        Scale the y coordinates by a factor.

        Parameters
        ----------
        factor : float
            The factor by which to scale the y-coordinates.

        Returns
        -------
        None
        """
        self.xy_coordinates.scale_y(factor)

    def scale_z(self, factor):
        """
        Scale the z coordinates by a factor.

        Parameters
        ----------
        factor : float or units.Quantity
            The factor by which to scale the z-coordinates.

        Returns
        -------
        None
        """
        self.z_coordinates.scale_x(factor)

    def invert(self):
        """
        Scale the (x, y, z) coordinates by -1.

        Returns
        -------
        None
        """
        self.invert_x()
        self.invert_y()
        self.invert_z()

    def invert_x(self):
        """
        Scale the x-coordinates by -1.

        Returns
        -------
        None
        """
        self.xy_coordinates.invert_x()

    def invert_y(self):
        """
        Scale the y-coordinates by -1.

        Returns
        -------
        None
        """
        self.xy_coordinates.invert_y()

    def invert_z(self):
        """
        Scale the z-coordinates by -1.

        Returns
        -------
        None
        """
        self.z_coordinates.invert_x()

    def parse_header(self, header, key_stem, alt='', default=None):
        """
        Parse a header and apply for the desired stem.

        Parameters
        ----------
        header : fits.Header
        key_stem : str
        alt : str, optional
        default : Coordinate2D1 or Coordinate2D or numpy.ndarray, optional

        Returns
        -------
        None
        """
        self.xy_coordinates.parse_header(header, key_stem, alt=alt,
                                         default=default)
        self.z_coordinates.parse_header(header, key_stem, alt=alt,
                                        default=default, dimension=3)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit a FITS header with the coordinate information.

        Parameters
        ----------
        header : fits.Header
        key_stem : str
            The name of the coordinate in the FITS header.  The name of the
            x coordinate in the header will be {key_stem}1, and y will be
            {key_stem}2.
        alt : str, optional
            The alternate FITS header system.

        Returns
        -------
        None
        """
        self.xy_coordinates.edit_header(header, key_stem, alt=alt)
        self.z_coordinates.edit_header(header, key_stem, alt=alt, dimension=3)

    def broadcast_to(self, thing):
        """
        Broadcast to a new shape if possible.

        If the coordinates are singular (a single coordinate value in each
        dimension), broadcasts that single value to a new shape in the internal
        coordinates.

        Parameters
        ----------
        thing : numpy.ndarray or tuple (int)
            An array from which to determine the broadcast shape, or a tuple
            containing the broadcast shape.

        Returns
        -------
        None
        """
        self.xy_coordinates.broadcast_to(thing)
        self.z_coordinates.broadcast_to(thing)

    @staticmethod
    def convert_indices(indices):
        """
        Convert the supplied indices to separate xy and z indices.

        Parameters
        ----------
        indices : None or slice or numpy.ndarray or tuple or list

        Returns
        -------
        xy_indices, z_indices
        """
        if indices is None:
            return None, None

        if isinstance(indices, (np.ndarray, slice)):
            return indices, None  # Assume not for the z-direction

        if len(indices) == 2:
            return indices[0], indices[1]

        else:
            raise ValueError("Could not convert indices for 2D1.")

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
        xy, z = self.convert_indices(indices)
        self.xy_coordinates.nan(xy)
        self.z_coordinates.nan(z)

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
        xy, z = self.convert_indices(indices)
        self.xy_coordinates.zero(xy)
        self.z_coordinates.zero(z)

    def is_nan(self):
        """
        Check whether coordinates are NaN.

        Returns
        -------
        xy_nan, z_nan : 2-tuple (bool or numpy.ndarray)
        """
        return self.xy_coordinates.is_nan(), self.z_coordinates.is_nan()

    def is_null(self):
        """
        Check whether coordinates are zero.

        Returns
        -------
        xy_null, z_null : 2-tuple (bool or numpy.ndarray)
        """
        return self.xy_coordinates.is_null(), self.z_coordinates.is_null()

    def is_finite(self):
        """
        Check whether coordinates are finite.

        Returns
        -------
        xy_finite, z_finite : 2-tuple (bool or numpy.ndarray)
        """
        return self.xy_coordinates.is_finite(), self.z_coordinates.is_finite()

    def is_infinite(self):
        """
        Check whether coordinates are infinite.

        Returns
        -------
        xy_finite, z_finite : 2-tuple (bool or numpy.ndarray)
        """
        return (self.xy_coordinates.is_infinite(),
                self.z_coordinates.is_infinite())

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
        xy, z = self.convert_indices(insert_indices)
        self.xy_coordinates.insert_blanks(xy)
        self.z_coordinates.insert_blanks(z)

    def merge(self, other):
        """
        Append other coordinates to the end of these.

        Parameters
        ----------
        other : Coordinate2D1

        Returns
        -------
        None
        """
        self.xy_coordinates.merge(other.xy_coordinates)
        self.z_coordinates.merge(other.z_coordinates)

    def paste(self, coordinates, indices):
        """
        Paste new coordinate values at the given indices.

        Parameters
        ----------
        coordinates : Coordinate2D1
        indices : numpy.ndarray (int)

        Returns
        -------
        None
        """
        xy, z = self.convert_indices(indices)
        self.xy_coordinates.paste(coordinates.xy_coordinates, xy)
        self.z_coordinates.paste(coordinates.z_coordinates, z)

    def shift(self, n, fill_value=np.nan):
        """
        Shift the coordinates by a given number of elements.

        Parameters
        ----------
        n : Iterable (int)
            The shift values for the (xy, z) coordinates respectively
        fill_value : float or int or units.Quantity, optional

        Returns
        -------
        None
        """
        self.xy_coordinates.shift(n[0], fill_value=fill_value)
        self.z_coordinates.shift(n[1], fill_value=fill_value)

    def mean(self):
        """
        Return the mean coordinates.

        Returns
        -------
        mean_coordinates : Coordinate2D1
        """
        return Coordinate2D1(xy=self.xy_coordinates.mean(),
                             z=self.z_coordinates.mean())

    def to_coordinate_3d(self):
        """
        Convert the 2D+1 coordinates to full 3D coordinates.

        Coordinates will be flattened and cannot maintain the current shape.
        The output shape will be (nx * nz,).  Can only occur when the z and
        xy units are equivalent.

        Returns
        -------
        Coordinate3D
        """
        if self.x is None or self.z is None:
            raise ValueError("Can only convert populated coordinates")
        if self.xy_unit != self.z_unit:
            raise ValueError("Cannot convert to 3D coordinates when xy and z "
                             "units are not convertable.")

        x, y, z = self.x, self.y, self.z
        if self.singular:
            return Coordinate3D([x, y, z])

        nx, nz, unit = x.size, z.size, self.xy_unit
        if unit is not None:
            x = x.ravel()
            y = y.ravel()
            z = z.to(unit).ravel()
        else:
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

        z2 = np.empty(nx * nz, dtype=float)
        if unit is not None:
            z2 = z2 * unit
        for i in range(nz):
            z2[i * nx:(i + 1) * nx] = z[i]

        return Coordinate3D(
            [np.hstack([x.copy() for _ in range(nz)]),
             np.hstack([y.copy() for _ in range(nz)]),
             z2]
        )
