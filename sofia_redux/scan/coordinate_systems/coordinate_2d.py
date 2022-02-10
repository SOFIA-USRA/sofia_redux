# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.coordinate import Coordinate

__all__ = ['Coordinate2D']


class Coordinate2D(Coordinate):

    default_dimensions = 2  # Used when setting up empty templates

    def empty_copy(self):
        """
        Return an unpopulated instance of the coordinates.

        Returns
        -------
        Coordinate2D
        """
        return super().empty_copy()

    def copy(self):
        """
        Return a copy of the Coordinate2D.

        Returns
        -------
        Coordinate2D
        """
        return super().copy()

    @property
    def ndim(self):
        """
        Return the number of dimensions in the coordinate.

        Returns
        -------
        dimensions : int
        """
        return 2

    @property
    def x(self):
        """
        Return the x coordinate.

        Returns
        -------
        float or numpy.ndarray (float)
        """
        if self.coordinates is None:
            return None
        return self.coordinates[0]

    @x.setter
    def x(self, value):
        """
        Set the x coordinate.

        Parameters
        ----------
        value : float or numpy.ndarray

        Returns
        -------
        None
        """
        self.set_x(value, copy=True)

    @property
    def y(self):
        """
        Return the y coordinate.

        Returns
        -------
        float or numpy.ndarray (float)
        """
        if self.coordinates is None:
            return None
        return self.coordinates[1]

    @y.setter
    def y(self, value):
        """
        Set the y coordinate.

        Parameters
        ----------
        value : float or numpy.ndarray

        Returns
        -------
        None
        """
        self.set_y(value, copy=True)

    @property
    def max(self):
        """
        Return the (x, y) maximum values.

        Returns
        -------
        Coordinate2D
        """
        new = self.empty_copy()
        max_x = np.nanmax(self.x)
        max_y = np.nanmax(self.y)
        new.set([max_x, max_y])
        return new

    @property
    def min(self):
        """
        Return the (x, y) minimum values.

        Returns
        -------
        Coordinate2D
        """
        new = self.empty_copy()
        max_x = np.nanmin(self.x)
        max_y = np.nanmin(self.y)
        new.set([max_x, max_y])
        return new

    @property
    def span(self):
        """
        Return the range of x and y values.

        Returns
        -------
        Coordinate2D
        """
        c_range = self.max
        c_range.subtract(self.min)
        return c_range

    @property
    def length(self):
        """
        Return the distance of the coordinate from (0, 0).

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
        if self.coordinates is None:
            return True
        return self.coordinates.ndim == 1

    def __str__(self):
        """
        Create a string representation of the equatorial coordinates.

        Returns
        -------
        str
        """
        if self.coordinates is None:
            return 'Empty coordinates'

        if self.singular:
            return f'x={self.x} y={self.y}'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return (f'x={np.nanmin(self.x)}->{np.nanmax(self.x)} '
                        f'y={np.nanmin(self.y)}->{np.nanmax(self.y)}')

    def __repr__(self):
        """
        Return a string representation of the Coordinate.

        Returns
        -------
        str
        """
        return f'{self.__str__()} {object.__repr__(self)}'

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        Coordinate2D
        """
        return super().__getitem__(indices)

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        Coordinate2D
        """
        return super().get_indices(indices)

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
        if empty:
            coordinates = np.empty(self.ndim, dtype=float)
        else:
            coordinates = np.zeros(self.ndim, dtype=float)
        if self.unit is not None:
            coordinates = coordinates * self.unit
        self.coordinates = coordinates

    def copy_coordinates(self, coordinates):
        """
        Copy the coordinates from another system to this system.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        None
        """
        if coordinates.coordinates is None:
            self.coordinates = None
        else:
            self.set_x(coordinates.x)
            self.set_y(coordinates.y)

    def set_x(self, coordinates, copy=True):
        """
        Set the x coordinates.

        Parameters
        ----------
        coordinates : float or numpy.ndarray
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        coordinates, original = self.check_coordinate_units(coordinates)
        self.set_shape_from_coordinates(coordinates, single_dimension=True)
        copy &= original

        if copy and isinstance(coordinates, np.ndarray):
            self.coordinates[0] = coordinates.copy()
        else:
            self.coordinates[0] = coordinates

    def set_y(self, coordinates, copy=True):
        """
        Set the y coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        coordinates, original = self.check_coordinate_units(coordinates)
        self.set_shape_from_coordinates(coordinates, single_dimension=True)
        copy &= original

        if copy and isinstance(coordinates, np.ndarray):
            self.coordinates[1] = coordinates.copy()
        else:
            self.coordinates[1] = coordinates

    def set(self, coordinates, copy=True):
        """
        Set the (x, y) coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or list
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        coordinates, original = self.check_coordinate_units(coordinates)
        copy &= original
        self.set_shape_from_coordinates(coordinates, empty=True)
        self.set_x(coordinates[0], copy=copy)
        self.set_y(coordinates[1], copy=copy)

    def add_x(self, x):
        """
        Add x to coordinates.

        Parameters
        ----------
        x : float or numpy.ndarray
            The value(s) to add.

        Returns
        -------
        None
        """
        self.broadcast_to(x)
        self.coordinates[0] += x

    def subtract_x(self, x):
        """
        Subtract x from coordinates.

        Parameters
        ----------
        x : float or numpy.ndarray
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.broadcast_to(x)
        self.coordinates[0] -= x

    def add_y(self, y):
        """
        Add y to coordinates.

        Parameters
        ----------
        y : float or numpy.ndarray
            The value(s) to add.

        Returns
        -------
        None
        """
        self.broadcast_to(y)
        self.coordinates[1] += y

    def subtract_y(self, y):
        """
        Subtract y from coordinates.

        Parameters
        ----------
        y : float or numpy.ndarray or astropy.units.Quantity
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.broadcast_to(y)
        self.coordinates[1] -= y

    def scale(self, factor):
        """
        Scale the coordinates by a factor.

        Parameters
        ----------
        factor : int or float or Coordinate2D

        Returns
        -------
        None
        """
        if not isinstance(factor, Coordinate2D):
            factor = self.convert_factor(factor)
            self.scale_x(factor)
            self.scale_y(factor)
        else:
            self.scale_x(factor.x)
            self.scale_y(factor.y)

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
        factor = self.convert_factor(factor)
        self.coordinates[0] *= factor

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
        factor = self.convert_factor(factor)
        self.coordinates[1] *= factor

    def invert_x(self):
        """
        Scale the x-coordinates by -1.

        Returns
        -------
        None
        """
        self.coordinates[0] *= -1

    def invert_y(self):
        """
        Scale the y-coordinates by -1.

        Returns
        -------
        None
        """
        self.coordinates[1] *= -1

    def invert(self):
        """
        Scale the x and y coordinates by -1.

        Returns
        -------
        None
        """
        self.invert_x()
        self.invert_y()

    def add(self, coordinates):
        """
        Add other coordinates to these.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        None
        """
        self.add_x(coordinates.x)
        self.add_y(coordinates.y)

    def subtract(self, coordinates):
        """
        Subtract other coordinates from these.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        None
        """
        self.subtract_x(coordinates.x)
        self.subtract_y(coordinates.y)

    def rotate(self, angle):
        """
        Rotate the coordinates by a given angle.

        Internal coordinates are rotated anti-clockwise about zero.

        Parameters
        ----------
        angle : units.Quantity

        Returns
        -------
        None
        """
        self.rotate_offsets(self, angle)

    @classmethod
    def rotate_offsets(cls, offsets, angle):
        """
        Rotate zero-centered offsets in-place by an angle.

        Offsets are rotated anti-clockwise.

        Parameters
        ----------
        offsets : Coordinate2D or numpy.ndarray or units.Quantity or None
            The (x, y) offset coordinates to rotate.
        angle : astropy.units.Quantity or float
            The angle by which to rotate the offsets.  If a float value is
            provided, it should be in radians.

        Returns
        -------
        None
        """
        if isinstance(offsets, np.ndarray):
            if offsets.ndim > 1 and offsets.shape[0] > 0:
                angle = cls.correct_factor_dimensions(angle, offsets[0])
        elif isinstance(offsets, Coordinate2D):
            if offsets.coordinates is None:
                return
            if offsets.coordinates.ndim > 1 and offsets.coordinates.size > 1:
                angle = cls.correct_factor_dimensions(
                    angle, offsets.coordinates[0])

        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
        if isinstance(offsets, Coordinate2D):
            x, y = offsets.x.copy(), offsets.y
            offsets.x = (x * cos_a) - (y * sin_a)
            offsets.y = (x * sin_a) + (y * cos_a)
        else:
            x, y = offsets
            x = x.copy()
            offsets[0] = (x * cos_a) - (y * sin_a)
            offsets[1] = (x * sin_a) + (y * cos_a)

    def angle(self, center=None):
        """
        Return the angle between the x and y coordinates w.r.t a center.

        Parameters
        ----------
        center : Coordinate2D, optional
            The center about which to find an origin.

        Returns
        -------
        units.Quantity
        """
        if center is None:
            coordinates = self
        else:
            coordinates = self.copy()
            coordinates.subtract(center)

        angle = np.arctan2(coordinates.y, coordinates.x)
        if not isinstance(angle, units.Quantity):
            angle = angle * units.Unit('radian')
        return angle

    def parse_header(self, header, key_stem, alt='', default=None):
        """
        Parse a header and return a Coordinate2D for the desired stem.

        Parameters
        ----------
        header : fits.Header
        key_stem : str
        alt : str, optional
        default : Coordinate2D or numpy.ndarray, optional

        Returns
        -------
        None
        """
        if alt is None:
            alt = ''
        if isinstance(default, Coordinate2D):
            dx, dy = default.x, default.y
        elif isinstance(default, np.ndarray):
            dx, dy = default
        else:
            dx = dy = 0.0

        self.x = float(header.get(f'{key_stem}1{alt}', dx))
        self.y = float(header.get(f'{key_stem}2{alt}', dy))

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
        if not self.singular:
            return
        if self.unit is None:
            x, y = self.x, self.y
            unit_str = '.'
        else:
            x, y = self.x.value, self.y.value
            unit_str = f' ({str(self.unit)}).'

        header[f'{key_stem}1{alt}'] = (
            x, f"The reference x coordinate{unit_str}")
        header[f'{key_stem}2{alt}'] = (
            y, f"The reference y coordinate{unit_str}")

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
        if self.coordinates is None:
            return
        import matplotlib.pyplot as plt
        plt.ion()
        c_args = self.coordinates[0].ravel(), self.coordinates[1].ravel()
        if args is not None:
            c_args += args

        plt.plot(*c_args, **kwargs)
        x_label = 'X'
        y_label = 'Y'
        if self.unit is not None:
            x_label += f' ({self.unit})'
            y_label += f' ({self.unit})'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
