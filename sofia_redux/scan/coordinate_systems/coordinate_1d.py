# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.coordinate import Coordinate

__all__ = ['Coordinate1D']


class Coordinate1D(Coordinate):
    """
    The Coordinate1D is a specialized extension of the `Coordinate` class
    for 1-dimensional data
    """
    default_dimensions = 1  # Used when setting up empty templates

    def empty_copy(self):
        """
        Return an unpopulated instance of the coordinates.

        Returns
        -------
        Coordinate1D
        """
        return super().empty_copy()

    def copy(self):
        """
        Return a copy of the Coordinate2D.

        Returns
        -------
        Coordinate1D
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
        return 1

    @property
    def size(self):
        """
        Return the number of coordinates.

        Returns
        -------
        int
        """
        if self.coordinates is None:
            return 0
        elif self.singular:
            return 1
        else:
            return self.coordinates.size

    @property
    def shape(self):
        """
        Return the shape of the coordinates.

        Returns
        -------
        tuple (int)
        """
        if self.coordinates is None or self.singular:
            return ()
        return self.coordinates.shape

    @property
    def x(self):
        """
        Return the x coordinate.

        Returns
        -------
        float or units.Quantity or numpy.ndarray (float)
        """
        if self.coordinates is None:
            return None
        return self.coordinates

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
    def max(self):
        """
        Return the (x, y) maximum values.

        Returns
        -------
        Coordinate2D
        """
        new = self.empty_copy()
        max_x = np.nanmax(self.x)
        new.set(max_x)
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
        min_x = np.nanmin(self.x)
        new.set(min_x)
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
        Return the distance of the coordinate from 0.

        Returns
        -------
        distance : float or numpy.ndarray or astropy.units.Quantity
        """
        return np.abs(self.x)

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
        if isinstance(self.coordinates, np.ndarray):
            return self.coordinates.shape == ()
        return True

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
            return f'x={self.x}'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return f'x={np.nanmin(self.x)}->{np.nanmax(self.x)}'

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
        Coordinate1D
        """
        return super().__getitem__(indices)

    def set_shape(self, shape, empty=False):
        """
        Set the shape of the coordinates.

        If the current coordinates are blank, dimensionality will be inferred
        from the input shape.  If a single integer value or 1-tuple is passed
        in, the number of dimensions will be set to 1.  Otherwise, the
        dimensionality will be permanently set to the first element of `shape`.

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
        if self.coordinates is not None:
            self.coordinates = self.coordinates[0]

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : slice or list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        Coordinate
        """
        new = self.empty_copy()
        if self.coordinates is None:
            return new

        if self.singular:
            raise KeyError(
                "Cannot retrieve indices for singular coordinates.")

        if isinstance(indices, np.ndarray) and indices.shape == ():
            indices = int(indices)

        coordinates = self.coordinates[indices]
        new.coordinates = coordinates
        return new

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
        coordinates = 0.0
        if self.unit is not None:
            coordinates = coordinates * self.unit
        self.coordinates = coordinates

    def copy_coordinates(self, coordinates):
        """
        Copy the coordinates from another system to this system.

        Parameters
        ----------
        coordinates : Coordinate1D

        Returns
        -------
        None
        """
        if coordinates.coordinates is None:
            self.coordinates = None
        else:
            self.set_x(coordinates.x)

    def set_x(self, coordinates, copy=True):
        """
        Set the x coordinates.

        Parameters
        ----------
        coordinates : float or numpy.ndarray or None
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        if coordinates is None:
            return
        coordinates, original = self.check_coordinate_units(coordinates)
        self.set_shape_from_coordinates(coordinates, single_dimension=True)

        copy &= original
        if isinstance(coordinates, np.ndarray) and coordinates.dtype != float:
            coordinates = coordinates.astype(float)
            copy = False
        elif not isinstance(coordinates, np.ndarray):
            if np.asarray(coordinates).dtype != float:
                coordinates = float(coordinates)
                copy = False

        if copy and isinstance(coordinates, np.ndarray):
            self.coordinates = coordinates.copy()
        else:
            self.coordinates = coordinates

    def set(self, coordinates, copy=True):
        """
        Set the coordinates.

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
        self.set_x(coordinates, copy=copy)

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
        if not self.singular or self.coordinates is None:
            return

        if isinstance(thing, np.ndarray):
            shape = thing.shape
        elif isinstance(thing, tuple):
            shape = thing
        else:
            return

        if shape == ():
            return

        self.coordinates = np.full(shape, self.coordinates)

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
        self.coordinates += x

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
        self.coordinates -= x

    def scale(self, factor):
        """
        Scale the coordinates by a factor.

        Parameters
        ----------
        factor : int or float or numpy.ndarray or Coordinate1D

        Returns
        -------
        None
        """
        if not isinstance(factor, Coordinate1D):
            factor = self.convert_factor(factor)
            self.scale_x(factor)
        else:
            self.scale_x(factor.x)

    def scale_x(self, factor):
        """
        Scale the x coordinates by a factor.

        Parameters
        ----------
        factor : float or units.Quantity or numpy.ndarray
            The factor by which to scale the x-coordinates.

        Returns
        -------
        None
        """
        factor = self.convert_factor(factor)
        self.coordinates *= factor

    def invert_x(self):
        """
        Scale the x-coordinates by -1.

        Returns
        -------
        None
        """
        self.coordinates *= -1

    def invert(self):
        """
        Scale the x and y coordinates by -1.

        Returns
        -------
        None
        """
        self.invert_x()

    def add(self, coordinates):
        """
        Add other coordinates to these.

        Parameters
        ----------
        coordinates : Coordinate1D or int or float or units.Quantity

        Returns
        -------
        None
        """
        if isinstance(coordinates, Coordinate1D):
            self.add_x(coordinates.x)
        else:
            self.add_x(coordinates)

    def subtract(self, coordinates):
        """
        Subtract other coordinates from these.

        Parameters
        ----------
        coordinates : Coordinate1D or int or float or units.Quantity

        Returns
        -------
        None
        """
        if isinstance(coordinates, Coordinate1D):
            self.subtract_x(coordinates.x)
        else:
            self.subtract_x(coordinates)

    def parse_header(self, header, key_stem, alt='', default=None,
                     dimension=1):
        f"""
        Parse a header and return a Coordinate2D for the desired stem.

        Parameters
        ----------
        header : fits.Header
        key_stem : str
        alt : str, optional
        default : Coordinate1D or numpy.ndarray or int or float, optional
        dimension : int, optional
            By default, the extracted header key will be key_stem + dimension.
            This will select another dimension examine

        Returns
        -------
        None
        """
        if alt is None:
            alt = ''
        if isinstance(default, Coordinate1D):
            dx = default.x
        elif isinstance(default, (np.ndarray, int, float, units.Quantity)):
            dx = default
        else:
            dx = 0.0

        x = header.get(f'{key_stem}{dimension}{alt}', dx)
        if isinstance(x, int):
            x = float(x)
        self.x = x

    def edit_header(self, header, key_stem, alt='', dimension=1):
        """
        Edit a FITS header with the coordinate information.

        Parameters
        ----------
        header : fits.Header
        key_stem : str
            The name of the coordinate in the FITS header.  The name of the
            x coordinate in the header will be {key_stem}{dimension}.
        alt : str, optional
            The alternate FITS header system.
        dimension : int, optional
            The dimension for which the coordinate is relevant.

        Returns
        -------
        None
        """
        if not self.singular:
            return
        x = self.x
        if self.unit is None:
            unit_str = '.'
        else:
            if isinstance(x, units.Quantity):
                x = x.value
            unit_str = f' ({str(self.unit)}).'

        header[f'{key_stem}{dimension}{alt}'] = (
            x, f"The reference x coordinate{unit_str}")

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
        c_args = np.asarray(self.coordinates).ravel(),
        if args is not None:
            c_args += args

        plt.plot(*c_args, **kwargs)
        x_label = 'X'
        if self.unit is not None:
            x_label += f' ({self.unit})'
        plt.xlabel(x_label)

    def mean(self):
        """
        Return the mean coordinates.

        Returns
        -------
        mean_coordinates : Coordinate1D
        """
        new = super().mean()
        if isinstance(new.coordinates, np.ndarray) and new.shape == (1,):
            new.coordinates = new.coordinates[0]

        return new
