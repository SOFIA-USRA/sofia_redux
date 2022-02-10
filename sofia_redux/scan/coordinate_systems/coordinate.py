# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
from copy import deepcopy
import importlib
import numpy as np

from sofia_redux.scan.coordinate_systems import \
    coordinate_systems_numba_functions as csnf
from sofia_redux.scan.utilities.class_provider import (
    to_module_name, to_class_name)

__all__ = ['Coordinate']


class Coordinate(ABC):

    default_dimensions = 1

    def __init__(self, coordinates=None, unit=None, copy=True):
        """
        Initialize a coordinate.

        The Coordinate class is a generic container to store and operate on
        a set of coordinates.

        Parameters
        ----------
        coordinates : list or tuple or array-like or units.Quantity, optional
            The coordinates used to populate the object during initialization.
        unit : str or units.Unit, optional
            The units of the internal coordinates.
        copy : bool, optional
            If `True`, populate these coordinates with a copy of `coordinates`
            rather than the actual coordinates.
        """
        self.coordinates = None
        if unit is None:
            self.unit = None
        else:
            self.unit = units.Unit(unit)
        if coordinates is None:
            return

        coordinates, original = self.check_coordinate_units(coordinates)
        if not original:
            copy = False
        if isinstance(coordinates, Coordinate):
            self.copy_coordinates(coordinates)
        else:
            self.set(coordinates, copy=copy)

    def empty_copy(self):
        """
        Return an unpopulated instance of the coordinates.

        Returns
        -------
        Coordinate
        """
        new = self.__class__()
        for attribute, value in self.__dict__.items():
            if attribute not in self.empty_copy_skip_attributes:
                setattr(new, attribute, value)
            else:
                setattr(new, attribute, None)
        return new

    def copy(self):
        """
        Return a copy of the Coordinate.

        Returns
        -------
        Coordinate
        """
        return deepcopy(self)

    @property
    def empty_copy_skip_attributes(self):
        """
        Return attributes that are set to None on an empty copy.

        Returns
        -------
        attributes : set (str)
        """
        return {'coordinates'}

    @property
    def ndim(self):
        """
        Return the number of dimensions for the coordinate.

        Returns
        -------
        int
        """
        if self.coordinates is None:
            return 0
        elif self.coordinates.ndim == 1:
            return 1
        else:
            return self.coordinates.shape[0]

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
        return self.coordinates.shape[1:]

    @shape.setter
    def shape(self, new_shape):
        """
        Set a new shape for the coordinates

        Parameters
        ----------
        new_shape : int or tuple (int)

        Returns
        -------
        None
        """
        self.set_shape(new_shape)

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
            return int(np.prod(self.coordinates.shape[1:]))

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
        if self.coordinates.ndim == 1:
            return self.coordinates.size == 1
        if self.coordinates.ndim > 1:
            return np.prod(self.coordinates.shape[1:]) <= 1
        return True  # pragma: no cover

    @property
    def length(self):
        """
        Return the Euclidean distance of the coordinates from (0, 0).

        Returns
        -------
        distance : float or numpy.ndarray or astropy.units.Quantity
        """
        if self.ndim == 0:
            return np.nan if self.unit is None else np.nan * self.unit
        elif self.ndim == 1:
            return np.abs(self.coordinates)
        else:
            return np.linalg.norm(self.coordinates, axis=0)

    def __eq__(self, other):
        """
        Test if these coordinates are equal to another.

        Parameters
        ----------
        other : Coordinate

        Returns
        -------
        bool
        """
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
            return np.allclose(self.coordinates, other.coordinates,
                               equal_nan=True)
        except units.UnitConversionError:
            return False

    def __len__(self):
        """
        Return the number of stored coordinates.

        Returns
        -------
        int
        """
        return self.size

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        Coordinate
        """
        return self.get_indices(indices)

    def __setitem__(self, indices, value):
        """
        Set the coordinates for given indices.

        Parameters
        ----------
        indices : slice or int or numpy.ndarray (int or bool)
        value : Coordinate

        Returns
        -------
        None
        """
        self.paste(value, indices)

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
            raise KeyError("Cannot retrieve indices for singular coordinates.")

        if isinstance(indices, np.ndarray) and indices.shape == ():
            indices = int(indices)

        all_indices = slice(None),  # dimensions
        if not isinstance(indices, tuple):
            all_indices += indices,
        else:
            all_indices += indices

        coordinates = self.coordinates[all_indices]
        if new.ndim == 0 and self.ndim > 0 and coordinates.ndim <= 1:
            # For the case of base coordinates and multi-dimensional
            coordinates = coordinates[..., None]

        new.coordinates = coordinates
        return new

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
        if isinstance(shape, int):
            shape = shape,

        shape_dim = len(shape)

        if self.ndim == 0:
            # Dimensionality is inferred from this shape if not set.
            if shape_dim == 1:
                new_shape = (1,) + shape
            else:
                new_shape = shape
        else:
            new_shape = (self.ndim,) + shape

        if empty:
            self.coordinates = np.empty(new_shape, dtype=float)
        else:
            self.coordinates = np.zeros(new_shape, dtype=float)

        if self.unit is not None:
            self.coordinates = self.coordinates * self.unit

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
        if self.ndim <= 1:
            shape = (1,)
        else:
            shape = (self.ndim, 1)

        if empty:
            coordinates = np.empty(shape, dtype=float)
        else:
            coordinates = np.zeros(shape, dtype=float)

        if self.unit is not None:
            coordinates = coordinates * self.unit
        self.coordinates = coordinates

    def set_shape_from_coordinates(self, coordinates, single_dimension=False,
                                   empty=False):
        """
        Set the new shape and units from the given coordinates.

        The dimensionality of this coordinate will be determined from the input
        `coordinates` if not previously defined.  Otherwise, any previously
        determined dimensionality will remain fixed and cannot be altered.

        Parameters
        ----------
        coordinates : numpy.ndarray (float) or astropy.units.Quantity
        single_dimension : bool, optional
            If `True`, the coordinates consist of data from only one of the
            dimensions.  Note that this is only applicable if the current
            dimensionality has been determined.  If `False`, the shape of the
            coordinates will be determined from `coordinates[0]`.  In the case
            where coordinates.ndim <= 1, the new shape will be set to a
            singular value.
        empty : bool, optional
            If `True` and a new array should be created, that array will be
            empty.  Otherwise, a zero array will be created.

        Returns
        -------
        None
        """
        if isinstance(coordinates, units.Quantity):
            new_unit = coordinates.unit
        else:
            new_unit = None

        coordinates = np.asarray(coordinates)

        if single_dimension or self.ndim == 0:
            if coordinates.shape == ():
                singular = True
                shape = ()
            else:
                singular = False
                shape = coordinates.shape
        else:
            if coordinates.ndim <= 1:
                singular = True
                shape = ()
            else:
                singular = False
                shape = coordinates[0].shape

        if self.unit is None and new_unit is not None:
            self.unit = new_unit

        if singular:
            if not self.singular or self.coordinates is None:
                self.set_singular(empty=empty)
        else:
            if self.shape != shape or self.coordinates is None:
                self.set_shape(shape, empty=empty)

    def check_coordinate_units(self, coordinates):
        """
        Check the coordinate units and update parameters if necessary.

        This method takes in a set of coordinates and returns a more
        standardized version consistent with these coordinates.  Coordinate
        units will be converted to these coordinates, or if no units exist for
        these coordinates, they will be inferred from the input `coordinates`.

        If the given coordinates are a Coordinate subclass, they will be
        converted to the current units, and any other array like input will
        be converted to an numpy array or units.Quantity depending on whether
        units for these coordinates have been defined (or inferred from
        `coordinates`).

        Parameters
        ----------
        coordinates : list or tuple or numpy.ndarray or units.Quantity or None
            The coordinates to check.

        Returns
        -------
        coordinates, original : numpy.ndarray or units.Quantity or Coordinate
            Returns the coordinates in the same unit as the coordinates, and
            whether the coordinates are the original coordinates.
        """
        if coordinates is None:
            return None, True

        original = True
        coordinate_unit = None
        if isinstance(coordinates, (Coordinate, units.Quantity)):
            coordinate_unit = coordinates.unit

        else:  # In the case that coordinates have been supplied weirdly
            if hasattr(coordinates, '__len__') and len(coordinates) > 0:
                if isinstance(coordinates[0], units.Quantity):
                    coordinate_unit = coordinates[0].unit
                    if coordinates[0].shape != ():
                        shape = (len(coordinates),) + coordinates[0].shape
                    else:
                        shape = len(coordinates),
                    new = np.empty(shape, dtype=float) * coordinate_unit
                    for dimension in range(shape[0]):
                        new[dimension] = coordinates[dimension]
                    coordinates = new
                    original = False
                elif not isinstance(coordinates, np.ndarray):
                    coordinates = np.asarray(coordinates)
                    original = False

        # Convert string values to floats
        if isinstance(coordinates, np.ndarray):
            if coordinates.dtype.kind in ['S', 'U']:
                coordinates = coordinates.astype(float)

        if coordinate_unit == units.dimensionless_unscaled:
            coordinate_unit = None
            if isinstance(coordinates, Coordinate):
                coordinates = coordinates.copy()
                original = False
                coordinates.unit = None
                coordinates.coordinates = coordinates.coordinates.value
            elif isinstance(coordinates, units.Quantity):
                coordinates = coordinates.value  # Still references data

        if self.unit is None and coordinate_unit is not None:
            self.unit = coordinate_unit

        if self.unit is not None:
            if coordinate_unit is None:
                if isinstance(coordinates, Coordinate):
                    coordinates = coordinates.copy()
                    coordinates.coordinates = (
                        coordinates.coordinates * self.unit)
                    coordinates.unit = self.unit
                else:
                    coordinates = coordinates * self.unit
                original = False
            elif coordinate_unit != self.unit:
                if isinstance(coordinates, Coordinate):
                    coordinates = coordinates.copy()
                    coordinates.change_unit(self.unit)
                elif isinstance(coordinates, units.Quantity):
                    coordinates = coordinates.to(self.unit)
                original = False

        return coordinates, original

    def change_unit(self, unit):
        """
        Change the coordinate units.

        Parameters
        ----------
        unit : str or units.Unit

        Returns
        -------
        None
        """
        unit = units.Unit(unit)
        if unit == self.unit:
            return
        self.unit = unit

        if self.coordinates is None:
            return
        if isinstance(self.coordinates, units.Quantity):
            self.coordinates = self.coordinates.to(unit)
        elif isinstance(self.coordinates, np.ndarray):
            self.coordinates = self.coordinates * unit

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
        if not self.singular:
            return
        if isinstance(thing, np.ndarray):
            shape = thing.shape
        elif isinstance(thing, tuple):
            shape = thing
        else:
            return

        if shape == ():
            return
        singular_coordinates = self.coordinates
        n_dimensions = self.ndim
        real_shape = (self.ndim,) + shape
        self.coordinates = np.empty_like(self.coordinates, shape=real_shape)
        for dimension in range(n_dimensions):
            self.coordinates[dimension] = singular_coordinates[dimension]

    def convert_factor(self, factor):
        """
        Returns a float factor in the correct units for multiplication.

        If the current coordinates are not quantities, they will be converted
        to such if the factor is a quantity.  Otherwise, the factor scaled to
        units of the coordinates will be returned.

        Parameters
        ----------
        factor : float or numpy.ndarray or units.Quantity

        Returns
        -------
        factor : float
        """
        if not isinstance(factor, units.Quantity):
            return factor

        if factor.unit == units.dimensionless_unscaled:
            return factor.value

        if self.unit is None or self.unit == units.dimensionless_unscaled:
            self.unit = factor.unit
            if self.coordinates is not None:  # Change coordinates
                self.coordinates = self.coordinates * factor.unit

            factor_value = factor.value
        else:
            factor_value = factor.to(self.unit).value

        return factor_value

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
        if self.coordinates is None:
            return
        if indices is None:
            self.coordinates.fill(np.nan)
        elif self.coordinates.ndim == 1:
            self.coordinates[indices] = np.nan
        else:
            self.coordinates[:, indices] = np.nan

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
        if self.coordinates is None:
            return
        if indices is None:
            self.coordinates.fill(0.0)
        elif self.coordinates.ndim == 1:
            self.coordinates[indices] = 0.0
        else:
            self.coordinates[:, indices] = 0.0

    @staticmethod
    def apply_coordinate_mask_function(coordinates, func):
        """
        Apply a masking function to a given set of coordinates.

        Parameters
        ----------
        coordinates : units.Quantity or numpy.ndarray
            The coordinates to check.  Must consist of an array with 2 or
            more dimensions.  I.e., of shape (ndim, x1, x2, ...).
        func : function
            A function that returns a boolean mask given a numpy array.  For
            these purposes, it should take in a two dimensional array of shape
            (ndim, n) where n = product(x1, x2, ...).

        Returns
        -------
        mask : numpy.ndarray (bool)
            The boolean mask output by `func` of shape (x1, x2, ...).
        """
        if isinstance(coordinates, units.Quantity):
            data = coordinates.value
        else:
            data = coordinates

        if data.ndim > 2:
            shape = data.shape
            shape2d = shape[0], int(np.prod(shape[1:]))
            mask = func(data.reshape(shape2d)).reshape(shape[1:])
        else:
            mask = func(data)
        return mask

    def is_null(self):
        """
        Check whether coordinates are zero.

        Returns
        -------
        bool or numpy.ndarray (bool)
        """
        if self.coordinates is None:
            return False
        elif self.singular:
            return np.all(self.coordinates == 0)
        elif self.coordinates.ndim == 1:
            return self.coordinates == 0
        else:
            return self.apply_coordinate_mask_function(
                self.coordinates, csnf.check_null)

    def is_nan(self):
        """
        Check whether coordinates are NaN.

        Returns
        -------
        bool or numpy.ndarray (bool)
        """
        if self.coordinates is None:
            return False
        elif self.singular:
            return np.all(np.isnan(self.coordinates))
        elif self.coordinates.ndim == 1:
            return np.isnan(self.coordinates)
        else:
            return self.apply_coordinate_mask_function(
                self.coordinates, csnf.check_nan)

    def is_finite(self):
        """
        Check whether coordinates are finite.

        Returns
        -------
        bool or numpy.ndarray (bool)
        """
        if self.coordinates is None:
            return False
        elif self.singular:
            return np.all(np.isfinite(self.coordinates))
        elif self.coordinates.ndim == 1:
            return np.isfinite(self.coordinates)
        else:
            return self.apply_coordinate_mask_function(
                self.coordinates, csnf.check_finite)

    def is_infinite(self):
        """
        Check whether coordinates are infinite.

        Returns
        -------
        bool or numpy.ndarray (bool)
        """
        if self.coordinates is None:
            return False
        elif self.singular:
            return np.all(np.isinf(self.coordinates))
        elif self.coordinates.ndim == 1:
            return np.isinf(self.coordinates)
        else:
            return self.apply_coordinate_mask_function(
                self.coordinates, csnf.check_infinite)

    def convert_from(self, coordinates):
        """
        Convert coordinates from another or same system to this.

        Parameters
        ----------
        coordinates : Coordinate

        Returns
        -------
        None
        """
        self.copy_coordinates(coordinates)

    def convert_to(self, coordinates):
        """
        Convert these coordinates to another coordinate system.

        Parameters
        ----------
        coordinates : Coordinate

        Returns
        -------
        None
        """
        coordinates.convert_from(self)

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
        factor : int or float or numpy.ndarray or astropy.units.Quantity
            The factor to check.
        array : numpy.ndarray or astropy.units.Quantity or Coordinate
            The array to check against

        Returns
        -------
        working_factor : numpy.ndarray or astropy.units.Quantity
        """
        if not isinstance(factor, np.ndarray):
            return factor
        if factor.shape == ():
            return factor

        if isinstance(array, Coordinate):
            array_ndim = len(array.shape)
        else:
            array_ndim = array.ndim

        add_dimensions = array_ndim - factor.ndim
        if add_dimensions == 0:
            return factor
        elif add_dimensions > 0:
            for i in range(add_dimensions):
                factor = factor[..., None]

        return factor

    @classmethod
    def get_class(cls, class_name=None):
        """
        Return a coordinate class for a given name.

        Parameters
        ----------
        class_name : str, optional
            The name of the class not including the "Coordinates" suffix.  The
            default is *this* class.

        Returns
        -------
        coordinate_class : class``
        """
        if class_name is None:
            return cls

        base_module_name = '.'.join(Coordinate.__module__.split('.')[:-1])

        # Determine if it's a class from whether the class name is mixed case
        is_class = not (class_name.isupper() or class_name.islower())
        user_class_name = class_name
        if is_class:
            module_name = to_module_name(class_name)
        else:
            module_name = class_name.lower()

        if module_name.endswith('_d'):
            module_name = module_name[:-2] + 'd'

        possibilities = ['%s', 'coordinate_%s', '%s_coordinates']
        for possibility in possibilities:
            module_basename = possibility % module_name
            try_module = f'{base_module_name}.{module_basename}'
            try:
                module = importlib.import_module(try_module)
                class_name = to_class_name(module_basename)
                if class_name[-1] == 'd' and class_name[-2].isdigit():
                    class_name = class_name[:-1] + class_name[-1].upper()
                break
            except ModuleNotFoundError:
                pass
        else:
            module = class_name = None

        if module is None:
            raise ValueError(f"Could not find {user_class_name} class module.")

        coordinate_class = getattr(module, class_name)
        if coordinate_class is None:  # pragma: no cover
            # If this gets hit it's because modules/classes weren't named well.
            raise ValueError(f"Could not find {class_name} in {module} given "
                             f"user input {user_class_name}.")

        if not issubclass(coordinate_class, Coordinate):
            raise ValueError(f"Retrieved class {coordinate_class} is not a "
                             f"{Coordinate} sub class.")

        return coordinate_class

    @classmethod
    def get_instance(cls, class_name=None):
        """
        Return a coordinate instance for a given name.

        Parameters
        ----------
        class_name : str, optional
            The name of the class not including the "Coordinates" suffix.  The
            default is *this* class.

        Returns
        -------
        object : Coordinate2D or SphericalCoordinates or CelestialCoordinates
        """
        return cls.get_class(class_name=class_name)()

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
        if self.coordinates is None or self.singular:
            return

        if self.coordinates.ndim == 1:
            self.coordinates = np.insert(self.coordinates, insert_indices,
                                         np.nan)
        else:
            self.coordinates = np.insert(
                self.coordinates, insert_indices, np.nan, axis=1)

    def merge(self, other):
        """
        Append other coordinates to the end of these.

        Parameters
        ----------
        other : Coordinate

        Returns
        -------
        None
        """
        if other.coordinates is None:
            return

        if self.coordinates is None:
            self.copy_coordinates(other)
            return

        if other.ndim != self.ndim:
            raise ValueError("Coordinate dimensions do not match.")

        coordinates = self.coordinates
        if self.ndim == 1 and coordinates.ndim == 1:
            coordinates = coordinates[None]
        elif self.ndim > 1 and coordinates.ndim == 1:
            coordinates = coordinates[:, None]

        other_c = other.coordinates
        if other.ndim == 1 and other_c.ndim == 1:
            other_c = other_c[None]
        elif other.ndim > 1 and other_c.ndim == 1:
            other_c = other_c[:, None]

        self.coordinates = np.concatenate((coordinates, other_c), axis=1)

    def paste(self, coordinates, indices):
        """
        Paste new coordinate values at the given indices.

        Parameters
        ----------
        coordinates : Coordinate
        indices : numpy.ndarray (int)

        Returns
        -------
        None
        """
        if self.singular or self.coordinates is None:
            raise ValueError("Cannot paste onto singular "
                             "or empty coordinates.")
        elif coordinates.coordinates is None:
            raise ValueError("Cannot paste empty coordinates.")
        elif self.ndim != coordinates.ndim:
            raise ValueError("Coordinate dimensions do not match.")

        if self.coordinates.ndim == 1:
            self.coordinates[indices] = coordinates.coordinates
        else:
            self.coordinates[:, indices] = coordinates.coordinates

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
        elif n == 0:
            return

        if (self.unit is not None
                and not isinstance(fill_value, units.Quantity)):
            fill_value = fill_value * self.unit

        if self.coordinates.ndim == 1:
            self.coordinates = np.roll(self.coordinates, n)
        else:
            self.coordinates = np.roll(self.coordinates, n, axis=1)

        blank = slice(0, n) if n > 0 else slice(n, None)
        if self.coordinates.ndim > 1:
            blank = slice(None), blank
        self.coordinates[blank] = fill_value

    def mean(self):
        """
        Return the mean coordinates.

        Returns
        -------
        mean_coordinates : Coordinate
        """
        new = self.empty_copy()
        if self.coordinates is None:
            return new
        if self.singular:
            mean_coordinates = self.coordinates.copy()
        elif self.ndim == 1:
            mean_coordinates = np.atleast_1d(np.nanmean(self.coordinates))
        else:
            mean_coordinates = np.nanmean(self.coordinates, axis=1)

        if new.ndim == 0:
            mean_coordinates = mean_coordinates[:, None]

        new.set(mean_coordinates, copy=False)
        return new

    def copy_coordinates(self, coordinates):
        """
        Copy the coordinates from another system to this system.

        Parameters
        ----------
        coordinates : Coordinate

        Returns
        -------
        None
        """
        self.set(coordinates.coordinates)

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

        if self.coordinates is not None:
            if self.coordinates.ndim > 1 and coordinates.ndim == 1:
                coordinates = coordinates[:, None]

        if copy and isinstance(coordinates, np.ndarray):
            self.coordinates = coordinates.copy()
        else:
            self.coordinates = coordinates
