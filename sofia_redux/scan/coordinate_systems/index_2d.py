# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems import \
    coordinate_systems_numba_functions as csnf
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import round_values


__all__ = ['Index2D']


class Index2D(Coordinate2D):

    def __init__(self, coordinates=None, copy=True):
        super().__init__(coordinates=coordinates, copy=copy, unit=None)
        if self.coordinates is not None:
            self.coordinates = round_values(coordinates)
            self.coordinates, _ = self.check_coordinate_units(self.coordinates)

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
        self.coordinates[1] = coordinates

    def check_coordinate_units(self, coordinates):
        """
        Check the coordinate units and update parameters if necessary.

        Parameters
        ----------
        coordinates : float or numpy.ndarray or units.Quantity or None

        Returns
        -------
        coordinates, original : float or numpy.ndarray or units.Quantity
            Returns the coordinates in the same unit as the coordinates, and
            whether the coordinates are the original coordinates.
        """
        self.unit = None

        if coordinates is None:
            return None, True

        input_coordinates = coordinates

        if isinstance(coordinates, units.Quantity):
            coordinates = coordinates.decompose()
            if coordinates.unit == units.dimensionless_unscaled:
                coordinates = coordinates.value
            else:
                raise ValueError("Indices must be dimensionless units.")

        coordinates = round_values(coordinates)
        return coordinates, coordinates is input_coordinates

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
        raise NotImplementedError("Cannot give indices unit dimensions.")

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
            self.coordinates.fill(-1)
        else:  # Assume non-singular
            self.coordinates[:, indices] = -1

    def is_nan(self):
        """
        Check whether indices are valid.

        Returns
        -------
        bool or numpy.ndarray (bool)
        """
        if self.coordinates is None:
            return False
        elif self.singular:
            return np.allclose(self.coordinates, -1)
        else:
            return self.apply_coordinate_mask_function(
                self.coordinates, self.is_neg1)

    @staticmethod
    def is_neg1(coordinates):
        """
        Check if coordinates are all equal to -1 over all dimensions.

        Parameters
        ----------
        coordinates : numpy.ndarray (float)
            The coordinate array to check of shape (n_dimensions, n).

        Returns
        -------
        mask : numpy.ndarray (bool)
            The result of shape (n,) where `True` indicates that coordinates
            are equal to -1 in all dimensions.
        """
        return csnf.check_value(-1, coordinates)

    def insert_blanks(self, insert_indices):
        """
        Insert blank (-1) values at the requested indices.

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
        self.coordinates = np.insert(self.coordinates, insert_indices, -1,
                                     axis=1)

    @classmethod
    def rotate_offsets(cls, offsets, angle):
        """
        Rotate zero-centered offsets in-place by an angle.

        Parameters
        ----------
        offsets : Coordinate2D or numpy.ndarray or units.Quantity
            The (x, y) offset coordinates to rotate.
        angle : astropy.units.Quantity or float
            The angle by which to rotate the offsets.

        Returns
        -------
        None
        """
        if isinstance(offsets, np.ndarray) and offsets.dtype == int:
            converted = True
            original_offsets = offsets
            offsets = offsets.astype(float)
        else:
            converted = original_offsets = False

        super().rotate_offsets(offsets, angle)
        if converted:
            offsets = round_values(offsets)
            original_offsets[...] = offsets

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
        self.coordinates[0] = round_values(self.coordinates[0] + x)

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
        self.coordinates[0] = round_values(self.coordinates[0] - x)

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
        self.coordinates[1] = round_values(self.coordinates[1] + y)

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
        self.coordinates[1] = round_values(self.coordinates[1] - y)

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
        self.coordinates[0] = round_values(factor * self.coordinates[0])

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
        self.coordinates[1] = round_values(factor * self.coordinates[1])
