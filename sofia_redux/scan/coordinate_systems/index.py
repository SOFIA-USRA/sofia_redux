# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems import \
    coordinate_systems_numba_functions as csnf
from sofia_redux.scan.coordinate_systems.coordinate import Coordinate
from sofia_redux.scan.utilities.utils import round_values


__all__ = ['Index']


class Index(Coordinate):

    def __init__(self, coordinates=None, copy=True):
        """
        Initialize 2-dimensional indexing coordinates.

        The indexing coordinates are used to represent integer (column,
        row) indices on a 2-dimensional array.  As an extension of the
        :class:`Coordinate2D`, x represents columns and y represents rows.
        Invalid indices are usually represented by an entry of -1.

        Parameters
        ----------
        coordinates : list or tuple or array-like, optional
            The coordinates used to populate the object during initialization.
            The first (0) value or index should represent column
            coordinates, and the second should represent the rows.
        copy : bool, optional
            Whether to explicitly perform a copy operation on the input
            coordinates when storing them into these coordinates.
        """
        super().__init__(coordinates=coordinates, copy=copy, unit=None)
        if self.coordinates is not None:
            self.coordinates = round_values(coordinates)
            self.coordinates, _ = self.check_coordinate_units(self.coordinates)

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
        self.coordinates = round_values(self.coordinates)
