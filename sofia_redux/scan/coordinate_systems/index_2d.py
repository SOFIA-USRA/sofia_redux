# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.index import Index
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import round_values


__all__ = ['Index2D']


class Index2D(Coordinate2D, Index):

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
        super().__init__(coordinates=coordinates, copy=copy)

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
