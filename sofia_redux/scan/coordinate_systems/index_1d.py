# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.index import Index
from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.utilities.utils import round_values


__all__ = ['Index1D']


class Index1D(Coordinate1D, Index):

    def __init__(self, coordinates=None, copy=True):
        """
        Initialize 1-dimensional indexing coordinates.

        The indexing coordinates are used to represent integer (column,
        row) indices on a 1-dimensional array.  As an extension of the
        :class:`Coordinate1D`, x represents columns and y represents rows.
        Invalid indices are usually represented by an entry of -1.

        Parameters
        ----------
        coordinates : list or tuple or array-like, optional
            The coordinates used to populate the object during initialization.
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
        self.coordinates = coordinates

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
        self.coordinates = round_values(self.coordinates + x)

    def subtract_x(self, x):
        """
        Subtract x from coordinates.

        Parameters
        ----------
        x : float or numpy.ndarray or iterable
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.broadcast_to(x)
        self.coordinates = round_values(self.coordinates - x)

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
        self.coordinates = round_values(factor * self.coordinates)

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
