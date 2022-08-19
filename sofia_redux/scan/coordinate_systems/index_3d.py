# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D
from sofia_redux.scan.utilities.utils import round_values


__all__ = ['Index3D']


class Index3D(Coordinate3D, Index2D):

    def __init__(self, coordinates=None, copy=True):
        """
        Initialize 3-dimensional indexing coordinates.

        The indexing coordinates are used to represent integer (x, y, z)
        indices on a 3-dimensional array.  As an extension of the
        :class:`Coordinate3D`. Invalid indices are usually represented by an
        entry of -1.

        Parameters
        ----------
        coordinates : list or tuple or array-like, optional
            The coordinates used to populate the object during initialization.
            The first (0) value should represent x-coordinates.
        copy : bool, optional
            Whether to explicitly perform a copy operation on the input
            coordinates when storing them into these coordinates.
        """
        super().__init__(coordinates=coordinates, copy=copy)

    def set_z(self, coordinates, copy=True):
        """
        Set the z coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or int
        copy : bool, optional
            If `True`, copy the coordinates.  Otherwise do a reference.

        Returns
        -------
        None
        """
        coordinates, original = self.check_coordinate_units(coordinates)
        self.set_shape_from_coordinates(coordinates, single_dimension=True)
        self.coordinates[2] = coordinates

    @classmethod
    def rotate_offsets(cls, offsets, angle, axis=2):
        """
        Rotate zero-centered offsets in-place by an angle.

        Parameters
        ----------
        offsets : Coordinate3D or numpy.ndarray or units.Quantity
            The (x, y) offset coordinates to rotate.
        angle : astropy.units.Quantity or float
            The angle by which to rotate the offsets.
        axis : int or str, optional
            The axis about

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

        super().rotate_offsets(offsets, angle, axis=axis)
        if converted:
            offsets = round_values(offsets)
            original_offsets[...] = offsets

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
        self.broadcast_to(z)
        self.coordinates[2] = round_values(self.coordinates[2] + z)

    def subtract_z(self, z):
        """
        Subtract z from coordinates.

        Parameters
        ----------
        z : float or numpy.ndarray
            The value(s) to subtract.

        Returns
        -------
        None
        """
        self.broadcast_to(z)
        self.coordinates[2] = round_values(self.coordinates[2] - z)

    def scale_z(self, factor):
        """
        Scale the z coordinates by a factor.

        Parameters
        ----------
        factor : float or units.Quantity
            The factor by which to scale the x-coordinates.

        Returns
        -------
        None
        """
        factor = self.convert_factor(factor)
        self.coordinates[2] = round_values(factor * self.coordinates[2])

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
