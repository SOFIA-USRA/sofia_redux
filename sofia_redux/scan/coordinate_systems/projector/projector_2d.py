# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from copy import deepcopy

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['Projector2D']


class Projector2D(ABC):

    def __init__(self, projection):
        """
        Initialize a 2-D projector.

        The 2-dimensional projector is a wrapper around a given projector to
        store given coordinates, projection offsets, and translate between the
        two if necessary.  The default coordinates are taken from the projector
        reference position, and a clean set of offsets are created on
        initialization.

        Parameters
        ----------
        projection : Projection2D
        """
        self.projection = projection
        self.offset = Coordinate2D()
        self.coordinates = projection.reference.copy()

    def copy(self):
        """
        Return a full copy of the Projector2D

        Returns
        -------
        Projector2D
        """
        return deepcopy(self)

    def set_reference_coordinates(self):
        """
        Sets the projector coordinates from the projection reference position.

        Returns
        -------
        None
        """
        self.coordinates.copy_coordinates(self.projection.reference)

    def project(self, coordinates=None, offsets=None):
        """
        Project the coordinates to an offset.

        Parameters
        ----------
        coordinates : Coordinate2D, optional
            The coordinates to project.  If not supplied, defaults to
            the stored coordinates.
        offsets : Coordinate2D, optional
            The output offsets to update with the results.  If not supplied,
            defaults to a set stored in the projector.

        Returns
        -------
        offsets : Coordinate2D
        """
        if offsets is None:
            offsets = self.offset
        if coordinates is None:
            coordinates = self.coordinates

        return self.projection.project(coordinates, projected=offsets)

    def deproject(self, offsets=None, coordinates=None):
        """
        Deproject offsets onto coordinates.

        Parameters
        ----------
        offsets : Coordinate2D, optional
            The offsets to deproject.  If not supplied, defaults to the offsets
            stored in the projector.
        coordinates : Coordinate2D, optional
            The coordinates to hold the results.  If not supplied, defaults to
            the coordinates stored in the projector.

        Returns
        -------
        coordinates : Coordinate2D
        """
        if offsets is None:
            offsets = self.offset
        if coordinates is None:
            coordinates = self.coordinates
        return self.projection.deproject(offsets, coordinates=coordinates)
