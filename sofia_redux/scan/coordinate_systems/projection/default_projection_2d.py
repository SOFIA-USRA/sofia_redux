# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.coordinate_systems.projection.projection_2d import \
    Projection2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['DefaultProjection2D']


class DefaultProjection2D(Projection2D):

    def __init__(self):
        """
        Initialize a default 2-dimensional Cartesian projection.

        The default 2-dimensional projection is a very simple projection in
        which projected and deprojected native coordinates are equivalent in a
        Cartesian (x, y) system.
        """
        super().__init__()
        self.set_reference(Coordinate2D(np.zeros(2, dtype=float)))

    def get_coordinate_instance(self):
        """
        Return a coordinate instance for this grid type.

        For the DefaultProjection2D, a Coordinate2D instance will always be
        returned.

        Returns
        -------
        Coordinate2D
        """
        return Coordinate2D()

    def project(self, coordinates, projected=None):
        """
        Project the coordinates.

        Converts coordinates to offsets w.r.t. a reference position.  Note that
        the projected coordinates will always be a Coordinate2D representation
        of the input coordinates.

        Parameters
        ----------
        coordinates : Coordinate2D
            The coordinates to project.
        projected : Coordinate2D, optional
            The output coordinates.  Will be created if not supplied.

        Returns
        -------
        projected : Coordinate2D
            The projected coordinates.
        """
        if projected is None:
            projected = Coordinate2D()
        projected.copy_coordinates(coordinates)
        return projected

    def deproject(self, projected, coordinates=None):
        """
        Deproject a projection onto coordinates.

        Converts offsets w.r.t a reference position to coordinates.  Note that
        the deprojected coordinates will always be a Coordinate2D
        representation of the input coordinates.

        Parameters
        ----------
        projected : Coordinate2D
            The projected coordinates to deproject.
        coordinates : Coordinate2D, optional
            The output deprojected coordinates.  Will be created if not
            supplied.

        Returns
        -------
        coordinates : Coordinate2D
            The deprojected coordinates.
        """
        if coordinates is None:
            coordinates = self.get_coordinate_instance()
        coordinates.copy_coordinates(projected)
        return coordinates

    def get_fits_id(self):
        """
        Return the FITS ID for the projection.

        Returns
        -------
        str
        """
        return ''

    def get_full_name(self):
        """
        Return the full name for the projection.

        Returns
        -------
        str
        """
        return "Cartesian"

    def parse_header(self, header, alt=''):
        """
        Parse and apply a FITS header to the projection.

        Parameters
        ----------
        header : fits.Header
            The FITS header to parse.
        alt : str, optional
            The alternate FITS system.

        Returns
        -------
        None
        """
        pass

    def edit_header(self, header, alt=''):
        """
        Edit a FITS header with the projection information.

        Parameters
        ----------
        header : fits.Header
            The FITS header to edit.
        alt : str, optional
            The alternate FITS system.

        Returns
        -------
        None
        """
        pass
