# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['GlobalSinusoidalProjection']


class GlobalSinusoidalProjection(SphericalProjection):

    def __init__(self):
        """
        Initialize a global sinusoidal projection.

        The global spherical projection is a very simple projection that
        converts between coordinates and offsets when the projection is also
        in spherical coordinates and celestial and native poles are equivalent.
        I.e, the only parameter of importance is the reference position of the
        projection.
        """
        super().__init__()

    @classmethod
    def get_fits_id(cls):
        """
        Return the FITS ID for the projection.

        Returns
        -------
        str
        """
        return "GLS"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Global Sinusoidal'

    def project(self, coordinates, projected=None):
        """
        Project the coordinates.

        The forward projection for the global sinusoidal projection converts
        spherical coordinates to cartesian offsets from a reference position as
        follows:

            dx = (x - x_reference) * cos(y)
            dy = y - y_reference

        Parameters
        ----------
        coordinates : SphericalCoordinates
            The coordinates to project.
        projected : Coordinate2D, optional
            The output coordinates.  Will be created if not supplied.

        Returns
        -------
        projected : Coordinate2D
            The projected coordinates.
        """
        if projected is None:
            projected = Coordinate2D(unit='degree')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            x = np.fmod(coordinates.x - self.reference.x, self.full_circle
                        ) * coordinates.cos_lat
        y = coordinates.y - self.reference.y
        projected.set([x, y])
        return projected

    def deproject(self, projected, coordinates=None):
        """
        Deproject global sinusoidal projection offsets to coordinates.

        The reverse projection (deprojection) is used to convert offsets in
        relation to a reference position into spherical coordinates as follows:

            y = y_reference + dy
            x = x_reference + (dx / cos(y))

        Parameters
        ----------
        projected : Coordinate2D
            The projected coordinates to deproject.
        coordinates : SphericalCoordinates, optional
            The output deprojected coordinates.  Will be created if not
            supplied.

        Returns
        -------
        coordinates : Coordinate2D
            The deprojected coordinates.
        """
        if coordinates is None:
            coordinates = SphericalCoordinates(unit='degree')
        coordinates.set_y(self.reference.y + projected.y)
        coordinates.set_x(self.reference.x
                          + (projected.x / coordinates.cos_lat))
        return coordinates

    def get_phi_theta(self, offset, phi_theta=None):
        """
        Return the phi_theta coordinates.

        Parameters
        ----------
        offset : Coordinate2D
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        raise NotImplementedError(
            f"Not implemented for {self.__class__} projection.")

    def get_offsets(self, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Parameters
        ----------
        theta : units.Quantity
            The theta angle.
        phi : units.Quantity
            The phi angle.
        offsets : Coordinate2D, optional
            An optional coordinate system in which to place the results.

        Returns
        -------
        offsets : Coordinate2D
        """
        raise NotImplementedError(
            f"Not implemented for {self.__class__} projection.")
