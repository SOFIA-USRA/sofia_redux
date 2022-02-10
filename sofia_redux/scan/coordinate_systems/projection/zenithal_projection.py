# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['ZenithalProjection']


class ZenithalProjection(SphericalProjection):

    def __init__(self):
        """
        Initialization for the abstract zenithal projection.

        A zenithal projection (azimuthal projection) is one in which the
        surface of projection is a plane.  A native coordinate system is
        defined such that the polar axis is orthogonal to the plane of
        projection.  Meridians are projected as equispaced rays from a
        central point, and parallels are concentric circles centered on
        that same point.

        By default, the native pole is set to (0, 90) degrees LON/LAT.
        """
        super().__init__()
        self.native_reference.set_native(
            [0.0 * units.Unit('degree'), self.right_angle])

    def calculate_celestial_pole(self):
        """
        Calculate the celestial pole.

        Returns
        -------
        None
        """
        self.set_celestial_pole(self.reference)

    def get_phi_theta(self, offsets, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For a zenithal projection these are given as:

            phi = arctan(x, -y)
            theta = angle_from_radius_function(sqrt(x^2 + y^2))

        where `angle_from_radius_function` depends on the exact zenithal
        projection model.

        Parameters
        ----------
        offsets : Coordinate2D
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        if phi_theta is None:
            phi_theta = SphericalCoordinates(unit='degree')

        x, y = self.offset_to_xy_radians(offsets)
        x, y = x.value, y.value

        phi = np.arctan2(x, -y) * units.Unit('radian')
        theta = self.theta_of_r(np.hypot(x, y))
        phi_theta.set_native([phi, theta])
        return phi_theta

    def get_offsets(self, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For a zenithal projection, this is given by:

            r = radius_from_angle_function(theta)
            x = r * sin(phi)
            y = -r * cos(phi)

        where `radius_from_angle_function` depends on the zenithal projection
        model.

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
        if offsets is None:
            offsets = Coordinate2D(unit='degree')

        # From Calabretta and Greisen 2002
        r = self.r(theta)
        x = r * np.sin(phi)
        y = -r * np.cos(phi)

        offsets.set([x, y])
        return offsets

    @abstractmethod
    def r(self, theta):  # pragma: no cover
        """
        Return the radius of a point from the center of the projection.

        Parameters
        ----------
        theta : float or numpy.ndarray or units.Quantity
            The latitude angle.

        Returns
        -------
        r : units.Quantity
            The distance of the point from the central point.
        """
        pass

    @abstractmethod
    def theta_of_r(self, r):  # pragma: no cover
        """
        Return theta (latitude) given a radius from the central point.

        Parameters
        ----------
        r : float or numpy.ndarray or units.Quantity
            The distance of the point from the central point.

        Returns
        -------
        theta : units.Quantity
        """
        pass
