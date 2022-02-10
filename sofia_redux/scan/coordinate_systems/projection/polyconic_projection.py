# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['PolyconicProjection']


class PolyconicProjection(SphericalProjection):

    def __init__(self):
        """
        Initialize a polyconic projection.

        The polyconic projection is also known as the American polyconic
        projection.  Each parallel is a circular arc at true scale with a
        straight equator and the center of each circle lying on the central
        axis.  The longitude of the central meridian lies at zero, with the
        latitude of the origin at the central meridian also equal to zero.

        Notes
        -----
        Deprojection is not possible for the polyconic projection.
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
        return "PCO"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Polyconic'

    def get_phi_theta(self, offset, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  This method is not implemented for the polyconic
        projection.

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
            f"Deprojection not implemented for {self.__class__} projection.")

    def get_offsets(self, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the polyconic projection when theta is non-zero, this is given by:

            x = cot(theta) * sin(phi * sin(theta))
            y = theta + cot(theta) * (1 - cos(phi * sin(theta)))

        If theta is equal to zero, then to avoid zero division:

            x = phi
            y = 0

        Parameters
        ----------
        theta : units.Quantity or float
            The theta (latitude) angle.  If a float value is provided, it
            should be in radians.
        phi : units.Quantity or float
            The phi (longitude) angle.  If a float value is provided, it should
            be in radians.
        offsets : Coordinate2D, optional
            An optional coordinate system in which to place the results.

        Returns
        -------
        offsets : Coordinate2D
        """
        if offsets is None:
            offsets = Coordinate2D(unit='degree')

        phi, theta = self.phi_theta_to_radians(phi, theta)
        rad = units.Unit('radian')
        if theta == 0:
            x = phi
            y = 0 * rad
        else:
            cot_theta = 1 / np.tan(theta)
            t = phi * np.sin(theta)
            x = cot_theta * np.sin(t) * rad
            dy = cot_theta * (1.0 - np.cos(t)) * rad
            y = theta + dy

        offsets.set([x, y])
        return offsets
