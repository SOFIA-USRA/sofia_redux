# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['HammerAitoffProjection']


class HammerAitoffProjection(CylindricalProjection):

    def __init__(self):
        """
        Initialize a Hammer-Aitoff projection.

        The Hammer-Aitoff projection is an equal-area cylindrical projection
        where the graticule takes the form of an ellipse, and is suitable for
        mapping on a small scale.  It is a modified azimuthal projection where
        the central meridian is a straight line, half the length of the
        projected equator.
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
        return "AIT"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Hammer-Aitoff'

    @classmethod
    def z2(cls, offset):
        """
        Return the Z2 factor for an offset.

        The z2 parameter (z squared) is used when calculating the deprojection
        (inverse projection) and is given by:

            z2 = 1 - (x^2)/16 - (y^2)/4

        Parameters
        ----------
        offset : Coordinate2D

        Returns
        -------
        z2 : units.Quantity
            The z2 factor in radian^2 units.
        """
        x, y = cls.offset_to_xy_radians(offset)
        x, y = x.value, y.value
        z2 = 1 - ((x ** 2) / 16) - ((y ** 2) / 4)
        return z2 * units.Unit('radian2')

    @classmethod
    def gamma(cls, theta, phi):
        """
        Return the gamma factor.

        Gamma is used during the forward projection and is given by:

            gamma = sqrt(2 / (1 + cos(theta) + cos(phi/2)))

        Parameters
        ----------
        theta : float or units.Quantity or numpy.ndarray
        phi : float or units.Quantity or numpy.ndarray

        Returns
        -------
        float or numpy.ndarray
        """
        phi, theta = cls.phi_theta_to_radians(phi, theta)
        g = np.sqrt(2 / (1 + (np.cos(theta) * np.cos(0.5 * phi)))).value
        return g

    @classmethod
    def get_phi_theta(cls, offset, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For the Hammer projection these are given as:

            phi = 2 * arctan(z * x, 2(2z^2 - 1))
            theta = arcsin(z * y)

        where

            z = sqrt(1 - (0.25x)^2 - (0.5y)^2)

        Parameters
        ----------
        offset : Coordinate2D
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        phi_theta : SphericalCoordinates
        """
        z2 = cls.z2(offset)
        z = np.sqrt(z2)
        if phi_theta is None:
            phi_theta = SphericalCoordinates(unit='degree')

        x, y = cls.offset_to_xy_radians(offset)
        x, y = x.value, y.value
        z, z2 = z.value, z2.value

        phi = 2 * np.arctan2(0.5 * z * x, (2 * z2) - 1) * units.Unit('radian')
        theta = cls.asin(y * z)  # in radians
        phi_theta.set_native([phi, theta])
        return phi_theta

    @classmethod
    def get_offsets(cls, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the Hammer projection, this is given by:

            dx = 2 * cos(theta) * sin(phi/2) * gamma
            dy = sin(theta) * gamma

        where

            gamma = sqrt(2) / sqrt(1 + cos(theta) * cos(phi/2))

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
        rad = units.Unit('radian')
        phi, theta = cls.phi_theta_to_radians(phi, theta)
        gamma = cls.gamma(theta, phi)
        x = 2 * gamma * np.cos(theta) * np.sin(0.5 * phi) * rad
        y = gamma * np.sin(theta) * rad
        offsets.set([x, y])
        return offsets
