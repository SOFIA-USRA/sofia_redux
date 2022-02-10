# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.projection.zenithal_projection \
    import ZenithalProjection

__all__ = ['ZenithalEqualAreaProjection']


class ZenithalEqualAreaProjection(ZenithalProjection):

    def __init__(self):
        """
        Initialize a zenithal equal-area projection.

        The zenithal equal-area projection (also known as the Lambert azimuthal
        equal-area projection) maps points from a sphere to a disk, accurately
        representing area in all regions of the sphere, but not angles.

        The forward projection is given by:

            x = sqrt(2 * (1 - sin(theta))) * sin(phi)
            y = -sqrt(2 * (1 - sin(theta))) * cos(phi)

        and the inverse transform (deprojection) is given by:

            phi = arctan(x, -y)
            theta = pi/2 - (2 * asin(sqrt(x^2 + y^2)/2))
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
        return "ZEA"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Zenithal Equal-Area'

    @classmethod
    def r(cls, theta):
        """
        Return the radius of a point from the center of the projection.

        For the zenithal equal-area projection, the radius of a point from
        the center of the projection, given the latitude (theta) is:

            r = sqrt(2 * (1 - sin(theta)))

        Parameters
        ----------
        theta : float or numpy.ndarray or units.Quantity
            The latitude angle.

        Returns
        -------
        r : units.Quantity
            The distance of the point from the central point.
        """
        return np.sqrt(2 * (1 - np.sin(theta))) * units.Unit('radian')

    @classmethod
    def theta_of_r(cls, r):
        """
        Return theta (latitude) given a radius from the central point.

        For the zenithal equal-area projection, the latitude (theta) of a point
        at a distance r from the center of the projection is given as:

            theta = pi/2 - (2 * asin(r/2))

        Parameters
        ----------
        r : float or numpy.ndarray or units.Quantity
            The distance of the point from the central point.

        Returns
        -------
        theta : units.Quantity
            The latitude angle.
        """
        return cls.right_angle - (2 * cls.asin(0.5 * r))
