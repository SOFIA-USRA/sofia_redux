# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units

from sofia_redux.scan.coordinate_systems.projection.zenithal_projection \
    import ZenithalProjection

__all__ = ['SlantOrthographicProjection']


class SlantOrthographicProjection(ZenithalProjection):

    def __init__(self):
        """
        Initialize a slant orthographic projection.

        The slant orthographic projection is a zenithal (or azimuthal)
        projection depicting a one hemisphere from an infinite distance, with
        the horizon as a great circle, with the origin at (0, 90) degrees
        LON/LAT.

        The forward projection is given by:

            x = cos(theta)sin(phi)
            y = -cos(theta)cos(phi)

        and the inverse transform (deprojection) is given by:

            phi = arctan(x, -y)
            theta = acos(sqrt(x^2 + y^2))
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
        return 'SIN'

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Slant Orthographic'

    @classmethod
    def r(cls, theta):
        """
        Return the radius of a point from the center of the projection.

        For the slant orthographic projection, the radius of a point from
        the center of the projection, given the latitude (theta) is:

            r = cos(theta)

        Parameters
        ----------
        theta : float or numpy.ndarray or units.Quantity
            The latitude angle.

        Returns
        -------
        r : units.Quantity
            The distance of the point from the central point.
        """
        return np.cos(theta) * units.Unit('radian')

    @classmethod
    def theta_of_r(cls, r):
        """
        Return theta (latitude) given a radius from the central point.

        For the slant orthographic projection, the latitude (theta) of a point
        at a distance r from the center of the projection is given as:

            theta = acos(r)

        Parameters
        ----------
        r : float or numpy.ndarray or units.Quantity
            The distance of the point from the central point.

        Returns
        -------
        theta : units.Quantity
            The latitude angle.
        """
        return cls.acos(r)
