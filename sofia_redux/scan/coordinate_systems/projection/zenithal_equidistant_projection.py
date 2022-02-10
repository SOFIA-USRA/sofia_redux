# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.projection.zenithal_projection \
    import ZenithalProjection

__all__ = ['ZenithalEquidistantProjection']


class ZenithalEquidistantProjection(ZenithalProjection):

    def __init__(self):
        """
        Initialize a zenithal equidistant projection.

        A useful property of the zenithal equidistant projection is that all
        points are at proportionally correct distances from the center point,
        and all points are at the correct azimuth from the center point.  All
        meridians are straight lines and distances from the pole are all
        correct.

        The forward projection is given by:

            x = (pi/2 - theta)sin(phi)
            y = -(theta - pi/2)cos(phi)

        with cot(theta) evaluating to zero at theta=90 degrees, and the inverse
        transform (deprojection) is given by:

            phi = arctan(x, -y)
            theta = pi/2 - sqrt(x^2 + y^2)
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
        return "ARC"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Zenithal Equidistant'

    @classmethod
    def r(cls, theta):
        """
        Return the radius of a point from the center of the projection.

        For the zenithal equidistant projection, the radius of a point from
        the center of the projection, given the latitude (theta) is:

            r = pi/2 - theta

        Parameters
        ----------
        theta : float or numpy.ndarray or units.Quantity
            The latitude angle.

        Returns
        -------
        value : units.Quantity
            The distance of the point from the central point.
        """
        return cls.right_angle - theta

    @classmethod
    def theta_of_r(cls, r):
        """
        Return theta (latitude) given a radius from the central point.

        For the zenithal equidistant projection, the latitude (theta) of a
        point at a distance r from the center of the projection is given as:

            theta = pi/2 - r

        Parameters
        ----------
        r : float or numpy.ndarray or units.Quantity
            The distance of the point from the central point.

        Returns
        -------
        theta : units.Quantity
            The latitude angle.
        """
        if not isinstance(r, units.Quantity):
            r = r * units.Unit('radian')
        return cls.right_angle - r
