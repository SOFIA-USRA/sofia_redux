# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.projection.zenithal_projection \
    import ZenithalProjection

__all__ = ['StereographicProjection']


class StereographicProjection(ZenithalProjection):

    def __init__(self):
        """
        Initialize a stereographic projection.

        The stereographic projection is a zenithal projection that is
        also known as the planisphere projection or azimuthal conformal
        projection.

        The forward projection is given by::

            x = 2 * tan(pi/4 - theta/2) * sin(phi)
            y = -2 * tan(pi/4 - theta/2) * cos(phi)

        and the inverse transform (deprojection) is given by::

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
        return "STG"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Stereographic'

    @classmethod
    def r(cls, theta):
        """
        Return the radius of a point from the center of the projection.

        For the stereographic projection, the radius of a point from the center
        of the projection, given the latitude (theta) is:

            r = 2 * tan(pi/4 - theta/2)

        Parameters
        ----------
        theta : float or numpy.ndarray or units.Quantity
            The latitude angle.

        Returns
        -------
        r : units.Quantity
            The distance of the point from the central point.
        """
        r = 2 * np.tan(0.5 * (cls.right_angle - theta)) * units.Unit('radian')
        return r

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
            The distance of the point from the central point.  If a float value
            is given, it should be in provided in radians.

        Returns
        -------
        theta : units.Quantity
            The latitude angle.
        """
        if isinstance(r, units.Quantity):
            if r.unit == units.dimensionless_unscaled:
                r = r.value
            else:
                r = r.to('radian').value

        return (cls.right_angle
                - (2 * np.arctan(0.5 * r))
                * units.Unit('radian'))
