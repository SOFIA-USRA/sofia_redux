# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.projection.zenithal_projection \
    import ZenithalProjection
from sofia_redux.scan.coordinate_systems.spherical_coordinates \
    import SphericalCoordinates

__all__ = ['GnomonicProjection']


class GnomonicProjection(ZenithalProjection):

    def __init__(self):
        """
        Initialize a gnomonic projection.

        A gnomonic projection displays all great circles as straight lines by
        converting surface points on a sphere to a tangent plane where a ray
        from the center of the sphere passes through the point on the sphere
        and then onto the plane.  Distortion is extreme away from the tangent
        point.

        The forward projection is given by::

            x = cot(theta)sin(phi)
            y = -cot(theta)cos(phi)

        with cot(theta) evaluating to zero at theta=90 degrees, and the inverse
        transform (deprojection) is given by::

            phi = arctan(x, -y)
            theta = arctan(1, sqrt(x^2 + y^2))
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
        return "TAN"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Gnomonic'

    @classmethod
    def r(cls, theta):
        """
        Return the distance of a point from the pole on the projection.

        Calculates the distance of a point from the celestial pole.  Since the
        projection defined by create circles on a sphere, this only depends on
        the latitude (theta), and is given as::

            r = cot(theta) ; |theta| > 0
            r = 0 ; theta = 90 degrees

        Parameters
        ----------
        theta : float or numpy.ndarray or units.Quantity

        Returns
        -------
        value : units.Quantity
        """
        radian = units.Unit('radian')
        if not isinstance(theta, units.Quantity):
            theta = theta * radian

        if theta.shape != ():
            equal = SphericalCoordinates.equal_angles(theta, cls.right_angle)
            result = np.empty(theta.shape, dtype=float) * radian
            result[equal] = 0.0 * radian
            result[~equal] = (1.0 / np.tan(theta[~equal])) * radian
            return result
        else:
            if SphericalCoordinates.equal_angles(theta, cls.right_angle):
                return 0.0 * radian

            return (1.0 / np.tan(theta)) * radian

    @classmethod
    def theta_of_r(cls, value):
        """
        Return the latitude (theta) given a distance from the celestial pole.

        Calculates the latitude of a point from the celestial pole.  Since the
        projection defined by create circles on a sphere, this only depends on
        the distance of the point from the celestial pole, and is given as:

            theta = arctan(1, r)

        Parameters
        ----------
        value : float or numpy.ndarray or units.Quantity

        Returns
        -------
        theta : units.Quantity
        """
        if isinstance(value, units.Quantity):
            value = value.to('radian').value

        return np.arctan2(1.0, value) * units.Unit('radian')
