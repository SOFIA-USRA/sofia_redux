# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['MercatorProjection']


class MercatorProjection(CylindricalProjection):

    def __init__(self):
        """
        Initialize a Mercator projection.

        The Mercator projection is a conformal projection in which lines of
        constant bearing are displayed as straight lines while somewhat
        preserving local directions and shapes.
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
        return "MER"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Mercator'

    @classmethod
    def get_phi_theta(cls, offset, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For the Mercator projection these are given as:

            phi = x
            theta = 2 * arctan(exp(y)) - pi/2

        Parameters
        ----------
        offset : Coordinate2D
            The projected offset to convert.  If the units are undefined or
            dimensionless, they are assumed to be in radians.
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        if phi_theta is None:
            phi_theta = SphericalCoordinates(unit='degree')

        phi, y = cls.offset_to_xy_radians(offset)

        y = y.value
        right_angle = cls.right_angle.to('radian').value
        theta = (2 * np.arctan(np.exp(y)) - right_angle) * units.Unit('radian')
        phi_theta.set_native([phi, theta])
        return phi_theta

    @classmethod
    def get_offsets(cls, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the Mercator projection, this is given by:

            x = phi
            y = ln(tan( pi/4 + theta/2 ))

        Parameters
        ----------
        theta : units.Quantity or float
            The theta (latitude) angle.  If a dimensionless or float value is
            supplied, it is assumed to be in radians.
        phi : units.Quantity or float
            The phi (longitude) angle.  If a dimensionless or float value is
            supplied, it is assumed to be in radians.
        offsets : Coordinate2D, optional
            An optional coordinate system in which to place the results.

        Returns
        -------
        offsets : Coordinate2D
        """
        if offsets is None:
            offsets = Coordinate2D(unit='degree')

        x, theta = cls.phi_theta_to_radians(phi, theta)

        rad = units.Unit('radian')
        dy = (cls.right_angle + theta).to(rad).value
        y = np.log(np.tan(0.5 * dy)) * rad
        offsets.set([x, y])
        return offsets
