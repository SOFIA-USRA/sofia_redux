# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['SansonFlamsteedProjection']


class SansonFlamsteedProjection(CylindricalProjection):

    def __init__(self):
        """
        Initialize a Sanson-Flamsteed projection.

        The Sanson-Flamsteed projection is also known as the sinusoidal
        projection, and is a pseudo-cylindrical equal-area map projection.
        Poles are represented as points.  Scale is constant along the central
        meridian, and the east-west scale is constant.
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
        return "SFL"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Sanson-Flamsteed'

    @classmethod
    def get_phi_theta(cls, offset, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For the Sanson-Flamsteed projection these are given as:

            phi = x / cos(y)
            theta = y

        Parameters
        ----------
        offset : Coordinate2D
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        if phi_theta is None:
            phi_theta = SphericalCoordinates(unit='degree')
        x, y = cls.offset_to_xy_radians(offset)

        phi_theta.set_y(y)
        phi_theta.set_x(x / phi_theta.cos_lat)
        return phi_theta

    @classmethod
    def get_offsets(cls, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the Sanson-Flamsteed projection, this is given by:

            x = phi * cos(theta)
            y = theta

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

        phi, theta = cls.phi_theta_to_radians(phi, theta)
        offsets.set([phi * np.cos(theta), theta])
        return offsets
