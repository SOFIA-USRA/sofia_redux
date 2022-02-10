# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['ParabolicProjection']


class ParabolicProjection(CylindricalProjection):

    def __init__(self):
        """
        Initialize a parabolic projection.

        The parabolic projection is an equal-area pseudo-cylindrical projection
        where a meridian follows a section of a parabolic curve, and the
        projected equator and central meridian are straight lines.  Parallels
        are unequally spaced straight lines, with spacing decreasing away from
        the equator.
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
        return "PAR"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Parabolic Projection'

    @classmethod
    def get_phi_theta(cls, offset, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For the parabolic projection these are given as:

            y0 = y / pi
            phi = x / (1 - 4(y0^2))
            theta = 3 * arcsin(y0)

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

        y0 = (y / cls.pi).decompose().value
        theta = 3 * cls.asin(y0)
        phi = x / (1 - (4 * y0 * y0))
        phi_theta.set_native([phi, theta])
        return phi_theta

    @classmethod
    def get_offsets(cls, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the parabolic projection, this is given by:

            x = phi * 2 * cos(cos(2 * theta) / 3) - 1
            y = pi * sin(theta / 3)

        Parameters
        ----------
        theta : units.Quantity
            The theta (latitude) angle.
        phi : units.Quantity
            The phi (longitude) angle.
        offsets : Coordinate2D, optional
            An optional coordinate system in which to place the results.

        Returns
        -------
        offsets : Coordinate2D
        """
        if offsets is None:
            offsets = Coordinate2D(unit='degree')

        phi, theta = cls.phi_theta_to_radians(phi, theta)
        x0 = (2 * np.cos(2 * theta / 3)).decompose().value - 1
        x = phi * x0
        y = cls.pi * np.sin(theta / 3)
        offsets.set([x, y])
        return offsets
