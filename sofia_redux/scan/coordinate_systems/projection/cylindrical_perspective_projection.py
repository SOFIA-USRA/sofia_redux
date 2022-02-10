# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['CylindricalPerspectiveProjection']


class CylindricalPerspectiveProjection(CylindricalProjection):

    def __init__(self):
        """
        Create a cylindrical perspective projection.

        The cylindrical perspective projection is constructed geometrically by
        projecting a sphere onto a tangent cylinder from the point on the
        equatorial plane opposite a given meridian.  The attributes `mu` and
        `la` (lambda) give the relative scaling of such a cylinder to the
        sphere, so that if `r` is the radius of the sphere:

           cylinder_radius = lambda * r

        and a point of projection moves around a circle of radius

           circle_radius = mu * r

        in the equatorial plane of the sphere, depending on the projected
        meridian.

        """
        super().__init__()
        self.mu = 1.0
        self.la = 1.0

    @classmethod
    def get_fits_id(cls):
        """
        Return the FITS ID for the projection.

        Returns
        -------
        str
        """
        return "CYP"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Cylindrical Perspective'

    def get_phi_theta(self, offset, phi_theta=None):
        """
        Return the phi (longitude) theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.

        For the cylindrical perspective projection, phi and theta are given by:

            phi = x / lambda
            theta = arctan(eta, 1) + arcsin(eta * mu / sqrt(1 + eta^2))

        where

            eta = y / (mu + lambda)

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

        x, y = self.offset_to_xy_radians(offset)
        phi = x / self.la

        eta = (y / (self.mu + self.la)).value
        theta = np.arctan2(eta, 1.0) * units.Unit('radian')
        theta += self.asin(eta * self.mu / np.hypot(eta, 1.0))

        phi_theta.set_native([phi, theta])
        return phi_theta

    def get_offsets(self, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the cylindrical perspective projection, this is given as:

            x = lambda * phi
            y = (mu + lambda) / (mu + cos(theta) * sin(theta))

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

        phi, theta = self.phi_theta_to_radians(phi, theta)
        x = self.la * phi
        y = (self.mu + self.la) / (self.mu + np.cos(theta)) * np.sin(theta)
        y = y * units.Unit('radian')
        offsets.set([x, y])
        return offsets

    def parse_header(self, header, alt=''):
        """
        Parse and apply a FITS header to the projection.

        Parameters
        ----------
        header : fits.Header
            The FITS header to parse.
        alt : str, optional
            The alternate FITS system.

        Returns
        -------
        None
        """
        super().parse_header(header, alt=alt)
        mu_name = f"{self.get_latitude_parameter_prefix()}1{alt}"
        if mu_name in header:
            self.mu = float(header.get(mu_name))
        lambda_name = f"{self.get_latitude_parameter_prefix()}2{alt}"
        if lambda_name in header:
            self.la = float(header.get(lambda_name))

    def edit_header(self, header, alt=''):
        """
        Edit a FITS header with the projection information.

        Parameters
        ----------
        header : fits.Header
            The FITS header to edit.
        alt : str, optional
            The alternate FITS system.

        Returns
        -------
        None
        """
        super().edit_header(header, alt=alt)
        lat_prefix = self.get_latitude_parameter_prefix()
        header[f"{lat_prefix}1{alt}"] = (
            self.mu, 'mu parameter for cylindrical perspective.')
        header[f"{lat_prefix}2{alt}"] = (
            self.la, 'lambda parameter for cylindrical perspective.')
