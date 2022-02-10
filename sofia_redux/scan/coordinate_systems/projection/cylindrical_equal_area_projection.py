# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['CylindricalEqualAreaProjection']


class CylindricalEqualAreaProjection(CylindricalProjection):

    def __init__(self):
        """
        Initialize a cylindrical equal-area projection.

        The cylindrical equal-area projection maps spherical coordinates onto
        a stretched vertical cylinder, with meridians as equally spaced
        vertical lines, and parallels as horizontal lines.  In this model,
        the stretch parameter is applied to the vertical axis, but is typically
        set to 1 (Lambert projection).
        """
        super().__init__()
        self.stretch = 1.0

    @classmethod
    def get_fits_id(cls):
        """
        Return the FITS ID for the projection.

        Returns
        -------
        str
        """
        return "CEA"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Cylindrical Equal Area'

    def get_phi_theta(self, offset, phi_theta=None):
        """
        Return the phi (longitude) theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.

        The cylindrical equal-area transform is very simple, of the form:

            phi = x
            theta = arcsin(stretch * y)

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

        phi, y = self.offset_to_xy_radians(offset)
        theta = self.asin(self.stretch * y)
        phi_theta.set_native([phi, theta])
        return phi_theta

    def get_offsets(self, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the cylindrical equal-area projection, this is given as:

            x = phi
            y = stretch * sin(theta)

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
        x = phi
        y = (np.sin(theta) / self.stretch) * units.Unit('radian')
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
        stretch_key = f"{self.get_latitude_parameter_prefix()}1{alt}"
        if stretch_key in header:
            self.stretch = float(header.get(stretch_key))

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
            self.stretch,
            'lambda parameter for cylindrical equal area projection.')
