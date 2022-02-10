# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['BonnesProjection']


class BonnesProjection(SphericalProjection):

    def __init__(self):
        """
        Initialize a Bonne spherical projection.

        The Bonne projection is a pseudo-canonical equal-area projection
        designed to maintain accurate shapes of areas along the central
        meridian (y0) and standard parallel (theta1).  Distortion is noticeable
        from this region, so it is best used to map "T"-shaped regions.

        Notes
        -----
        By default, the theta1 parameter (standard parallel) is set to zero.
        As such, forward and reverse projections will be inconsistent as
        cot(0) is undefined.  Therefore, the user should always make sure to
        explicitly set theta 1.   If a Bonne projection is required at
        theta1=0 (equator), the global sinusoidal projection should be used
        instead.
        """
        super().__init__()
        self._theta_1 = 0.0 * units.Unit('radian')
        self._y0 = 0.0 * units.Unit('radian')

    @classmethod
    def get_fits_id(cls):
        """
        Return the FITS ID for the projection.

        Returns
        -------
        str
        """
        return "BON"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return "Bonne's Projection"

    @property
    def theta1(self):
        """
        Return the theta1 angle.

        The theta1 angle is the standard parallel of the projection (line with
        no distortion on the projection).

        Returns
        -------
        units.Quantity
        """
        return self._theta_1

    @theta1.setter
    def theta1(self, value):
        """
        Set the theta1 angle.

        The theta1 angle is the standard parallel of the projection (line with
        no distortion on the projection).  Note that setting theta1 also sets
        the native reference latitude to the same value, and the central
        meridian (y0) to:

            y0 = theta1 + cot(theta1)

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.set_theta_1(value)

    @property
    def y0(self):
        """
        Return the y0 angle.

        The y0 defines the central meridian of the Bonne projection.

        Returns
        -------
        units.Quantity
        """
        return self._y0

    @y0.setter
    def y0(self, value):
        """
        Set the y0 value.

        The y0 defines the central meridian of the Bonne projection.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if not isinstance(value, units.Quantity) or (
                value.unit == units.dimensionless_unscaled):
            self._y0 = value * units.Unit('radian')
        else:
            self._y0 = value

    def set_theta_1(self, value):
        """
        Set the theta_1 value.

        The theta 1 parameter in the Bonne projection is the standard parallel
        (line where there is no distortion in the map projection).  For this
        projection, setting theta1 sets the projection native reference
        latitude to the same value, and the central meridian (y0) to:

           y0 = theta1 + cot(theta1)

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if value == self.theta1:
            return
        self._theta_1 = value
        self.native_reference.set_y(self.theta1)
        dy = (1.0 / np.tan(self.theta1)) * units.Unit('radian')
        self.y0 = self.theta1 + dy

    def get_phi_theta(self, offset, phi_theta=None):
        """
        Return the phi_theta coordinates for the Bonne projection.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets to the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For the Bonne projection, phi and theta are given by:

            dy = y0 - y
            r = sign(theta1) * sqrt(x^2 + dy^2)
            a = arctan(x, dy)

            theta = y0 - r
            phi = a * r / cos(theta)

        where y0 is the central meridian and theta1 is the standard parallel.

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

        r = np.hypot(x, self.y0 - y)
        if self.theta1 < 0:
            r = -r

        theta = self.y0 - r
        a = np.arctan2(x, self.y0 - y)
        phi_theta.set_y(theta)

        phi = (a * r / phi_theta.cos_lat).decompose().value
        phi = phi * units.Unit('radian')
        phi_theta.set_x(phi)
        return phi_theta

    def get_offsets(self, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the Bonne projection these are given by:

            r = y0 - theta
            a = phi * cos(theta) / r
            dx = r * sin(a)
            dy = y0 - (r * cos(a))

        where y0 is the central meridian.

        Parameters
        ----------
        theta : units.Quantity
            The theta angle (latitude about the celestial pole).
        phi : units.Quantity
            The phi angle (longitude about the celestial pole).
        offsets : Coordinate2D, optional
            An optional coordinate system in which to place the results.

        Returns
        -------
        offsets : Coordinate2D
        """
        if offsets is None:
            offsets = Coordinate2D(unit='degree')

        phi, theta = self.phi_theta_to_radians(phi, theta)
        r = self.y0 - theta
        a = phi * np.cos(theta) / r
        a = a.decompose().value * units.Unit('radian')

        x = r * np.sin(a)
        y = self.y0 - r * np.cos(a)
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
        name = f'{self.get_latitude_parameter_prefix()}1{alt}'
        if name in header:
            self.set_theta_1(header[name] * units.Unit('degree'))

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
        name = f'{self.get_latitude_parameter_prefix()}1{alt}'
        header[name] = (self.theta1.to('degree').value,
                        "Theta1 parameter for Bonne's projection")
