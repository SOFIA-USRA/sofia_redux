# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.cylindrical_projection \
    import CylindricalProjection

__all__ = ['PlateCarreeProjection']


class PlateCarreeProjection(CylindricalProjection):

    def __init__(self):
        """
        Initialize a plate carree projection.

        The plate carree (French for "flat square") projection is an
        equirectangular or equidistant cylindrical projection in which the
        standard parallel is zero (the equator).  Meridians are vertical
        straight lines of constant spacing, and circles of latitude are
        horizontal straight lines of constant spacing.  It is neither
        equal-area nor conformal.
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
        return "CAR"

    @classmethod
    def get_full_name(cls):
        """
        Return the full name of the projection.

        Returns
        -------
        str
        """
        return 'Plate carree'

    @classmethod
    def get_phi_theta(cls, offset, phi_theta=None):
        """
        Return the phi (longitude) and theta (latitude) coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.  For the plate carree projection these are given as:

            phi = x
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
        phi_theta.copy_coordinates(offset)
        return phi_theta

    @classmethod
    def get_offsets(cls, theta, phi, offsets=None):
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.
        For the plate carree projection, this is given by:

            x = phi
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
            offsets = Coordinate2D()

        offsets.set([phi, theta])
        return offsets
