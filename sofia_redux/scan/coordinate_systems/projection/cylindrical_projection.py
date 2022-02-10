# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod

from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection

__all__ = ['CylindricalProjection']


class CylindricalProjection(SphericalProjection):  # pragma: no cover
    """
    An abstract cylindrical projection class.

    A cylindrical projection normally defines one in which meridians are mapped
    to equally spaced vertical lines and parallels are mapped to horizontal
    lines.
    """

    @abstractmethod
    def get_phi_theta(self, offset, phi_theta=None):  # pragma: no cover
        """
        Return the phi_theta coordinates.

        Parameters
        ----------
        offset : Coordinate2D
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        pass

    @abstractmethod
    def get_offsets(self, theta, phi, offsets=None):  # pragma: no cover
        """
        Get the offsets given theta and phi.

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
        pass
