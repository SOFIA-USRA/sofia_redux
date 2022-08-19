# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.grid.grid_2d1 import Grid2D1
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates

__all__ = ['SphericalGrid2D1']


class SphericalGrid2D1(Grid2D1, SphericalGrid):

    def __init__(self, reference=None):
        """
        Initialize a spherical grid.

        The spherical grid is used to convert between longitude/latitude
        coordinates on a sphere and offsets in relation to a specified
        reference position onto a regular grid, and the reverse operation.

        Forward transform: grid projection -> offsets -> coordinates
        Reverse transform: coordinates -> offsets -> grid projection

        Parameters
        ----------
        reference : Coordinates2D1
            The reference coordinate from which to base any projections.
        """
        super().__init__()
        if reference is not None:
            self.set_reference(reference)

    @property
    def reference(self):
        """
        Return the reference value for the spherical grid.

        Returns
        -------
        Coordinate2D1
        """
        return super().reference

    @reference.setter
    def reference(self, value):
        """
        Return the reference value for the spherical grid.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        self.set_reference(value)

    @property
    def fits_z_unit(self):
        """
        Return the unit for the FITS z-axis

        Returns
        -------
        units.Unit
        """
        return self.z.unit

    @classmethod
    def get_default_xy_unit(cls):
        """
        Return the default unit for the grid (xy) dimensions.

        Returns
        -------
        unit : units.Unit
        """
        return units.Unit('degree')

    def get_default_z_unit(self):
        """
        Return the default unit for the grid z-dimension.

        Returns
        -------
        unit : units.Unit
        """
        return self.z.get_default_unit()

    @classmethod
    def get_coordinate_instance_for(cls, ctype):
        """
        Return a coordinate instance for the given name.

        Parameters
        ----------
        ctype : str

        Returns
        -------
        SphericalCoordinates
        """
        xy = super().get_coordinate_instance_for(ctype)
        return Coordinate2D1(xy=xy)

    def set_reference(self, reference):
        """
        Set the reference position of the grid.

        Parameters
        ----------
        reference : Coordinate2D1

        Returns
        -------
        None
        """
        super().set_reference(reference)
        self.z.set_reference(reference.z_coordinates)

    def is_reverse_x(self):
        """
        Returns if the x-axis is reversed.

        Returns
        -------
        bool
        """
        return self.reference.xy_coordinates.reverse_longitude

    def is_reverse_y(self):
        """
        Returns if the y-axis is reversed.

        Returns
        -------
        bool
        """
        return self.reference.xy_coordinates.reverse_latitude

    def is_reverse_z(self):
        """
        Returns if the x-axis is reversed.

        Returns
        -------
        bool
        """
        return self.z_axis.reverse

    def parse_projection(self, header):
        """
        Parse the projection from the header.

        The projection is taken from the CTYPE1 value in the header (if no
        alternate designation is prescribed), beginning with the 5th character.
        For example, RA---TAN parses "-TAN" to create a gnomonic projection.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        super().parse_projection(header)
        self.z.parse_header(header)
