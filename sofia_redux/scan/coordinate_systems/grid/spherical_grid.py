# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection

__all__ = ['SphericalGrid']


class SphericalGrid(Grid2D):

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
        reference : SphericalCoordinates
            The reference coordinate from which to base any projections.
        """
        super().__init__()
        # default Gnomonic projection
        self.projection = SphericalProjection.for_name('TAN')
        if isinstance(reference, SphericalCoordinates):
            self.set_reference(reference)
        elif reference is not None:
            raise ValueError(f"Reference must be {SphericalCoordinates} "
                             f"class. Received {reference}.")

    def set_defaults(self):
        """
        Set the default values for the grid.

        Returns
        -------
        None
        """
        super().set_defaults()
        self.set_coordinate_system(
            SphericalCoordinates.get_default_system()[0])

    @property
    def fits_x_unit(self):
        """
        Return the unit for the FITS x-axis.

        Returns
        -------
        units.Unit
        """
        return units.Unit('degree')

    @property
    def fits_y_unit(self):
        """
        Return the unit for the FITS y-axis.

        Returns
        -------
        units.Unit
        """
        return units.Unit('degree')

    @classmethod
    def get_default_unit(cls):
        """
        Return the default unit for the grid class.

        Returns
        -------
        unit : units.Unit
        """
        return units.Unit('degree')

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
        return SphericalCoordinates.get_fits_class(ctype)()

    def set_reference(self, reference):
        """
        Set the reference position of the grid.

        Parameters
        ----------
        reference : SphericalCoordinates

        Returns
        -------
        None
        """
        super().set_reference(reference)
        self.set_coordinate_system(reference.coordinate_system)

    def is_reverse_x(self):
        """
        Returns if the x-axis is reversed.

        Returns
        -------
        bool
        """
        return self.reference.reverse_longitude

    def is_reverse_y(self):
        """
        Returns if the y-axis is reversed.

        Returns
        -------
        bool
        """
        return self.reference.reverse_latitude

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
        alt = self.get_fits_id()
        ctype = header.get(f'CTYPE1{alt}')
        if len(ctype) < 6:
            raise ValueError(f"Cannot extract projection from CTYPE={ctype}")
        projection = ctype[5:]
        try:
            self.set_projection(SphericalProjection.for_name(projection))
        except ValueError:
            raise ValueError(f"Unknown projection: {projection}.")
