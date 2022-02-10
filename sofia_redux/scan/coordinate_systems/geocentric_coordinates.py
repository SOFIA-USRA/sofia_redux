# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates

__all__ = ['GeocentricCoordinates']


class GeocentricCoordinates(SphericalCoordinates):

    NORTH = 1
    SOUTH = -1
    EAST = 1
    WEST = -1

    def __init__(self, coordinates=None, unit='degree', copy=True):
        """
        Initialize a set of Geocentric coordinates.

        Geocentric coordinates (also known as the Earth-centered, Earth-fixed
        coordinate system, or ECEF) represent locations on the Earth surface as
        longitude latitude positions.


        Parameters
        ----------
        coordinates : list or tuple or array-like or units.Quantity, optional
            The coordinates used to populate the object during initialization.
            The first (0) value or index should represent longitudinal
            coordinates, and the second should represent latitude.
        unit : units.Unit or str, optional
            The angular unit for the spherical coordinates.  The default is
            'degree'.
        copy : bool, optional
            Whether to explicitly perform a copy operation on the input
            coordinates when storing them into these coordinates.  Note that it
            is extremely unlikely for the original coordinates to be passed in
            as a reference due to the significant checks performed on them.
        """
        super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the geocentric coordinates.

        Returns
        -------
        GeocentricCoordinates
        """
        return super().copy()

    @classmethod
    def get_default_system(cls):
        """
        Return the default and local default coordinate system.

        Returns
        -------
        system, local_system : (CoordinateSystem, CoordinateSystem)
        """
        system, local_system = super().get_default_system()
        system.name = 'Geocentric Coordinates'
        local_system.name = 'Geocentric Offsets'
        return system, local_system

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the Geodetic coordinate system.

        Returns
        -------
        code : str
        """
        return 'GC'

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        GeocentricCoordinates
        """
        return super().__getitem__(indices)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with geocentric coordinate information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to modify.
        key_stem : str
            The name of the header key to update.
        alt : str, optional
            The alternative coordinate system.

        Returns
        -------
        None
        """
        if not self.singular:
            return  # Can't do this for multiple coordinates
        super().edit_header(header, key_stem, alt=alt)
        header[f'WCSNAME{alt}'] = (self.coordinate_system.name,
                                   'coordinate system description.')
