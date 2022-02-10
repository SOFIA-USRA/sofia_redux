# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.geocentric_coordinates import \
    GeocentricCoordinates

__all__ = ['GeodeticCoordinates']


class GeodeticCoordinates(SphericalCoordinates):

    A = 6378137.0 * units.Unit('meter')  # Earth major axis
    B = 6356752.3 * units.Unit('meter')  # Earth minor axis
    F = (1.0 / 298257.0)  # Flattening of Earth (Marik: Csillagaszat)

    # Approximation term for geodesic conversion (Marik: Csillagaszat)
    Z = 103132.4 * units.Unit('degree') * (2 * F - (F ** 2))
    NORTH = 1
    SOUTH = -1
    EAST = 1
    WEST = -1

    def __init__(self, coordinates=None, unit='degree', copy=True):
        """
        Initialize a set of Geodetic coordinates.

        Geodetic coordinates are based on a reference ellipsoid model of the
        Earth using geodetic longitude, latitude, and height.

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
        if isinstance(coordinates, GeocentricCoordinates):
            super().__init__(coordinates=None, unit=unit)
            self.unit = units.Unit(unit)
            self.from_geocentric(coordinates)
        else:
            super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the geodetic coordinates.

        Returns
        -------
        GeodeticCoordinates
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
        system.name = 'Geodetic Coordinates'
        local_system.name = 'Geodetic Offsets'
        return system, local_system

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the Geodetic coordinate system.

        Returns
        -------
        code : str
        """
        return 'GD'

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        GeodeticCoordinates
        """
        return super().__getitem__(indices)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with geodetic coordinate information.

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

    def from_geocentric(self, geocentric):
        """
        Convert from a geocentric representation.

        Parameters
        ----------
        geocentric : GeocentricCoordinates

        Returns
        -------
        None
        """
        self.set_native_longitude(geocentric.x)
        self.set_native_latitude(
            geocentric.y + (self.Z * np.sin(2 * geocentric.y)))
        return geocentric
