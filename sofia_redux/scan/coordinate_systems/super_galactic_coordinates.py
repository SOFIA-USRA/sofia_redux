# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates
from sofia_redux.scan.coordinate_systems.galactic_coordinates import \
    GalacticCoordinates

__all__ = ['SuperGalacticCoordinates', 'PHI0']


class SuperGalacticCoordinates(CelestialCoordinates):

    GALACTIC_POLE = GalacticCoordinates([47.37, 6.32], unit='degree')
    GALACTIC_ZERO = GalacticCoordinates([137.37, 0.0], unit='degree')
    EQUATORIAL_POLE = GALACTIC_POLE.to_equatorial()

    def __init__(self, coordinates=None, unit='degree', copy=True):
        """
        Initialize a set of super-galactic coordinates.

        The super-galactic coordinates reference the plane of the local
        super-cluster of galaxies as observed from Earth.  The super-galactic
        plane intersects with the galactic plane at (137.37, 0) degrees in
        galactic coordinates, and the super-galactic pole is at (47.37, 6.32)
        degrees in galactic coordinates.

        Parameters
        ----------
        coordinates : list or tuple or array-like or units.Quantity, optional
            The coordinates used to populate the object during initialization.
            The first (0) value or index should represent longitudinal
            coordinates, and the second should represent latitude.
        unit : units.Unit or str, optional
            The angular unit for the coordinates.  The default is 'degree'.
        copy : bool, optional
            Whether to explicitly perform a copy operation on the input
            coordinates when storing them into these coordinates.  Note that it
            is extremely unlikely for the original coordinates to be passed in
            as a reference due to the significant checks performed on them.
        """
        super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the super-galactic coordinates.

        Returns
        -------
        SuperGalacticCoordinates
        """
        return super().copy()

    def setup_coordinate_system(self):
        """
        Setup the system for the coordinates.

        Returns
        -------
        None
        """
        self.default_coordinate_system = CoordinateSystem(
            name='Supergalactic Coordinates')
        self.default_local_coordinate_system = CoordinateSystem(
            name='Supergalactic Offsets')
        lon_axis = self.create_axis('Supergalactic Longitude', 'SGL')
        lon_axis.reverse = True
        lat_axis = self.create_axis('Supergalactic Latitude', 'SGB')
        lon_offset_axis = self.create_offset_axis(
            'Galactic Longitude Offset', 'dSGL')
        lon_offset_axis.reverse = True
        lat_offset_axis = self.create_offset_axis(
            'Galactic Latitude Offset', 'dSGB')
        self.default_coordinate_system.add_axis(lon_axis)
        self.default_coordinate_system.add_axis(lat_axis)
        self.default_local_coordinate_system.add_axis(lon_offset_axis)
        self.default_local_coordinate_system.add_axis(lat_offset_axis)

    @property
    def fits_longitude_stem(self):
        """
        Return the string prefix for longitude (RA).

        Returns
        -------
        str
        """
        return 'SLON'

    @property
    def fits_latitude_stem(self):
        """
        Return the string prefix for latitude (DEC).

        Returns
        -------
        str
        """
        return 'SLAT'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the coordinate system.

        Returns
        -------
        str
        """
        return 'SG'

    @property
    def equatorial_pole(self):
        """
        Return an equatorial pole.

        Returns
        -------
        pole : EquatorialCoordinates
        """
        return self.get_equatorial_pole()

    @classmethod
    def get_equatorial_pole(cls):
        """
        Return the equatorial pole.

        Returns
        -------
        pole : EquatorialCoordinates
        """
        return cls.EQUATORIAL_POLE

    @classmethod
    def get_zero_longitude(cls):
        """
        Return the zero longitude value.

        Returns
        -------
        astropy.units.Quantity
        """
        return PHI0

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with super-galactic coordinate information.

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


PHI0 = CelestialCoordinates.get_zero_longitude_from(
    SuperGalacticCoordinates.GALACTIC_ZERO, GalacticCoordinates())
