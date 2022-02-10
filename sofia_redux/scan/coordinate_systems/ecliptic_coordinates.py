# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.precessing_coordinates import \
    PrecessingCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates

__all__ = ['EclipticCoordinates']


class EclipticCoordinates(PrecessingCoordinates):

    INCLINATION = ((23 * units.Unit('degree'))
                   + (26 * units.Unit('arcmin'))
                   + (30 * units.Unit('arcsec')))
    EQUATORIAL_POLE = CelestialCoordinates.get_pole(
        INCLINATION, 0.0 * units.Unit('degree'))

    def __init__(self, coordinates=None, unit='degree',
                 copy=True, epoch=J2000):
        super().__init__(coordinates=coordinates, unit=unit, copy=copy,
                         epoch=epoch)

    def setup_coordinate_system(self):
        """
        Setup the system for the coordinates.

        Returns
        -------
        None
        """
        self.default_coordinate_system = CoordinateSystem(
            name='Ecliptic Coordinates')
        self.default_local_coordinate_system = CoordinateSystem(
            name='Ecliptic Offsets')
        lon_axis = self.create_axis('Ecliptic Longitude', 'ELON')
        lon_axis.reverse = True
        lat_axis = self.create_axis('Ecliptic Latitude', 'ELAT')
        lon_offset_axis = self.create_offset_axis(
            'Ecliptic Longitude Offset', 'dELON')
        lon_offset_axis.reverse = True
        lat_offset_axis = self.create_offset_axis(
            'Ecliptic Latitude Offset', 'dELAT')
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
        return 'ELON'

    @property
    def fits_latitude_stem(self):
        """
        Return the string prefix for latitude (DEC).

        Returns
        -------
        str
        """
        return 'ELAT'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the coordinate system.

        Returns
        -------
        str
        """
        return 'EC'

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
        return cls.right_angle

    def precess_to_epoch(self, new_epoch):
        """
        Precess from one epoch to another.

        Parameters
        ----------
        new_epoch : Epoch

        Returns
        -------
        None
        """
        equatorial = self.to_equatorial()
        equatorial.precess(new_epoch)
        self.from_equatorial(equatorial)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with ecliptic coordinate information.

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
