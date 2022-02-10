# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import B1950, J2000
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates

__all__ = ['GalacticCoordinates']


class GalacticCoordinates(CelestialCoordinates):

    def __init__(self, coordinates=None, unit='degree', copy=True):
        super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def setup_coordinate_system(self):
        """
        Setup the system for the coordinates.

        Returns
        -------
        None
        """
        self.default_coordinate_system = CoordinateSystem(
            name='Galactic Coordinates')
        self.default_local_coordinate_system = CoordinateSystem(
            name='Galactic Offsets')
        lon_axis = self.create_axis('Galactic Longitude', 'GLON')
        lon_axis.reverse = True
        lat_axis = self.create_axis('Galactic Latitude', 'GLAT')
        lon_offset_axis = self.create_offset_axis(
            'Galactic Longitude Offset', 'dGLON')
        lon_offset_axis.reverse = True
        lat_offset_axis = self.create_offset_axis(
            'Galactic Latitude Offset', 'dGLAT')
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
        return 'GLON'

    @property
    def fits_latitude_stem(self):
        """
        Return the string prefix for latitude (DEC).

        Returns
        -------
        str
        """
        return 'GLAT'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the coordinate system.

        Returns
        -------
        str
        """
        return 'GA'

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
        return EQUATORIAL_POLE

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
        Edit the header with galactic coordinate information.

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


# The following is used to determine the equatorial pole and zero longitude
_pole_ra = ((12 * units.Unit('hourangle'))
            + (49 * units.Unit('hourangle') / 60))
_pole_dec = 27.4 * units.Unit('degree')

EQUATORIAL_POLE = EquatorialCoordinates([_pole_ra, _pole_dec], epoch=B1950)
PHI0 = 123 * units.Unit('degree')
zero = GalacticCoordinates([PHI0, 0.0 * units.Unit('degree')])
equatorial_zero = zero.to_equatorial()
equatorial_zero.precess(J2000)
zero.from_equatorial(equatorial_zero)
PHI0 = -zero.x
EQUATORIAL_POLE.precess(J2000)
