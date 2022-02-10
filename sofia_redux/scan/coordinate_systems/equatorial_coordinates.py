# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.coordinates import Angle
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.precessing_coordinates import \
    PrecessingCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.coordinate_systems.epoch.precession import Precession
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem

__all__ = ['EquatorialCoordinates', 'EQUATORIAL_POLE']


class EquatorialCoordinates(PrecessingCoordinates):

    NORTH = 1
    SOUTH = -1
    EAST = -1
    WEST = 1
    ZERO_LONGITUDE = 0.0 * units.Unit('degree')

    def __init__(self, coordinates=None, unit='degree',
                 copy=True, epoch=J2000):
        super().__init__(coordinates=coordinates, unit=unit, copy=copy,
                         epoch=epoch)

    def copy(self):
        """
        Return a copy of the equatorial coordinates.

        Returns
        -------
        EquatorialCoordinates
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
            name='Equatorial Coordinates')
        self.default_local_coordinate_system = CoordinateSystem(
            name='Equatorial Offsets')
        ra_axis = self.create_axis('Right Ascension', 'RA')
        ra_axis.reverse = True
        dec_axis = self.create_axis('Declination', 'DEC')
        ra_offset_axis = self.create_offset_axis(
            'Right Ascension Offset', 'dRA')
        ra_offset_axis.reverse = True
        dec_offset_axis = self.create_offset_axis(
            'Declination Offset', 'dDEC')
        self.default_coordinate_system.add_axis(ra_axis)
        self.default_coordinate_system.add_axis(dec_axis)
        self.default_local_coordinate_system.add_axis(ra_offset_axis)
        self.default_local_coordinate_system.add_axis(dec_offset_axis)

    @property
    def fits_longitude_stem(self):
        """
        Return the string prefix for longitude (RA).

        Returns
        -------
        str
        """
        return 'RA--'

    @property
    def fits_latitude_stem(self):
        """
        Return the string prefix for latitude (DEC).

        Returns
        -------
        str
        """
        return 'DEC-'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the coordinate system.

        Returns
        -------
        str
        """
        return 'EQ'

    @property
    def ra(self):
        """
        Return the Right Ascension coordinate(s).

        Returns
        -------
        right_ascension : units.Quantity
        """
        if self.coordinates is None:
            return None
        return self.zero_to_two_pi(self.longitude)

    @ra.setter
    def ra(self, values):
        """
        Set the Right Ascension.

        Parameters
        ----------
        values : units.Quantity

        Returns
        -------
        None
        """
        self.set_ra(values, copy=True)

    @property
    def dec(self):
        """
        Return the Declination coordinate(s).

        Returns
        -------
        declination : units.Quantity
        """
        if self.coordinates is None:
            return None
        return self.latitude

    @dec.setter
    def dec(self, values):
        """
        Set the Declination.

        Parameters
        ----------
        values : units.Quantity

        Returns
        -------
        None
        """
        self.set_dec(values, copy=True)

    @property
    def equatorial_pole(self):
        """
        Return an equatorial pole.

        Returns
        -------
        pole : EquatorialCoordinates
        """
        return self.get_equatorial_pole()

    @property
    def zero_longitude(self):
        """
        Return the zero longitude coordinate.

        Returns
        -------
        longitude : units.Quantity
        """
        return self.get_zero_longitude()

    def __str__(self):
        """
        Create a string representation of the equatorial coordinates.

        Returns
        -------
        str
        """
        if self.coordinates is None:
            return f'Empty coordinates ({self.epoch})'

        ra = self.ra
        dec = self.dec

        if self.singular:
            if np.isnan(ra):
                ra_string = 'NaN'
            else:
                ra_string = Angle(self.ra).to_string(unit='hourangle')
            if np.isnan(dec):
                dec_string = 'NaN'
            else:
                dec_string = Angle(self.dec).to_string(unit='degree')
            return f'RA={ra_string} DEC={dec_string} ({self.epoch})'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                min_ra = np.nanmin(ra)
                if np.isnan(min_ra):
                    min_ra = max_ra = 'NaN'
                else:
                    min_ra = Angle(min_ra).to_string(unit='hourangle')
                    max_ra = Angle(np.nanmax(self.ra)).to_string(
                        unit='hourangle')
                min_dec = np.nanmin(dec)
                if np.isnan(min_dec):
                    min_dec = max_dec = 'NaN'
                else:
                    min_dec = Angle(min_dec).to_string(unit='degree')
                    max_dec = Angle(np.nanmax(self.dec)).to_string(
                        unit='degree')
                return (f'RA={min_ra}->{max_ra} DEC={min_dec}->{max_dec} '
                        f'({self.epoch})')

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        EquatorialCoordinates
        """
        return super().__getitem__(indices)

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
        Return the zero longitude coordinate.

        Returns
        -------
        lon : units.Quantity (float)
        """
        return cls.ZERO_LONGITUDE

    @classmethod
    def to_horizontal_offset(cls, offset, position_angle, in_place=True):
        """
        Convert an equatorial offset to a horizontal offset.

        Parameters
        ----------
        offset : Coordinate2D
            The 2-dimensional equatorial offsets to convert
        position_angle : units.Quantity
            The position angle as a scalar or shape (n,).
        in_place : bool, optional
            If `True`, perform the conversion in-place.  Otherwise, return a
            copy while leaving the original `offset` unchanged

        Returns
        -------
        horizontal_offset : Coordinate2D
            The (x, y) horizontal offsets.
        """
        if not in_place:
            offset = offset.copy()
        offset.scale_x(-1)
        cls.rotate_offsets(offset, -position_angle)
        return offset

    def set_ra(self, ra, copy=True):
        """
        Set the Right Ascension values.

        Parameters
        ----------
        ra : units.Quantity
        copy : bool, optional
            If `True`, copy the coordinates.

        Returns
        -------
        None
        """
        self.set_longitude(ra, copy=copy)

    def set_dec(self, dec, copy=True):
        """
        Set the Declination values.

        Parameters
        ----------
        dec : units.Quantity
        copy : bool, optional
            If `True`, copy the coordinates.

        Returns
        -------
        None
        """
        self.set_latitude(dec, copy=copy)

    def get_parallactic_angle(self, site, lst):
        """
        Return the parallactic angle for the coordinates.

        Parameters
        ----------
        site : GeodeticCoordinates
            The site coordinates.
        lst : units.Quantity
            The local sidereal time.

        Returns
        -------
        angle : units.Quantity
        """
        if self.coordinates is None:
            return None
        ra = self.ra
        lst = self.correct_factor_dimensions(lst, ra)
        site_cos_lat = self.correct_factor_dimensions(site.cos_lat, ra)
        site_sin_lat = self.correct_factor_dimensions(site.sin_lat, ra)
        h = lst - ra
        y = site_cos_lat * np.sin(h).value
        x = site_sin_lat * self.cos_lat
        x -= site_cos_lat * self.sin_lat * np.cos(h).value
        return np.arctan2(y, x) * units.Unit('radian')

    def get_equatorial_position_angle(self):
        """
        Return the equatorial position angle.

        Returns
        -------
        position_angle : units.Quantity
        """
        if self.singular:
            return 0.0 * units.Unit('radian')
        return np.zeros(self.shape) * units.Unit('radian')

    def to_equatorial(self, coordinates=None):
        """
        Set the given coordinates to these equatorial coordinates.

        Parameters
        ----------
        coordinates : EquatorialCoordinates, optional
            The coordinates to convert.  If not supplied, defaults to *this*.

        Returns
        -------
        coordinates : EquatorialCoordinates
        """
        if coordinates is None:
            coordinates = self.get_instance('equatorial')
        coordinates.copy_coordinates(self)
        return coordinates

    def from_equatorial(self, coordinates):
        """
        Set the equatorial coordinates from those given.

        Parameters
        ----------
        coordinates : EquatorialCoordinates

        Returns
        -------
        None
        """
        self.copy_coordinates(coordinates)

    def to_horizontal(self, site, lst, equatorial=None, horizontal=None):
        """
        Convert equatorial coordinates to horizontal coordinates.

        Parameters
        ----------
        site : GeodeticCoordinates
            The site coordinates.
        lst : units.Quantity
            The local sidereal time.
        equatorial : EquatorialCoordinates, optional
            The equatorial coordinates to convert.  The default are *these*
            coordinates.
        horizontal : HorizontalCoordinates, optional
            The horizontal coordinates to convert to.  The default is a fresh
            set of HorizontalCoordinates.

        Returns
        -------
        horizontal : HorizontalCoordinates
            The equatorial coordinates as a horizontal representation.
        """
        if equatorial is None:
            equatorial = self
        if horizontal is None:
            horizontal = self.get_instance('horizontal')

        ra = equatorial.ra
        lst = self.correct_factor_dimensions(lst, ra)
        if lst.unit == units.Unit('hour'):
            lst = lst.value * units.Unit('hourangle')

        h = lst - ra
        cos_h = np.cos(h)

        site_sin_lat = self.correct_factor_dimensions(
            site.sin_lat, equatorial.sin_lat)
        site_cos_lat = self.correct_factor_dimensions(
            site.cos_lat, equatorial.cos_lat)

        sin2_lat = equatorial.sin_lat * site_sin_lat
        cos2_lat = equatorial.cos_lat * site_cos_lat

        horizontal.set_latitude(np.arcsin(sin2_lat + (cos2_lat * cos_h)),
                                copy=False)
        asin_a = -np.sin(h) * cos2_lat
        acos_a = equatorial.sin_lat - (site_sin_lat * horizontal.sin_lat)
        horizontal.set_longitude(np.arctan2(asin_a, acos_a), copy=False)
        return horizontal

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
        if self.epoch == new_epoch:
            return
        precession = Precession(self.epoch, new_epoch)
        precession.precess(self)

    def convert(self, from_coordinates, to_coordinates):
        """
        Convert one type of coordinates to another.

        The `to_coordinates` will be updated in-place.

        Parameters
        ----------
        from_coordinates : CelestialCoordinates
        to_coordinates : CelestialCoordinates

        Returns
        -------
        None
        """
        super().convert(from_coordinates, to_coordinates)
        if isinstance(from_coordinates, EquatorialCoordinates):
            from_coordinates.to_equatorial(to_coordinates)
        elif isinstance(to_coordinates, EquatorialCoordinates):
            to_coordinates.from_equatorial(from_coordinates)
        else:
            temp_equatorial = self.get_equatorial_class()()
            temp_equatorial.from_equatorial(from_coordinates)
            temp_equatorial.to_equatorial(to_coordinates)

    def plot(self, *args, **kwargs):  # pragma: no cover
        """
        Plot the coordinates.

        Parameters
        ----------
        args : values
            Optional positional parameters to pass into pyplot.plot.
        kwargs : dict, optional
            Optional keyword arguments.

        Returns
        -------
        None
        """
        from matplotlib import pyplot as plt
        plt.ion()
        c_args = self.ra.ravel(), self.dec.ravel()
        if args is not None:
            c_args += args

        plt.plot(*c_args, **kwargs)
        plt.xlabel(f'{self.longitude_axis.label} ({self.unit})')
        plt.ylabel(f'{self.latitude_axis.label} ({self.unit})')

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with equatorial coordinate information.

        Parameters
        ----------
        header : fits.Header
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


EQUATORIAL_POLE = EquatorialCoordinates(
    np.array([0.0, 90.0]) * units.Unit('degree'), epoch=J2000)
