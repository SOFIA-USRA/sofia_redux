# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.coordinates import Angle
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem

__all__ = ['HorizontalCoordinates']


class HorizontalCoordinates(SphericalCoordinates):

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
            name='Horizontal Coordinates')
        self.default_local_coordinate_system = CoordinateSystem(
            name='Horizontal Offsets')
        az_axis = self.create_axis('Azimuth', 'AZ')
        el_axis = self.create_axis('Elevation', 'EL')
        az_offset_axis = self.create_offset_axis('Azimuth Offset', 'dAZ')
        el_offset_axis = self.create_offset_axis('Elevation Offset', 'dEL')
        self.default_coordinate_system.add_axis(az_axis)
        self.default_coordinate_system.add_axis(el_axis)
        self.default_local_coordinate_system.add_axis(az_offset_axis)
        self.default_local_coordinate_system.add_axis(el_offset_axis)

    def copy(self):
        """
        Return a copy of the horizontal coordinates.

        Returns
        -------
        HorizontalCoordinates
        """
        return super().copy()

    @property
    def fits_longitude_stem(self):
        """
        Return the FITS header longitude stem string.

        Returns
        -------
        stem : str
        """
        return 'ALON'

    @property
    def fits_latitude_stem(self):
        """
        Return the FITS header latitude stem string.

        Returns
        -------
        stem : str
        """
        return 'ALAT'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code representing the horizontal system.

        Returns
        -------
        code : str
        """
        return 'HO'

    @property
    def az(self):
        """
        Return the azimuth coordinates.

        Returns
        -------
        azimuth : astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        return self.native_longitude

    @az.setter
    def az(self, values):
        """
        Set the azimuth coordinates.

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_az(values)

    @property
    def el(self):
        """
        Return the elevation coordinates.

        Returns
        -------
        elevation : astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        return self.native_latitude

    @el.setter
    def el(self, values):
        """
        Set the elevation coordinates.

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_el(values)

    @property
    def za(self):
        """
        Return the Zenith Angle (ZA).

        Returns
        -------
        astropy.units.Quantity (float or numpy.ndarray)
        """
        if self.coordinates is None:
            return None
        return self.right_angle - self.native_latitude

    @za.setter
    def za(self, values):
        """
        Set the Zenith Angle (ZA).

        Parameters
        ----------
        values : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_za(values)

    def __str__(self):
        """
        Create a string representation of the equatorial coordinates.

        Returns
        -------
        str
        """
        if self.coordinates is None:
            return 'Empty coordinates'

        if self.singular:
            return (f"Az={Angle(self.az).to_string(unit='degree')} "
                    f"El={Angle(self.el).to_string(unit='degree')}")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                s = (f"Az="
                     f"{Angle(np.nanmin(self.az)).to_string(unit='degree')}->"
                     f"{Angle(np.nanmax(self.az)).to_string(unit='degree')} "
                     f"El="
                     f"{Angle(np.nanmin(self.el)).to_string(unit='degree')}->"
                     f"{Angle(np.nanmax(self.el)).to_string(unit='degree')}")
                return s

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        HorizontalCoordinates
        """
        return super().__getitem__(indices)

    @classmethod
    def convert_horizontal_to_equatorial(cls, horizontal, equatorial, site,
                                         lst):
        """
        Convert horizontal coordinates to an equatorial frame.

        Parameters
        ----------
        horizontal : HorizontalCoordinates
        equatorial : EquatorialCoordinates
            The frame in which to hold the equatorial coordinates.
        site : GeodeticCoordinates
        lst : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        hx = horizontal.x
        lst = cls.correct_factor_dimensions(lst, hx)
        site_cos_lat = cls.correct_factor_dimensions(site.cos_lat, hx)
        site_sin_lat = cls.correct_factor_dimensions(site.sin_lat, hx)
        c = np.cos(hx) * horizontal.cos_lat

        equatorial.set_native_latitude(
            np.arcsin((horizontal.sin_lat * site_sin_lat)
                      + (site_cos_lat * c)))

        asin_h = -np.sin(hx) * horizontal.cos_lat
        acos_h = (site_cos_lat * horizontal.sin_lat) - (site_sin_lat * c)
        equatorial.set_longitude(lst - np.arctan2(asin_h, acos_h))

    @classmethod
    def to_equatorial_offset(cls, offset, position_angle, in_place=True):
        """
        Convert horizontal offsets to an equatorial offset.

        The offsets are updated in-place.

        Parameters
        ----------
        offset : Coordinate2D
            The (x, y) horizontal offsets to rotate.
        position_angle : astropy.units.Quantity (float or numpy.ndarray)
            The position angle as a scalar or array of shape (n,).
        in_place : bool, optional
            If `True`, update `offset` in-place.  Otherwise, return a new
            offset without modifying the original.

        Returns
        -------
        equatorial_offset : Coordinate2D
        """
        if not in_place:
            offset = offset.copy()  # Create a local copy

        cls.rotate_offsets(offset, position_angle)
        offset.scale_x(-1)
        return offset

    def set_az(self, azimuth):
        """
        Set the azimuth coordinates.

        Parameters
        ----------
        azimuth : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_native_longitude(azimuth)

    def set_el(self, elevation):
        """
        Set the elevation coordinates.

        Parameters
        ----------
        elevation : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_native_latitude(elevation)

    def set_za(self, zenith_angle):
        """
        Set the zenith angle.

        Parameters
        ----------
        zenith_angle : astropy.units.Quantity (float or numpy.ndarray)

        Returns
        -------
        None
        """
        self.set_native_latitude(self.right_angle - zenith_angle)

    def get_parallactic_angle(self, site):
        """
        Return the parallactic angle for a given site.

        Parameters
        ----------
        site : GeodeticCoordinates

        Returns
        -------
        pa : astropy.units.Quantity (float or numpy.ndarray)
        """
        site_cos_lat = self.correct_factor_dimensions(site.cos_lat, self.x)
        site_sin_lat = self.correct_factor_dimensions(site.sin_lat, self.x)

        y = -site_cos_lat * np.sin(self.x)
        x = (site_sin_lat * self.cos_lat) - (
            site_cos_lat * self.sin_lat * np.cos(self.x))
        return np.arctan2(y, x)

    def to_equatorial(self, site, lst, equatorial=None):
        """
        Return these coordinates as equatorial coordinates.

        Parameters
        ----------
        site : GeodeticCoordinates
        lst : astropy.units.Quantity
        equatorial : EquatorialCoordinates, optional

        Returns
        -------
        EquatorialCoordinates
        """
        if equatorial is None:
            equatorial = self.get_instance('equatorial')
        self.convert_horizontal_to_equatorial(self, equatorial, site, lst)
        return equatorial

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with horizontal coordinate information.

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
                                   'coordinate system description')

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
        import matplotlib.pyplot as plt
        plt.ion()
        c_args = self.az.ravel(), self.el.ravel()
        if args is not None:
            c_args += args

        plt.plot(*c_args, **kwargs)
        plt.xlabel(f'{self.longitude_axis.label} ({self.unit})')
        plt.ylabel(f'{self.latitude_axis.label} ({self.unit})')
