# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['TelescopeCoordinates']


class TelescopeCoordinates(SphericalCoordinates):

    def __init__(self, coordinates=None, unit='degree', copy=True):
        """
        Initialize a set of telescope coordinates.

        Telescope coordinates are spherical coordinates relating to the
        telescope elevation and cross-elevation.

        Parameters
        ----------
        coordinates : list or tuple or array-like or units.Quantity, optional
            The coordinates used to populate the object during initialization.
            The first (0) value or index should represent longitudinal
            coordinates, and the second should represent latitude.
        unit : units.Unit or str, optional
            The angular unit for the telescope coordinates.  The default is
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
        Return a copy of the telescope coordinates.

        Returns
        -------
        TelescopeCoordinates
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
            name='Telescope Coordinates')
        self.default_local_coordinate_system = CoordinateSystem(
            name='Telescope Offsets')
        xel_axis = self.create_axis('Telescope Cross-elevation', 'XEL')
        el_axis = self.create_axis('Telescope Elevation', 'EL')
        xel_offset_axis = self.create_offset_axis(
            'Telescope Cross-elevation Offset', 'dXEL')
        el_offset_axis = self.create_offset_axis(
            'Telescope Elevation Offset', 'dEL')
        self.default_coordinate_system.add_axis(xel_axis)
        self.default_coordinate_system.add_axis(el_axis)
        self.default_local_coordinate_system.add_axis(xel_offset_axis)
        self.default_local_coordinate_system.add_axis(el_offset_axis)

    @property
    def fits_longitude_stem(self):
        """
        Return the string prefix for cross elevation longitude.

        Returns
        -------
        str
        """
        return 'TLON'

    @property
    def fits_latitude_stem(self):
        """
        Return the string prefix for the elevation latitude.

        Returns
        -------
        str
        """
        return 'TLAT'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the telescope coordinate system.

        Returns
        -------
        code : str
        """
        return 'TE'

    @property
    def xel(self):
        """
        Return the cross elevation.

        Returns
        -------
        astropy.units.Quantity
        """
        if self.coordinates is None:
            return None
        return self.native_longitude

    @xel.setter
    def xel(self, values):
        """
        Set the cross elevation.

        Parameters
        ----------
        values : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.set_native_longitude(values, copy=True)

    @property
    def cross_elevation(self):
        """
        Return the cross elevation.

        Returns
        -------
        astropy.units.Quantity
        """
        if self.coordinates is None:
            return None
        return self.native_longitude

    @cross_elevation.setter
    def cross_elevation(self, values):
        """
        Set the cross elevation.

        Parameters
        ----------
        values : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.set_native_longitude(values, copy=True)

    @property
    def el(self):
        """
        Return the elevation.

        Returns
        -------
        astropy.units.Quantity
        """
        if self.coordinates is None:
            return None
        return self.native_latitude

    @el.setter
    def el(self, values):
        """
        Set the elevation.

        Parameters
        ----------
        values : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.set_native_latitude(values, copy=True)

    @property
    def elevation(self):
        """
        Return the elevation.

        Returns
        -------
        astropy.units.Quantity
        """
        if self.coordinates is None:
            return None
        return self.native_latitude

    @elevation.setter
    def elevation(self, values):
        """
        Set the elevation.

        Parameters
        ----------
        values : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.set_native_latitude(values, copy=True)

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        TelescopeCoordinates
        """
        return super().__getitem__(indices)

    def to_equatorial_offset(self, offset, telescope_vpa, in_place=True):
        """
        Return these coordinates as equatorial coordinates.

        Parameters
        ----------
        offset : Coordinate2D
        telescope_vpa : astropy.units.Quantity
            The telescope VPA angle(s).
        in_place : bool, optional
            If `True`, update the offsets in-place.  Otherwise, return a fresh
            frame.

        Returns
        -------
        Coordinate2D
        """
        if not in_place:
            offset = offset.copy()
        telescope_vpa = self.correct_factor_dimensions(telescope_vpa, offset.x)
        Coordinate2D.rotate_offsets(offset, telescope_vpa)
        offset.scale_x(-1.0)
        return offset

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with telescope coordinate information.

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
