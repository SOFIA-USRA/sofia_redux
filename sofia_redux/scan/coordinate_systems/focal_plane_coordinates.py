# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem

__all__ = ['FocalPlaneCoordinates']


class FocalPlaneCoordinates(SphericalCoordinates):

    def __init__(self, coordinates=None, unit='degree', copy=True):
        super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the focal-plane coordinates.

        Returns
        -------
        FocalPlaneCoordinates
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
            name='Focal Plane Coordinates')
        x_axis = self.create_axis('Focal-plane X', 'X')
        y_axis = self.create_axis('Focal-plane Y', 'Y')
        self.default_coordinate_system.add_axis(x_axis)
        self.default_coordinate_system.add_axis(y_axis)

        self.default_local_coordinate_system = CoordinateSystem(
            name='Focal Plane Offsets')
        dx_axis = self.create_offset_axis('Focal-plane dX', 'dX')
        dy_axis = self.create_offset_axis('Focal-plane dY', 'dY')
        self.default_local_coordinate_system.add_axis(dx_axis)
        self.default_local_coordinate_system.add_axis(dy_axis)

    @property
    def fits_longitude_stem(self):
        """
        Return the FITS longitude stem string.

        Returns
        -------
        stem : str
        """
        return 'FLON'

    @property
    def fits_latitude_stem(self):
        """
        Return the FITS latitude stem string.

        Returns
        -------
        stem : str
        """
        return 'FLAT'

    @property
    def two_letter_code(self):
        """
        Return the two-letter code for the coordinate system.

        Returns
        -------
        code : str
        """
        return 'FP'

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        FocalPlaneCoordinates
        """
        return super().__getitem__(indices)

    def edit_header(self, header, key_stem, alt=''):
        """
        Edit the header with focal plane coordinate information.

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
        super().edit_header(header, key_stem, alt=alt)
        header[f'WCSNAME{alt}'] = (self.coordinate_system.name,
                                   'coordinate system description.')
