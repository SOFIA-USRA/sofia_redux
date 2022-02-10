# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log, units
import numpy as np

from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000

__all__ = ['CelestialCoordinates']


class CelestialCoordinates(SphericalCoordinates):

    def __init__(self, coordinates=None, unit='degree', copy=True):
        """
        Initialize a CelestialCoordinates object.

        Celestial coordinates are used to represent spherical coordinates
        on the sky with respect to a given pole.  The functionality implemented
        here allows for transformations between spherical coordinates with
        differing poles or zero-longitude definitions.

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
        if isinstance(coordinates, CelestialCoordinates):
            super().__init__(unit=unit)
            self.convert(coordinates, self)
        else:
            super().__init__(coordinates=coordinates, unit=unit, copy=copy)

    def copy(self):
        """
        Return a copy of the celestial coordinates.

        Returns
        -------
        CelestialCoordinates
        """
        return super().copy()

    def __getitem__(self, indices):
        """
        Return a section of the coordinates

        Parameters
        ----------
        indices : int or numpy.ndarray or slice

        Returns
        -------
        CelestialCoordinates
        """
        return super().__getitem__(indices)

    @abstractmethod
    def get_equatorial_pole(self):  # pragma: no cover
        """
        Return the equatorial pole coordinates.

        Returns
        -------
        EquatorialCoordinates
        """
        pass

    @abstractmethod
    def get_zero_longitude(self):  # pragma: no cover
        """
        Return the zero longitude value.

        Returns
        -------
        astropy.units.Quantity
        """
        pass

    @classmethod
    def get_pole(cls, inclination, rising_ra, reference=None):
        """
        Get the pole from inclination and rising RA angles.

        Parameters
        ----------
        inclination : astropy.units.Quantity
            The inclination angle.
        rising_ra : astropy.units.Quantity
            The rising Right Ascension.
        reference : CelestialCoordinates, optional
            If supplied sets the coordinates in the reference system and
            converts it to equatorial coordinates.

        Returns
        -------
        pole : EquatorialCoordinates
        """
        ra = rising_ra - cls.right_angle
        dec = cls.right_angle - inclination
        coords = np.stack((ra, dec))

        if reference is None:
            return cls.get_equatorial_class()(coords)
        else:
            reference.set(coords, copy=False)
            reference.to_equatorial()
            return reference

    @classmethod
    def get_zero_longitude_from(cls, from_coordinates, to_coordinates):
        """
        Return the zero longitude of one coordinates system in another.

        Parameters
        ----------
        from_coordinates : EquatorialCoordinates
            The coordinates from which to determine zero longitude.
        to_coordinates : CelestialCoordinates
            The coordinate system in which to return the zero longitude.

        Returns
        -------
        zero_longitude : astropy.units.Quantity
            The zero longitude angle.
        """
        equatorial_zero = from_coordinates.to_equatorial()
        to_coordinates.from_equatorial(equatorial_zero)
        return to_coordinates.native_longitude

    @classmethod
    def get_equatorial_class(cls):
        """
        Return the equatorial class.

        Returns
        -------
        EquatorialCoordinates
        """
        return cls.get_class('equatorial')

    def get_equatorial_position_angle(self):
        """
        Return the equatorial position angle of the celestial coordinates.

        Returns
        -------
        astropy.units.Quantity
        """
        pole = self.get_equatorial_pole()
        y = -pole.cos_lat * np.sin(self.x).value
        x = (pole.sin_lat * self.cos_lat)
        x -= (pole.cos_lat * self.sin_lat * np.cos(self.x).value)
        angle = np.arctan2(y, x) * units.Unit('radian')
        return angle

    def get_equatorial(self):
        """
        Return an equatorial representation of the celestial coordinates.

        Returns
        -------
        EquatorialCoordinates
        """
        equatorial_class = self.get_equatorial_class()
        equatorial = equatorial_class()
        self.to_equatorial(equatorial)
        return equatorial

    def to_equatorial(self, equatorial=None):
        """
        Convert these celestial coordinates to equatorial coordinates.

        Parameters
        ----------
        equatorial : EquatorialCoordinates, optional
            The equatorial coordinates that will hold these coordinates.  If

        Returns
        -------
        equatorial : EquatorialCoordinates
        """
        if equatorial is None:
            equatorial = self.get_instance('equatorial')

        if equatorial.epoch is None:
            equatorial.epoch = J2000
        pole = self.get_equatorial_pole()
        phi0 = self.get_zero_longitude()
        self.inverse_transform(pole, phi0, from_coordinates=self,
                               to_coordinates=equatorial)

        if equatorial.epoch != pole.epoch:
            epoch = equatorial.epoch
            equatorial.epoch = pole.epoch
            try:
                equatorial.precess(epoch)
            except Exception as err:  # pragma: no cover
                log.warning(f"Could not precess: {err}")

        return equatorial

    def from_equatorial(self, equatorial):
        """
        Set the celestial coordinates from those given.

        Parameters
        ----------
        equatorial : EquatorialCoordinates

        Returns
        -------
        None
        """
        pole = self.get_equatorial_pole()
        if equatorial.epoch != pole.epoch:
            equatorial = equatorial.copy()
            try:
                equatorial.precess(pole.epoch)
            except Exception as err:  # pragma: no cover
                log.warning(f"Could not precess: {err}")

        phi0 = self.get_zero_longitude()
        self.transform(pole, phi0, from_coordinates=equatorial,
                       to_coordinates=self)

    def convert_from(self, coordinates):
        """
        Convert coordinates from another (or same) system to these coordinates.

        Parameters
        ----------
        coordinates : CelestialCoordinates or Coordinate2D

        Returns
        -------
        None
        """
        if not isinstance(coordinates, CelestialCoordinates):
            super().convert_from(coordinates)
        else:
            self.convert_from_celestial(coordinates)

    def convert_to(self, coordinates):
        """
        Convert coordinates to another system.

        Parameters
        ----------
        coordinates : CelestialCoordinates or Coordinate2D

        Returns
        -------
        None
        """
        if not isinstance(coordinates, CelestialCoordinates):
            super().convert_to(coordinates)
        else:
            self.convert_to_celestial(coordinates)

    def convert_from_celestial(self, celestial):
        """
        Convert coordinates from another celestial frame onto this one.

        Parameters
        ----------
        celestial : CelestialCoordinates

        Returns
        -------
        None
        """
        self.convert(celestial, self)

    def convert_to_celestial(self, celestial):
        """
        Convert these coordinates to another celestial system.

        Parameters
        ----------
        celestial : CelestialCoordinates

        Returns
        -------
        None
        """
        self.convert(self, celestial)

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
        to_coordinates.copy_coordinates(from_coordinates)
