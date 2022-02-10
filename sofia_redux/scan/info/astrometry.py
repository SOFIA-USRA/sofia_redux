# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.time import Time
import numpy as np

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.bracketed_values import BracketedValues
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import (
    JulianEpoch, Epoch, J2000)
from sofia_redux.scan.coordinate_systems.epoch.precession import Precession
from sofia_redux.scan.coordinate_systems.focal_plane_coordinates import \
    FocalPlaneCoordinates

__all__ = ['AstrometryInfo']


class AstrometryInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.mjd = np.nan
        self.lst = np.nan * units.Unit('hour')
        self.utc = BracketedValues()
        self.time_stamp = None
        self.date = None

        self.object_name = None
        self.object_coordinates = None
        self.apparent = EquatorialCoordinates()
        self.equatorial = EquatorialCoordinates()
        self.horizontal = HorizontalCoordinates()
        self.site = GeodeticCoordinates()
        self.pointing_correction = None
        self.epoch = None
        self.apparent_epoch = None
        self.from_apparent = None
        self.to_apparent = None
        self.is_nonsidereal = False
        self.pointing = None
        self.ground_based = True

    def calculate_precessions(self, equinox):
        """
        Given an equinox, update the epoch and apparent epoch.

        Parameters
        ----------
        equinox : int or float or str or Epoch

        Returns
        -------

        """
        t_apparent = Time(self.mjd, format='mjd')
        self.epoch = Epoch(equinox=equinox)
        self.apparent_epoch = JulianEpoch(equinox=t_apparent)
        self.from_apparent = Precession(self.apparent_epoch, self.epoch)
        self.to_apparent = Precession(self.epoch, self.apparent_epoch)

    def precess(self, epoch, scan=None):
        """
        Precess equatorial to a new epoch.

        Parameters
        ----------
        epoch : BaseCoordinateFrame or int or float or str.
        scan : Scan, optional
            If provided, precess all integration data.

        Returns
        -------
        None
        """
        to_epoch = Precession(self.equatorial.epoch, epoch)
        to_epoch.precess(self.equatorial)
        if scan is not None and scan.integrations is not None:
            for integration in scan.integrations:
                equatorial = integration.frames.equatorial
                if equatorial is not None:
                    to_epoch.precess(equatorial)

        self.calculate_precessions(epoch)

    def calculate_equatorial(self):
        """
        Calculate equatorial coordinates from current horizontal coordinates.

        Returns
        -------
        None
        """
        self.equatorial = self.horizontal.to_equatorial(
            site=self.site, lst=self.lst)
        self.equatorial.epoch = JulianEpoch(
            equinox=Time(self.mjd, format='mjd'))
        if self.from_apparent is None:
            self.calculate_precessions(J2000)
        self.from_apparent.precess(self.equatorial)

    def calculate_apparent(self):
        """
        Calculate the apparent equatorial coordinates.

        Returns
        -------
        None
        """
        self.apparent = self.equatorial.copy()
        if self.to_apparent is None:
            self.calculate_precessions(self.equatorial.epoch)
        self.to_apparent.precess(self.apparent)

    def calculate_horizontal(self):
        """
        Calculate horizontal coordinates from apparent equatorial coordinates.

        The apparent equatorial coordinates will be calculated from the
        equatorial coordinates if not available.

        Returns
        -------
        None
        """
        if self.apparent is None:
            self.calculate_apparent()
        self.horizontal = self.apparent.to_horizontal(
            site=self.site, lst=self.lst)

    def get_native_coordinates(self):
        """
        The native coordinates of the scan astrometry.

        The native coordinates returned in this case are horizontal.

        Returns
        -------
        coordinates : SkyCoord
            The native coordinates.
        """
        return self.horizontal

    def get_position_reference(self, system=None):
        """
        Return position reference in the defined coordinate frame.

        By default, the equatorial coordinates are returned, but many other
        frame systems may be specified.  All astropy coordinate frames may
        be used but may raise conversion errors depending on the type.  If
        an error is encountered during conversion, or the frame system is
        unavailable, equatorial coordinates will be returned.

        Parameters
        ----------
        system : str
            Name of the coordinate frame.  Available values are:
            {'horizontal', 'native', 'focalplane'} and all Astropy frame
            type names.

        Returns
        -------
        coordinates : SphericalCoordinates
            Coordinates of the specified type.
        """
        system = str(system).lower().strip()
        if system == 'horizontal':
            return self.horizontal
        elif system == 'native':
            return self.get_native_coordinates()
        elif system == 'focalplane':
            coordinates = FocalPlaneCoordinates()
            coordinates.set([0, 0])  # Single zero coordinates
            return coordinates
        elif self.is_nonsidereal:
            ra = self.configuration.get_float('reference.ra')
            dec = self.configuration.get_float('reference.dec')
            if np.isfinite(ra) and np.isfinite(dec):
                equatorial = EquatorialCoordinates(epoch=self.equatorial.epoch)
                equatorial.ra = ra * units.Unit('hourangle')
                equatorial.dec = dec * units.Unit('degree')
                return equatorial

            elif np.isfinite(ra) or np.isfinite(dec):
                log.warning("The reference.ra or reference.dec configuration "
                            "was given without the other.  Defaulting to scan "
                            "reference points.")
                return self.equatorial
            else:
                return self.equatorial
        elif system in ['eliptic', 'galatic', 'supergalatic']:
            raise NotImplementedError(f"Have not built {system} coordinates.")
        else:
            return self.equatorial

    def apply_scan(self, scan):
        self.is_nonsidereal |= self.configuration.get_bool('moving')

    def set_mjd(self, mjd):
        """
        Set the Modified Julian Date and apply any configuration options.

        Parameters
        ----------
        mjd : float

        Returns
        -------
        None
        """
        self.mjd = float(mjd)
        if self.configuration is not None:
            self.configuration.set_date(self.mjd, validate=True)
