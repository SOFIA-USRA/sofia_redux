# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.custom.sofia.info.astrometry import SofiaAstrometryInfo
from sofia_redux.scan.coordinate_systems.epoch.epoch import Epoch

__all__ = ['FifiLsAstrometryInfo']


degree = units.Unit('degree')
arcsec = units.Unit('arcsec')


class FifiLsAstrometryInfo(SofiaAstrometryInfo):

    def __init__(self):
        """
        Initialize astrometry information for FIFI-LS observations.

        An extension of the SOFIA astrometry information with additional
        handling for FIFI-LS.
        """
        super().__init__()
        self.scan_equatorial = EquatorialCoordinates()
        self.scan_equatorial.set_singular()
        self.scan_equatorial.nan()
        self.delta_map = Coordinate2D(unit='arcsec')
        self.delta_map.set_singular()
        self.delta_map.zero()
        self.obs_equatorial = EquatorialCoordinates()
        self.obs_equatorial.set_singular()
        self.obs_equatorial.nan()

    def apply_configuration(self):
        """
        Update astrometry information with the FITS header configuration data.

        Returns
        -------
        None
        """
        # No object coordinates are available
        super().apply_configuration()
        self.is_nonsidereal = False

    def parse_astrometry(self):
        """
        Parse and apply coordinate related keywords from the FITS header.

        The following FITS header keywords are used::

          KEY       DESCRIPTION
          --------  -----------
          EQUINOX   The equinox of the observation (year)
          OBSRA     The right-ascension of the observed source (hours)
          OBSDEC    The declination of the observed source (degrees)

        The modified Julian date (MJD) is calculated, and precessions are
        initialized relative to the J2000 equinox.

        Returns
        -------
        None
        """
        if self.options is None:
            return
        if "EQUINOX" in self.options:
            self.epoch = Epoch.get_epoch(self.options.get_float("EQUINOX"))

        scan_ra = self.options.get_hms_time('OBSRA', angle=True)
        scan_dec = self.options.get_dms_angle('OBSDEC')
        self.requested_equatorial = EquatorialCoordinates(
            [scan_ra, scan_dec], unit='degree', epoch=self.epoch)
        self.equatorial = self.requested_equatorial.copy()
        self.scan_equatorial = self.requested_equatorial.copy()

        obs_lambda = self.options.get_float('OBSLAM', default=0) * degree
        obs_beta = self.options.get_float('OBSBET', default=0) * degree
        self.obs_equatorial = EquatorialCoordinates(
            [obs_lambda, obs_beta], epoch=self.epoch)

        map_lambda = self.options.get_float('DLAM_MAP', default=0) * arcsec
        map_beta = self.options.get_float('DBET_MAP', default=0) * arcsec
        self.delta_map = Coordinate2D([map_lambda, map_beta])

        self.scan_equatorial.subtract_offset(self.delta_map)

        self.calculate_precessions(self.equatorial.epoch)
