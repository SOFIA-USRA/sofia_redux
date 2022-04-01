# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.time import Time

from sofia_redux.scan.info.astrometry import AstrometryInfo
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.utilities.utils import safe_sidereal_time


__all__ = ['ExampleAstrometryInfo']


class ExampleAstrometryInfo(AstrometryInfo):

    default_fits_date = "1970-01-01T00:00:00.0"

    def __init__(self):
        """
        Initialize the astrometry info for the example instrument.

        The example instrument is assumed to be ground based using equatorial
        coordinates in the J2000 equinox.
        """
        super().__init__()
        self.date = None
        self.epoch = J2000
        self.horizontal = None
        self.equatorial = None
        self.ground_based = True

    def apply_configuration(self):
        """
        Apply the configuration options related to the FITS header.

        Returns
        -------
        None
        """
        deg = units.Unit('degree')
        if self.options is None:
            return
        self.time_stamp = self.options.get_string('DATE-OBS')
        if self.time_stamp is None:
            log.warning('DATE-OBS not found')
            self.time_stamp = self.default_fits_date

        date_obs = Time(self.time_stamp, format='isot', scale='utc')
        self.set_mjd(date_obs.mjd)
        self.calculate_precessions(J2000)
        self.equatorial = EquatorialCoordinates(epoch=self.epoch)
        self.equatorial.ra = self.options.get_hms_time('OBSRA', angle=True)
        self.equatorial.dec = self.options.get_dms_angle('OBSDEC')

        if 'SITELON' in self.options and 'SITELAT' in self.options:
            self.site.latitude = self.options.get_float('SITELAT') * deg
            self.site.longitude = self.options.get_float('SITELON') * deg
        else:
            # NASA Ames
            self.site.latitude = 37.4089 * deg
            self.site.longitude = -122.0644 * deg

        if 'LST' in self.options:
            self.lst = self.options.get_float('LST') * units.Unit('hourangle')
        else:
            self.lst = safe_sidereal_time(
                date_obs, 'mean', longitude=self.site.longitude)

        log.info(f"Equatorial: {self.equatorial}")
        log.info(f"Site: {self.site}")
        log.info(f"Observed on: {self.time_stamp}")
