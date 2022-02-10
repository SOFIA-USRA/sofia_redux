# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.time import Time

from sofia_redux.scan.info.astrometry import AstrometryInfo
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates

__all__ = ['SofiaAstrometryInfo']


class SofiaAstrometryInfo(AstrometryInfo):

    default_fits_date = "1970-01-01T00:00:00.0"

    def __init__(self):
        super().__init__()
        self.date = None
        self.start_time = None
        self.file_date = None
        self.epoch = J2000
        self.requested_equatorial = None
        self.horizontal = None
        self.equatorial = None
        self.ground_based = True

    def apply_configuration(self):
        if self.options is None:
            return
        self.parse_time()
        self.parse_astrometry()

    def parse_time(self):
        options = self.options
        if options is None:
            return
        self.file_date = options.get_string("DATE")
        self.date = options.get_string("DATE-OBS",
                                       default=self.default_fits_date)

        start_time = options.get_string("UTCSTART")
        if start_time is not None:
            self.utc.start = utils.parse_time(start_time)

        end_time = utils.get_string(options.get("UTCEND"))
        if end_time is not None:
            self.utc.end = utils.parse_time(end_time)

        t_ind = self.date.find('T')
        if t_ind != -1:
            self.time_stamp = self.date
            self.date = self.time_stamp[:t_ind]
            start_time = self.time_stamp[(t_ind + 1):]
        elif start_time is None:
            self.time_stamp = self.date + 'T' + start_time
        else:
            self.time_stamp = self.date

        self.start_time = start_time
        self.set_mjd(Time(self.time_stamp, format='isot', scale='utc').mjd)
        self.calculate_precessions(J2000)

    def parse_astrometry(self):
        options = self.options
        if options is None:
            return
        if "EQUINOX" in options:
            self.epoch = utils.get_epoch(options.get_float("EQUINOX"))

        if "OBSRA" in options and "OBSDEC" in options:
            self.requested_equatorial = EquatorialCoordinates(epoch=self.epoch)
            self.requested_equatorial.ra = options.get_hms_time(
                'OBSRA', angle=True)
            self.requested_equatorial.dec = options.get_dms_angle('OBSDEC')

        if self.is_requested_valid():
            self.equatorial = self.requested_equatorial.copy()
            self.calculate_precessions(self.equatorial.epoch)
        else:
            log.warning("No valid OBSRA/OBSDEC in header.")
            self.requested_equatorial = None
            self.equatorial = self.guess_reference_coordinates()
            self.calculate_precessions("J2000")

    def is_requested_valid(self, header=None):
        """
        Check whether the requested OBSRA/DEC values in the header are valid.

        Parameters
        ----------
        header : astropy.io.fits.header.Header, optional
            The header to read.  If not supplied, the FITS header for the scan
            will be read instead.

        Returns
        -------
        valid : bool
        """
        if header is None:
            if self.options is None:
                return False
            ra = self.options.get_float('OBSRA')
            dec = self.options.get_float('OBSDEC')
        else:
            ra = header.get('OBSRA')
            dec = header.get('OBSDEC')

        if not self.valid_header_value(ra):
            return False
        elif not self.valid_header_value(dec):
            return False
        elif ra == 0 and dec == 0:
            return False
        else:
            return True

    @classmethod
    def coordinate_valid(cls, coordinate):
        """
        Check whether a coordinate frame is valid.

        Parameters
        ----------
        coordinate : SphericalCoordinates

        Returns
        -------
        valid : bool
        """
        if coordinate is None:
            return False
        elif not cls.valid_header_value(coordinate.x.value):
            return False
        elif not cls.valid_header_value(coordinate.y.value):
            return False
        else:
            return True

    def guess_reference_coordinates(self, header=None, telescope=None):
        """
        Guess the reference coordinates of a scan.

        Parameters
        ----------
        header : astropy.io.fits.Header, optional
            The header to read.  The default is to read stored OBSRA/OBSDEC
            values in the configuration.
        telescope : SofiaTelescopeInfo, optional
            A telescope object to extract the boresight equatorial coordinates
            if all other avenues to the coordinates failed.

        Returns
        -------
        coordinates : EquatorialCoordinates
        """

        if self.coordinate_valid(self.object_coordinates):
            log.debug("Referencing scan to object coordinates OBJRA/OBJDEC.")
            return self.object_coordinates.copy()

        elif self.is_requested_valid(header=header):
            log.debug("Referencing scan to requested coordinates.")
            return self.requested_equatorial.copy()

        elif telescope is not None and self.coordinate_valid(
                telescope.boresight_equatorial):
            log.debug(
                "Referencing scan to initial telescope boresight TELRA/TELDEC")
            return telescope.boresight_equatorial.copy()

        else:
            log.warning("Referencing scan to initial scan position.")
            return None

    def validate_astrometry(self, scan):
        """
        Validate astrometry information from a scan.

        Returns
        -------
        None
        """
        if scan is None:
            return
        first_integration = scan.get_first_integration()
        if first_integration is None:
            raise ValueError("No integrations exist for scan.")
        last_integration = scan.get_last_integration()
        first = first_integration.get_first_frame_index()
        last = last_integration.get_last_frame_index()

        if self.is_nonsidereal:
            oe0 = first_integration.frames.object_equatorial[first]
            oe1 = last_integration.frames.object_equatorial[last]
            self.object_coordinates = EquatorialCoordinates(epoch=oe0.epoch)
            self.object_coordinates.ra = 0.5 * (oe0.ra + oe1.ra)
            self.object_coordinates.dec = 0.5 * (oe0.dec + oe1.dec)
            self.equatorial = self.object_coordinates.copy()

        h0 = first_integration.frames.horizontal[first]
        h1 = last_integration.frames.horizontal[last]
        self.horizontal = HorizontalCoordinates()
        self.horizontal.az = 0.5 * (h0.x + h1.x)
        self.horizontal.el = 0.5 * (h0.y + h1.y)

        s0 = first_integration.frames.site[first]
        s1 = last_integration.frames.site[last]
        self.site = GeodeticCoordinates()
        self.site.longitude = 0.5 * (s0.x + s1.x)
        self.site.latitude = 0.5 * (s0.y + s1.y)
        log.info(f"Location: {self.site}")

        log.info(f"Mean telescope VPA is {scan.get_telescope_vpa():.6f}")
