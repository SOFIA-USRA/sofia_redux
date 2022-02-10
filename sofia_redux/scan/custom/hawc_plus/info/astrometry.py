# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log

from sofia_redux.scan.custom.sofia.info.astrometry import SofiaAstrometryInfo
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import Epoch

__all__ = ['HawcPlusAstrometryInfo']


class HawcPlusAstrometryInfo(SofiaAstrometryInfo):

    def __init__(self):
        super().__init__()

    def apply_configuration(self):
        if self.configuration is None:
            return
        options = self.options
        if options is None:
            return

        if 'OBJRA' in self.configuration and 'OBJDEC' in self.configuration:
            self.object_coordinates = EquatorialCoordinates(
                epoch=Epoch.get_epoch(options.get_float("EQUINOX")))
            self.object_coordinates.ra = options.get_hms_time(
                "OBJRA", angle=True)
            self.object_coordinates.dec = options.get_dms_angle("OBJDEC")

        super().apply_configuration()

        self.is_nonsidereal |= options.get_bool("NONSIDE")
        self.is_nonsidereal |= self.configuration.get_bool('rtoc')

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
        if not super().is_requested_valid(header=header):
            return False
        if header is None:
            if self.options is None:
                return False
            ra = self.options.get_float('OBSRA')
            dec = self.options.get_float('OBSDEC')
        else:
            ra = header.get('OBSRA')
            dec = header.get('OBSDEC')

        if ra == 1 and dec == 2:
            return False
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
        if self.is_nonsidereal:
            log.info("Referencing images to real-time object coordinates.")
            return
        return super().guess_reference_coordinates(header=header,
                                                   telescope=telescope)
