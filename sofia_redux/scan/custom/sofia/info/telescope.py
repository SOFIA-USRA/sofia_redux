# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units
from astropy.time import Time

from sofia_redux.scan.utilities.bracketed_values import BracketedValues
from sofia_redux.scan.info.telescope import TelescopeInfo
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000, Epoch
from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaTelescopeInfo']


class SofiaTelescopeInfo(TelescopeInfo):

    telescope_diameter = 2.5 * units.Unit('m')

    def __init__(self):
        super().__init__()
        self.telescope = "SOFIA 2.5m"
        self.tel_config = None
        self.vpa = np.nan * units.Unit('deg')
        self.last_rewind = None
        self.focus_t = BracketedValues(np.nan * units.Unit('um'),
                                       np.nan * units.Unit('um'))
        self.rel_elevation = np.nan * units.Unit('deg')
        self.cross_elevation = np.nan * units.Unit('deg')
        self.line_of_sight_angle = np.nan * units.Unit('deg')
        self.tascu_status = None
        self.fbc_status = None
        self.zenith_angle = BracketedValues(np.nan * units.Unit('um'),
                                            np.nan * units.Unit('um'))
        self.tracking_mode = None
        self.has_tracking_error = False
        self.is_tracking = False

        self.epoch = J2000
        self.boresight_equatorial = EquatorialCoordinates(
            np.full(2, np.nan), unit='degree', epoch=self.epoch)

        self.requested_equatorial = self.boresight_equatorial.copy()

    def apply_configuration(self):
        options = self.options
        if options is None:
            return

        deg = units.Unit('deg')
        um = units.Unit('um')

        self.telescope = options.get_string("TELESCOP", default=self.telescope)
        self.vpa = options.get_float("TELVPA") * deg
        self.last_rewind = options.get_string("LASTREW")
        self.focus_t.start = options.get_float("FOCUS_ST") * um
        self.focus_t.end = options.get_float("FOCUS_EN") * um
        self.rel_elevation = options.get_float("TELEL") * deg
        self.cross_elevation = options.get_float("TELXEL") * deg
        self.line_of_sight_angle = options.get_float("TELLOS") * deg
        self.tascu_status = options.get_string("TSC-STAT")
        self.fbc_status = options.get_string("FBC-STAT")
        self.zenith_angle.start = options.get_float("ZA_START") * deg
        self.zenith_angle.end = options.get_float("ZA_END") * deg
        self.tracking_mode = options.get_string("TRACMODE")
        self.has_tracking_error = options.get_bool("TRACERR")
        self.is_tracking = str(self.tracking_mode).strip().upper() != 'OFF'

        self.tel_config = options.get_string('TELCONF')

        if "EQUINOX" in options:
            self.epoch = Epoch(equinox=options.get_float("EQUINOX"))

        self.requested_equatorial = EquatorialCoordinates(
            np.full(2, np.nan), epoch=self.epoch, unit='degree')
        boresight_epoch = options.get_string('TELEQUI', default=None)
        if boresight_epoch is None:
            boresight_epoch = self.epoch
        else:
            boresight_epoch = Epoch(equinox=boresight_epoch)
        self.boresight_equatorial = EquatorialCoordinates(
            np.full(2, np.nan), epoch=boresight_epoch, unit='degree')

        if "TELRA" in options and "TELDEC" in options:
            self.boresight_equatorial.ra = options.get_hms_time(
                'TELRA', angle=True)
            self.boresight_equatorial.dec = options.get_dms_angle('TELDEC')

        if 'OBSRA' in options and 'OBSDEC' in options:
            self.requested_equatorial.ra = options.get_hms_time(
                'OBSRA', angle=True)
            self.requested_equatorial.dec = options.get_dms_angle('OBSDEC')

    @staticmethod
    def get_telescope_name():
        """
        Return the telescope name.

        Returns
        -------
        name : str
        """
        return "SOFIA"

    def edit_image_header(self, header, scans=None):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.
        scans : list (Scan), optional
            A list of scans to use during editing.

        Returns
        -------
        None
        """
        header['TELESCOP'] = self.get_telescope_name()
        header.comments['TELESCOP'] = 'Telescope name.'

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        if self.boresight_equatorial is not None:
            telra = self.boresight_equatorial.ra.to(
                units.Unit('hourangle')).value
            teldec = self.boresight_equatorial.dec.to(units.Unit('deg')).value
        else:
            telra = teldec = np.nan

        if self.requested_equatorial is not None:
            obsra = self.requested_equatorial.ra.to(units.Unit('hourangle'))
            obsdec = self.requested_equatorial.dec.to(units.Unit('deg'))
        else:
            obsra = np.nan * units.Unit('hourangle')
            obsdec = np.nan * units.Unit('degree')

        info = [
            ('COMMENT', "<------ SOFIA Telescope Data ------>"),
            ('TELESCOP', self.telescope, 'Observatory name.'),
            ('TELCONF', self.tel_config, 'Telescope configuration.'),
            ('TELRA', telra, '(hour) Boresight RA.'),
            ('TELDEC', teldec, '(deg) Boresight DEC.'),
            ('TELEQUI', str(self.boresight_equatorial.epoch),
             'Boresight epoch.'),
            ('TELVPA', to_header_float(self.vpa, 'deg'),
             '(deg) Boresight position angle.'),
            ('LASTREW', self.last_rewind,
             'UTC time of last telescope rewind.'),
            ('FOCUS_ST', to_header_float(self.focus_t.start, 'um'),
             '(um) Focus T value at start.'),
            ('FOCUS_EN', to_header_float(self.focus_t.end, 'um'),
             '(um) Focus T value at end.'),
            ('TELEL', to_header_float(self.rel_elevation, 'deg'),
             '(deg) Telescope elevation in cavity.'),
            ('TELXEL', to_header_float(self.cross_elevation, 'deg'),
             '(deg) Telescope cross elevation in cavity.'),
            ('TELLOS', to_header_float(self.line_of_sight_angle, 'deg'),
             '(deg) Telescope line-of-sight angle in cavity.'),
            ('TSC-STAT', self.tascu_status, 'TASCU system status at end.'),
            ('FBC-STAT', self.fbc_status,
             'flexible body compensation system status at end'),
            ('OBSRA', to_header_float(obsra, 'hourangle'),
             '(hour) Requested RA.'),
            ('OBSDEC', to_header_float(obsdec, 'deg'), '(deg) Requested DEC.'),
            ('EQUINOX', Time(self.epoch.equinox, format='jyear').value,
             '(yr) The coordinate epoch.'),
            ('ZA_START', to_header_float(self.zenith_angle.start, 'deg'),
             '(deg) Zenith angle at start.'),
            ('ZA_END', to_header_float(self.zenith_angle.end, 'deg'),
             '(deg) Zenith angle at end.')
        ]

        if self.tracking_mode is not None:
            info.extend([
                ('TRACMODE', self.tracking_mode, 'SOFIA tracking mode.'),
                ('TRACERR', self.has_tracking_error,
                 'Was there a tracking error during the scan?')
            ])
        insert_info_in_header(header, info, delete_special=True)

    def get_table_entry(self, name):
        """
        Return a parameter value for the given name.

        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.

        Returns
        -------
        value
        """
        if name == 'focus':
            return self.focus_t.midpoint.to('um')
        elif name == 'bra':
            return self.boresight_equatorial.ra.to('hourangle')
        elif name == 'bdec':
            return self.boresight_equatorial.dec.to('degree')
        elif name == 'rra':
            return self.requested_equatorial.ra.to('hourangle')
        elif name == 'rdec':
            return self.requested_equatorial.dec.to('degree')
        elif name == 'epoch':
            return str(self.epoch)
        elif name == 'vpa':
            return self.vpa.to('degree')
        elif name == 'za':
            return self.zenith_angle.midpoint.to('degree')
        elif name == 'los':
            return self.line_of_sight_angle.to('degree')
        elif name == 'el':
            return self.rel_elevation.to('degree')
        elif name == 'xel':
            return self.cross_elevation.to('degree')
        elif name == 'trkerr':
            return self.has_tracking_error
        elif name == 'trkmode':
            return self.tracking_mode
        elif name == 'cfg':
            return self.tel_config
        elif name == 'fbc':
            return self.fbc_status
        elif name == 'rew':
            return self.last_rewind
        else:
            return super().get_table_entry(name)
