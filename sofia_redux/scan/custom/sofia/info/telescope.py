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


degree = units.Unit('degree')
um = units.Unit('um')
hourangle = units.Unit('hourangle')


class SofiaTelescopeInfo(TelescopeInfo):

    telescope_diameter = 2.5 * units.Unit('m')

    def __init__(self):
        """
        Initialize the SOFIA telescope information.

        Contains information on the SOFIA specific telescope parameters such as
        zenith angle, boresight coordinates, and tracking status.
        """
        super().__init__()
        self.telescope = "SOFIA 2.5m"
        self.tel_config = None
        self.vpa = np.nan * degree
        self.last_rewind = None
        self.focus_t = BracketedValues(np.nan * um,
                                       np.nan * um)
        self.rel_elevation = np.nan * degree
        self.cross_elevation = np.nan * degree
        self.line_of_sight_angle = np.nan * degree
        self.tascu_status = None
        self.fbc_status = None
        self.zenith_angle = BracketedValues(np.nan * degree,
                                            np.nan * degree)
        self.tracking_mode = None
        self.has_tracking_error = False
        self.is_tracking = False

        self.epoch = J2000
        self.boresight_equatorial = EquatorialCoordinates(
            np.full(2, np.nan), unit='degree', epoch=self.epoch)

        self.requested_equatorial = self.boresight_equatorial.copy()

    def apply_configuration(self):
        """
        Update telescope information with FITS header information.

        Updates the information by taking the following keywords from the
        FITS header::

          TELESCOP - The observatory name (str)
          TELVPA - The boresight position angle (degrees)
          LASTREW - The UTC time of last telescope rewind (str)
          FOCUS_ST - The focus T value at start (um)
          FOCUS_EN - The focus T value at end (um)
          TELEL - The telescope elevation in cavity (degrees)
          TELXEL - The telescope cross elevation in cavity (degrees)
          TELLOS - The telescope line-of-sight angle in cavity (degrees)
          TSC-STAT - The TASCU system status at end (str)
          FBC-STAT - The flexible body compensation system status at end (str)
          ZA_START - The zenith angle at start (degrees)
          ZA_END - The zenith angle at end (degrees)
          TRACMODE - The SOFIA tracking mode (str)
          TRACERR - Whether there was a tracking error in the scan (bool)
          TELCONF - The telescope configuration (str)
          EQUINOX - The coordinate epoch (year)
          TELEQUI - The boresight epoch (year)
          TELRA - The boresight RA (hourangle)
          TELDEC - The boresight DEC (degrees)
          OBSRA - The requested RA (hourangle)
          OBSDEC - The requested DEC (degrees)

        Returns
        -------
        None
        """
        options = self.options
        if options is None:
            return
        self.telescope = options.get_string("TELESCOP", default=self.telescope)
        self.vpa = options.get_float("TELVPA") * degree
        self.last_rewind = options.get_string("LASTREW")
        self.focus_t.start = options.get_float("FOCUS_ST") * um
        self.focus_t.end = options.get_float("FOCUS_EN") * um
        self.rel_elevation = options.get_float("TELEL") * degree
        self.cross_elevation = options.get_float("TELXEL") * degree
        self.line_of_sight_angle = options.get_float("TELLOS") * degree
        self.tascu_status = options.get_string("TSC-STAT")
        self.fbc_status = options.get_string("FBC-STAT")
        self.zenith_angle.start = options.get_float("ZA_START") * degree
        self.zenith_angle.end = options.get_float("ZA_END") * degree
        self.tracking_mode = options.get_string("TRACMODE")
        self.has_tracking_error = options.get_bool("TRACERR")
        self.is_tracking = str(self.tracking_mode).strip().upper() != 'OFF'

        self.tel_config = options.get_string('TELCONF')

        if "EQUINOX" in options:
            self.epoch = Epoch(equinox=options.get_float("EQUINOX"))

        self.requested_equatorial = EquatorialCoordinates(
            np.full(2, np.nan), epoch=self.epoch, unit='degree')
        boresight_epoch = options.get_string('TELEQUI', default=None)

        if (boresight_epoch is None or
                (isinstance(boresight_epoch, str) and
                 boresight_epoch.lower().startswith('unk'))):
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
        header['TELESCOP'] = self.get_telescope_name(), 'Telescope name.'

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
            telra = self.boresight_equatorial.ra.to(hourangle).value
            teldec = self.boresight_equatorial.dec.to(degree).value
            tel_epoch = self.boresight_equatorial.epoch
        else:
            telra = teldec = np.nan
            tel_epoch = None

        if self.requested_equatorial is not None:
            obsra = self.requested_equatorial.ra.to(hourangle)
            obsdec = self.requested_equatorial.dec.to(degree)
        else:
            obsra = np.nan * hourangle
            obsdec = np.nan * degree

        info = [
            ('COMMENT', "<------ SOFIA Telescope Data ------>"),
            ('TELESCOP', self.telescope, 'Observatory name.'),
            ('TELCONF', self.tel_config, 'Telescope configuration.'),
            ('TELRA', to_header_float(telra, 'hourangle'),
             '(hour) Boresight RA.'),
            ('TELDEC', to_header_float(teldec, 'degree'),
             '(deg) Boresight DEC.')]

        if tel_epoch is not None:
            info.append(('TELEQUI', str(tel_epoch), 'Boresight epoch.'))

        info.extend([
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
        ])

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
