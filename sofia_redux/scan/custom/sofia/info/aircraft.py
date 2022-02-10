# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.bracketed_values import BracketedValues
import numpy as np
from astropy import units
from astropy.units import imperial

from sofia_redux.scan.utilities.utils import to_header_float
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaAircraftInfo']


class SofiaAircraftInfo(InfoBase):

    knots = imperial.kn
    ft = imperial.ft
    kft = 1000 * ft

    def __init__(self):
        super().__init__()
        self.altitude = BracketedValues(unit=self.kft)
        self.latitude = BracketedValues(unit='degree')
        self.longitude = BracketedValues(unit='degree')
        self.air_speed = np.nan * self.knots
        self.ground_speed = np.nan * self.knots
        self.heading = np.nan * units.Unit('deg')
        self.track_ang = np.nan * units.Unit('deg')

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'ac'

    def apply_configuration(self):
        if self.options is None:
            return
        options = self.options
        self.latitude.start = options.get_dms_angle("LAT_STA")
        self.latitude.end = options.get_dms_angle("LAT_END")
        self.longitude.start = options.get_dms_angle("LON_STA")
        self.longitude.end = options.get_dms_angle("LON_END")
        self.heading = options.get_dms_angle("HEADING")
        self.track_ang = options.get_dms_angle("TRACKANG")
        self.air_speed = options.get_float("AIRSPEED") * self.knots
        self.ground_speed = options.get_float("GRDSPEED") * self.knots
        self.altitude.start = options.get_float("ALTI_STA") * self.ft
        self.altitude.end = options.get_float("ALTI_END") * self.ft

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
        info = [
            ('COMMENT', "<------ SOFIA Aircraft Data ------>"),
            ('LON_STA', to_header_float(self.longitude.start, 'deg'),
             '(deg) Longitude at start of observation.'),
            ('LON_END', to_header_float(self.longitude.end, 'deg'),
             '(deg) Longitude at end of observation.'),
            ('LAT_STA', to_header_float(self.latitude.start, 'deg'),
             '(deg) Latitude at start of observation.'),
            ('LAT_END', to_header_float(self.latitude.end, 'deg'),
             '(deg) Latitude at end of observation.'),
            ('ALTI_STA', to_header_float(self.altitude.start, self.ft),
             '(ft) Altitude at start of observation.'),
            ('ALTI_END', to_header_float(self.altitude.end, self.ft),
             '(ft) Altitude at end of observation.'),
            ('AIRSPEED', to_header_float(self.air_speed, self.knots),
             '(kn) Airspeed at start of observation.'),
            ('GRDSPEED', to_header_float(self.ground_speed, self.knots),
             '(kn) Ground speed at start of observation.'),
            ('HEADING', to_header_float(self.heading, 'deg'),
             '(deg) True aircraft heading at start.'),
            ('TRACKANG', to_header_float(self.track_ang, 'deg'),
             '(deg) Aircraft tracking angle at start.')
        ]
        insert_info_in_header(header, info, delete_special=True)

    def get_table_entry(self, name):
        """
        Given a name, return the parameter stored in the information object.

        Note that names do not exactly match to attribute names.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        if name == 'alt':
            return self.altitude.midpoint.to('meter')
        elif name == 'altkft':
            return self.altitude.midpoint.to(self.kft)
        elif name == 'lon':
            return self.longitude.midpoint
        elif name == 'lat':
            return self.latitude.midpoint
        elif name == 'lond':
            return self.longitude.midpoint.to('degree')
        elif name == 'latd':
            return self.latitude.midpoint.to('degree')
        elif name == 'airspeed':
            return self.air_speed.to('km/h')
        elif name == 'gndspeed':
            return self.ground_speed.to('km/h')
        elif name == 'dir':
            return self.heading.to('degree')
        elif name == 'trkangle':
            return self.track_ang.to('degree')
        else:
            return super().get_table_entry(name)
