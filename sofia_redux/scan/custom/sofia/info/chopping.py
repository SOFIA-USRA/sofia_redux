# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase
import numpy as np
from astropy import units

from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaChoppingInfo']


class SofiaChoppingInfo(InfoBase):

    volts_to_angle = 33.394 * units.Unit('arcsec') / units.Unit('V')

    def __init__(self):
        super().__init__()
        self.chopping = None
        self.frequency = np.nan * units.Unit('Hz')
        self.profile_type = None
        self.symmetry_type = None
        self.amplitude = np.nan * units.Unit('arcsec')
        self.amplitude2 = np.nan * units.Unit('arcsec')
        self.coordinate_system = None
        self.angle = np.nan * units.Unit('deg')
        self.tip = np.nan * units.Unit('arcsec')
        self.tilt = np.nan * units.Unit('arcsec')
        self.phase = np.nan * units.Unit('ms')

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'chop'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return

        self.chopping = options.get_bool("CHOPPING")
        self.frequency = options.get_float("CHPFREQ") * units.Unit('Hz')
        self.profile_type = options.get_string("CHPPROF")
        self.symmetry_type = options.get_string("CHPSYM")
        self.amplitude = options.get_float("CHPAMP1") * units.Unit('arcsec')
        self.amplitude2 = options.get_float("CHPAMP2") * units.Unit('arcsec')
        self.coordinate_system = options.get_string("CHPCRSYS")
        self.angle = options.get_float("CHPANGLE") * units.Unit('deg')
        self.tip = options.get_float("CHPTIP") * units.Unit('arcsec')
        self.tilt = options.get_float("CHPTILT") * units.Unit('arcsec')
        self.phase = options.get_float("CHPPHASE") * units.Unit('ms')

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
            ('COMMENT', "<------ SOFIA Chopping Data ------>"),
            ('CHPFREQ', to_header_float(self.frequency, 'Hz'),
             '(Hz) Chop frequency.'),
            ('CHPAMP1', to_header_float(self.amplitude, 'arcsec'),
             '(arcsec) Chop amplitude on sky.'),
            ('CHPAMP2', to_header_float(self.amplitude2, 'arcsec'),
             '(arcsec) Second chop amplitude on sky.'),
            ('CHPANGLE', to_header_float(self.angle, 'deg'),
             '(deg) Chop angle on sky.'),
            ('CHPTIP', to_header_float(self.tip, 'arcsec'),
             '(arcsec) Chopper tip on sky.'),
            ('CHPTILT', to_header_float(self.tilt, 'arcsec'),
             '(arcsec) Chop tilt on sky.'),
            ('CHPPROF', self.profile_type, 'Chop profile from MCCS.'),
            ('CHPSYM', self.symmetry_type, 'Chop symmetry mode.'),
            ('SHPCRSYS', self.coordinate_system, 'Chop coordinate system.'),
            ('CHPPHASE', to_header_float(self.phase, 'ms'), '(ms) Chop phase.')
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
        if name == 'flag':
            if self.amplitude == 0 or np.isnan(self.amplitude):
                return '-'
            else:
                return 'C'
        elif name == 'amp':
            return self.amplitude.to('arcsec')
        elif name == 'angle':
            return self.angle.to('degree')
        elif name == 'frequency':
            return self.frequency.to('Hz')
        elif name == 'tip':
            return self.tip.to('arcsec')
        elif name == 'tilt':
            return self.tilt.to('arcsec')
        elif name == 'profile':
            return self.profile_type
        elif name == 'sys':
            return self.coordinate_system
        else:
            return super().get_table_entry(name)
