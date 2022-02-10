# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase
import numpy as np
from astropy import units

from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaNoddingInfo']


class SofiaNoddingInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.nodding = None
        self.dwell_time = np.nan * units.s
        self.cycles = -1
        self.settling_time = np.nan * units.s
        self.amplitude = np.nan * units.Unit('arcsec')
        self.angle = np.nan * units.Unit('deg')
        self.beam_position = None
        self.pattern = None
        self.style = None
        self.coordinate_system = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'nod'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.nodding = options.get_bool("NODDING")
        self.dwell_time = options.get_float("NODTIME") * units.Unit('s')
        self.cycles = options.get_int("NODN")
        self.settling_time = options.get_float("NODSETL") * units.Unit('s')
        self.amplitude = options.get_float("NODAMP") * units.Unit('arcsec')
        self.angle = options.get_float("NODANGLE") * units.Unit('deg')
        self.beam_position = options.get_string("NODBEAM")
        self.pattern = options.get_string("NODPATT")
        self.style = options.get_string("NODSTYLE")
        self.coordinate_system = options.get_string("NODCRSYS")

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
            ('COMMENT', "<------ SOFIA Nodding Data ------>"),
            ('NODN', self.cycles, 'Number of nod cycles.'),
            ('NODAMP', to_header_float(self.amplitude, 'arcsec'),
             '(arcsec) Nod amplitude on sky.'),
            ('NODANGLE', to_header_float(self.angle, 'deg'),
             '(deg) Nod angle on sky.'),
            ('NODTIME', to_header_float(self.dwell_time, 'second'),
             '(s) Total dwell time per nod position.'),
            ('NODSETL', to_header_float(self.settling_time, 'second'),
             '(s) Nod settling time.'),
            ('NODPATT', self.pattern, 'Ponting sequence for one nod cycle.'),
            ('NODCRSYS', self.coordinate_system, 'Nodding coordinate system.'),
            ('NODBEAM', self.beam_position, 'Nod beam position.'),
        ]

        if self.style is not None:
            info.append(('NODSTYLE', self.style, 'Nodding style'))

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
        if name == 'amp':
            return self.amplitude.to('arcsec')
        elif name == 'angle':
            return self.angle.to('degree')
        elif name == 'dwell':
            return self.dwell_time.to('second')
        elif name == 'settle':
            return self.settling_time.to('second')
        elif name == 'n':
            return str(self.cycles)
        elif name == 'pos':
            return self.beam_position
        elif name == 'sys':
            return self.coordinate_system
        elif name == 'pattern':
            return self.pattern
        elif name == 'style':
            return self.style
        else:
            return super().get_table_entry(name)
