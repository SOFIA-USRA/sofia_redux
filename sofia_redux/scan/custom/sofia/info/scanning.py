# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.bracketed_values import BracketedValues
from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaScanningInfo']


class SofiaScanningInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.scanning = None
        self.ra = BracketedValues(np.nan, np.nan, unit='hourangle')
        self.dec = BracketedValues(np.nan, np.nan, unit='degree')
        self.speed = np.nan * units.Unit('arcsec/second')
        self.angle = np.nan * units.Unit('deg')
        self.scan_type = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'scan'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.scanning = options.get_bool("SCANNING")
        self.ra.start = options.get_hms_time("SCNRA0", angle=True)
        self.ra.end = options.get_hms_time("SCNRAF", angle=True)
        self.dec.start = options.get_dms_angle("SCNDEC0")
        self.dec.end = options.get_dms_angle("SCNDECF")
        self.speed = options.get_float(
            "SCNRATE", default=np.nan) * units.Unit('arcsec/second')
        self.angle = options.get_float("SCNDIR") * units.Unit('deg')
        self.scan_type = options.get_string("SCANTYPE")

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
            ('COMMENT', "<------ SOFIA Scanning Data ------>"),
            ('SCNRA0', to_header_float(self.ra.start, 'hourangle'),
             '(hour) Initial scan RA.'),
            ('SCNDEC0', to_header_float(self.dec.start, 'degree'),
             '(deg) Initial scan DEC.'),
            ('SCNRAF', to_header_float(self.ra.start, 'hourangle'),
             '(hour) Final scan RA.'),
            ('SCNDECF', to_header_float(self.dec.start, 'degree'),
             '(deg) Final scan DEC.'),
            ('SCNRATE', to_header_float(self.speed, 'arcsec/second'),
             '(arcsec/s) Commanded slew rate on sky.'),
            ('SCNDIR', to_header_float(self.angle, 'degree'),
             '(deg) Scan direction on sky.'),
            ('SCANTYPE', self.scan_type, 'Scan type.')
        ]
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
        if name == 'angle':
            return self.angle.to('degree')
        elif name == 'ra':
            return self.ra.midpoint.to('hourangle')
        elif name == 'dec':
            return self.dec.midpoint.to('degree')
        elif name == 'speed':
            return self.speed.to('arcsec/second')
        elif name == 'type':
            return self.scan_type
        else:
            return super().get_table_entry(name)
