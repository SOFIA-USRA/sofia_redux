# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.bracketed_values import BracketedValues
from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaEnvironmentInfo']


class SofiaEnvironmentInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.pwv = BracketedValues()
        self.ambient_t = np.nan * units.Unit('K')
        self.primary_t1 = np.nan * units.Unit('K')
        self.primary_t2 = np.nan * units.Unit('K')
        self.primary_t3 = np.nan * units.Unit('K')
        self.secondary_t = np.nan * units.Unit('K')

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'env'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return

        self.pwv.start = options.get_float("WVZ_STA") * units.Unit('um')
        self.pwv.end = options.get_float("WVZ_END") * units.Unit('um')
        self.ambient_t = options.get_float("TEMP_OUT") * units.Unit('deg_C')
        self.primary_t1 = options.get_float("TEMPPRI1") * units.Unit('deg_C')
        self.primary_t2 = options.get_float("TEMPPRI2") * units.Unit('deg_C')
        self.primary_t3 = options.get_float("TEMPPRI3") * units.Unit('deg_C')
        self.secondary_t = options.get_float("TEMPSEC1") * units.Unit('deg_C')

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
            ('COMMENT', "<------ SOFIA Environmental Data ------>"),
            ('WVZ_STA', to_header_float(self.pwv.start, 'um'),
             '(um) Precipitable Water Vapor at start.'),
            ('WVZ_END', to_header_float(self.pwv.end, 'um'),
             '(um) Precipitable Water Vapor at end.'),
            ('TEMP_OUT', to_header_float(self.ambient_t, 'deg_C'),
             '(C) Ambient air temperature.'),
            ('TEMPPRI1', to_header_float(self.primary_t1, 'deg_C'),
             '(C) Primary mirror temperature #1.'),
            ('TEMPPRI2', to_header_float(self.primary_t2, 'deg_C'),
             '(C) Primary mirror temperature #2.'),
            ('TEMPPRI3', to_header_float(self.primary_t3, 'deg_C'),
             '(C) Primary mirror temperature #3.'),
            ('TEMPSEC1', to_header_float(self.secondary_t, 'deg_C'),
             '(C) Secondary mirror temperature.')
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
        if name == 'tamb':
            return self.ambient_t.to('Kelvin')
        elif name == 'pwv':
            return self.pwv.midpoint.to('um')
        elif name == 't1':
            return self.primary_t1.to('Kelvin')
        elif name == 't2':
            return self.primary_t2.to('Kelvin')
        elif name == 't3':
            return self.primary_t3.to('Kelvin')
        elif name == 'sect':
            return self.secondary_t.to('Kelvin')
        else:
            return super().get_table_entry(name)
