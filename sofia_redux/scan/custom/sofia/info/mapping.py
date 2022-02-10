# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaMappingInfo']


class SofiaMappingInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.mapping = None
        self.coordinate_system = None
        self.size_x = -1
        self.size_y = -1
        self.step = Coordinate2D(unit='arcmin')

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'map'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.mapping = options.get_bool("MAPPING")
        self.coordinate_system = options.get_string("MAPCRSYS")
        self.size_x = options.get_int("MAPNXPOS")
        self.size_y = options.get_int("MAPNYPOS")
        self.step.x = options.get_float("MAPINTX", default=np.nan)
        self.step.y = options.get_float("MAPINTY", default=np.nan)

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
        if self.step is None:
            step = Coordinate2D(np.full(2, np.nan) * units.Unit('arcmin'))
        else:
            step = self.step

        info = [
            ('COMMENT', "<------ SOFIA Mapping Data ------>"),
            ('MAPCRSYS', self.coordinate_system, 'Mapping coordinate system.'),
            ('MAPNXPOS', self.size_x, 'Number of map positions in X.'),
            ('MAPNYPOS', self.size_y, 'Number of map positions in Y.'),
            ('MAPINTX', to_header_float(step.x, unit='arcmin'),
             '(arcmin) Map step interval in X.'),
            ('MAPINTY', to_header_float(step.y, unit='arcmin'),
             '(arcmin) Map step interval in Y.')
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
        if name == 'stepx':
            return self.step.x.to('arcmin')
        elif name == 'stepy':
            return self.step.y.to('arcmin')
        elif name == 'nx':
            return self.size_x
        elif name == 'ny':
            return self.size_y
        elif name == 'sys':
            return self.coordinate_system
        else:
            return super().get_table_entry(name)
