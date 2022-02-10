# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.utils import to_header_float
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaDitheringInfo']


class SofiaDitheringInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.dithering = None
        self.coordinate_system = None
        self.pattern_shape = None
        self.offset = Coordinate2D(np.full(2, np.nan), unit='arcsec')
        self.positions = -1
        self.index = -1

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'dither'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return

        self.dithering = options.get_bool("DITHER")
        self.coordinate_system = options.get_string("DTHCRSYS")
        self.pattern_shape = options.get_string("DTHPATT")
        self.offset.x = options.get_float('DTHXOFF')
        self.positions = options.get_int("DTHNPOS")
        self.index = options.get_int("DTHINDEX")

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
        if self.offset is None:
            offset = Coordinate2D(np.full(2, np.nan) * units.Unit('arcsec'))
        else:
            offset = self.offset

        info = [
            ('COMMENT', "<------ SOFIA Dithering Data ------>"),
            ('DTHCRSYS', self.coordinate_system, 'Dither coordinate system.'),
            ('DTHXOFF', to_header_float(offset.x, 'arcsec'),
             '(arcsec) Dither X offset.'),
            ('DTHYOFF', to_header_float(offset.y, 'arcsec'),
             '(arcsec) Dither Y offset.'),
            ('DTHPATT', self.pattern_shape,
             'Approximate shape of dither pattern.'),
            ('DTHNPOS', self.positions, 'Number of dither positions.'),
            ('DTHINDEX', self.index, 'Dither position index.')
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
        if name == 'dx':
            return self.offset.x.to('arcsec')
        elif name == 'dy':
            return self.offset.y.to('arcsec')
        elif name == 'index':
            return self.index
        elif name == 'pattern':
            return self.pattern_shape
        elif name == 'npos':
            return self.positions
        elif name == 'sys':
            return self.coordinate_system
        else:
            return super().get_table_entry(name)
