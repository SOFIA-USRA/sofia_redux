# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaModeInfo']


class SofiaModeInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.is_chopping = False
        self.is_nodding = False
        self.is_dithering = False
        self.is_mapping = False
        self.is_scanning = False

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'mode'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.is_chopping = options.get_bool("CHOPPING")
        self.is_nodding = options.get_bool("NODDING")
        self.is_dithering = options.get_bool("DITHER")
        self.is_mapping = options.get_bool("MAPPING")
        self.is_scanning = options.get_bool("SCANNING")

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
            ('COMMENT', "<------ SOFIA Mode Data ------>"),
            ('CHOPPING', self.is_chopping, "Was chopper in use?"),
            ('NODDING', self.is_nodding, 'Was nodding used?'),
            ('DITHER', self.is_dithering, 'Was dithering used?'),
            ('MAPPING', self.is_mapping, 'Was mapping?'),
            ('SCANNING', self.is_scanning, 'Was scanning?')
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
        if name == 'chop':
            return self.is_chopping
        elif name == 'nod':
            return self.is_nodding
        elif name == 'dither':
            return self.is_dithering
        elif name == 'map':
            return self.is_mapping
        elif name == 'scan':
            return self.is_scanning
        else:
            return super().get_table_entry(name)
