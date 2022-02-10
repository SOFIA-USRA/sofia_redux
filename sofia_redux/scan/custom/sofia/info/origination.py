# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.origination import OriginationInfo
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaOriginationInfo']


class SofiaOriginationInfo(OriginationInfo):

    def __init__(self):
        self.checksum = None
        self.checksum_version = None
        super().__init__()

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.organization = options.get_string("ORIGIN")
        self.observer = options.get_string("OBSERVER")
        self.creator = options.get_string("CREATOR")
        self.operator = options.get_string("OPERATOR")
        self.filename = options.get_string("FILENAME")
        self.checksum = options.get_string("DATASUM")
        self.checksum_version = options.get_string("CHECKVER")

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
        filename = 'UNKNOWN' if self.filename is None else self.filename
        info = [
            ('COMMENT', "<------ SOFIA Origination Data ------>"),
            ('ORIGIN', self.organization, 'Creator organization / node.'),
            ('CREATOR', self.creator, 'Creator software / task.'),
            ('OBSERVER', self.observer, 'Name(s) of observer(s).'),
            ('OPERATOR', self.operator, 'Name(s) of operator(s).'),
            ('FILENAME', filename, 'Original file name.')
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
        if name == 'creator':
            return self.creator
        elif name == 'file':
            return self.filename
        elif name == 'org':
            return self.organization
        elif name == 'observer':
            return self.observer
        elif name == 'operator':
            return self.operator
        else:
            return super().get_table_entry(name)
