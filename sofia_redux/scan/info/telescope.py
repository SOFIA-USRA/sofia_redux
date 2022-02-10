# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000

__all__ = ['TelescopeInfo']


class TelescopeInfo(InfoBase):

    def __init__(self):
        super().__init__()
        self.is_tracking = False
        self.epoch = J2000

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'tel'

    @staticmethod
    def get_telescope_name():
        """
        Return the telescope name.

        Returns
        -------
        str
        """
        return 'UNKNOWN'

    def edit_image_header(self, header, scans=None):
        """
        Edit a FITS image header with the telescope information.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The header to edit.
        scans : Scan or list (Scan), optional
            Optional scans used for editing.

        Returns
        -------
        None
        """
        header['TELESCOP'] = self.get_telescope_name(), 'Telescope name.'
