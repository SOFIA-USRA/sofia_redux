# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.sofia.info.telescope import SofiaTelescopeInfo

from astropy import units, log
import numpy as np

__all__ = ['HawcPlusTelescopeInfo']


class HawcPlusTelescopeInfo(SofiaTelescopeInfo):

    def __init__(self):
        """
        Initialize the HAWC+ telescope information.

        Contains information on the SOFIA specific telescope parameters such as
        zenith angle, boresight coordinates, and tracking status and the focus
        offset for HAWC+.
        """
        super().__init__()
        self.focus_t_offset = np.nan * units.Unit('um')

    def apply_configuration(self):
        """
        Update telescope information with FITS header information.

        Updates the information by taking the following keywords from the
        FITS header::

          FCSTOFF - The total focus offset (um)

        Returns
        -------
        None
        """
        options = self.options
        if options is None:
            return

        self.focus_t_offset = options.get_float("FCSTOFF") * units.Unit('um')
        if not np.isnan(self.focus_t_offset):
            log.debug(f"Focus T Offset: {self.focus_t_offset}")
        super().apply_configuration()
