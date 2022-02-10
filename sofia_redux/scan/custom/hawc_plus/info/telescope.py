# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.sofia.info.telescope import SofiaTelescopeInfo

from astropy import units, log
import numpy as np

__all__ = ['HawcPlusTelescopeInfo']


class HawcPlusTelescopeInfo(SofiaTelescopeInfo):

    def __init__(self):
        super().__init__()
        self.focus_t_offset = np.nan * units.Unit('um')

    def apply_configuration(self):
        options = self.options
        if options is None:
            return

        self.focus_t_offset = options.get_float("FCSTOFF")
        if not np.isnan(self.focus_t_offset):
            log.debug(f"Focus T Offset: {self.focus_t_offset}")
        super().apply_configuration()
