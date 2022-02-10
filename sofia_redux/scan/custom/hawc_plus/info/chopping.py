# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.custom.sofia.info.chopping import SofiaChoppingInfo

__all__ = ['HawcPlusChoppingInfo']


class HawcPlusChoppingInfo(SofiaChoppingInfo):

    def __init__(self):
        super().__init__()
        self.transit_tolerance = np.nan * units.Unit('arcsec')
