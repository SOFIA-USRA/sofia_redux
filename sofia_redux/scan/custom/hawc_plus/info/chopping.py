# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.custom.sofia.info.chopping import SofiaChoppingInfo

__all__ = ['HawcPlusChoppingInfo']


class HawcPlusChoppingInfo(SofiaChoppingInfo):

    def __init__(self):
        """
        Initialize the HAWC+ chopping information.

        Contains information on the SOFIA chop parameters with an additional
        parameter for the minimum transit tolerance.
        """
        super().__init__()
        self.transit_tolerance = np.nan * units.Unit('arcsec')
