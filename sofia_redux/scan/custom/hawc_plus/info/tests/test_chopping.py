# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.custom.hawc_plus.info.chopping import \
    HawcPlusChoppingInfo


def test_init():
    info = HawcPlusChoppingInfo()
    assert np.isclose(info.transit_tolerance, np.nan * units.Unit('arcsec'),
                      equal_nan=True)
