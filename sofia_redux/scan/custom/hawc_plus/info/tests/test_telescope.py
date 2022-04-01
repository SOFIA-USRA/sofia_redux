# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy import units
import numpy as np

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.hawc_plus.info.telescope import \
    HawcPlusTelescopeInfo


def test_init():
    info = HawcPlusTelescopeInfo()
    assert np.isclose(info.focus_t_offset, np.nan * units.Unit('um'),
                      equal_nan=True)


def test_apply_configuration():
    info = HawcPlusTelescopeInfo()
    info.apply_configuration()
    assert np.isnan(info.focus_t_offset)
    c = Configuration()
    c.read_configuration('default.cfg')
    h = fits.Header()
    h['FCSTOFF'] = 11.0
    c.read_fits(h)
    info.configuration = c
    info.apply_configuration()
    assert info.focus_t_offset == 11 * units.Unit('um')
