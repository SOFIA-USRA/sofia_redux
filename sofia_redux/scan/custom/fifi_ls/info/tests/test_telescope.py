# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.fifi_ls.info.telescope import FifiLsTelescopeInfo


def test_init():
    info = FifiLsTelescopeInfo()
    assert info.delta_map is None


def test_apply_configuration():
    c = Configuration()
    c.instrument_name = 'fifi_ls'
    c.read_configuration('default.cfg')
    info = FifiLsTelescopeInfo()
    h = fits.Header()
    h['DLAM_MAP'] = 1.0
    h['DBET_MAP'] = 2.0
    c.read_fits(h)

    info.apply_configuration()
    assert info.delta_map is None

    info.configuration = c

    with pytest.raises(ValueError) as err:
        info.apply_configuration()
    assert 'No valid OBSRA/OBDEC' in str(err.value)

    h['OBSRA'] = 10.0
    h['OBSDEC'] = 20.0
    c.read_fits(h)
    info.apply_configuration()

    assert info.delta_map.x == 1 * units.Unit('arcsec')
    assert info.delta_map.y == 2 * units.Unit('arcsec')
    assert np.isclose(info.boresight_equatorial.ra,
                      149.9997044 * units.Unit('degree'), atol=1e-4)
    assert np.isclose(info.boresight_equatorial.dec,
                      19.99944444 * units.Unit('degree'), atol=1e-4)
