# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.fifi_ls.info.astrometry import \
    FifiLsAstrometryInfo


@pytest.fixture
def fifi_header():
    header = fits.Header()
    header['DATE'] = '2022-03-18T21:39:23.016'
    header['DATE-OBS'] = '2022-03-18T21:39:32.953'
    header['UTCSTART'] = '21:39:32.953'
    header['UTCEND'] = '21:39:42.953'  # 10 seconds
    header['EQUINOX'] = 2000.0
    header['OBSRA'] = 20.0
    header['OBSDEC'] = 30.0
    header['OBJRA'] = 21.0
    header['OBJDEC'] = 31.0
    header['NONSIDE'] = True
    return header


@pytest.fixture
def fifi_configuration(fifi_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(fifi_header)
    return c


def test_init():
    info = FifiLsAstrometryInfo()
    assert info.epoch.equinox == 'J2000'


def test_apply_configuration(fifi_configuration):
    info = FifiLsAstrometryInfo()
    info.apply_configuration()
    assert info.date is None
    info.configuration = fifi_configuration
    info.apply_configuration()
    assert info.date == '2022-03-18'
    assert not info.is_nonsidereal
    assert info.requested_equatorial is not None
    assert info.equatorial is not None
    assert info.scan_equatorial is not None
    assert info.delta_map.is_null()


def test_parse_astrometry():
    # Just need to check case when no options are available
    info = FifiLsAstrometryInfo()
    info.parse_astrometry()
    assert info.requested_equatorial is None
