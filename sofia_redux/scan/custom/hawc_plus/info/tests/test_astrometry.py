# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.hawc_plus.info.astrometry import \
    HawcPlusAstrometryInfo

hourangle = units.Unit('hourangle')
degree = units.Unit('degree')


@pytest.fixture
def hawc_header():
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
    header['NONSIDE'] = False
    return header


@pytest.fixture
def hawc_configuration(hawc_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(hawc_header)
    c.parse_key_value('rtoc', 'False')
    return c


@pytest.fixture
def hawc_astrometry(hawc_configuration):
    info = HawcPlusAstrometryInfo()
    info.configuration = hawc_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = HawcPlusAstrometryInfo()
    assert info.equatorial is None


def test_apply_configuration(hawc_astrometry):
    info = HawcPlusAstrometryInfo()
    info.apply_configuration()
    assert info.object_coordinates is None
    info = hawc_astrometry.copy()
    info.apply_configuration()
    assert info.object_coordinates is None
    assert not info.is_nonsidereal
    info.configuration.parse_key_value(
        'OBJRA', str(info.configuration.fits.header['OBJRA']))
    info.configuration.parse_key_value(
        'OBJDEC', str(info.configuration.fits.header['OBJDEC']))
    info.apply_configuration()
    assert info.object_coordinates.ra == 21 * hourangle
    assert info.object_coordinates.dec == 31 * degree
    assert not info.is_nonsidereal
    info.configuration.parse_key_value('rtoc', 'True')
    info.apply_configuration()
    assert info.is_nonsidereal
    info.configuration.parse_key_value('rtoc', 'False')
    header = info.configuration.fits.header.copy()
    header['NONSIDE'] = True
    info.configuration.read_fits(header)
    info.apply_configuration()
    assert info.is_nonsidereal


def test_is_requested_valid(hawc_astrometry):
    info = HawcPlusAstrometryInfo()
    assert not info.is_requested_valid()
    info = hawc_astrometry.copy()
    assert info.is_requested_valid()
    header = fits.Header()
    header['OBSRA'] = 1.0  # ra=1, dec=2 is considered bad
    header['OBSDEC'] = 2.0
    assert not info.is_requested_valid(header=header)
    header['OBSRA'] = 3.0
    header['OBSDEC'] = 4.0
    assert info.is_requested_valid(header=header)


def test_guess_reference_coordinates(hawc_astrometry):
    info = hawc_astrometry.copy()
    info.is_nonsidereal = True
    assert info.guess_reference_coordinates() is None
    info.is_nonsidereal = False
    guess = info.guess_reference_coordinates()
    assert guess.ra == 20 * hourangle
    assert guess.dec == 30 * degree
