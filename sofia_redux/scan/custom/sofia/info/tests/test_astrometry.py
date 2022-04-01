# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.astrometry import SofiaAstrometryInfo
from sofia_redux.scan.custom.sofia.info.telescope import SofiaTelescopeInfo


h = units.Unit('hour')
s = units.Unit('second')
degree = units.Unit('degree')


@pytest.fixture
def sofia_header():
    header = fits.Header()
    header['DATE'] = '2022-03-18T21:39:23.016'
    header['DATE-OBS'] = '2022-03-18T21:39:32.953'
    header['UTCSTART'] = '21:39:32.953'
    header['UTCEND'] = '21:39:42.953'  # 10 seconds
    header['EQUINOX'] = 2000.0
    header['OBSRA'] = 20.0
    header['OBSDEC'] = 30.0
    return header


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_astrometry(sofia_configuration):
    info = SofiaAstrometryInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_class():
    assert SofiaAstrometryInfo.default_fits_date == "1970-01-01T00:00:00.0"


def test_init():
    info = SofiaAstrometryInfo()
    assert info.date is None
    assert info.start_time is None
    assert info.file_date is None
    assert info.epoch.equinox == 'J2000'
    assert info.requested_equatorial is None
    assert info.horizontal is None
    assert info.equatorial is None
    assert info.ground_based


def test_apply_configuration(sofia_configuration):
    info = SofiaAstrometryInfo()
    info.apply_configuration()
    assert info.equatorial is None
    info.configuration = sofia_configuration
    info.apply_configuration()
    assert info.equatorial is not None
    assert info.date is not None


def test_parse_time(sofia_configuration):
    info = SofiaAstrometryInfo()
    info.configuration = sofia_configuration.copy()
    info.parse_time()
    assert info.file_date == '2022-03-18T21:39:23.016'
    assert info.date == '2022-03-18'
    assert np.isclose(info.utc.start, 21.65915361 * h)
    assert np.isclose(info.utc.end, 21.66193139 * h)
    assert info.time_stamp == '2022-03-18T21:39:32.953'
    assert np.isclose(info.mjd, 59656.902465, atol=1e-6)
    assert info.start_time == '21:39:32.953'
    assert info.epoch.equinox == 'J2000'
    assert info.apparent_epoch
    assert info.apparent_epoch.mjd == info.mjd
    assert info.from_apparent.from_epoch.equinox.mjd == info.mjd
    assert info.from_apparent.to_epoch.equinox == 'J2000'
    assert info.to_apparent.to_epoch.equinox.mjd == info.mjd
    assert info.to_apparent.from_epoch.equinox == 'J2000'

    options = info.configuration.fits.options
    options['DATE-OBS'] = '2022-03-18'
    info.parse_time()
    assert info.time_stamp == '2022-03-18T21:39:32.953'
    del options['UTCSTART']
    info.parse_time()
    assert info.time_stamp == '2022-03-18'

    info = SofiaAstrometryInfo()
    info.parse_time()
    assert info.start_time is None


def test_parse_astrometry(sofia_configuration):
    info = SofiaAstrometryInfo()
    info.configuration = sofia_configuration.copy()
    info.parse_time()
    info.parse_astrometry()
    ra = (20 * units.Unit('hourangle')).to('degree')
    dec = 30 * units.Unit('degree')
    assert info.equatorial == EquatorialCoordinates([ra, dec])
    info.configuration.fits.options['OBSRA'] = -9999.0
    info.parse_astrometry()
    assert info.equatorial is None
    info = SofiaAstrometryInfo()
    info.parse_astrometry()
    assert info.equatorial is None


def test_is_requested_valid(sofia_astrometry):
    info = sofia_astrometry.copy()
    assert info.is_requested_valid()
    info.configuration = None
    assert not info.is_requested_valid()
    header = fits.Header()
    header['OBSRA'] = 10
    header['OBSDEC'] = 10
    assert info.is_requested_valid(header=header)
    header['OBSRA'] = 0
    header['OBSDEC'] = 0
    assert not info.is_requested_valid(header=header)
    header['OBSDEC'] = -9999
    assert not info.is_requested_valid(header=header)
    header['OBSDEC'] = 10
    header['OBSRA'] = -9999
    assert not info.is_requested_valid(header=header)


def test_coordinate_valid():
    c = Coordinate2D([1, 1], unit='degree')
    assert SofiaAstrometryInfo.coordinate_valid(c)
    assert not SofiaAstrometryInfo.coordinate_valid(None)
    assert not SofiaAstrometryInfo.coordinate_valid(
        Coordinate2D([np.nan, 1], unit='degree'))
    assert not SofiaAstrometryInfo.coordinate_valid(
        Coordinate2D([1, np.nan], unit='degree'))


def test_guess_reference_coordinates():
    info = SofiaAstrometryInfo()
    e = EquatorialCoordinates([1, 1])
    info.object_coordinates = e
    assert info.guess_reference_coordinates() == e
    info.object_coordinates = None
    assert info.guess_reference_coordinates() is None
    info.requested_equatorial = e
    header = fits.Header()
    header['OBSRA'] = 1
    header['OBSDEC'] = 1
    assert info.guess_reference_coordinates(header=header) == e
    info.requested_equatorial = None
    assert info.guess_reference_coordinates(header=header) == e
    telescope = SofiaTelescopeInfo()
    telescope.boresight_equatorial = e
    assert info.guess_reference_coordinates(telescope=telescope) == e


def test_validate_astrometry(sofia_astrometry, populated_hawc_scan):
    info = sofia_astrometry.copy()
    info.validate_astrometry(None)  # Does nothing
    scan = populated_hawc_scan.copy()
    integrations = scan.integrations
    scan.integrations = [None]
    with pytest.raises(ValueError) as err:
        info.validate_astrometry(scan)
    assert "No integrations exist" in str(err.value)
    scan.integrations = integrations
    info.is_nonsidereal = True
    info.validate_astrometry(scan)
    assert np.isclose(info.object_coordinates.ra, 19.09026 * degree,
                      atol=1e-3)
    assert np.isclose(info.object_coordinates.dec, 7.406657 * degree,
                      atol=1e-3)
    assert info.object_coordinates == info.equatorial
    assert np.isclose(info.horizontal.az, -110.779 * degree, atol=1e-3)
    assert np.isclose(info.horizontal.el, 28.047 * degree, atol=1e-3)
    assert np.isclose(info.site.longitude, -108.164 * degree, atol=1e-3)
    assert np.isclose(info.site.latitude, 47.011 * degree, atol=1e-3)
