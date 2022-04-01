# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.aircraft import SofiaAircraftInfo

ft = SofiaAircraftInfo.ft
kn = SofiaAircraftInfo.knots
degree = units.Unit('degree')
kft = SofiaAircraftInfo.kft


@pytest.fixture
def sofia_header():
    header = fits.Header()
    header['LAT_STA'] = 30.1
    header['LAT_END'] = 30.2
    header['LON_STA'] = 40.1
    header['LON_END'] = 40.2
    header['HEADING'] = 15.5
    header['TRACKANG'] = 1.0
    header['AIRSPEED'] = 500.0
    header['GRDSPEED'] = 510.0
    header['ALTI_STA'] = 39000.0
    header['ALTI_END'] = 40000.0
    return header


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def aircraft_info(sofia_configuration):
    info = SofiaAircraftInfo()
    info.configuration = sofia_configuration
    info.apply_configuration()
    return info


def test_class():
    assert SofiaAircraftInfo.knots == 'kn'
    assert SofiaAircraftInfo.ft == 'ft'
    assert SofiaAircraftInfo.kft == 1000 * SofiaAircraftInfo.ft


def test_init():
    info = SofiaAircraftInfo()
    assert info.altitude.unit == SofiaAircraftInfo.kft
    assert np.isnan(info.altitude.midpoint)
    assert info.latitude.unit == 'degree'
    assert np.isnan(info.latitude.midpoint)
    assert info.longitude.unit == 'degree'
    assert np.isnan(info.longitude.midpoint)
    assert np.isnan(info.air_speed) and info.air_speed.unit == kn
    assert np.isnan(info.ground_speed) and info.ground_speed.unit == kn
    assert np.isnan(info.heading) and info.heading.unit == 'degree'
    assert np.isnan(info.track_ang) and info.track_ang.unit == 'degree'


def test_log_id():
    info = SofiaAircraftInfo()
    assert info.log_id == 'ac'


def test_apply_configuration(sofia_configuration):
    info = SofiaAircraftInfo()
    info.configuration = sofia_configuration
    info.apply_configuration()
    assert np.isclose(info.latitude.midpoint, 30.15 * degree)
    assert np.isclose(info.longitude.midpoint, 40.15 * degree)
    assert info.heading == 15.5 * degree
    assert info.track_ang == 1 * degree
    assert info.air_speed == 500 * kn
    assert info.ground_speed == 510 * kn
    assert np.isclose(info.altitude.midpoint, 39500 * ft)
    info = SofiaAircraftInfo()
    info.apply_configuration()
    assert np.isnan(info.latitude.midpoint)


def test_edit_header(aircraft_info, sofia_header):
    info = aircraft_info
    header = fits.Header()
    info.edit_header(header)
    for k, v in header.items():
        if k in ['COMMENT', 'HISTORY']:
            continue
        assert np.isclose(sofia_header[k], v)


def test_get_table_entry(aircraft_info):
    info = aircraft_info
    kmh = units.Unit('km/h')
    assert np.isclose(info.get_table_entry('alt'), 12039.6 * units.Unit('m'))
    assert np.isclose(info.get_table_entry('altkft'), 39.5 * kft)
    assert np.isclose(info.get_table_entry('lon'), 40.15 * degree)
    assert np.isclose(info.get_table_entry('lat'), 30.15 * degree)
    assert np.isclose(info.get_table_entry('lond'), 40.15 * degree)
    assert np.isclose(info.get_table_entry('latd'), 30.15 * degree)
    assert np.isclose(info.get_table_entry('airspeed'), 926 * kmh)
    assert np.isclose(info.get_table_entry('gndspeed'), 944.52 * kmh, atol=0.1)
    assert np.isclose(info.get_table_entry('dir'), 15.5 * degree)
    assert np.isclose(info.get_table_entry('trkangle'), 1 * degree)
    assert info.get_table_entry('foo') is None
