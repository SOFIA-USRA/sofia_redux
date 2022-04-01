# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.units import imperial
import numpy as np
import pytest

from sofia_redux.scan.custom.sofia.simulation.aircraft import \
    AircraftSimulation

imperial.enable()
ft = units.Unit('ft')
kft = ft * 1000
knot = units.Unit('knot')
second = units.Unit('second')
hourangle = units.Unit('hourangle')
degree = units.Unit('degree')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['ALTI_STA'] = 40000
    h['ALTI_END'] = 42000
    h['AIRSPEED'] = 510.0
    h['GRDSPEED'] = 500.0
    h['DATE-OBS'] = '2022-03-24T20:28:05.386'
    h['EXPTIME'] = 60.0
    h['UTCEND'] = '20:29:05.386'
    h['OBJRA'] = 12.0
    h['OBJDEC'] = 30.0
    h['LON_STA'] = 45.0
    h['LAT_STA'] = 10.0
    return h


@pytest.fixture
def sofia(sofia_header):
    sim = AircraftSimulation()
    sim.initialize_from_header(sofia_header)
    return sim


def test_init():
    sim = AircraftSimulation()
    assert sim.start_altitude == 41 * kft
    assert sim.end_altitude == 41 * kft
    assert sim.airspeed == 500 * knot
    for attr in ['heading', 'ground_speed', 'start_utc', 'end_utc',
                 'flight_time', 'source', 'start_location', 'end_location',
                 'start_lst', 'end_lst', 'start_horizontal', 'end_horizontal']:
        assert getattr(sim, attr) is None


def test_initialize_from_header(sofia_header):
    h = sofia_header.copy()
    sim = AircraftSimulation()
    sim.initialize_from_header(h)
    assert sim.start_altitude == 40 * kft
    assert sim.end_altitude == 42 * kft
    assert sim.airspeed == 510 * knot
    assert sim.ground_speed == 500 * knot
    assert sim.start_utc.isot == '2022-03-24T20:28:05.386'
    assert sim.end_utc.isot == '2022-03-24T20:29:05.386'
    assert np.isclose(sim.source.ra, 180 * degree)
    assert np.isclose(sim.source.dec, 30 * degree)
    assert sim.start_location.lon == 45 * degree
    assert sim.start_location.lat == 10 * degree
    assert np.isclose(sim.start_lst, 11.62104962 * hourangle, atol=1e-4)
    assert np.isclose(sim.start_horizontal.az, 14.0498866 * degree, atol=1e-4)
    assert np.isclose(sim.start_horizontal.el, 69.3088984 * degree, atol=1e-4)
    assert sim.heading is not None
    assert sim.end_lst is not None
    assert sim.end_horizontal is not None
    assert sim.end_location is not None

    sim.end_utc = None
    del h['EXPTIME']
    del h['GRDSPEED']
    sim.initialize_from_header(h)
    assert sim.ground_speed == 510 * knot
    assert sim.end_utc.isot == '2022-03-24T20:29:05.386'
    sim.end_utc = None
    del h['UTCEND']
    with pytest.raises(ValueError) as err:
        sim.initialize_from_header(h)
    assert "Cannot determine flight length" in str(err.value)


def test_orient_to_source(sofia):
    sim = sofia
    sim.start_horizontal.az = 80 * degree
    sim.orient_to_source()
    assert sim.heading == -10 * degree


def test_calculate_end_position(sofia):
    sim = sofia
    sim.heading = 30 * degree
    sim.calculate_end_position()
    assert np.isclose(sim.end_location.lon, 45.07168424 * degree, atol=1e-5)
    assert np.isclose(sim.end_location.lat, 10.1222206 * degree, atol=1e-5)
    assert np.isclose(sim.end_lst, 11.64254087 * hourangle, atol=1e-5)
    assert np.isclose(sim.end_horizontal.az, 13.36259847 * degree, atol=1e-5)
    assert np.isclose(sim.end_horizontal.el, 69.50285143 * degree, atol=1e-5)
