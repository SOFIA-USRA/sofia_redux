# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.chopper.chopper import Chopper
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D


hz = units.Unit('Hz')
arcsec = units.Unit('arcsec')
second = units.Unit('second')
degree = units.Unit('degree')


@pytest.fixture
def signal():
    angle = np.deg2rad(30)
    n = 1000
    periods = 5.5
    amplitude = 5 * arcsec
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    t = np.linspace(0, 2 * np.pi * periods, n)
    r = np.sin(t) * amplitude
    dx = cos_a * r
    dy = sin_a * r
    position = Coordinate2D([dx, dy])
    return position


@pytest.fixture
def initialized_chopper(signal):
    time = np.arange(signal.size) * 0.1 * second
    chopper = Chopper(x=signal.x, y=signal.y, time=time, threshold=1 * arcsec)
    return chopper


def test_init(signal):
    time = np.arange(signal.size) * 0.1 * second
    chopper = Chopper(x=signal.x, y=signal.y, time=time, threshold=1 * arcsec)
    assert np.isclose(chopper.frequency, 0.056352 * hz,
                      atol=1e-6)
    assert chopper.amplitude > 3 * arcsec
    assert np.isclose(chopper.angle, 30 * degree)
    assert chopper.offset is None
    assert chopper.phases is None
    assert chopper.is_chopping
    assert np.isclose(chopper.efficiency, 0.394737, atol=1e-6)


def test_frequency():
    chopper = Chopper()
    assert np.isclose(chopper.frequency, np.nan * hz, equal_nan=True)
    chopper.frequency = 1
    assert chopper.frequency == 1 * hz
    chopper.frequency = 2 * hz
    assert chopper.frequency == 2 * hz


def test_amplitude():
    chopper = Chopper()
    assert chopper.amplitude == 0 * arcsec
    chopper.amplitude = 1
    assert chopper.amplitude == 1 * arcsec
    chopper.amplitude = 2 * arcsec
    assert chopper.amplitude == 2 * arcsec


def test_angle():
    chopper = Chopper()
    assert np.isclose(chopper.angle, np.nan * degree, equal_nan=True)
    chopper.angle = 1
    assert chopper.angle == 1 * degree
    chopper.angle = 2 * degree
    assert chopper.angle == 2 * degree


def test_stare_duration(initialized_chopper):
    chopper = Chopper()
    assert np.isclose(chopper.stare_duration, np.nan * second, equal_nan=True)
    chopper = initialized_chopper
    assert np.isclose(chopper.stare_duration, 3.502392 * second, atol=1e-6)


def test_str(initialized_chopper):
    s = str(initialized_chopper)
    assert s == ('chop +/- 3.566 arcsec at 30.000 deg, 0.056352 Hz, '
                 '39.5% efficiency')


def test_analyze_xy(signal):
    x, y = signal.coordinates
    t = np.arange(signal.size) * 0.1 * second
    chopper = Chopper()
    chopper.analyze_xy(x, y, t, 1 * arcsec)
    assert np.isclose(chopper.amplitude, 3.566 * arcsec, atol=1e-3)
    assert chopper.positions == 2
    assert np.isclose(chopper.frequency, 0.056 * hz, atol=1e-3)
    assert np.isclose(chopper.angle, 30 * degree, atol=1)
    assert np.isclose(chopper.efficiency, 0.395, atol=1e-3)
    assert chopper.is_chopping

    chopper.analyze_xy(x, y, t, 10 * arcsec)
    assert not chopper.is_chopping
    x2, y2 = x * 0, y * 0
    for i in range(100, 106):
        if i % 2 == 1:
            x2[i] = -2 * arcsec
        else:
            x2[i] = 2 * arcsec

    chopper.analyze_xy(x2, y2, t, 1 * arcsec)
    assert not chopper.is_chopping


def test_get_chop_table_entry(initialized_chopper):
    chopper = initialized_chopper
    assert np.isclose(chopper.get_chop_table_entry('chopfreq'),
                      0.056 * hz, atol=1e-3)
    assert np.isclose(chopper.get_chop_table_entry('chopthrow'),
                      3.566 * arcsec, atol=1e-3)
    assert np.isclose(chopper.get_chop_table_entry('chopeff'),
                      0.395, atol=1e-3)
    assert chopper.get_chop_table_entry('foo') is None
