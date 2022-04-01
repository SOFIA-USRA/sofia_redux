# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.custom.example.info.astrometry import \
    ExampleAstrometryInfo


degree = units.Unit('degree')


@pytest.fixture
def example_configuration(populated_scan):
    return populated_scan.configuration.copy()


@pytest.fixture
def configured_astrometry(example_configuration):
    info = ExampleAstrometryInfo()
    info.configuration = example_configuration.copy()
    return info


def test_init():
    info = ExampleAstrometryInfo()
    assert info.date is None
    assert info.epoch.equinox == 'J2000'
    assert info.horizontal is None
    assert info.equatorial is None
    assert info.ground_based


def test_apply_configuration(configured_astrometry):
    info = ExampleAstrometryInfo()
    info.apply_configuration()
    assert info.time_stamp is None  # Nothing happens (no configuration)
    info = configured_astrometry.copy()
    info.apply_configuration()
    assert info.time_stamp == '2021-12-06T18:48:25.876'
    assert np.isclose(info.mjd, 59554.783633, atol=1e-6)
    assert info.apparent_epoch.equinox.value == info.mjd
    assert np.isclose(info.equatorial.ra, 266.415009 * degree, atol=1e-6)
    assert np.isclose(info.equatorial.dec, -29.006111 * degree, atol=1e-6)
    assert np.isclose(info.site.latitude, 37.4089 * degree)
    assert np.isclose(info.site.longitude, -122.0644 * degree)
    assert np.isclose(info.lst, 15.721237 * units.Unit('hourangle'), atol=1e-6)
    for key in ['DATE-OBS', 'SITELON', 'SITELAT', 'LST']:
        del info.options[key]
    info.time_stamp = None
    info.apply_configuration()
    assert info.time_stamp == '1970-01-01T00:00:00.0'
    assert info.site.latitude == 37.4089 * degree
    assert info.site.longitude == -122.0644 * degree
    assert np.isclose(info.lst, 22.544576 * units.Unit('hourangle'), atol=1e-6)
