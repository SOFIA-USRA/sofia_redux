# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.units import imperial
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.sofia.integration.integration import \
    SofiaIntegration

imperial.enable()
arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
kft = 1000 * units.Unit('ft')
hz = units.Unit('Hz')
um = units.Unit('um')


@pytest.fixture
def sofia_integration(initialized_hawc_scan):
    scan = initialized_hawc_scan
    scan.hdul = fits.HDUList()
    integration = SofiaIntegration(scan=scan)
    integration.frames.integration = integration
    integration.frames.set_frame_size(1)
    integration.frames.horizontal.el[0] = 45 * degree
    integration.frames.horizontal.sin_lat[0] = 1 / np.sqrt(2)
    integration.info.aircraft.altitude.start = 41 * kft
    integration.info.aircraft.altitude.end = 41 * kft
    integration.frames.chopper_position.x[0] = 1 * arcsec
    integration.frames.chopper_position.y[0] = 1 * arcsec
    integration.frames.pwv[0] = 2 * um
    return integration


@pytest.fixture
def atran_options():
    return {
        'amcoeffs': ['0.9995', '-0.1089', '0.02018', '0.008359', '-0.006565'],
        'altcoeffs': ['0.9994', '0.01921', '-0.0001924', '-0.0003502',
                      '-2.141e-05', '1.974e-05'],
        'reference': '0.682'}


@pytest.fixture
def pwv_options():
    return {'pwv': {'a': '1.0', 'b': '0.0', 'value': '62.49575959996947'},
            'hawc_plus': {'a': '0.0020', 'b': '0.181'}}


@pytest.fixture
def configured_integration(sofia_integration, atran_options, pwv_options):
    integration = sofia_integration
    c = integration.configuration
    for key, value in atran_options.items():
        c.parse_key_value(f'atran.{key}', value)
    for key, value in pwv_options.items():
        c.parse_key_value(f'tau.{key}', value)
    c.parse_key_value('pwvscale', '4.38')
    c.parse_key_value('pwv41k', '22.0')
    return integration


def test_init(initialized_hawc_scan):
    scan = initialized_hawc_scan
    scan.hdul = fits.HDUList()
    integration = SofiaIntegration(scan=scan)
    assert integration.scan is scan


def test_set_tau(configured_integration):
    integration = configured_integration.copy()
    c = integration.configuration
    c.parse_key_value('tau', 'atran')
    integration.set_tau()
    assert np.isclose(integration.zenith_tau, 0.2714, atol=1e-4)
    c.parse_key_value('tau', 'pwvmodel')
    integration.set_tau()
    assert np.isclose(integration.zenith_tau, 0.225)
    del c['tau']
    integration.zenith_tau = 0.0
    integration.set_tau()
    assert integration.zenith_tau == 0


def test_get_atran_tau(configured_integration):
    integration = configured_integration
    integration.set_atran_tau()
    np.isclose(integration.zenith_tau, 0.2714, atol=1e-4)


def test_set_pwv_model_tau(configured_integration):
    integration = configured_integration
    integration.set_pwv_model_tau()
    assert np.isclose(integration.zenith_tau, 0.225)


def test_get_modulation_frequency(sofia_integration):
    integration = sofia_integration.copy()
    assert integration.get_modulation_frequency(1) == 0 * hz
    integration.info.chopping.chopping = True
    integration.info.chopping.frequency = 2 * hz
    assert integration.get_modulation_frequency(1) == 2 * hz


def test_get_mean_pwv(sofia_integration):
    integration = sofia_integration.copy()
    assert integration.get_mean_pwv() == 2 * um
    integration.frames.valid[0] = False
    assert np.isclose(integration.get_mean_pwv(), np.nan * um, equal_nan=True)


def test_get_mid_elevation(sofia_integration):
    assert np.isclose(sofia_integration.get_mid_elevation(), 45 * degree)


def test_get_mean_chopper_position(sofia_integration):
    assert sofia_integration.get_mean_chopper_position() == Coordinate2D(
        [1, 1], unit='arcsec')


def test_get_model_pwv(configured_integration):
    assert np.isclose(configured_integration.get_model_pwv(), 22 * um)


def test_validate(configured_integration):
    integration = configured_integration.copy()
    integration.validate()
    assert np.isclose(float(integration.configuration['tau.pwv']), 2)
    assert not np.any(integration.frames.valid)


def test_validate_pwv(configured_integration):
    integration = configured_integration.copy()
    integration.validate_pwv()
    assert np.isclose(float(integration.configuration['tau.pwv']), 2)
    integration.frames.valid[0] = False
    integration.validate_pwv()
    assert np.isclose(float(integration.configuration['tau.pwv']), 22)
