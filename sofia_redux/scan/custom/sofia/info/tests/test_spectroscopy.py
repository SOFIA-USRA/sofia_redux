# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.spectroscopy import \
    SofiaSpectroscopyInfo


mhz = units.Unit('MHz')
ghz = units.Unit('GHz')
kelvin = units.Unit('Kelvin')
kms = units.Unit('km/s')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['FRONTEND'] = 'front'
    h['BACKEND'] = 'back'
    h['BANDWID'] = 25.0
    h['FREQRES'] = 1.0
    h['TSYS'] = 70.0
    h['OBSFREQ'] = 1422.0
    h['IMAGFREQ'] = 1421.0
    h['RESTFREQ'] = 1420.0
    h['VELDEF'] = 'RADI-OBS'
    h['VFRAME'] = 100.0
    h['RVSYS'] = 10.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaSpectroscopyInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_class():
    assert SofiaSpectroscopyInfo.velocity_unit == kms


def test_init():
    info = SofiaSpectroscopyInfo()
    assert info.front_end is None
    assert info.back_end is None
    assert np.isnan(info.bandwidth) and info.bandwidth.unit == mhz
    assert np.isnan(info.frequency_resolution)
    assert info.frequency_resolution.unit == mhz
    assert np.isnan(info.tsys) and info.tsys.unit == kelvin
    assert np.isnan(info.observing_frequency)
    assert info.observing_frequency.unit == mhz
    assert np.isnan(info.image_frequency) and info.image_frequency.unit == mhz
    assert np.isnan(info.rest_frequency) and info.rest_frequency.unit == mhz
    assert info.velocity_type is None
    assert np.isnan(info.frame_velocity) and info.frame_velocity.unit == kms
    assert np.isnan(info.source_velocity) and info.source_velocity.unit == kms


def test_log_id():
    assert SofiaSpectroscopyInfo().log_id == 'spec'


def test_apply_configuration(sofia_configuration):
    info = SofiaSpectroscopyInfo()
    info.apply_configuration()
    assert info.front_end is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.front_end == 'front'
    assert info.back_end == 'back'
    assert info.bandwidth == 25 * mhz
    assert info.frequency_resolution == 1 * mhz
    assert info.tsys == 70 * kelvin
    assert info.observing_frequency == 1422 * mhz
    assert info.image_frequency == 1421 * mhz
    assert info.rest_frequency == 1420 * mhz
    assert info.velocity_type == 'RADI-OBS'
    assert info.frame_velocity == 100 * kms
    assert info.source_velocity == 10 * kms


def test_get_redshift(sofia_info):
    info = sofia_info.copy()
    assert np.isclose(info.get_redshift(), 3.336e-5, rtol=1e-3)
    info.source_velocity = 200000 * kms
    assert np.isclose(info.get_redshift(), 1.238, atol=1e-3)


def test_edit_header(sofia_info, sofia_header):
    h = fits.Header()
    sofia_info.edit_header(h)
    for key, value in sofia_header.items():
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('bw') == 0.025 * ghz
    assert info.get_table_entry('df') == 1 * mhz
    assert info.get_table_entry('tsys') == 70 * kelvin
    assert info.get_table_entry('fobs') == 1.422 * ghz
    assert info.get_table_entry('frest') == 1.42 * ghz
    assert info.get_table_entry('vsys') == 'RADI-OBS'
    assert info.get_table_entry('vframe') == 100 * kms
    assert info.get_table_entry('vrad') == 10 * kms
    assert np.isclose(info.get_table_entry('z'), 3.336e-5, rtol=1e-3)
    assert info.get_table_entry('foo') is None
