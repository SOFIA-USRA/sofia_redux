# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.chopping import SofiaChoppingInfo


arcsec = units.Unit('arcsec')
volts = units.Unit('V')
degree = units.Unit('degree')
ms = units.Unit('ms')
hz = units.Unit('Hz')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['CHOPPING'] = True
    h['CHPFREQ'] = 0.1
    h['CHPPROF'] = '2-POINT'
    h['CHPSYM'] = 'no_chop'
    h['CHPAMP1'] = 5.0
    h['CHPAMP2'] = 0.0
    h['CHPCRSYS'] = 'tarf'
    h['CHPANGLE'] = 45.0
    h['CHPTIP'] = 2.0
    h['CHPTILT'] = 3.0
    h['CHPPHASE'] = 4.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


def test_class():
    v2a = SofiaChoppingInfo.volts_to_angle
    assert v2a == 33.394 * arcsec / volts


def test_init():
    info = SofiaChoppingInfo()
    assert info.chopping is None
    assert np.isnan(info.frequency) and info.frequency.unit == 'Hz'
    assert info.profile_type is None
    assert info.symmetry_type is None
    assert np.isnan(info.amplitude) and info.amplitude.unit == 'arcsec'
    assert np.isnan(info.amplitude2) and info.amplitude2.unit == 'arcsec'
    assert info.coordinate_system is None
    assert np.isnan(info.angle) and info.angle.unit == 'degree'
    assert np.isnan(info.tip) and info.tip.unit == 'arcsec'
    assert np.isnan(info.tilt) and info.tilt.unit == 'arcsec'
    assert np.isnan(info.phase) and info.phase.unit == 'ms'


def test_log_id():
    info = SofiaChoppingInfo()
    assert info.log_id == 'chop'


def test_apply_configuration(sofia_configuration):
    info = SofiaChoppingInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.chopping
    assert info.frequency == 0.1 * hz
    assert info.profile_type == '2-POINT'
    assert info.symmetry_type == 'no_chop'
    assert info.amplitude == 5 * arcsec
    assert info.amplitude2 == 0 * arcsec
    assert info.coordinate_system == 'tarf'
    assert info.angle == 45 * degree
    assert info.tip == 2 * arcsec
    assert info.tilt == 3 * arcsec
    assert info.phase == 4 * ms
    info = SofiaChoppingInfo()
    info.apply_configuration()
    assert np.isnan(info.frequency)


def test_edit_header(sofia_configuration, sofia_header):
    info = SofiaChoppingInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    h = fits.Header()
    info.edit_header(h)
    for key, value in sofia_header.items():
        if key == 'CHOPPING':
            continue
        assert h[key] == value


def test_get_table_entry(sofia_configuration):
    info = SofiaChoppingInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.get_table_entry('flag') == 'C'
    assert info.get_table_entry('amp') == 5 * arcsec
    info.amplitude = 0 * arcsec
    assert info.get_table_entry('flag') == '-'
    assert info.get_table_entry('angle') == 45 * degree
    assert info.get_table_entry('frequency') == 0.1 * hz
    assert info.get_table_entry('tip') == 2 * arcsec
    assert info.get_table_entry('tilt') == 3 * arcsec
    assert info.get_table_entry('profile') == '2-POINT'
    assert info.get_table_entry('sys') == 'tarf'
    assert info.get_table_entry('foo') is None
