# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.nodding import SofiaNoddingInfo


s = units.Unit('second')
arcsec = units.Unit('arcsec')
degree = units.Unit('degree')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['NODDING'] = True
    h['NODTIME'] = 5.0
    h['NODN'] = 3
    h['NODSETL'] = 0.1
    h['NODAMP'] = 30.0
    h['NODANGLE'] = -90.0
    h['NODBEAM'] = 'a'
    h['NODPATT'] = 'ABBA'
    h['NODSTYLE'] = 'NMC'
    h['NODCRSYS'] = 'erf'
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaNoddingInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaNoddingInfo()
    assert info.nodding is None
    assert np.isnan(info.dwell_time) and info.dwell_time.unit == 's'
    assert info.cycles == -1
    assert np.isnan(info.settling_time) and info.settling_time.unit == 's'
    assert np.isnan(info.amplitude) and info.amplitude.unit == 'arcsec'
    assert np.isnan(info.angle) and info.angle.unit == 'degree'
    assert info.beam_position is None
    assert info.pattern is None
    assert info.style is None
    assert info.coordinate_system is None


def test_log_id():
    assert SofiaNoddingInfo().log_id == 'nod'


def test_apply_configuration(sofia_configuration):
    info = SofiaNoddingInfo()
    info.apply_configuration()
    assert info.nodding is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.nodding
    assert info.dwell_time == 5 * s
    assert info.cycles == 3
    assert info.settling_time == 0.1 * s
    assert info.amplitude == 30 * arcsec
    assert info.angle == -90 * degree
    assert info.beam_position == 'a'
    assert info.pattern == 'ABBA'
    assert info.style == 'NMC'
    assert info.coordinate_system == 'erf'


def test_edit_header(sofia_info, sofia_header):
    info = sofia_info
    h = fits.Header()
    info.edit_header(h)
    for key, value in sofia_header.items():
        if key == 'NODDING':
            continue
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('amp') == 30 * arcsec
    assert info.get_table_entry('angle') == -90 * degree
    assert info.get_table_entry('dwell') == 5 * s
    assert info.get_table_entry('settle') == 0.1 * s
    assert info.get_table_entry('n') == '3'
    assert info.get_table_entry('pos') == 'a'
    assert info.get_table_entry('sys') == 'erf'
    assert info.get_table_entry('pattern') == 'ABBA'
    assert info.get_table_entry('style') == 'NMC'
    assert info.get_table_entry('foo') is None
