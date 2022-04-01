# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.mission import SofiaMissionInfo


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['PLANID'] = '9999_99'
    h['DEPLOY'] = 'DAOF'
    h['MISSN-ID'] = '2016-12-14_HA_F358'
    h['FLIGHTLG'] = 2
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaMissionInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaMissionInfo()
    assert info.obs_plan_id == ''
    assert info.base == ''
    assert info.mission_id == ''
    assert info.flight_leg == -1


def test_log_id():
    assert SofiaMissionInfo().log_id == 'missn'


def test_apply_configuration(sofia_configuration):
    info = SofiaMissionInfo()
    info.apply_configuration()
    assert info.obs_plan_id == ''
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.obs_plan_id == '9999_99'
    assert info.base == 'DAOF'
    assert info.mission_id == '2016-12-14_HA_F358'
    assert info.flight_leg == 2


def test_edit_header(sofia_info, sofia_header):
    h = fits.Header()
    sofia_info.edit_header(h)
    for key, value in sofia_header.items():
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('leg') == 2
    assert info.get_table_entry('id') == '2016-12-14_HA_F358'
    assert info.get_table_entry('plan') == '9999_99'
    assert info.get_table_entry('foo') is None
