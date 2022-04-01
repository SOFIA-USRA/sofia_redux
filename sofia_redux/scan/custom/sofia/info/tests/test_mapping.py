# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.mapping import SofiaMappingInfo


arcmin = units.Unit('arcmin')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['MAPPING'] = True
    h['MAPCRSYS'] = 'equatorial'
    h['MAPNXPOS'] = 5
    h['MAPNYPOS'] = 7
    h['MAPINTX'] = 8.0
    h['MAPINTY'] = 9.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaMappingInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaMappingInfo()
    assert info.mapping is None
    assert info.coordinate_system is None
    assert info.size_x == -1
    assert info.size_y == -1
    assert info.step.size == 0


def test_log_id():
    assert SofiaMappingInfo().log_id == 'map'


def test_apply_configuration(sofia_configuration):
    info = SofiaMappingInfo()
    info.apply_configuration()
    assert info.mapping is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.mapping
    assert info.coordinate_system == 'equatorial'
    assert info.size_x == 5
    assert info.size_y == 7
    assert info.step.x == 8 * arcmin
    assert info.step.y == 9 * arcmin


def test_edit_header(sofia_info, sofia_header):
    info = sofia_info.copy()
    h = fits.Header()
    info.edit_header(h)
    for key in ['MAPCRSYS', 'MAPNXPOS', 'MAPNYPOS', 'MAPINTX', 'MAPINTY']:
        assert h[key] == sofia_header[key]

    info.step = None
    info.edit_header(h)
    assert h['MAPINTX'] == -9999
    assert h['MAPINTY'] == -9999


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('stepx') == 8 * arcmin
    assert info.get_table_entry('stepy') == 9 * arcmin
    assert info.get_table_entry('nx') == 5
    assert info.get_table_entry('ny') == 7
    assert info.get_table_entry('sys') == 'equatorial'
    assert info.get_table_entry('foo') is None
