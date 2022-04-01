# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.mode import SofiaModeInfo


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['CHOPPING'] = True
    h['NODDING'] = True
    h['DITHER'] = True
    h['MAPPING'] = True
    h['SCANNING'] = True
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaModeInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaModeInfo()
    assert not info.is_chopping
    assert not info.is_nodding
    assert not info.is_dithering
    assert not info.is_mapping
    assert not info.is_scanning


def test_log_id():
    assert SofiaModeInfo().log_id == 'mode'


def test_apply_configuration(sofia_configuration):
    info = SofiaModeInfo()
    info.apply_configuration()
    assert not info.is_chopping
    info.configuration = sofia_configuration
    info.apply_configuration()
    assert info.is_chopping
    assert info.is_nodding
    assert info.is_dithering
    assert info.is_mapping
    assert info.is_scanning


def test_edit_header(sofia_info, sofia_header):
    h = fits.Header()
    sofia_info.edit_header(h)
    for key, value in sofia_header.items():
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info
    for key in ['chop', 'nod', 'dither', 'map', 'scan']:
        assert info.get_table_entry(key)
    assert info.get_table_entry('foo') is None
