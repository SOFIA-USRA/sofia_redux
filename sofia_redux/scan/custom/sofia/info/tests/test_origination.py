# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.origination import SofiaOriginationInfo


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['ORIGIN'] = 'HAWC+ CDH'
    h['OBSERVER'] = 'Dan Perera'
    h['CREATOR'] = 'SOFSCAN'
    h['OPERATOR'] = 'Auto'
    h['FILENAME'] = 'file1.fits'
    h['DATASUM'] = 'abcdef'
    h['CHECKVER'] = 'a1'
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaOriginationInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaOriginationInfo()
    assert info.checksum is None
    assert info.checksum_version is None


def test_apply_configuration(sofia_configuration):
    info = SofiaOriginationInfo()
    info.apply_configuration()
    assert info.checksum is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.organization == 'HAWC+ CDH'
    assert info.observer == 'Dan Perera'
    assert info.creator == 'SOFSCAN'
    assert info.operator == 'Auto'
    assert info.filename == 'file1.fits'
    assert info.checksum == 'abcdef'
    assert info.checksum_version == 'a1'


def test_edit_header(sofia_info, sofia_header):
    h = fits.Header()
    sofia_info.edit_header(h)
    for key, value in sofia_header.items():
        if key in ['DATASUM', 'CHECKVER']:
            continue
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('creator') == 'SOFSCAN'
    assert info.get_table_entry('file') == 'file1.fits'
    assert info.get_table_entry('org') == 'HAWC+ CDH'
    assert info.get_table_entry('observer') == 'Dan Perera'
    assert info.get_table_entry('operator') == 'Auto'
    assert info.get_table_entry('foo') is None
