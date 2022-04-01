# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.dithering import SofiaDitheringInfo


arcsec = units.Unit('arcsec')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['DITHER'] = True
    h['DTHCRSYS'] = 'UNKNOWN'
    h['DTHPATT'] = 'NONE'
    h['DTHXOFF'] = 2.0
    h['DTHYOFF'] = 3.0
    h['DTHNPOS'] = 4
    h['DTHINDEX'] = 5
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


def test_init():
    info = SofiaDitheringInfo()
    assert info.dithering is None
    assert info.coordinate_system is None
    assert info.offset.is_nan()
    assert info.positions == -1
    assert info.index == -1


def test_log_id():
    info = SofiaDitheringInfo()
    assert info.log_id == 'dither'


def test_apply_configuration(sofia_configuration):
    info = SofiaDitheringInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.dithering
    assert info.coordinate_system == 'UNKNOWN'
    assert info.pattern_shape == 'NONE'
    assert info.offset.x == 2.0 * arcsec
    assert info.offset.y == 3.0 * arcsec
    assert info.positions == 4
    assert info.index == 5
    info = SofiaDitheringInfo()
    info.apply_configuration()
    assert info.dithering is None


def test_edit_header(sofia_configuration):
    info = SofiaDitheringInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    h = fits.Header()
    info.edit_header(h)
    assert h['DTHCRSYS'] == 'UNKNOWN'
    assert h['DTHXOFF'] == 2
    assert h['DTHYOFF'] == 3
    assert h['DTHPATT'] == 'NONE'
    assert h['DTHNPOS'] == 4
    assert h['DTHINDEX'] == 5
    info.offset = None
    info.edit_header(h)
    assert h['DTHXOFF'] == -9999
    assert h['DTHYOFF'] == -9999


def test_get_table_entry(sofia_configuration):
    info = SofiaDitheringInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.get_table_entry('dx') == 2 * arcsec
    assert info.get_table_entry('dy') == 3 * arcsec
    assert info.get_table_entry('index') == 5
    assert info.get_table_entry('pattern') == 'NONE'
    assert info.get_table_entry('npos') == 4
    assert info.get_table_entry('sys') == 'UNKNOWN'
    assert info.get_table_entry('foo') is None
