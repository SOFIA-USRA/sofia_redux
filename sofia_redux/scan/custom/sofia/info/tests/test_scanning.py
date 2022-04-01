# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.scanning import SofiaScanningInfo


s = units.Unit('second')
arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
hourangle = units.Unit('hourangle')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['SCANNING'] = True
    h['SCNRA0'] = 5.0  # hourangle
    h['SCNRAF'] = 6.0  # hourangle
    h['SCNDEC0'] = 20.0  # degrees
    h['SCNDECF'] = 21.0  # degrees
    h['SCNRATE'] = 100.0  # arcsec/sec
    h['SCNDIR'] = 45.0  # degrees
    h['SCANTYPE'] = 'LISSAJOUS'
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaScanningInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaScanningInfo()
    assert info.scanning is None
    assert np.isnan(info.ra.midpoint) and info.ra.midpoint.unit == hourangle
    assert np.isnan(info.dec.midpoint) and info.dec.midpoint.unit == degree
    assert np.isnan(info.speed) and info.speed.unit == 'arcsec/s'
    assert np.isnan(info.angle) and info.angle.unit == degree
    assert info.scan_type is None


def test_log_id():
    assert SofiaScanningInfo().log_id == 'scan'


def test_apply_configuration(sofia_configuration):
    info = SofiaScanningInfo()
    info.apply_configuration()
    assert info.scanning is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.scanning
    assert np.isclose(info.ra.midpoint, 5.5 * hourangle)
    assert np.isclose(info.dec.midpoint, 20.5 * degree)
    assert np.isclose(info.speed, 100 * arcsec / s)
    assert info.angle == 45 * degree
    assert info.scan_type == 'LISSAJOUS'


def test_edit_header(sofia_info, sofia_header):
    h = fits.Header()
    sofia_info.edit_header(h)
    for key, value in sofia_header.items():
        if key == 'SCANNING':
            continue
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('angle') == 45 * degree
    assert np.isclose(info.get_table_entry('ra'), 5.5 * hourangle)
    assert np.isclose(info.get_table_entry('dec'), 20.5 * degree)
    assert np.isclose(info.get_table_entry('speed'), 100 * arcsec / s)
    assert info.get_table_entry('type') == 'LISSAJOUS'
    assert info.get_table_entry('foo') is None
