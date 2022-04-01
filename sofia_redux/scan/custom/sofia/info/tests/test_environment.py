# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.environment import SofiaEnvironmentInfo


k = units.Unit('Kelvin')
um = units.Unit('um')
deg_c = units.Unit('deg_C')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['WVZ_STA'] = 10.0
    h['WVZ_END'] = 12.0
    h['TEMP_OUT'] = 0.0
    h['TEMPPRI1'] = 1.0
    h['TEMPPRI2'] = 2.0
    h['TEMPPRI3'] = 3.0
    h['TEMPSEC1'] = 4.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


def test_init():
    info = SofiaEnvironmentInfo()
    assert np.isnan(info.pwv.midpoint)
    for attr in ['ambient_t', 'primary_t1', 'primary_t2', 'primary_t3',
                 'secondary_t']:
        value = getattr(info, attr)
        assert np.isnan(value) and value.unit == k


def test_log_id():
    info = SofiaEnvironmentInfo()
    assert info.log_id == 'env'


def test_apply_configuration(sofia_configuration):
    info = SofiaEnvironmentInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.pwv.midpoint == 11 * um
    assert info.ambient_t == 0 * deg_c
    assert info.primary_t1 == 1 * deg_c
    assert info.primary_t2 == 2 * deg_c
    assert info.primary_t3 == 3 * deg_c
    assert info.secondary_t == 4 * deg_c
    info = SofiaEnvironmentInfo()
    info.apply_configuration()
    assert np.isnan(info.ambient_t)


def test_edit_header(sofia_configuration):
    info = SofiaEnvironmentInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    h = fits.Header()
    info.edit_header(h)
    assert h['WVZ_STA'] == 10
    assert h['WVZ_END'] == 12
    assert h['TEMP_OUT'] == 0
    assert h['TEMPPRI1'] == 1
    assert h['TEMPPRI2'] == 2
    assert h['TEMPPRI3'] == 3
    assert h['TEMPSEC1'] == 4


def test_get_table_entry(sofia_configuration):
    info = SofiaEnvironmentInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert np.isclose(info.get_table_entry('tamb'), 273.15 * k)
    assert np.isclose(info.get_table_entry('pwv'), 11 * um)
    assert np.isclose(info.get_table_entry('t1'), 274.15 * k)
    assert np.isclose(info.get_table_entry('t2'), 275.15 * k)
    assert np.isclose(info.get_table_entry('t3'), 276.15 * k)
    assert np.isclose(info.get_table_entry('sect'), 277.15 * k)
    assert info.get_table_entry('foo') is None
