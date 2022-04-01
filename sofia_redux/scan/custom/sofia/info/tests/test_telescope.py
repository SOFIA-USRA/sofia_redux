# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.telescope import SofiaTelescopeInfo


s = units.Unit('second')
degree = units.Unit('degree')
hourangle = units.Unit('hourangle')
um = units.Unit('um')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['TELESCOP'] = 'SOFIA'
    h['TELVPA'] = 1.0
    h['LASTREW'] = '2022-03-22T21:41:46.678Z'
    h['FOCUS_ST'] = 400.0
    h['FOCUS_EN'] = 401.0
    h['TELEL'] = 2.0
    h['TELXEL'] = 3.0
    h['TELLOS'] = 4.0
    h['TSC-STAT'] = 'STAB_INERTIAL_ONGOING'
    h['FBC-STAT'] = 'FBC_ON'
    h['ZA_START'] = 5.0
    h['ZA_END'] = 6.0
    h['TRACMODE'] = 'offset'
    h['TRACERR'] = True
    h['TELCONF'] = 'NASMYTH'
    h['EQUINOX'] = 2000.0
    h['TELEQUI'] = 2001.0
    h['TELRA'] = 7.0
    h['TELDEC'] = 8.0
    h['OBSRA'] = 9.0
    h['OBSDEC'] = 10.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaTelescopeInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaTelescopeInfo()
    assert info.telescope == 'SOFIA 2.5m'
    assert info.tel_config is None
    assert np.isnan(info.vpa) and info.vpa.unit == degree
    assert info.last_rewind is None
    focus = info.focus_t.midpoint
    assert np.isnan(focus) and focus.unit == um
    assert np.isnan(info.rel_elevation) and info.rel_elevation.unit == degree
    assert np.isnan(info.cross_elevation)
    assert info.cross_elevation.unit == degree
    assert np.isnan(info.line_of_sight_angle)
    assert info.line_of_sight_angle.unit == degree
    assert info.tascu_status is None
    assert info.fbc_status is None
    z = info.zenith_angle.midpoint
    assert np.isnan(z) and z.unit == degree
    assert info.tracking_mode is None
    assert not info.has_tracking_error
    assert not info.is_tracking
    assert info.epoch.equinox == 'J2000'
    assert info.boresight_equatorial.is_nan()
    assert info.requested_equatorial.is_nan()


def test_apply_configuration(sofia_configuration):
    info = SofiaTelescopeInfo()
    info.apply_configuration()
    assert info.tel_config is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.telescope == 'SOFIA'
    assert info.vpa == 1 * degree
    assert info.last_rewind == '2022-03-22T21:41:46.678Z'
    assert np.isclose(info.focus_t.midpoint, 400.5 * um)
    assert info.rel_elevation == 2 * degree
    assert info.cross_elevation == 3 * degree
    assert info.line_of_sight_angle == 4 * degree
    assert info.tascu_status == 'STAB_INERTIAL_ONGOING'
    assert info.fbc_status == 'FBC_ON'
    assert np.isclose(info.zenith_angle.midpoint, 5.5 * degree)
    assert info.tracking_mode == 'offset'
    assert info.has_tracking_error
    assert info.is_tracking
    assert info.epoch.equinox == 'J2000'
    assert info.boresight_equatorial.ra == 7 * hourangle
    assert info.boresight_equatorial.dec == 8 * degree
    assert info.boresight_equatorial.epoch.equinox.jyear == 2001
    assert info.requested_equatorial.ra == 9 * hourangle
    assert info.requested_equatorial.dec == 10 * degree
    del info.options['TELEQUI']
    info.apply_configuration()
    assert info.boresight_equatorial.epoch.equinox.jyear == 2000


def test_get_telescope_name():
    assert SofiaTelescopeInfo.get_telescope_name() == 'SOFIA'


def test_edit_image_header(sofia_info):
    h = fits.Header()
    sofia_info.edit_image_header(h)
    assert h['TELESCOP'] == 'SOFIA'
    assert h.comments['TELESCOP'] == 'Telescope name.'


def test_edit_header(sofia_info, sofia_header):
    info = sofia_info.copy()
    h = fits.Header()
    info.edit_header(h)
    for key, value in sofia_header.items():
        if key == 'TELEQUI':
            assert h[key] == 'J2001.0'
            continue
        assert h[key] == value

    h = fits.Header()
    info.boresight_equatorial = None
    info.requested_equatorial = None
    info.edit_header(h)
    assert 'TELEQUI' not in h
    assert h['TELRA'] == -9999
    assert h['TELDEC'] == -9999
    assert h['OBSRA'] == -9999
    assert h['OBSDEC'] == -9999


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert np.isclose(info.get_table_entry('focus'), 400.5 * um)
    assert info.get_table_entry('bra') == 7 * hourangle
    assert info.get_table_entry('bdec') == 8 * degree
    assert info.get_table_entry('rra') == 9 * hourangle
    assert info.get_table_entry('rdec') == 10 * degree
    assert info.get_table_entry('epoch') == '2000.0'
    assert info.get_table_entry('vpa') == 1 * degree
    assert np.isclose(info.get_table_entry('za'), 5.5 * degree)
    assert info.get_table_entry('los') == 4 * degree
    assert info.get_table_entry('el') == 2 * degree
    assert info.get_table_entry('xel') == 3 * degree
    assert info.get_table_entry('trkerr')
    assert info.get_table_entry('trkmode') == 'offset'
    assert info.get_table_entry('cfg') == 'NASMYTH'
    assert info.get_table_entry('fbc') == 'FBC_ON'
    assert info.get_table_entry('rew') == '2022-03-22T21:41:46.678Z'
    assert info.get_table_entry('foo') is None
