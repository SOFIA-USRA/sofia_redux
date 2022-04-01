# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.extended_scanning import \
    SofiaExtendedScanningInfo


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
second = units.Unit('second')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['SCNPATT'] = 'Lissajous'
    h['SCNCRSYS'] = 'equatorial'
    h['SCNAMPXL'] = 4.0
    h['SCNAMPEL'] = 5.0
    h['SCNANGLC'] = 1.0
    h['SCNANGLS'] = 2.0
    h['SCNANGLF'] = 3.0
    h['SCNDUR'] = 10.0
    h['SCNITERS'] = 11
    h['SCNNSUBS'] = 12
    h['SCNLEN'] = 13.0
    h['SCNSTEP'] = 14.0
    h['SCNSTEPS'] = 15
    h['SCNCROSS'] = True
    h['SCNFQRAT'] = 16.0
    h['SCNPHASE'] = 17.0
    h['SCNTOFF'] = 18.0
    h['SCNTWAIT'] = 19.0
    h['SCNTRKON'] = 1
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


def test_init():
    info = SofiaExtendedScanningInfo()
    assert info.pattern == ''
    assert info.coordinate_system == ''
    assert info.amplitude is None
    assert np.isnan(info.current_position_angle)
    assert np.isnan(info.duration)
    assert np.isnan(info.rel_frequency)
    assert np.isnan(info.rel_phase)
    assert np.isnan(info.t0)
    assert np.isnan(info.gyro_time_window)
    assert info.subscans == -9999
    assert info.iterations == -9999
    assert np.isnan(info.raster_length)
    assert np.isnan(info.raster_step)
    assert not info.is_cross_scanning
    assert info.n_steps == -9999
    assert info.tracking_enabled == -9999
    assert np.isnan(info.position_angle.midpoint)


def test_log_id():
    info = SofiaExtendedScanningInfo()
    assert info.log_id == 'scan'


def test_apply_configuration(sofia_configuration):
    info = SofiaExtendedScanningInfo()
    info.apply_configuration()
    assert info.pattern == ''
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.pattern == 'Lissajous'
    assert info.coordinate_system == 'equatorial'
    assert np.allclose(info.amplitude.coordinates, [4, 5] * arcsec)
    assert info.current_position_angle == 1 * degree
    assert np.isclose(info.position_angle.midpoint, 2.5 * degree)
    assert info.duration == 10 * second
    assert info.iterations == 11
    assert info.subscans == 12
    assert info.raster_length == 13 * arcsec
    assert info.raster_step == 14 * arcsec
    assert info.n_steps == 15
    assert info.is_cross_scanning
    assert info.rel_frequency == 16.0
    assert info.rel_phase == 17.0 * degree
    assert info.t0 == 18 * second
    assert info.gyro_time_window == 19 * second
    assert info.tracking_enabled == 1


def test_edit_header(sofia_configuration, sofia_header):
    info = SofiaExtendedScanningInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    h = fits.Header()
    info.edit_header(h)
    for key, value in sofia_header.items():
        assert h[key] == value
    info.amplitude = None
    info.edit_header(h)
    assert h['SCNAMPXL'] == -9999
    assert h['SCNAMPEL'] == -9999


def test_get_table_entry(sofia_configuration):
    info = SofiaExtendedScanningInfo()
    assert info.get_table_entry('trk') == '?'
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.get_table_entry('pattern') == 'Lissajous'
    assert info.get_table_entry('sys') == 'equatorial'
    assert info.get_table_entry('PA') == 1 * degree
    assert info.get_table_entry('T') == 10 * second
    assert info.get_table_entry('iters') == 11
    assert info.get_table_entry('nsub') == 12
    assert info.get_table_entry('trk')
    assert info.get_table_entry('X') == 13 * arcsec
    assert info.get_table_entry('dY') == 14 * arcsec
    assert info.get_table_entry('strips') == 15
    assert info.get_table_entry('cross?')
    assert info.get_table_entry('Ax') == 4 * arcsec
    assert info.get_table_entry('Ay') == 5 * arcsec
    assert info.get_table_entry('frel') == 16.0
    assert info.get_table_entry('phi0') == 17 * degree
    assert info.get_table_entry('t0') == 18 * second
    assert info.get_table_entry('foo') is None
