# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.gyro_drifts import GyroDrift
from sofia_redux.scan.custom.sofia.info.gyro_drifts import SofiaGyroDriftsInfo


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['DBRA0'] = '00:00:00.0'
    h['DBDEC0'] = 30.0
    h['DARA0'] = '00:08:00.0'
    h['DADEC0'] = 31.0
    h['EQUINOX'] = 2000.0
    h['DBTIME0'] = 10.0
    h['DATIME0'] = 11.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def gyro_drift(sofia_header):
    return GyroDrift(sofia_header.copy(), 0)


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaGyroDriftsInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaGyroDriftsInfo()
    assert info.drifts is None


def test_log_id():
    info = SofiaGyroDriftsInfo()
    assert info.log_id == 'gyro'


def test_n_drifts():
    info = SofiaGyroDriftsInfo()
    assert info.n_drifts == 0
    info.drifts = [1, 2, 3]
    assert info.n_drifts == 3


def test_lengths(sofia_info):
    info = SofiaGyroDriftsInfo()
    lengths = info.lengths
    assert lengths.size == 0 and lengths.unit == 'degree'
    info = sofia_info
    lengths = info.lengths
    assert np.allclose(lengths, [2] * degree)


def test_apply_configuration(sofia_configuration):
    info = SofiaGyroDriftsInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.n_drifts == 1


def test_add_drifts(sofia_configuration):
    info = SofiaGyroDriftsInfo()
    info.add_drifts()
    assert info.n_drifts == 0
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    info.drifts = None
    info.add_drifts()
    assert info.n_drifts == 1
    info.drifts = None
    info.options.header['DBDEC0'] = 'NaN'
    info.add_drifts()
    assert info.n_drifts == 0


def test_get_max(sofia_info):
    assert np.isnan(SofiaGyroDriftsInfo().get_max())
    assert np.isclose(sofia_info.get_max(), 2 * degree)


def test_get_rms(sofia_info):
    assert np.isnan(SofiaGyroDriftsInfo().get_rms())
    info = sofia_info
    assert np.isclose(info.get_rms(), 2 * degree)


def test_get_drift_utc_ranges(sofia_info):
    info = sofia_info
    assert np.allclose(info.get_drift_utc_ranges(),
                       [[-np.inf, 10]])
    assert np.all(np.isnan(SofiaGyroDriftsInfo().get_drift_utc_ranges()))


def test_get_drift_deltas(sofia_info):
    assert SofiaGyroDriftsInfo().get_drift_deltas().size == 0
    assert np.allclose(sofia_info.get_drift_deltas(),
                       [[6235.38, 3600.]] * arcsec, atol=1e-2)


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert np.isclose(info.get_table_entry('max'), 7200 * arcsec)
    assert np.isclose(info.get_table_entry('rms'), 7200 * arcsec)
    assert info.get_table_entry('foo') is None


def test_validate_time_range(populated_hawc_scan, sofia_info):
    scan = populated_hawc_scan.copy()
    info = sofia_info
    lower = info.drifts[0].utc_range.min
    upper = info.drifts[0].utc_range.max
    info.validate_time_range(None)
    assert np.isclose(info.drifts[0].utc_range.min, lower, equal_nan=True)
    assert np.isclose(info.drifts[0].utc_range.max, upper, equal_nan=True)
    info.validate_time_range(scan)
    assert np.isclose(info.drifts[0].utc_range.min, 1481697690.45)


def test_correct(populated_hawc_scan, sofia_info, capsys):
    scan = populated_hawc_scan.copy()
    integration = scan[0].copy()
    equatorial = integration.frames.equatorial.copy()
    SofiaGyroDriftsInfo().correct(integration)
    assert integration.frames.equatorial == equatorial
    assert 'No data' in capsys.readouterr().err

    integration.configuration.parse_key_value('gyrocorrect.max', '1.0')
    info = sofia_info
    info.correct(integration)
    assert integration.frames.equatorial == equatorial
    assert 'Drifts are too large' in capsys.readouterr().err

    integration.configuration.parse_key_value('gyrocorrect.max', '10000.0')
    info.drifts[0].utc_range.min = integration.frames.utc[0]
    info.drifts[0].utc_range.max = integration.frames.utc[-100]
    info.correct(integration)
    assert integration.frames.equatorial != equatorial
    assert 'Extrapolated drift correction' in capsys.readouterr().err


def test_init_drift(sofia_header):
    h = sofia_header.copy()
    drift = GyroDrift(h, 0)
    assert drift.index == 0
    assert drift.valid
    assert np.isclose(drift.delta.x, 6235.38 * arcsec, atol=1e-2)
    assert np.isclose(drift.delta.y, 3600 * arcsec)
    assert drift.epoch.equinox == 'J2000'
    assert drift.utc_range.min == -np.inf
    assert drift.utc_range.max == 10.0
    assert drift.next_utc == 11.0


def test_str(gyro_drift):
    s = str(gyro_drift)
    assert s.startswith('index: 0')
    assert 'after: RA=0h08m00s DEC=31d00m00s (J2000.0)' in s
    assert 'before: RA=0h00m00s DEC=30d00m00s (J2000.0)' in s


def test_length(gyro_drift):
    drift = gyro_drift
    assert np.isclose(drift.length, 2 * degree)
    drift.delta = None
    assert np.isnan(drift.length)


def test_parse_header(gyro_drift, sofia_header):
    drift = gyro_drift
    header = sofia_header.copy()
    drift.epoch = None
    drift.index = 2
    drift.parse_header(header)
    header['EQUINOX'] = 2001.0
    assert not drift.valid and drift.epoch is None
    drift.index = 0
    drift.parse_header(header)
    assert drift.valid and drift.epoch.equinox.jyear == 2001
    del header['EQUINOX']
    drift.parse_header(header)
    assert drift.valid and drift.epoch.equinox == 'J2000'
    header['DBDEC0'] = 'NaN'
    drift.parse_header(header)
    assert not drift.valid
