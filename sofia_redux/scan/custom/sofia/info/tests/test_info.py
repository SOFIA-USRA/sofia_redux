# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.custom.sofia.info.info import SofiaInfo
from sofia_redux.scan.custom.hawc_plus.simulation.simulation \
    import HawcPlusSimulation
from sofia_redux.scan.reduction.reduction import Reduction


class Info(SofiaInfo):  # pragma: no cover

    def get_si_pixel_size(self):
        return -1

    def get_file_id(self):
        return 'test_file'

    def max_pixels(self):
        return -1


class DummyScan(object):  # pragma: no cover
    def __init__(self):
        self.info = Info()
        self.id = 'scan'
        self.obstime = 1 * units.Unit('second')

    @property
    def configuration(self):
        return self.info.configuration

    def get_id(self):
        return self.id

    def get_observing_time(self):
        return self.obstime

    @staticmethod
    def is_valid():
        return False


@pytest.fixture
def hawc_hdul():
    h = fits.Header()
    h['CHPNOISE'] = 3.0  # Chopper noise (arcsec)
    h['SRCAMP'] = 20.0  # NEFD estimate
    h['SRCS2N'] = 30.0  # source signal to noise
    h['OBSDEC'] = 7.406657  # declination (degree)
    h['OBSRA'] = 1.272684  # ra (hours)
    h['SPECTEL1'] = 'HAW_C'  # sets band
    h['SRCSIZE'] = 20  # source FWHM (arcsec)
    h['ALTI_STA'] = 41993.0
    h['ALTI_END'] = 41998.0
    h['LON_STA'] = -108.182373
    h['LAT_STA'] = 47.043457
    h['EXPTIME'] = 1.0  # scan length (seconds)
    h['DATE-OBS'] = '2016-12-14T06:41:30.450'
    reduction = Reduction('hawc_plus')
    sim = HawcPlusSimulation(reduction.info)
    hdul = sim.create_simulated_hdul(header_options=h)
    return hdul


@pytest.fixture
def hawc_header(hawc_hdul):
    return hawc_hdul[0].header.copy()


@pytest.fixture
def hawc_configuration(hawc_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(hawc_header)
    return c


@pytest.fixture
def initialized_sofia_info(hawc_configuration):
    info = Info()
    info.configuration = hawc_configuration.copy()
    return info


@pytest.fixture
def configured_sofia_info(initialized_sofia_info):
    info = initialized_sofia_info.copy()
    info.apply_configuration()
    return info


def test_init():
    info = Info()
    assert info.name == 'sofia'
    assert info.history == []
    assert info.configuration_files == set([])
    for x in ['instrument', 'astrometry', 'aircraft', 'chopping',
              'detector_array', 'dithering', 'environment', 'mapping',
              'mission', 'mode', 'nodding', 'observation', 'origin',
              'processing', 'scanning', 'spectroscopy', 'telescope']:
        assert isinstance(getattr(info, x), InfoBase)


def test_register_config_file():
    info = Info()
    info.register_config_file(None)
    assert info.history == []
    info.register_config_file('foo')
    assert info.history == ['AUX: foo']
    assert info.configuration_files == {'foo'}


def test_read_configuration():
    info = Info()
    info.read_configuration('foo.cfg')
    assert info.history == ['AUX: foo.cfg']


def test_get_name():
    info = Info()
    assert info.get_name() == 'sofia'
    info.instrument.name = 'foo'
    assert info.get_name() == 'foo'


def test_apply_configuration(initialized_sofia_info, capsys):
    info = initialized_sofia_info.copy()
    info.configuration.parse_key_value('hwp', '1.5')
    info.configuration.fits.header['HISTORY'] = 'history 1'
    info.apply_configuration()
    output = capsys.readouterr().out
    assert 'Equatorial:' in output
    assert 'Boresight:' in output
    assert 'Requested:' in output
    assert 'Altitude: 42.00 kft, Tamb: 226.150 K' in output
    assert 'Focus: (800.0 --> 800.0 um)' in output
    assert info.configuration.fits.header['HWPINIT'] == '1.5'
    assert info.history == ['history 1']


def test_append_history_message():
    info = Info()
    info.history = None
    info.append_history_message(None)
    assert info.history is None
    info.append_history_message('message 1')
    assert info.history == ['message 1']
    info.append_history_message(['message 2', 'message 3'])
    assert info.history == ['message 1', 'message 2', 'message 3']
    info.append_history_message('message 2')  # Check duplicate
    assert info.history == ['message 1', 'message 2', 'message 3']


def test_edit_image_header(configured_sofia_info, populated_hawc_scan):
    info = configured_sofia_info.copy()
    h = fits.Header()
    info.edit_image_header(h)
    assert h['TELESCOP'] == 'SOFIA'
    h = fits.Header()
    scans = [populated_hawc_scan.copy(), populated_hawc_scan.copy()]
    first_scan = scans[0]
    first_scan.info.mode.is_chopping = True
    first_scan.info.mode.is_nodding = True
    first_scan.info.mode.is_dithering = True
    first_scan.info.mode.is_mapping = True
    first_scan.info.mode.is_scanning = True

    info.configuration.parse_key_value('organization', 'the firm')
    info.edit_image_header(h, scans=[scans[0]])
    assert h['UTCSTART'] == '06:41:30.450'
    assert h['UTCEND'] == '06:42:00.450'
    assert h['CREATOR'] == 'sofscan'
    assert h['ORIGIN'] == 'the firm'
    assert h['DTHINDEX'] == 0

    h = fits.Header()
    info.edit_image_header(h, scans=scans)
    assert h['DTHINDEX'] == -9999


def test_has_tracking_error():
    assert not SofiaInfo.has_tracking_error(None)
    scan = DummyScan()
    assert not SofiaInfo.has_tracking_error([scan])
    scan.info.telescope.has_tracking_error = True
    assert SofiaInfo.has_tracking_error([scan])


def test_edit_header(configured_sofia_info):
    info = configured_sofia_info.copy()
    h = fits.Header()
    info.edit_header(h)
    for key in ['CHPFREQ', 'NODN', 'DTHCRSYS', 'MAPCRSYS', 'SCNRA0']:
        assert key in h


def test_get_total_exposure_time():
    s = units.Unit('second')
    scan1 = DummyScan()
    scan2 = DummyScan()
    scan1.info.instrument.exposure_time = 1 * s
    scan2.info.instrument.exposure_time = 4 * s
    assert SofiaInfo.get_total_exposure_time(None) == 0 * s
    assert SofiaInfo.get_total_exposure_time([scan1, scan2]) == 5 * s


def test_get_lowest_quality():
    info = Info()
    q1 = info.processing.flagspace.convert_flag(1)
    q2 = info.processing.flagspace.convert_flag(4)
    scan1 = DummyScan()
    scan2 = DummyScan()
    scan1.info.processing.quality_level = q2
    scan2.info.processing.quality_level = q1
    q = SofiaInfo.get_lowest_quality([scan1, scan2])
    assert q == q1


def test_add_history():
    info = Info()
    info.read_configuration('default.cfg')
    scan = DummyScan()
    info.history.append('a message')
    h = fits.Header()
    info.add_history(h, scans=scan)
    history = h['HISTORY']
    aux, msg, pwd, scn = 0, 0, 0, 0
    for s in history:
        if 'AUX:' in s:
            aux += 1
        elif 'a message' in s:
            msg += 1
        elif 'PWD:' in s:
            pwd += 1
        elif 'OBS-ID' in s:
            scn += 1
    assert aux > 0 and msg == 1 and pwd == 1 and scn == 1


def test_parse_history():
    info = Info()
    h = {'HISTORY': 'message'}
    info.parse_history(h)
    assert info.history == ['message']
    h = fits.Header()
    h['HISTORY'] = 'a'
    h['HISTORY'] = 'b'
    info.parse_history(h)
    assert info.history == ['a', 'b']


def test_get_ambient_kelvins():
    k = units.Unit('Kelvin')
    info = Info()
    assert np.isclose(info.get_ambient_kelvins(), np.nan * k, equal_nan=True)
    info.environment.ambient_t = 10 * units.Unit('deg_C')
    assert np.isclose(info.get_ambient_kelvins(), 283.15 * k)


def test_get_ambient_pressure():
    assert np.isclose(Info().get_ambient_pressure(),
                      np.nan * units.Unit('Pascal'), equal_nan=True)


def test_get_ambient_humidity():
    assert np.isclose(Info().get_ambient_humidity(),
                      np.nan * units.Unit('gram/m3'), equal_nan=True)


def test_get_wind_direction():
    kmh = units.Unit('km/h')
    info = Info()
    info.aircraft.ground_speed = 490 * kmh
    info.aircraft.air_speed = 500 * kmh
    assert info.get_wind_direction() == 0 * units.Unit('degree')
    info.aircraft.ground_speed = 510 * kmh
    assert info.get_wind_direction() == -180 * units.Unit('degree')


def test_get_wind_speed():
    kmh = units.Unit('km/h')
    info = Info()
    info.aircraft.ground_speed = 500 * kmh
    info.aircraft.air_speed = 520 * kmh
    assert info.get_wind_speed() == 20 * kmh


def test_get_wind_peak():
    assert np.isclose(Info().get_wind_peak(), np.nan * units.Unit('m/s'),
                      equal_nan=True)


def test_validate_scans():
    info = Info()
    info.configuration.parse_key_value('point', 'False')
    info.validate_scans(None)  # Does nothing
    scans = [DummyScan()]
    info.validate_scans(scans)
    assert info.configuration['point']
    assert scans[0].configuration['point']


def test_get_plate_scale():
    angular_size = [2, 3] * units.Unit('arcsec')
    physical_size = [4, 5] * units.Unit('mm')
    plate_scale = SofiaInfo.get_plate_scale(angular_size, physical_size)
    assert np.isclose(plate_scale, 0.00265543 * units.Unit('radian/m'),
                      atol=1e-6)
    ps = SofiaInfo.get_plate_scale(angular_size.to('radian').value,
                                   physical_size.to('meter').value)
    assert np.isclose(ps, plate_scale)
