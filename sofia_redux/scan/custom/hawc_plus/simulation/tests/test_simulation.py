# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.units import imperial
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.time import Time
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.custom.hawc_plus.simulation.simulation import \
    HawcPlusSimulation as Sim
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.sofia.simulation.aircraft import \
    AircraftSimulation
from sofia_redux.scan.custom.hawc_plus.channels.channels import \
    HawcPlusChannels
from sofia_redux.scan.simulation.source_models.single_gaussian import \
    SingleGaussian
from sofia_redux.scan.custom.hawc_plus.scan.scan import HawcPlusScan
from sofia_redux.scan.custom.hawc_plus.integration.integration import \
    HawcPlusIntegration


imperial.enable()
degree = units.Unit('degree')
arcsec = units.Unit('arcsec')
second = units.Unit('second')
hourangle = units.Unit('hourangle')
ft = units.Unit('ft')


class DummyInfo(object):
    def __init__(self):
        self.sampling_interval = 0.1 * second


class DummyScan(object):
    def __init__(self):
        self.info = DummyInfo()


@pytest.fixture
def basic_hawc_info():
    info = HawcPlusInfo()
    info.read_configuration()
    return info


@pytest.fixture
def user_options():
    h = fits.Header()
    h['SRCAMP'] = 20.0  # NEFD estimate
    h['SRCS2N'] = 30.0  # source signal to noise
    h['OBSDEC'] = 30.0  # declination (degree)
    h['OBSRA'] = 12.0  # ra (hours)
    h['SPECTEL1'] = 'HAW_C'  # sets band
    h['SRCSIZE'] = 20  # source FWHM (arcsec)
    h['ALTI_STA'] = 41993.0
    h['ALTI_END'] = 41998.0
    h['LON_STA'] = -108.182373
    h['LAT_STA'] = 47.043457
    h['EXPTIME'] = 5.0  # scan length (seconds)
    h['DATE-OBS'] = '2016-12-14T06:41:30.450'
    h['SCNCONST'] = True
    return h


@pytest.fixture
def initialized_simulation(basic_hawc_info):
    return Sim(basic_hawc_info)


@pytest.fixture
def upto_non_astronomical(initialized_simulation, user_options):
    sim = initialized_simulation
    sim.create_basic_hdul(header_options=user_options)
    sim.scan = sim.channels.get_scan_instance()
    sim.scan.info.parse_header(sim.hdul[0].header)
    sim.scan.channels.read_data(sim.hdul)
    sim.scan.channels.validate_scan(sim)
    sim.scan.hdul = sim.hdul
    sim.integration = sim.scan.get_integration_instance()
    sim.update_non_astronomical_columns()
    return sim


@pytest.fixture
def upto_vpa(upto_non_astronomical):
    sim = upto_non_astronomical
    n_records = sim.column_values['FrameCounter'].size
    sim.integration.frames.initialize(sim.integration, n_records)
    return sim


def test_class():
    assert isinstance(Sim.sim_keys, set)
    for key, value in Sim.data_column_definitions.items():
        assert isinstance(key, str)
        assert isinstance(value, tuple) and len(value) == 2
        assert isinstance(value[1], str)

    for key in Sim.default_values.keys():
        assert key in Sim.default_comments


def test_init(basic_hawc_info):
    info = basic_hawc_info.copy()
    with pytest.raises(ValueError) as err:
        _ = Sim(1)
    assert "Simulation must be initialized with" in str(err.value)
    sim = Sim(info)
    assert sim.info is info
    assert isinstance(sim.primary_header, fits.Header)
    assert len(sim.primary_header) == 0
    assert isinstance(sim.channels, HawcPlusChannels)
    assert sim.channels.parent is sim
    assert isinstance(sim.aircraft, AircraftSimulation)
    assert isinstance(sim.user, str)
    for attribute in ['hdul', 'source_equatorial', 'start_utc', 'end_utc',
                      'start_site', 'scan', 'integration', 'column_values',
                      'equatorial', 'apparent_equatorial', 'horizontal',
                      'equatorial_corrected', 'horizontal_corrected',
                      'horizontal_offset_corrected',
                      'apparent_equatorial_corrected', 'horizontal_offset',
                      'lst', 'mjd', 'site', 'sin_pa', 'cos_pa',
                      'chopper_position', 'source_model', 'source_data',
                      'data_hdu', 'projection', 'projector', 'model_offsets']:
        assert getattr(sim, attribute, 1) is None


def test_default_value():
    assert Sim.default_value('all') == Sim.default_values
    assert Sim.default_value('TSC-STAT') == 'STAB_INERTIAL_ONGOING'


def test_default_comment():
    assert Sim.default_comment('all') == Sim.default_comments
    assert Sim.default_comment('CALMODE') == 'Diagnostic procedure mode'


def test_update_header_value(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    h['ORIGIN'] = 'foo', 'Initial comment'
    sim.update_header_value(h, 'TELESCOP', 'bar')
    assert h['TELESCOP'] == 'bar'
    assert h.comments['TELESCOP'] == 'Telescope name'
    sim.update_header_value(h, 'ORIGIN', 'baz')
    assert h['ORIGIN'] == 'baz'
    assert h.comments['ORIGIN'] == 'Initial comment'
    h['ORIGIN'] = 'foo', ''
    sim.update_header_value(h, 'ORIGIN', 'baz')
    assert h['ORIGIN'] == 'baz'
    assert h.comments['ORIGIN'] == 'Origin of FITS file'


def test_write_simulated_hdul(initialized_simulation, tmpdir):
    filename = str(tmpdir.mkdir('test_write_simulated_hdul').join('test.fits'))
    sim = initialized_simulation
    sim.write_simulated_hdul(filename)
    hdul = fits.open(filename)
    names = [hdu.header['EXTNAME'] for hdu in hdul[1:]]
    assert names == ['CONFIGURATION', 'Timestream']
    assert hdul[0].header['ORIGIN'] == 'SOFSCAN simulation'
    hdul.close()


def test_create_simulated_hdul(initialized_simulation):
    sim = initialized_simulation
    assert sim.hdul is None
    hdul = sim.create_simulated_hdul()
    assert isinstance(hdul, fits.HDUList) and hdul is sim.hdul


def test_create_basic_hdul(initialized_simulation):
    sim = initialized_simulation
    assert sim.hdul is None
    sim.create_basic_hdul()
    assert isinstance(sim.hdul, fits.HDUList)
    assert len(sim.hdul) == 2
    assert sim.primary_header is sim.hdul[0].header
    assert sim.hdul[1].header['EXTNAME'] == 'CONFIGURATION'


def test_create_primary_header(initialized_simulation, user_options):
    sim = initialized_simulation
    h = sim.create_primary_header(header_options=user_options)
    assert h['HEADSTAT'] == 'SIMULATED'
    assert h['SPECTEL1'] == 'HAW_C'
    assert np.isclose(sim.source_equatorial.ra, 12 * hourangle)
    assert np.isclose(sim.source_equatorial.dec, 30 * degree)
    assert np.isclose(sim.aircraft.start_location.lon, -108.182 * degree,
                      atol=1e-3)
    assert isinstance(sim.source_model, SingleGaussian)
    assert h is sim.primary_header


def test_create_configuration_hdu(initialized_simulation):
    sim = initialized_simulation
    hdu = sim.create_configuration_hdu()
    h = hdu.header
    for key in ['MCE0_TES_BIAS', 'MCE1_TES_BIAS', 'MCE2_TES_BIAS']:
        assert h[key] == '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1'
    assert h['EXTNAME'] == 'CONFIGURATION'


def test_update_hdul_with_data(initialized_simulation, user_options):
    sim = initialized_simulation
    sim.create_basic_hdul(header_options=user_options)
    sim.update_hdul_with_data()
    assert isinstance(sim.scan, HawcPlusScan)
    assert isinstance(sim.integration, HawcPlusIntegration)
    assert sim.integration.size == 1016
    assert len(sim.hdul) == 3
    assert sim.hdul[2].header['EXTNAME'] == 'Timestream'
    assert sim.hdul[2] is sim.data_hdu


def test_update_primary_header_with_data_hdu(initialized_simulation,
                                             user_options):
    sim = initialized_simulation
    sim.create_basic_hdul(header_options=user_options)
    sim.update_hdul_with_data()
    for key in ['TELVPA', 'TELEL', 'TELXEL', 'TELLOS', 'ZA_START', 'ZA_END']:
        if key in sim.primary_header:  # pragma: no cover
            del sim.primary_header[key]
    sim.update_primary_header_with_data_hdu()
    assert sim.primary_header['TELVPA'] == 0
    assert np.isclose(sim.primary_header['TELEL'], 12.639, atol=1e-3)
    assert np.isclose(sim.primary_header['TELXEL'], -0.075, atol=1e-3)
    assert sim.primary_header['TELLOS'] == 0
    assert np.isclose(sim.primary_header['ZA_START'], 77.557, atol=1e-3)
    assert np.isclose(sim.primary_header['ZA_END'], 77.547, atol=1e-3)


def test_default_primary_header(user_options, initialized_simulation):
    sim = initialized_simulation
    header = user_options.copy()
    h = sim.default_primary_header(user_options)
    dict_options = dict(header)
    dict_options['SPECTEL1'] = ('HAW_C', 'custom comment')
    h2 = sim.default_primary_header(dict_options)
    assert h['SPECTEL1'] == h2['SPECTEL1']
    assert h.comments['SPECTEL1'] == ''
    assert h2.comments['SPECTEL1'] == 'custom comment'


def test_update_header_band(initialized_simulation):
    sim = initialized_simulation
    centers = {'A': 53.0, 'B': 63.0, 'C': 89.0, 'D': 154.0, 'E': 214.0}
    for band, center in centers.items():
        h = fits.Header()
        h['SPECTEL1'] = f'HAW_{band}'
        sim.update_header_band(h)
        assert h['MCCSMODE'] == f'band_{band.lower()}_foctest'
        assert h['SPECTEL2'] == 'HAW_HWP_Open'
        assert h['WAVECENT'] == center
        assert h['BSITE'] == f'band_{band.lower()}_foctest'

    h = fits.Header()

    h['SPECTEL1'] = 'bar'
    h['SPECTEL2'] = 'foo'
    sim.update_header_band(h)
    assert h['SPECTEL2'] == 'foo' and h['SPECTEL1'] == 'HAW_A'


def test_update_header_origin(initialized_simulation):
    h = fits.Header()
    date_obs = Time('2022-03-30T20:00:00.000')
    sim = initialized_simulation
    sim.start_utc = date_obs
    initialized_simulation.update_header_origin(h)
    assert h['PLANID'] == '99_9999'
    assert h['OBS_ID'] == '2022-03-30_HA_F999-sim-999'
    assert h['MISSN-ID'] == '2022-03-30_HA_F999'
    assert h['FILENAME'] == '2022-03-30_HA_F999_999_SIM_999999_HAWA_RAW.fits'


def test_update_header_chopping(initialized_simulation):
    h = fits.Header()
    expected = {'CHOPPING': False, 'CHPFREQ': 10.2, 'CHPPROF': '2-POINT',
                'CHPSYM': 'no_chop', 'CHPAMP1': 0.0, 'CHPAMP2': 0.0,
                'CHPCRSYS': 'tarf', 'CHPANGLE': 0.0, 'CHPTIP': 0.0,
                'CHPTILT': 0.0, 'CHPSRC': 'external', 'CHPONFPA': False}
    initialized_simulation.update_header_chopping(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_nodding(initialized_simulation):
    h = fits.Header()
    expected = {'NODDING': False, 'NODTIME': -9999.0, 'NODN': 1,
                'NODSETL': -9999.0, 'NODAMP': 150.0, 'NODBEAM': 'a',
                'NODPATT': 'ABBA', 'NODSTYLE': 'NMC', 'NODCRSYS': 'erf',
                'NODANGLE': -90.0}
    initialized_simulation.update_header_nodding(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_dithering(initialized_simulation):
    h = fits.Header()
    expected = {'DITHER': False, 'DTHCRSYS': 'UNKNOWN', 'DTHXOFF': -9999.0,
                'DTHYOFF': -9999.0, 'DTHPATT': 'NONE', 'DTHNPOS': -9999,
                'DTHINDEX': -9999, 'DTHUNIT': 'UNKNOWN', 'DTHSCALE': -9999.0}
    initialized_simulation.update_header_dithering(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_mapping(initialized_simulation):
    h = fits.Header()
    expected = {'MAPPING': False, 'MAPCRSYS': 'UNKNOWN', 'MAPNXPOS': -9999,
                'MAPNYPOS': -9999, 'MAPINTX': -9999.0, 'MAPINTY': -9999.0}
    initialized_simulation.update_header_mapping(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_hwp(initialized_simulation):
    h = fits.Header()
    expected = {'NHWP': 1, 'HWPSTART': -9999.0, 'HWPSPEED': -9999,
                'HWPSTEP': -9999.0, 'HWPSEQ': 'UNKNOWN', 'HWPON': 10.0,
                'HWPOFF': 9.0, 'HWPHOME': 8.0}
    initialized_simulation.update_header_hwp(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_focus(initialized_simulation):
    h = fits.Header()
    expected = {'FOCUS_ST': 800.0, 'FOCUS_EN': 800.0, 'FCSCOEFA': -13.8,
                'FCSCOEFB': 0.0, 'FCSCOEFC': 0.0, 'FCSCOEFK': -4.23,
                'FCSCOEFQ': 0.0, 'FCSDELTA': 381.569916, 'FCSTCALC': 800.0,
                'FCST1NM': 'ta_mcp.mcp_hk_pms.tmm_temp_1', 'FCST2NM': '',
                'FCST3NM': '', 'FCSXNM': '', 'FCST1': -27.649994,
                'FCST2': -9999.0, 'FCST3': -9999.0, 'FCSX': -9999.0,
                'FCSTOFF': 25.0}
    initialized_simulation.update_header_focus(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_skydip(initialized_simulation):
    h = fits.Header()
    expected = {'SDELSTEN': -9999.0, 'SDELMID': -9999.0, 'SDWTSTRT': -9999.0,
                'SDWTMID': -9999.0, 'SDWTEND': -9999.0}
    initialized_simulation.update_header_skydip(h)
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_scanning(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    h['SRCSIZE'] = 10.0
    h['SRCTYPE'] = 'point_source'
    sim.update_header_scanning(h)
    expected = {'SRCSIZE': 10.0, 'SRCTYPE': 'point_source', 'EXPTIME': 30.0,
                'TOTTIME': 30.0, 'SCANNING': True, 'OBSMODE': 'Scan',
                'SCNPATT': 'Daisy', 'SCNCRSYS': 'TARF', 'SCNITERS': 1,
                'SCNANGLS': 0.0, 'SCNANGLC': 0.0, 'SCNANGLF': 0.0,
                'SCNTWAIT': 0.0, 'SCNTRKON': 0, 'SCNRATE': 100.0,
                'SCNCONST': False, 'SCNAMPEL': -9999.0, 'SCNAMPXL': -9999.0,
                'SCNFQRAT': -9999.0, 'SCNPHASE': -9999.0, 'SCNTOFF': -9999.0,
                'SCNDRAD': 50.0, 'SCNDNOSC': 22.0, 'SCNNSUBS': -9999,
                'SCNLEN': -9999.0, 'SCNSTEP': -9999.0, 'SCNSTEPS': -9999.0,
                'SCNCROSS': False}

    for key, value in expected.items():
        assert h[key] == value

    h = fits.Header()
    h['SRCSIZE'] = 10.0
    h['SRCTYPE'] = 'point_source'
    h['TOTTIME'] = 10.0
    sim.update_header_scanning(h)
    assert h['EXPTIME'] == 10
    assert h['TOTTIME'] == 10

    h = fits.Header()
    h['SRCSIZE'] = 10.0
    h['SRCTYPE'] = 'point_source'
    h['EXPTIME'] = 15.0
    sim.update_header_scanning(h)
    assert h['EXPTIME'] == 15
    assert h['TOTTIME'] == 15

    h['SCNPATT'] = 'foo'
    with pytest.raises(ValueError) as err:
        sim.update_header_scanning(h)
    assert 'not currently supported' in str(err.value)


def test_update_header_lissajous(initialized_simulation):
    sim = initialized_simulation
    h0 = fits.Header()
    h0['SRCSIZE'] = 10.0
    h0['SRCTYPE'] = 'point_source'
    h0['SCNPATT'] = 'Lissajous'
    h = h0.copy()
    sim.update_header_lissajous(h)

    expected = {'SRCSIZE': 10.0, 'SRCTYPE': 'point_source',
                'SCNPATT': 'Lissajous', 'SCNCONST': False, 'SCNAMPXL': 50.0,
                'SCNAMPEL': 50.0, 'SCNFQRAT': np.sqrt(2),
                'SCNPHASE': 90.0, 'SCNTOFF': 0.0}
    for key, value in expected.items():
        if isinstance(value, float):
            assert np.isclose(h[key], value)
        else:
            assert h[key] == value

    h = h0.copy()
    h['SRCTYPE'] = 'extended'
    sim.update_header_lissajous(h)
    expected['SCNAMPXL'] = 25.0
    expected['SCNAMPEL'] = 25.0
    expected['SRCTYPE'] = 'extended'
    for key, value in expected.items():
        if isinstance(value, float):
            assert np.isclose(h[key], value)
        else:
            assert h[key] == value

    h['SCNAMPXL'] = 30.0
    del h['SCNAMPEL']
    sim.update_header_lissajous(h)
    assert h['SCNAMPXL'] == 30 and h['SCNAMPEL'] == 30

    h['SCNAMPEL'] = 40.0
    del h['SCNAMPXL']
    sim.update_header_lissajous(h)
    assert h['SCNAMPXL'] == 40 and h['SCNAMPEL'] == 40

    h['SCNAMPEL'] = 50.0
    sim.update_header_lissajous(h)
    assert h['SCNAMPXL'] == 40 and h['SCNAMPEL'] == 50

    h['SCNPATT'] = 'daisy'
    sim.update_header_lissajous(h)
    expected = {'SRCSIZE': 10.0, 'SRCTYPE': 'extended', 'SCNPATT': 'daisy',
                'SCNCONST': False, 'SCNFQRAT': -9999.0, 'SCNPHASE': -9999.0,
                'SCNTOFF': -9999.0, 'SCNAMPEL': -9999.0, 'SCNAMPXL': -9999.0}

    for key, value in expected.items():
        assert h[key] == value


def test_update_header_daisy(initialized_simulation):
    sim = initialized_simulation
    h0 = fits.Header()
    h0['SRCSIZE'] = 10.0
    h0['SRCTYPE'] = 'point_source'
    h0['SCNPATT'] = 'Lissajous'
    h0['EXPTIME'] = 33.0
    h = h0.copy()
    sim.update_header_daisy(h)
    for key, value in {'SRCSIZE': 10.0, 'SRCTYPE': 'point_source',
                       'SCNPATT': 'Lissajous', 'SCNDRAD': -9999.0,
                       'SCNDPER': -9999.0, 'SCNDNOSC': -9999.0}.items():
        assert h[key] == value

    h0['SCNPATT'] = 'Daisy'
    h = h0.copy()
    sim.update_header_daisy(h)
    expected = {'SCNDRAD': 50.0, 'SCNDNOSC': 22.0, 'SCNCONST': False,
                'SCNDPER': 1.5}
    for key, value in expected.items():
        assert h[key] == value

    h = h0.copy()
    h['SRCTYPE'] = 'extended'
    sim.update_header_daisy(h)
    expected['SCNDRAD'] = 25.0
    for key, value in expected.items():
        assert h[key] == value

    h = h0.copy()
    h['SCNDPER'] = 3.0
    sim.update_header_daisy(h)
    assert h['SCNDNOSC'] == 11

    h = h0.copy()
    h['SCNDPER'] = 1.5
    h['SCNDNOSC'] = 22
    sim.update_header_daisy(h)
    expected['SCNDRAD'] = 50.0
    for key, value in expected.items():
        assert h[key] == value


def test_update_header_raster(initialized_simulation):
    h = fits.Header()
    initialized_simulation.update_header_raster(h)

    for key, value in {'SCNNSUBS': -9999, 'SCNLEN': -9999.0,
                       'SCNSTEP': -9999.0, 'SCNSTEPS': -9999.0,
                       'SCNCROSS': False}.items():
        assert h[key] == value


def test_set_source(initialized_simulation):
    sim = initialized_simulation
    prime_header = fits.Header()
    prime_header['SPECTEL1'] = 'HAW_A'
    prime_header['SRCTYPE'] = 'extended'
    sim.primary_header = prime_header.copy()
    header_options = fits.Header()
    header_options['OBSRA'] = 12.0
    header_options['OBSDEC'] = 30.0
    sim.set_source('', '', header_options=header_options)
    assert np.isclose(sim.source_equatorial.ra, 12 * hourangle)
    assert np.isclose(sim.source_equatorial.dec, 30 * degree)
    expected = {'SPECTEL1': 'HAW_A', 'TELRA': 12.0, 'OBJRA': 12.0,
                'TELDEC': 30.0, 'OBJDEC': 30.0, 'OBJECT': 'simulated_source',
                'SRCTYPE': 'extended', 'SRCSIZE': 14.55}
    for key, value in expected.items():
        if isinstance(value, float):
            assert np.isclose(sim.primary_header[key], value)
        else:
            assert sim.primary_header[key] == value

    sim.set_source('13:00:00', '31:00:00.0')
    assert np.isclose(sim.source_equatorial.ra, 13 * hourangle)
    assert np.isclose(sim.source_equatorial.dec, 31 * degree)

    del sim.primary_header['SRCTYPE']
    del sim.primary_header['SRCSIZE']
    sim.set_source('', '', header_options=header_options)
    assert sim.primary_header['SRCTYPE'] == 'point_source'
    assert np.isclose(sim.primary_header['SRCSIZE'], 4.85)


def test_set_times(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    sim.primary_header = h
    h['EXPTIME'] = 30.0
    header_options = fits.Header()
    header_options['DATE-OBS'] = '2022-03-30T12:00:00.000'
    sim.set_times('', header_options=header_options)
    for key, value in {'EXPTIME': 30.0, 'DATE-OBS': '2022-03-30T12:00:00.000',
                       'DATE': '2022-03-30T12:00:00.000',
                       'UTCSTART': '12:00:00.000', 'UTCEND': '12:00:30.000',
                       'LASTREW': '2022-03-30T11:59:00.000'}.items():
        assert h[key] == value


def test_set_start_site(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    sim.primary_header = h
    header_options = fits.Header()
    header_options['LON_STA'] = 9.0
    header_options['LAT_STA'] = 10.0
    sim.set_start_site('', '', header_options=header_options)
    assert sim.primary_header['LON_STA'] == 9
    assert sim.primary_header['LAT_STA'] == 10
    assert np.isclose(sim.start_site.lon, 9 * degree)
    assert np.isclose(sim.start_site.lat, 10 * degree)

    sim.set_start_site('12d', '13d')
    assert np.isclose(sim.start_site.lon, 12 * degree)
    assert np.isclose(sim.start_site.lat, 13 * degree)


def test_initialize_aircraft(initialized_simulation, user_options):
    sim = initialized_simulation
    h = user_options.copy()
    sim.primary_header = h
    h['OBJRA'] = h['OBSRA']
    h['OBJDEC'] = h['OBSDEC']
    sim.initialize_aircraft()
    assert sim.aircraft.start_altitude == 41993 * ft
    expected = {'ALTI_STA': 41993.0, 'ALTI_END': 41998.0,
                'LON_STA': -108.182373, 'LAT_STA': 47.043457,
                'AIRSPEED': 500.0, 'GRDSPEED': 500.0,
                'LON_END': -108.19102289756896, 'LAT_END': 47.05336705548469,
                'HEADING': -30.738315389960356,
                'TRACKANG': -30.738315389960356}
    for key, value in expected.items():
        assert np.isclose(h[key], value, rtol=1e-3)


def test_update_header_weather(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    h['ALTI_STA'] = 40000.0
    h['ALTI_END'] = 41000.0
    sim.update_header_weather(h)
    for key, value in {'TEMP_OUT': -47.0, 'TEMPPRI1': -13.0, 'TEMPPRI2': -15.0,
                       'TEMPPRI3': -14.0, 'TEMPSEC1': -17.0,
                       'WVZ_STA': 27.64245887371262, 'WVZ_END': 22.0}.items():
        assert np.isclose(h[key], value)


def test_update_non_astronomical_columns(upto_non_astronomical):
    sim = upto_non_astronomical
    sim.update_non_astronomical_columns()
    for key in ['FrameCounter', 'Timestamp', 'hwpCounts', 'Flag', 'PWV', 'LOS',
                'ROLL', 'LON', 'LAT', 'LST', 'NonSiderealRA',
                'NonSiderealDec']:
        assert key in sim.column_values
    assert sim.lst.size == 1016
    assert sim.mjd.size == 1016


def test_get_hwp_column():
    assert np.allclose(Sim.get_hwp_column(np.arange(4)), [0, 0, 0, 0])


def test_get_pwv_column(initialized_simulation):
    sim = initialized_simulation
    sim.aircraft.start_altitude = 40000 * ft
    sim.aircraft.end_altitude = 41000 * ft
    pwv = sim.get_pwv_column(np.arange(5))
    assert np.allclose(
        pwv, [27.642, 26.109, 24.660, 23.292, 22], atol=1e-3)


def test_get_los_column():
    assert np.allclose(Sim.get_los_column(np.arange(4)), [0, 0, 0, 0])


def test_get_roll_column():
    assert np.allclose(Sim.get_roll_column(np.arange(4)), [0, 0, 0, 0])


def test_get_location_columns(initialized_simulation):
    sim = initialized_simulation
    sim.aircraft.start_location = GeodeticCoordinates([10, 11])
    sim.aircraft.end_location = GeodeticCoordinates([10.4, 11.4])
    location = sim.get_location_columns(np.arange(5))
    assert np.allclose(location,
                       [[10, 10.1, 10.2, 10.3, 10.4],
                        [11, 11.1, 11.2, 11.3, 11.4]])


def test_get_nonsidereal_columns(initialized_simulation):
    sim = initialized_simulation
    sim.source_equatorial = EquatorialCoordinates([15, 11])
    c = sim.get_nonsidereal_columns(np.arange(3))
    assert np.allclose(c, [[1, 1, 1], [11, 11, 11]])


def test_update_vpa_columns(upto_vpa):
    sim = upto_vpa
    sim.update_vpa_columns()
    for key in ['TABS_VPA', 'SIBS_VPA', 'Chop_VPA']:
        assert key in sim.column_values
    assert np.allclose(sim.column_values['TABS_VPA'], 0)
    sim.primary_header['TELVPA'] = 1.0
    sim.update_vpa_columns()
    assert np.allclose(sim.column_values['TABS_VPA'], 1)
    assert np.allclose(sim.sin_pa, 0.01745, atol=1e-5)
    assert np.allclose(sim.cos_pa, 0.99985, atol=1e-5)
    assert np.allclose(sim.column_values['Chop_VPA'], 1)


def test_update_chopper(upto_vpa):
    sim = upto_vpa
    sim.update_vpa_columns()
    sim.primary_header['CHOPPING'] = False
    sim.update_chopper()
    assert np.allclose(sim.column_values['sofiaChopR'], 0)
    assert np.allclose(sim.column_values['sofiaChopS'], 0)
    assert np.all(sim.chopper_position.is_null())

    sim.primary_header['CHPFREQ'] = 0.5
    sim.primary_header['CHOPPING'] = True
    sim.primary_header['CHPAMP1'] = 20.0
    sim.column_values['Chop_VPA'].fill(45)
    sim.column_values['TABS_VPA'].fill(15)
    sim.scan.configuration.parse_key_value('chopper.invert', 'True')

    sim.update_chopper()
    angle = np.median(sim.chopper_position.angle())
    assert np.isclose(angle, 150 * degree)
    assert np.isclose(sim.chopper_position.length.max(), 20 * arcsec, atol=0.1)

    p0 = sim.chopper_position.copy()
    sim.primary_header['CHPNOISE'] = 2.0
    sim.update_chopper()
    assert sim.chopper_position != p0


def test_update_astronomical_columns(upto_vpa):
    sim = upto_vpa
    sim.update_vpa_columns()
    sim.update_chopper()
    sim.integration.configuration.parse_key_value('chopper.shift', '1')

    sim.primary_header['SCNPATT'] = 'foo'
    with pytest.raises(ValueError) as err:
        sim.update_astronomical_columns()
    assert 'Scan pattern foo not implemented' in str(err.value)

    sim.primary_header['SCNPATT'] = 'Daisy'
    sim.scan.info.sampling_interval = 0.006 * second

    sim.update_astronomical_columns()
    for key in ['RA', 'DEC', 'AZ', 'EL']:
        assert key in sim.column_values
    # Check chopper shift
    assert sim.chopper_position.x[0] == 0 * arcsec
    assert sim.chopper_position.y[0] == 0 * arcsec

    for attr in ['horizontal_corrected', 'equatorial_corrected',
                 'horizontal_offset_corrected',
                 'apparent_equatorial_corrected', 'apparent_equatorial',
                 'site', 'horizontal', 'equatorial', 'horizontal_offset']:
        assert not np.any(getattr(sim, attr).is_nan())

    sim.primary_header['SCNPATT'] = 'Lissajous'
    sim.scan.info.sampling_interval = 0.004 * second
    sim.update_astronomical_columns()

    for attr in ['horizontal_corrected', 'equatorial_corrected',
                 'horizontal_offset_corrected',
                 'apparent_equatorial_corrected', 'apparent_equatorial',
                 'site', 'horizontal', 'equatorial', 'horizontal_offset']:
        assert not np.any(getattr(sim, attr).is_nan())


def test_get_daisy_equatorial(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    h['SCNDNOSC'] = 10
    h['SCNDRAD'] = 5
    h['SCNDPER'] = 1
    h['SCNCONST'] = False
    sim.primary_header = h
    sim.source_equatorial = EquatorialCoordinates([20, 20])
    sim.scan = DummyScan()
    p = sim.get_daisy_equatorial()
    assert isinstance(p, EquatorialCoordinates) and p.size == 100
    assert not np.any(p.is_nan())


def test_get_lissajous_equatorial(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    h['SCNAMPXL'] = 10
    h['SCNAMPEL'] = 20
    h['SCNFQRAT'] = np.sqrt(2)
    h['SCNPHASE'] = 180
    h['EXPTIME'] = 10
    h['SCNRATE'] = 1
    h['SCNNOSC'] = 20
    h['SCNCONST'] = False
    sim.primary_header = h
    sim.scan = DummyScan()
    sim.source_equatorial = EquatorialCoordinates([20, 20])
    p = sim.get_lissajous_equatorial()
    assert isinstance(p, EquatorialCoordinates) and p.size == 100
    assert not np.any(p.is_nan())


def test_get_data_hdu(upto_vpa):
    sim = upto_vpa
    sim.update_vpa_columns()
    sim.update_chopper()
    sim.update_astronomical_columns()

    del sim.column_values['LOS']
    hdu = sim.get_data_hdu()
    assert hdu.header['EXTNAME'] == 'Timestream'
    table = Table(hdu.data)
    h = hdu.header
    unit_type = {}
    data_type = {}
    fields = h['TFIELDS']
    sq_key = -1
    jump_key = -1
    for i in range(1, fields + 1):
        name = f'TTYPE{i}'
        fits_name = h.get(name)
        unit_type[fits_name] = h.get(f'TUNIT{i}')
        data_type[fits_name] = h.get(f'TFORM{i}')
        if fits_name == 'SQ1Feedback':
            sq_key = i
        elif fits_name == 'FluxJumps':
            jump_key = i

    for name, coldef in sim.data_column_definitions.items():
        if name == 'LOS':
            assert name not in table.columns
            continue
        assert name in table.columns
        if name in ['SQ1Feedback', 'FluxJumps']:
            continue
        check_unit = unit_type[name]
        if check_unit is not None:
            assert check_unit == coldef[0]
        check_type = data_type[name]
        if check_type is not None:
            assert check_type == coldef[1]

    assert data_type['SQ1Feedback'] == '41656J'
    assert data_type['FluxJumps'] == '41656I'

    n = sim.info.detector_array.FITS_COLS * sim.info.detector_array.FITS_ROWS
    sim.integration.frames.set_frame_size(n + 10)
    hdu = sim.get_data_hdu()

    assert hdu.header[f'TFORM{sq_key}'] == '5258J'
    assert hdu.header[f'TFORM{jump_key}'] == '5258I'


def test_create_source_model(initialized_simulation):
    sim = initialized_simulation
    h = fits.Header()
    sim.primary_header = h
    h['SRCSIZE'] = 10.0
    sim.source_model = None
    sim.create_source_model()
    assert isinstance(sim.source_model, SingleGaussian)
    h['SRCTYPE'] = 'extended'
    sim.create_source_model()
    assert np.isclose(sim.source_model.model.x_stddev * gaussian_sigma_to_fwhm,
                      10 * arcsec)

    h['SRCTYPE'] = 'foo'
    with pytest.raises(ValueError) as err:
        sim.create_source_model()
    assert 'foo simulated source is not implemented' in str(err.value)


def test_create_simulated_data(upto_vpa):
    sim = upto_vpa
    sim.update_vpa_columns()
    sim.update_chopper()
    sim.update_astronomical_columns()
    sim.integration.frames.apply_hdu(sim.get_data_hdu())
    sim.primary_header['SRCS2N'] = 30.0
    sim.source_model.model.amplitude = 1 * units.Unit('Jy')
    sim.create_simulated_data()
    dac = sim.column_values['SQ1Feedback']
    assert dac.shape == (1016, 41, 128)
    assert not np.allclose(dac, 0)
    assert np.all(np.isfinite(dac))


def test_create_simulated_jumps(upto_vpa):
    sim = upto_vpa
    sim.update_vpa_columns()
    sim.update_chopper()
    sim.update_astronomical_columns()
    sim.integration.frames.apply_hdu(sim.get_data_hdu())
    sim.create_simulated_data()
    h = sim.primary_header
    h['JUMPCHAN'] = '99999999'
    sim.create_simulated_jumps()
    assert np.allclose(sim.column_values['FluxJumps'], 0)
    h['JUMPCHAN'] = 'all'
    sim.create_simulated_jumps()

    jumps = sim.column_values['FluxJumps']
    has_jumps = jumps != jumps[0][None]
    has_jumps = np.any(has_jumps, axis=0)
    assert has_jumps.sum() == 2624

    h['JUMPFRMS'] = '5,10'
    sim.create_simulated_jumps()
    jumps = sim.column_values['FluxJumps']
    assert np.allclose(jumps[0:5, 0, 0], 1)
    assert np.allclose(jumps[5:10, 0, 0], 2)
    assert np.allclose(jumps[10:, 0, 0], 3)

    del h['JUMPCHAN']
    sim.create_simulated_jumps()
    assert np.allclose(sim.column_values['FluxJumps'], 0)
