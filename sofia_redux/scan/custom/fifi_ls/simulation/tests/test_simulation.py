# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.units import imperial
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.simulation.source_models.single_gaussian_2d1 import \
    SingleGaussian2d1
from sofia_redux.scan.custom.fifi_ls.simulation.simulation import \
    FifiLsSimulation as Sim
from sofia_redux.scan.custom.fifi_ls.info.info import FifiLsInfo

imperial.enable()
degree = units.Unit('degree')
arcsec = units.Unit('arcsec')
second = units.Unit('second')
hourangle = units.Unit('hourangle')
ft = units.Unit('ft')


def test_defaults():
    assert isinstance(Sim.sim_keys, set)
    for key in list(Sim.sim_keys):
        assert isinstance(key, str)
    assert isinstance(Sim.default_values, dict)
    for key in Sim.default_values.keys():
        assert isinstance(key, str)
    assert isinstance(Sim.default_comments, dict)
    for key, value in Sim.default_comments.items():
        assert isinstance(key, str) and isinstance(value, str)


def test_init():
    sim = Sim()
    assert isinstance(sim.info, FifiLsInfo)
    with pytest.raises(ValueError) as err:
        Sim(info='foo')
    assert 'Simulation must be initialized with' in str(err.value)


def test_default_value():
    for k, v in Sim.default_values.items():
        assert Sim.default_value(k) == v
    assert Sim.default_value('all') == Sim.default_values


def test_default_comment():
    for k, v in Sim.default_comments.items():
        assert Sim.default_comment(k) == v
    assert Sim.default_comment('all') == Sim.default_comments


def test_update_header_value():
    sim = Sim()
    h = fits.Header()
    sim.update_header_value(h, 'DATASRC', 'foo')
    assert h['DATASRC'] == 'foo'
    assert h.comments['DATASRC'] == 'Data source'
    h.comments['DATASRC'] = ''
    sim.update_header_value(h, 'DATASRC', 'foo')
    assert h['DATASRC'] == 'foo'
    assert h.comments['DATASRC'] == 'Data source'


def test_write_simulated_hdul(tmpdir):
    directory = tmpdir.mkdir('test_fifi_simulated_write')
    filename = str(directory.join('test_file.fits'))
    sim = Sim()
    sim.write_simulated_hdul(filename)
    hdul = fits.open(filename)
    assert 'FLUX' in hdul
    hdul.close()


def test_create_simulated_hdul():
    sim = Sim()
    hdul = sim.create_simulated_hdul()
    for extname in ['FLUX', 'STDDEV', 'UNCORRECTED_FLUX', 'UNCORRECTED_STDDEV',
                    'LAMBDA', 'UNCORRECTED_LAMBDA', 'XS', 'YS', 'RA', 'DEC',
                    'ATRAN', 'RESPONSE', 'UNSMOOTHED_ATRAN']:
        assert extname in hdul


def test_create_basic_hdul():
    sim = Sim()
    sim.create_basic_hdul()
    assert isinstance(sim.primary_header, fits.Header)
    assert isinstance(sim.hdul, fits.HDUList)


def test_create_primary_header():
    sim = Sim()
    h = sim.create_primary_header()
    assert isinstance(h, fits.Header)
    assert h is sim.primary_header


def test_default_primary_header():
    sim = Sim()
    h = sim.default_primary_header(None)
    h2 = sim.default_primary_header(h)
    assert h2 == h
    h3 = sim.default_primary_header({'FOO': ('bar', 'baz')})
    assert h3['FOO'] == 'bar'
    assert h3.comments['FOO'] == 'baz'
    h4 = sim.default_primary_header({'FOO': 'bar'})
    assert h4['FOO'] == 'bar'
    assert h4.comments['FOO'] == ''
    h4 = sim.default_primary_header({'DATASRC': 'foo'})
    assert h4['DATASRC'] == 'foo'
    assert h4.comments['DATASRC'] == 'Data source'


def test_set_source():
    sim = Sim()
    sim.primary_header = sim.default_primary_header(None)
    sim.info.configuration.read_fits(sim.primary_header)
    sim.info.instrument.set_configuration(sim.info.configuration)
    sim.info.instrument.apply_configuration()
    for key in ['OBSLAM', 'OBSBET', 'SKY_ANGLE', 'DET_ANGLE', 'DLAM_MAP',
                'DBET_MAP', 'OBSRA', 'OBSDEC', 'OBJECT', 'SRCTYPE',
                'SRCSIZE', 'SRCZSIZE']:
        if key in sim.primary_header:  # pragma: no cover
            del sim.primary_header[key]
    sim.set_source(ra=1 * degree, dec=2 * degree,
                   header_options={'OBSRA': 10, 'OBSDEC': 30})
    expected = {'OBSLAM': 150, 'OBSBET': 30, 'SKY_ANGL': 0, 'DET_ANGL': 0,
                'DLAM_MAP': 0, 'DBET_MAP': 0, 'OBSRA': 10, 'OBSDEC': 30,
                'TELRA': 10, 'OBJRA': 10, 'TELDEC': 30, 'OBJDEC': 30,
                'OBJECT': 'simulated_source', 'SRCTYPE': 'point_source'}
    for key, value in expected.items():
        if not isinstance(value, str):
            assert np.isclose(sim.primary_header[key], value)
        else:
            assert sim.primary_header[key] == value
    assert sim.primary_header['SRCSIZE'] > 0
    assert sim.primary_header['SRCZSIZE'] > 0
    fwhm = sim.primary_header['SRCSIZE']

    del sim.primary_header['SRCSIZE']
    del sim.primary_header['OBSRA']
    del sim.primary_header['OBSDEC']
    sim.primary_header['SRCTYPE'] = 'extended'
    sim.set_source(ra='2:00:00', dec='60:00:00',
                   header_options={'SRCTYPE': 'extended'})
    assert np.isclose(sim.primary_header['SRCSIZE'], 3 * fwhm)
    assert np.isclose(sim.primary_header['OBSRA'], 2)
    assert np.isclose(sim.primary_header['OBSDEC'], 60)


def test_set_times():
    sim = Sim()
    sim.primary_header = fits.Header()
    sim.primary_header['EXPTIME'] = 60.0
    timestamp = '2020-07-26T16:30:09.060'
    header_options = {'DATE-OBS': '2022-07-26T16:30:09.060'}
    sim.set_times(timestamp, header_options=header_options)
    h = dict(sim.primary_header)
    expected = {'EXPTIME': 60.0,
                'DATE-OBS': '2022-07-26T16:30:09.060',
                'DATE': '2022-07-26T16:30:09.060',
                'UTCSTART': '16:30:09.060',
                'UTCEND': '16:31:09.060',
                'LASTREW': '2022-07-26T16:29:09.060'}
    for k, v in expected.items():
        assert h[k] == v


def test_generate_scan_pattern():
    sim = Sim()
    sim.create_basic_hdul()
    sim.primary_header['SCNPATT'] = 'foo'
    with pytest.raises(ValueError) as err:
        sim.generate_scan_pattern()
    assert 'Scan pattern foo not implemented' in str(err.value)
    sim.primary_header['SCNPATT'] = 'daisy'
    sim.generate_scan_pattern()
    xs = sim.hdul['XS'].data
    ys = sim.hdul['YS'].data
    ra = sim.hdul['RA'].data
    dec = sim.hdul['DEC'].data
    for key in ['XS', 'YS', 'RA', 'DEC']:
        del sim.hdul[key]
    sim.primary_header['SCNPATT'] = 'lissajous'
    sim.generate_scan_pattern()
    assert not np.allclose(sim.hdul['XS'].data, xs)
    assert not np.allclose(sim.hdul['YS'].data, ys)
    assert not np.allclose(sim.hdul['RA'].data, ra)
    assert not np.allclose(sim.hdul['DEC'].data, dec)


def test_create_lambda_hdu():
    sim = Sim()
    sim.create_basic_hdul()
    options = sim.info.instrument.options.options
    options['CHANNEL'] = 'BLUE'
    options['DICHROIC'] = '105'
    options['G_ORD_B'] = '1'
    sim.primary_header['BARYSHIFT'] = 1.0
    sim.create_lambda_hdu()
    lam = sim.hdul['LAMBDA'].data
    ulam = sim.hdul['UNCORRECTED_LAMBDA'].data
    assert np.allclose(lam, ulam * 2)
    del sim.hdul['LAMBDA']
    del sim.hdul['UNCORRECTED_LAMBDA']
    options['G_ORD_B'] = 2
    sim.create_lambda_hdu()
    assert not np.allclose(sim.hdul['LAMBDA'].data, lam)
    assert not np.allclose(sim.hdul['UNCORRECTED_LAMBDA'].data, lam)
    options['CHANNEL'] = 'RED'
    del sim.hdul['LAMBDA']
    del sim.hdul['UNCORRECTED_LAMBDA']
    sim.create_lambda_hdu()
    assert not np.allclose(sim.hdul['LAMBDA'].data, lam)
    assert not np.allclose(sim.hdul['UNCORRECTED_LAMBDA'].data, lam)
    del sim.hdul['LAMBDA']
    del sim.hdul['UNCORRECTED_LAMBDA']
    options['DICHROIC'] = '130'
    sim.create_lambda_hdu()
    sim.create_lambda_hdu()
    assert not np.allclose(sim.hdul['LAMBDA'].data, lam)
    assert not np.allclose(sim.hdul['UNCORRECTED_LAMBDA'].data, lam)


def test_create_atran_response_hdu():
    sim = Sim()
    sim.create_basic_hdul()
    sim.create_lambda_hdu()
    sim.create_atran_response_hdu()
    assert sim.hdul['ATRAN'].data.shape == (16, 25)
    assert sim.hdul['RESPONSE'].data.shape == (16, 25)
    assert sim.hdul['UNSMOOTHED_ATRAN'].data.shape == (2, 100)


def test_update_header_origin():
    sim = Sim()
    sim.create_basic_hdul()
    h = fits.Header()
    sim.update_header_origin(h)
    expected = {'FILENAME': 'F0999_FI_IFS_09912345_BLU_WSH_00001.fits',
                'OBS_ID': 'P_2021-12-06_FI_F999B00001',
                'PLANID': 'UNKNOWN',
                'MISSN-ID': '2021-12-06_FI_F999'}
    for k, v in expected.items():
        assert h[k] == v
    sim.info.instrument.channel = 'red'
    h = fits.Header()
    sim.update_header_origin(h)
    expected = {'FILENAME': 'F0999_FI_IFS_09912345_RED_WSH_00001.fits',
                'OBS_ID': 'P_2021-12-06_FI_F999R00001',
                'PLANID': 'UNKNOWN',
                'MISSN-ID': '2021-12-06_FI_F999'}
    for k, v in expected.items():
        assert h[k] == v


def test_update_header_scanning():
    sim = Sim()
    sim.create_basic_hdul()
    h = fits.Header()
    h['SRCSIZE'] = 5.0
    h['SRCTYPE'] = 'extended'
    base_h = h.copy()
    sim.update_header_scanning(h)
    assert h['EXPTIME'] >= 45
    h = base_h.copy()
    h['EXPTIME'] = 30
    sim.update_header_scanning(h)
    assert h['EXPTIME'] >= 30
    h = base_h.copy()
    h['TOTTIME'] = 50
    sim.update_header_scanning(h)
    assert h['EXPTIME'] >= 50

    h = base_h.copy()
    h['SCNPATT'] = 'foo'
    with pytest.raises(ValueError) as err:
        sim.update_header_scanning(h)
    assert 'not currently supported' in str(err.value)


def test_update_header_lissajous():
    sim = Sim()
    h = fits.Header()
    h['SCNPATT'] = 'foo'
    sim.update_header_lissajous(h)
    for key in ['SCNAMPEL', 'SCNAMPXL', 'SCNFQRAT', 'SCNPHASE', 'SCNTOFF']:
        assert h[key] == -9999
    h = fits.Header()
    h['SCNPATT'] = 'LISSAJOUS'
    h['SRCSIZE'] = 10.0
    h['SRCTYPE'] = 'extended'
    h['SCNAMPXL'] = 30.0
    h['SCNAMPEL'] = 40.0
    sim.update_header_lissajous(h)
    assert not h['SCNCONST']
    assert np.isclose(h['SCNFQRAT'], np.sqrt(2))
    assert np.isclose(h['SCNPHASE'], 90)
    assert np.isclose(h['SCNTOFF'], 0)

    del h['SCNAMPEL']
    sim.update_header_lissajous(h)
    assert h['SCNAMPEL'] == 30
    del h['SCNAMPXL']
    sim.update_header_lissajous(h)
    assert h['SCNAMPXL'] == 30
    del h['SCNAMPEL']
    del h['SCNAMPXL']
    sim.update_header_lissajous(h)
    assert np.isclose(h['SCNAMPEL'], 25)
    assert np.isclose(h['SCNAMPXL'], 25)


def test_update_header_daisy():
    sim = Sim()
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


def test_update_header_raster():
    sim = Sim()
    h = fits.Header()
    sim.update_header_raster(h)
    for key in ['SCNNSUBS', 'SCNLEN', 'SCNSTEP', 'SCNSTEPS']:
        assert h[key] == -9999
    assert not h['SCNCROSS']


def test_set_start_site():
    sim = Sim()
    longitude = '12:30:30d'
    latitude = '10:30:30d'
    sim.primary_header = fits.Header()
    sim.set_start_site(longitude, latitude)
    assert np.isclose(sim.primary_header['LON_STA'], 12.5083333, atol=1e-5)
    assert np.isclose(sim.primary_header['LAT_STA'], 10.5083333, atol=1e-5)
    assert sim.start_site is not None
    header_options = {'LON_STA': 10.0, 'LAT_STA': 12.0}
    sim.set_start_site(longitude, latitude, header_options=header_options)
    assert np.isclose(sim.primary_header['LON_STA'], 10, atol=1e-5)
    assert np.isclose(sim.primary_header['LAT_STA'], 12, atol=1e-5)


def test_initialize_aircraft():
    sim = Sim()
    h = sim.default_primary_header({'DATE-OBS': '2020-02-02T22:22:22'})
    h['EXPTIME'] = 30.0
    h['OBJRA'] = 10.0
    h['OBJDEC'] = 12.0
    sim.primary_header = h
    sim.initialize_aircraft()
    for key in ['ALTI_STA', 'ALTI_END', 'AIRSPEED', 'GRDSPEED', 'LON_END',
                'LAT_END', 'HEADING', 'TRACKANG']:
        assert key in h


def test_update_header_weather():
    sim = Sim()
    h = fits.Header()
    h['ALTI_STA'] = 41000
    h['ALTI_END'] = 41002
    sim.update_header_weather(h)
    expected = {'ALTI_STA': 41000, 'ALTI_END': 41002, 'TEMP_OUT': -47.0,
                'TEMPPRI1': -13.0, 'TEMPPRI2': -15.0, 'TEMPPRI3': -14.0,
                'TEMPSEC1': -17.0, 'WVZ_STA': 22.0, 'WVZ_END': 21.9899566}
    for key, value in expected.items():
        assert np.isclose(h[key], value, atol=1e-3)


def test_get_daisy_equatorial():
    sim = Sim()
    h = fits.Header()
    h['SCNDNOSC'] = 1
    h['SCNDRAD'] = 10.0
    h['SCNDPER'] = 10
    h['SCNCONST'] = False
    sim.source_equatorial = EquatorialCoordinates([30, 40])
    sim.info.sampling_interval = 1.0 * units.Unit('second')
    sim.primary_header = h
    equatorial = sim.get_daisy_equatorial()
    assert np.allclose(equatorial.ra[:3], [30, 30.000961, 30.001948] * degree,
                       atol=1e-4)
    assert np.allclose(equatorial.dec[:3], [40, 40.001320, 40.001784] * degree,
                       atol=1e-4)


def test_get_lissajous_equatorial():
    sim = Sim()
    h = fits.Header()
    h['SCNAMPXL'] = 10.0
    h['SCNAMPEL'] = 10.0
    h['SCNFQRAT'] = np.sqrt(2)
    h['SCNPHASE'] = 90.0
    h['EXPTIME'] = 5.0
    h['SCNRATE'] = 10.0
    h['SCNNOSC'] = 1
    h['SCNCONST'] = False
    sim.source_equatorial = EquatorialCoordinates([30, 40])
    sim.info.sampling_interval = 1.0 * units.Unit('second')
    sim.primary_header = h
    equatorial = sim.get_lissajous_equatorial()
    assert np.allclose(equatorial.ra[:3],
                       [29.99861111, 29.99978341, 30.00132134] * degree,
                       atol=1e-4)
    assert np.allclose(equatorial.dec[:3],
                       [40.00138889, 39.99942202, 39.99909216] * degree,
                       atol=1e-4)


def test_create_source_model():
    sim = Sim()
    h = fits.Header()
    h['SRCSIZE'] = 10.0
    h['SRCZSIZE'] = 0.08
    h['WAVECENT'] = 150.0
    sim.primary_header = h
    sim.create_source_model()
    assert isinstance(sim.source_model, SingleGaussian2d1)
    sim.primary_header['SRCTYPE'] = 'foo'
    with pytest.raises(ValueError) as err:
        sim.create_source_model()
    assert 'not implemented' in str(err.value)


def test_create_simulated_data():
    sim = Sim()
    sim.create_basic_hdul()
    sim.create_lambda_hdu()
    sim.create_atran_response_hdu()
    sim.generate_scan_pattern()
    sim.create_simulated_data()
    for extname in ['FLUX', 'STDDEV', 'UNCORRECTED_FLUX',
                    'UNCORRECTED_STDDEV']:
        assert extname in sim.hdul


