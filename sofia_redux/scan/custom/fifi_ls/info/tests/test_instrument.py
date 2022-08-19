# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.fifi_ls.info.instrument import \
    FifiLsInstrumentInfo

arcsec = units.Unit('arcsec')
um = units.Unit('um')
second = units.Unit('second')


@pytest.fixture
def fifi_header():
    h = fits.Header()
    h['CHANNEL'] = 'BLUE'
    h['G_ORD_B'] = 1
    h['G_WAVE_B'] = 63.26
    h['G_WAVE_R'] = 157.93
    h['RAMPLN_B'] = 32
    h['RAMPLN_R'] = 32
    h['ALPHA'] = 0.004
    return h


@pytest.fixture
def fifi_configuration(fifi_header):
    c = Configuration()
    c.instrument_name = 'fifi_ls'
    c.read_configuration('default.cfg')
    c.read_fits(fifi_header)
    return c


@pytest.fixture
def fifi_info(fifi_configuration):
    info = FifiLsInstrumentInfo()
    info.configuration = fifi_configuration
    return info


def test_init():
    info = FifiLsInstrumentInfo()
    assert info.name == 'fifi_ls'
    assert info.channel is None
    assert np.isnan(info.alpha)
    assert info.ramps == -1
    assert info.resolution.x == 5 * arcsec
    assert info.resolution.y == 5 * arcsec
    assert info.resolution.z == 0 * um
    assert info.spectral_resolution == 1000


def test_xy_resolution(fifi_info):
    info = fifi_info.copy()
    info.xy_resolution = 4 * arcsec
    assert info.xy_resolution == 4 * arcsec


def test_z_resolution(fifi_info):
    info = fifi_info.copy()
    info.z_resolution = 3 * um
    assert info.z_resolution == 3 * um


def test_get_spectral_size(fifi_info):
    assert fifi_info.get_spectral_size() == 0 * um


def test_apply_configuration(fifi_info):
    info = FifiLsInstrumentInfo()
    info.apply_configuration()
    assert np.isnan(info.alpha)
    info = fifi_info.copy()
    info.apply_configuration()
    assert info.channel == 'BLUE'
    assert info.ramps == 32
    assert info.alpha == 0.004 * second
    assert info.sampling_interval == 0.128 * second
    assert info.integration_time == info.sampling_interval

    del info.options['ALPHA']
    with pytest.raises(ValueError) as err:
        info.apply_configuration()
    assert 'No ALPHA key in header' in str(err.value)

    del info.options['RAMPLN_B']
    with pytest.raises(ValueError) as err:
        info.apply_configuration()
    assert 'No RAMPLN_B key in header' in str(err.value)

    del info.options['CHANNEL']
    with pytest.raises(ValueError) as err:
        info.apply_configuration()
    assert 'No CHANNEL key in header' in str(err.value)


def test_read_resolution(fifi_info):
    info = fifi_info.copy()
    info.configuration.instrument_name = 'foo'
    info.read_resolution()
    default_resolution = 5 * arcsec
    default_spectral = 1000
    assert info.xy_resolution == default_resolution
    assert info.spectral_resolution == default_spectral

    info.channel = 'UNKNOWN'
    info.configuration.instrument_name = 'fifi_ls'
    info.read_resolution()
    assert info.xy_resolution == default_resolution
    assert info.spectral_resolution == default_spectral

    info.channel = 'BLUE'
    g_ord_b = info.options['G_ORD_B']
    del info.options['G_ORD_B']
    info.read_resolution()
    assert info.xy_resolution == default_resolution
    assert info.spectral_resolution == default_spectral
    info.options['G_ORD_B'] = g_ord_b

    g_wave_b = info.options['G_WAVE_B']
    del info.options['G_WAVE_B']
    info.read_resolution()
    assert info.xy_resolution == default_resolution
    assert info.spectral_resolution == default_spectral

    info.options['G_WAVE_B'] = g_wave_b
    info.read_resolution()
    assert info.xy_resolution == 6.9 * arcsec
    assert info.spectral_resolution == 545

    info.channel = 'RED'
    info.read_resolution()
    assert info.xy_resolution == 15.8 * arcsec
    assert info.spectral_resolution == 1180


def test_edit_header(fifi_info):
    info = fifi_info.copy()
    h = fits.Header()
    info.apply_configuration()
    info.edit_header(h)
    assert h['RESOLUN'] == 545
    assert np.isclose(h['SMPLFREQ'], 7.8125)
    assert h['CHANNEL'] == 'BLUE'
    assert h['ALPHA'] == 0.004
    assert h['RAMPLN'] == 32


def test_get_point_size(fifi_info):
    resolution = fifi_info.get_point_size()
    assert resolution.x == 5 * arcsec
    assert resolution.y == 5 * arcsec
    assert resolution.z == 0 * um


def test_get_source_size(fifi_info):
    info = fifi_info.copy()
    source_size = info.get_source_size()
    assert source_size.x == 5 * arcsec and source_size.y == 5 * arcsec
    assert source_size.z == 0 * um
    info.configuration.parse_key_value('sourcesize', '1.0')
    source_size = info.get_source_size()
    assert np.isclose(source_size.x, 5.09901951 * arcsec, atol=1e-3)
    assert np.isclose(source_size.y, 5.09901951 * arcsec, atol=1e-3)
    assert source_size.z == 0 * um
    info.configuration.parse_key_value('sourcesize', '1.0, 2.0')
    source_size = info.get_source_size()
    assert np.isclose(source_size.x, 5.09901951 * arcsec, atol=1e-3)
    assert np.isclose(source_size.y, 5.09901951 * arcsec, atol=1e-3)
    assert source_size.z == 2 * um
    info.configuration = None
    source_size = info.get_source_size()
    assert source_size.x == 5 * arcsec and source_size.y == 5 * arcsec
    assert source_size.z == 0 * um


def test_edit_image_header(fifi_info):
    h = fits.Header()
    fifi_info.edit_image_header(h)
    assert h['BEAM'] == 5.0
    assert h['BEAMZ'] == 0.0
