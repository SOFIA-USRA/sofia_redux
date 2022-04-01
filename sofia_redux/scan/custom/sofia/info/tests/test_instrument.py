# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.instrument import SofiaInstrumentInfo


arcsec = units.Unit('arcsec')
s = units.Unit('second')
um = units.Unit('um')
m = units.Unit('m')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['INSTRUME'] = 'HAWC_PLUS'
    h['DATATYPE'] = 'OTHER'
    h['INSTCFG'] = 'TOTAL_INTENSITY'
    h['INSTMODE'] = 'OTFMAP'
    h['MCCSMODE'] = 'band_a_foctest'
    h['SPECTEL1'] = 'HAW_A'
    h['SPECTEL2'] = 'HAW_HWP_Open'
    h['SLIT'] = 'UNKNOWN'
    h['DETCHAN'] = 'UNKNOWN'
    h['RESOLUN'] = -9999.0
    h['EXPTIME'] = 100.0
    h['TOTINT'] = 110.0
    h['WAVECENT'] = 53.0
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaInstrumentInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_class():
    assert SofiaInstrumentInfo.telescope_diameter == 2.5 * m


def test_init():
    info = SofiaInstrumentInfo()
    assert info.mount.name == 'NASMYTH_COROTATING'
    assert info.instrument_name is None
    assert info.data_type is None
    assert info.instrument_config is None
    assert info.instrument_mode is None
    assert info.mccs_mode is None
    assert info.spectral_element_1 is None
    assert info.spectral_element_2 is None
    assert info.slit_id is None
    assert info.detector_channel is None
    assert np.isnan(info.spectral_resolution)
    assert np.isnan(info.exposure_time) and info.exposure_time.unit == s
    assert np.isnan(info.total_integration_time)
    assert info.total_integration_time.unit == s
    assert np.isnan(info.wavelength) and info.wavelength.unit == um


def test_apply_configuration(sofia_configuration):
    info = SofiaInstrumentInfo()
    info.apply_configuration()
    assert info.instrument_name is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.instrument_name == 'HAWC_PLUS'
    assert info.data_type == 'OTHER'
    assert info.instrument_config == 'TOTAL_INTENSITY'
    assert info.mccs_mode == 'band_a_foctest'
    assert info.spectral_element_1 == 'HAW_A'
    assert info.spectral_element_2 == 'HAW_HWP_Open'
    assert info.slit_id == 'UNKNOWN'
    assert info.detector_channel == 'UNKNOWN'
    assert np.isnan(info.spectral_resolution)
    assert info.exposure_time == 100 * s
    assert info.total_integration_time == 110 * s
    assert info.wavelength == 53 * um
    assert np.isclose(info.angular_resolution, 5.335 * arcsec, atol=1e-3)
    assert np.isclose(info.frequency, 5.656e12 * units.Unit('Hz'), rtol=1e-3)
    info.configuration.parse_key_value('aperture', '2.0')
    info.apply_configuration()
    assert np.isclose(info.angular_resolution, 6.669 * arcsec, atol=1e-3)


def test_edit_header(sofia_info):
    info = sofia_info.copy()
    info.spectral_resolution = 15.0
    h = fits.Header()
    info.edit_header(h)
    expected = {
        'INSTRUME': 'HAWC_PLUS',
        'DATATYPE': 'OTHER',
        'INSTCFG': 'TOTAL_INTENSITY',
        'INSTMODE': 'OTFMAP',
        'MCCSMODE': 'band_a_foctest',
        'EXPTIME': 100.0,
        'SPECTEL1': 'HAW_A',
        'SPECTEL2': 'HAW_HWP_Open',
        'WAVECENT': 53.0,
        'SLIT': 'UNKNOWN',
        'RESOLUN': 15.0,
        'DETCHAN': 'UNKNOWN',
        'TOTINT': 110.0}
    for key, value in expected.items():
        assert h[key] == value


def test_get_table_entry(sofia_info):
    info = sofia_info.copy()
    assert info.get_table_entry('wave') == 53 * um
    assert info.get_table_entry('exp') == 100 * s
    assert info.get_table_entry('inttime') == 110 * s
    assert info.get_table_entry('datatype') == 'OTHER'
    assert info.get_table_entry('mode') == 'OTFMAP'
    assert info.get_table_entry('cfg') == 'TOTAL_INTENSITY'
    assert info.get_table_entry('slit') == 'UNKNOWN'
    assert info.get_table_entry('spec1') == 'HAW_A'
    assert info.get_table_entry('spec2') == 'HAW_HWP_Open'
    assert info.get_table_entry('foo') is None
