# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.hawc_plus.info.instrument import \
    HawcPlusInstrumentInfo


degree = units.Unit('degree')
second = units.Unit('second')


@pytest.fixture
def hawc_header():
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
    h['SMPLFREQ'] = 200.0
    return h


@pytest.fixture
def hawc_configuration(hawc_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(hawc_header)
    return c


def test_init():
    info = HawcPlusInstrumentInfo()
    assert info.name == 'hawc_plus'
    assert info.band_id is None
    assert info.hwp_step == 0.25 * degree
    assert info.hwp_telescope_vertical == 0 * degree


def test_apply_configuration(hawc_configuration):
    info = HawcPlusInstrumentInfo()
    info.apply_configuration()
    assert info.band_id is None
    info.configuration = hawc_configuration.copy()
    info.configuration.purge('filter')
    info.apply_configuration()
    assert info.band_id == 'A'
    assert info.integration_time == 0.005 * second
    assert info.sampling_interval == 0.005 * second
    assert info.configuration['filter'] == '53.0um'
    del info.options['SMPLFREQ']
    del info.options['SPECTEL1']
    info.spectral_element_1 = ''
    info.apply_configuration()
    assert info.band_id == '-'
    assert np.isclose(info.sampling_interval, 0.00492005 * second)


def test_edit_header():
    h = fits.Header()
    info = HawcPlusInstrumentInfo()
    info.sampling_interval = 0.1 * second
    info.edit_header(h)
    assert h['SMPLFREQ'] == 10
