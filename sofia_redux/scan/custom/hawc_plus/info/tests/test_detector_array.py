# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.hawc_plus.info.detector_array import \
    HawcPlusDetectorArrayInfo


degree = units.Unit('degree')
arcsec = units.Unit('arcsec')


@pytest.fixture
def hawc_header():
    h = fits.Header()
    h['DETECTOR'] = 'HAWC'
    h['DETSIZE'] = '64,40'
    h['PIXSCAL'] = 9.43
    h['SUBARRNO'] = 3
    h['SIBS_X'] = 15.5
    h['SIBS_Y'] = 19.5
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['SUBARR01'] = 10
    h['SUBARR02'] = 11
    h['MCEMAP'] = '0,2,1,-1'
    return h


@pytest.fixture
def hawc_configuration(hawc_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(hawc_header)
    c.parse_key_value('darkcorrect', 'True')
    c.parse_key_value('hwp', '2')
    return c


@pytest.fixture
def configured_info(hawc_configuration):
    info = HawcPlusDetectorArrayInfo()
    info.configuration = hawc_configuration.copy()
    info.apply_configuration()
    return info


def test_class():
    info = HawcPlusDetectorArrayInfo
    assert info.pol_arrays == 2
    assert info.pol_subarrays == 2
    assert info.subarrays == 4
    assert info.subarray_cols == 32
    assert info.rows == 41
    assert info.subarray_pixels == 1312
    assert info.pol_cols == 64
    assert info.pol_array_pixels == 2624
    assert info.pixels == 5248
    assert info.DARK_SQUID_ROW == 40
    assert info.MCE_BIAS_LINES == 20
    assert info.FITS_ROWS == 41
    assert info.FITS_COLS == 128
    assert info.FITS_CHANNELS == 5248
    assert info.JUMP_RANGE == 128
    assert info.R0 == 0
    assert info.R1 == 1
    assert info.T0 == 2
    assert info.T1 == 3
    assert info.R_ARRAY == 0
    assert info.T_ARRAY == 1
    assert info.POL_ID == ('R', 'T')
    assert info.hwp_step == 0.25 * degree
    assert info.default_boresight_index.x == 33.5
    assert info.default_boresight_index.y == 19.5


def test_init():
    info = HawcPlusDetectorArrayInfo()
    assert not info.dark_squid_correction
    assert info.dark_squid_lookup is None
    assert np.isnan(info.hwp_telescope_vertical)
    assert info.subarray_gain_renorm is None
    assert info.subarrays_requested == ''
    assert info.hwp_angle == -1
    assert np.allclose(info.mce_subarray, [-1, -1, -1, -1])
    assert np.allclose(info.has_subarray, [False] * 4)
    assert np.allclose(info.subarray_offset.is_nan(), [True] * 4)
    assert np.allclose(info.subarray_orientation, [np.nan] * 4 * degree,
                       equal_nan=True)
    assert np.allclose(info.pol_zoom, [np.nan] * 2, equal_nan=True)
    assert info.pixel_sizes.size == 0
    assert np.allclose(info.detector_bias, np.zeros((4, 20)))


def test_apply_configuration(hawc_configuration):
    configuration = hawc_configuration.copy()
    configuration.parse_key_value('darkcorrect', 'True')
    info = HawcPlusDetectorArrayInfo()
    info.apply_configuration()
    assert not info.dark_squid_correction
    info.configuration = configuration.copy()
    info.apply_configuration()
    assert info.dark_squid_correction
    assert np.allclose(info.has_subarray, [True, True, True, False])
    assert np.allclose(info.mce_subarray, [0, 2, 1, -1])
    assert info.hwp_angle == 2
    assert info.subarrays_requested == ''


def test_set_hwp_header(hawc_configuration):
    info = HawcPlusDetectorArrayInfo()
    info.configuration = hawc_configuration.copy()
    info.set_hwp_header()
    assert info.hwp_angle == 2
    info = HawcPlusDetectorArrayInfo()
    info.configuration = hawc_configuration.copy()
    del info.configuration['hwp']
    info.set_hwp_header()
    assert info.hwp_angle == -1


def test_load_detector_configuration(configured_info):
    info = configured_info.copy()
    config = info.configuration
    for i, sub in enumerate(['R0', 'R1', 'T0', 'T1']):
        rotation = float(i + 1)
        offset = ','.join([str(i + 10), str(i + 20)])
        config.parse_key_value(f'rotation.{sub}', rotation)
        config.parse_key_value(f'offset.{sub}', offset)
    config.parse_key_value('zoom.R', '2')
    config.parse_key_value('zoom.T', '3')
    config.parse_key_value('pixelsize', '3.5,4.5')

    info.load_detector_configuration()
    assert np.allclose(info.subarray_orientation, [1, 2, 3, 4] * degree)
    assert np.allclose(info.subarray_offset.x, [10, 11, 12, 13])
    assert np.allclose(info.subarray_offset.y, [20, 21, 22, 23])
    assert np.allclose(info.pol_zoom, [2, 3])
    assert np.isclose(info.pixel_size, np.sqrt(3.5 * 4.5) * arcsec)
    assert np.allclose(info.pixel_sizes.coordinates, [3.5, 4.5] * arcsec)

    config.parse_key_value('pixelsize', '')
    info.load_detector_configuration()
    assert info.pixel_sizes.is_nan()


def test_set_boresight(configured_info):
    info = configured_info.copy()
    info.boresight_index.nan()
    info.configuration.purge('pcenter')
    info.set_boresight()
    assert info.boresight_index.x == 33.5 and info.boresight_index.y == 19.5
    info.configuration.parse_key_value('pcenter', '10,11')
    info.set_boresight()
    assert info.boresight_index.x == 10 and info.boresight_index.y == 11
    info.configuration.parse_key_value('pcenter', '10')
    info.set_boresight()
    assert info.boresight_index.x == 10 and info.boresight_index.y == 10
    info.configuration.parse_key_value('pcenter', '10,11,12')
    with pytest.raises(ValueError) as err:
        info.set_boresight()
    assert 'wrong length' in str(err.value)


def test_select_subarrays(configured_info):
    info = configured_info.copy()
    info.configuration.purge('subarray')
    info.has_subarray.fill(True)
    info.select_subarrays()
    assert info.has_subarray.all()
    assert info.subarrays_requested == ''
    info.configuration.parse_key_value('subarray', 'R0,R1,T0,T1')
    info.select_subarrays()
    assert np.allclose(info.has_subarray, True)
    assert info.subarrays_requested == 'R0, R1, T0, T1'
    info.configuration.parse_key_value('subarray', 'R,T')
    info.select_subarrays()
    assert np.allclose(info.has_subarray, True)
    assert info.subarrays_requested == 'R0, R1, T0, T1'
    info.configuration.parse_key_value('subarray', 'R')
    info.select_subarrays()
    assert np.allclose(info.has_subarray, [True, True, False, False])
    assert info.subarrays_requested == 'R0, R1'
    info.has_subarray.fill(True)
    info.configuration.parse_key_value('subarray', 'T')
    info.select_subarrays()
    assert np.allclose(info.has_subarray, [False, False, True, True])
    assert info.subarrays_requested == 'T0, T1'
    info.has_subarray.fill(True)
    info.configuration.parse_key_value('subarray', 'X1')
    info.select_subarrays()
    assert not info.has_subarray.any()
    assert info.subarrays_requested == ''
    info.has_subarray.fill(True)
    info.configuration.parse_key_value('subarray', ',')
    info.select_subarrays()
    assert info.has_subarray.all()
    assert info.subarrays_requested == ''


def test_parse_configuration_hdu(configured_info):
    info = configured_info.copy()
    h = fits.Header()
    biases = np.arange(20)
    for sub in range(3):
        values = biases + 100 * sub
        if sub == 2:
            values = values[:-1]
        b = ','.join([str(x) for x in list(values)])
        h[f'MCE{sub}_TES_BIAS'] = b
    hdu = fits.BinTableHDU()
    hdu.header = h
    info.parse_configuration_hdu(hdu)
    expected = np.zeros((4, 20), dtype=int)
    expected[0] = np.arange(20)
    expected[1] = np.arange(20) + 100
    assert np.allclose(info.detector_bias, expected)

    h = fits.Header()
    for sub in range(2):
        values = biases + 100 * sub
        b = ','.join([str(x) for x in list(values)])
        h[f'MCE{sub}_TES_BIAS'] = b
    hdu.header = h
    info.parse_configuration_hdu(hdu)
    assert np.allclose(info.detector_bias, expected)


def test_get_sibs_positions(configured_info):
    info = configured_info.copy()
    info.configuration.parse_key_value('pixelsize', '1.0')
    info.load_detector_configuration()
    position = info.get_sibs_position(1, 1, 1)
    assert np.isclose(position.x, 66.03 * arcsec)
    assert np.isclose(position.y, 77 * arcsec)


def test_get_subarray_id(configured_info):
    ids = [configured_info.get_subarray_id(i) for i in range(4)]
    assert ids == ['R0', 'R1', 'T0', 'T1']


def test_create_dark_squid_lookup(configured_info, hawc_plus_channel_data):
    info = configured_info.copy()
    channels = hawc_plus_channel_data.channels
    channels.data = hawc_plus_channel_data
    assert info.dark_squid_lookup is None
    info.create_dark_squid_lookup(channels)
    assert info.dark_squid_lookup.shape == (4, 32)


def test_initialize_channel_data(configured_info, hawc_plus_channel_data):
    data = hawc_plus_channel_data.copy()
    configured_info.initialize_channel_data(data)
    assert data.channel_id[1] == 'R0[0,1]'
