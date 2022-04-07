# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.hawc_plus.simulation.simulation import \
    HawcPlusSimulation
from sofia_redux.scan.custom.hawc_plus.channels.channels import \
    HawcPlusChannels
from sofia_redux.toolkit.utilities.multiprocessing import in_windows_os

arcsec = units.Unit('arcsec')


@pytest.fixture
def jump_file_ones(tmpdir):
    filename = str(tmpdir.mkdir('fake_jump_data').join('jump.dat'))
    data = np.ones((32, 123), dtype=int)
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU(data=data))
    hdul.writeto(filename)
    return filename


@pytest.fixture
def hawc_plus_channels(jump_file_ones):
    info = HawcPlusInfo()
    info.read_configuration()
    h = HawcPlusSimulation.default_values.copy()
    h['SPECTEL1'] = 'HAW_A'
    h['SPECTEL2'] = 'HAW_HWP_Open'
    h['WAVECENT'] = 53.0
    header = fits.Header(h)
    info.configuration.parse_key_value('subarray', 'R0,T0,R1')
    info.configuration.parse_key_value('jumpdata', jump_file_ones)
    info.configuration.lock('subarray')
    info.configuration.lock('jumpdata')
    info.configuration.read_fits(header)
    info.apply_configuration()
    channels = info.get_channels_instance()
    channels.load_channel_data()
    return channels


@pytest.fixture
def initialized_channels(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.initialize()
    channels.normalize_array_gains()
    return channels


def test_init():
    info = HawcPlusInfo()
    parent = np.arange(10)
    channels = HawcPlusChannels(info=info, parent=parent, size=11, name='foo')
    assert channels.info is info
    assert channels.parent is parent
    assert channels.n_store_channels == 11
    assert channels.info.name == 'foo'
    assert channels.subarray_gain_renorm is None
    assert channels.subarray_groups is None


def test_copy(hawc_plus_channels):
    channels = hawc_plus_channels
    channels2 = channels.copy()
    assert channels2 is not channels
    assert isinstance(channels2, HawcPlusChannels)
    assert np.allclose(channels2.data.row, channels.data.row)


def test_detector(hawc_plus_channels):
    assert (hawc_plus_channels.detector is
            hawc_plus_channels.info.detector_array)


def test_band_id(hawc_plus_channels):
    assert hawc_plus_channels.band_id == 'A'


def test_pixel_sizes(hawc_plus_channels):
    assert hawc_plus_channels.pixel_sizes == Coordinate2D(
        [2.57, 2.57], unit='arcsec')


def test_dark_squid_lookup(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.info.detector_array.dark_squid_lookup = 2
    assert channels.dark_squid_lookup == 2


def test_init_groups(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.init_groups()
    for key in ['R0', 'R1', 'T0', 'T1']:
        assert key in channels.groups


def test_init_divisions(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.init_groups()

    dead_flag = channels.flagspace.flags.DEAD
    blind_flag = channels.flagspace.flags.BLIND
    channels.data.flag[0] = dead_flag.value
    channels.data.flag[1] = blind_flag.value
    channels.data.flag[2] = 0

    channels.configuration.parse_key_value('darkcorrect', 'True')
    channels.init_divisions()
    for key in ['polarrays', 'subarrays', 'bias', 'series', 'rows']:
        assert key in channels.divisions

    group = channels.divisions['mux'].groups[1]
    assert group.name == 'mux-1'
    assert 1 in group.indices

    channels.configuration.parse_key_value('darkcorrect', 'False')
    channels.init_divisions()
    group = channels.divisions['mux'].groups[1]
    assert group.name == 'mux-1'
    assert 1 not in group.indices


def test_init_modalities(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.init_groups()
    channels.init_divisions()
    channels.init_modalities()

    for modality in ['subarrays', 'bias', 'series', 'mux', 'rows', 'los',
                     'roll']:
        assert modality in channels.modalities


def test_load_channel_data(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.data.jump.fill(0)
    channels.data.position.nan()
    channels.load_channel_data()
    assert np.allclose(channels.data.jump, 1)
    assert not np.all(channels.data.position.is_nan())


def test_read_jump_levels(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.data.jump.fill(0)
    channels.read_jump_levels(None)
    assert np.allclose(channels.data.jump, 0)
    channels.read_jump_levels(channels.configuration['jumpdata'])
    assert np.allclose(channels.data.jump, 1)


def test_normalize_array_gains(initialized_channels):
    channels = initialized_channels.copy()
    for sub in range(3):
        idx = channels.data.sub == sub
        channels.data.gain[idx] = (sub + 1.0) / 2
    average = channels.normalize_array_gains()
    assert average == 1
    assert np.allclose(channels.subarray_gain_renorm, [0.5, 1, 1.5, np.nan],
                       equal_nan=True)


def test_test_nominal_pixel_positions(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.data.position.nan()
    channels.set_nominal_pixel_positions(Coordinate2D([1, 1], unit='arcsec'))
    assert np.allclose(channels.data.position[1].coordinates,
                       [-14.5, -19.5] * arcsec)


def test_max_pixels(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.n_store_channels = 100
    assert channels.max_pixels() == 100


def test_read_data(hawc_plus_channels, capsys):
    channels = hawc_plus_channels.copy()
    hdul = fits.HDUList()
    hdu = fits.BinTableHDU()
    hdu.header['EXTNAME'] = 'Configuration'
    hdul.append(hdu)
    channels.read_data(hdul)
    assert 'Missing TES bias values' in capsys.readouterr().err


def test_validate_scan(hawc_plus_channels):
    channels = hawc_plus_channels.copy()
    channels.configuration.parse_key_value('darkcorrect', 'False')
    channels.configuration.purge('filter')
    assert 'correlated.polarrays' not in channels.configuration.blacklisted
    channels.data.row.fill(-1)
    channels.validate_scan(None)
    assert not np.allclose(channels.data.row, -1)
    assert 'correlated.polarrays' not in channels.configuration.blacklisted
    assert channels.configuration['filter'] == '53um'


def test_slim(initialized_channels):
    channels = initialized_channels.copy()
    channels.data.flag.fill(0)
    dead = channels.flagspace.convert_flag('DEAD').value
    blind = channels.flagspace.convert_flag('BLIND').value
    channels.data.flag[:10] = dead
    channels.data.flag[10:20] = blind
    size1 = channels.size
    slimmed = channels.slim()
    assert slimmed
    assert size1 - channels.size == 10
    assert np.allclose(channels.dark_squid_lookup[0, :10], -1)
    assert np.allclose(channels.dark_squid_lookup[0, 10:20],
                       np.arange(10, 20))


def test_create_dark_squid_lookup(initialized_channels):
    channels = initialized_channels.copy()
    channels.data.flag.fill(0)
    blind = channels.flagspace.convert_flag('BLIND').value
    channels.data.flag[:10] = blind
    channels.data.flag[-10:] = blind
    channels.create_dark_squid_lookup()
    expected = np.full((4, 32), -1)
    expected[0, :10] = np.arange(10)
    expected[2, -10:] = np.arange(3926, 3936)
    assert np.allclose(channels.dark_squid_lookup, expected)


def test_get_si_pixel_size(hawc_plus_channels):
    assert hawc_plus_channels.get_si_pixel_size() == Coordinate2D(
        [2.57, 2.57], unit='arcsec')


@pytest.mark.skipif(in_windows_os(), reason='Path differences')
def test_write_flat_field(initialized_channels, tmpdir):
    channels = initialized_channels.copy()
    path = tmpdir.mkdir('test_write_flat_field')
    regular = str(path.join('regular.fits'))
    nonlinear = str(path.join('nonlinear.fits'))
    channels.write_flat_field(regular, include_nonlinear=False)
    channels.write_flat_field(nonlinear, include_nonlinear=True)

    hdul = fits.open(regular)
    extnames = [hdu.header['EXTNAME'] for hdu in hdul]
    assert extnames == ['R ARRAY GAIN', 'T ARRAY GAIN', 'R BAD PIXEL MASK',
                        'T BAD PIXEL MASK']
    for hdu in hdul:
        assert hdu.data.shape == (41, 64)

    assert not np.allclose(hdul['T ARRAY GAIN'].data, 1)
    assert not np.allclose(hdul['T BAD PIXEL MASK'].data, 2)

    hdul.close()

    hdul = fits.open(nonlinear)
    extnames = [hdu.header['EXTNAME'] for hdu in hdul]
    assert extnames == ['R ARRAY GAIN', 'T ARRAY GAIN', 'R BAD PIXEL MASK',
                        'T BAD PIXEL MASK', 'R ARRAY NONLINEARITY',
                        'T ARRAY NONLINEARITY']
    for hdu in hdul:
        assert hdu.data.shape == (41, 64)
    hdul.close()

    channels.data.pol.fill(0)
    channels.write_flat_field(regular, include_nonlinear=False)
    hdul = fits.open(regular)
    assert np.allclose(hdul['T ARRAY GAIN'].data, 1)
    assert np.allclose(hdul['T BAD PIXEL MASK'].data, 2)


def test_add_hdu(initialized_channels):
    channels = initialized_channels.copy()
    hdul = fits.HDUList()
    hdu = fits.PrimaryHDU()
    channels.add_hdu(hdul, hdu, 'FOO')

    hdu = hdul[0]
    assert hdu.header['EXTNAME'] == 'FOO'
    assert hdu.header['INSTRUME'] == 'HAWC_PLUS'


def test_get_table_entry(initialized_channels):
    channels = initialized_channels
    assert channels.get_table_entry('band') == 'A'
    assert channels.get_table_entry('foo') is None
