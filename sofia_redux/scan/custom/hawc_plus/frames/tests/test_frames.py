# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.table import Table
from copy import deepcopy
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.hawc_plus.frames.frames import HawcPlusFrames


degree = units.Unit('degree')
arcsec = units.Unit('arcsec')
hourangle = units.Unit('hourangle')


@pytest.fixture
def no_data_frames(no_data_scan):
    return no_data_scan[0].frames


@pytest.fixture
def scan_before_frame_read(hawc_scan_file):
    info = HawcPlusInfo()
    info.read_configuration()
    channels = info.get_channels_instance()
    scan = channels.get_scan_instance()
    scan.hdul = fits.open(hawc_scan_file)
    scan.info.parse_header(scan.hdul[0].header.copy())
    scan.channels.read_data(scan.hdul)
    scan.channels.validate_scan(scan)
    integration = scan.get_integration_instance()
    scan.integrations = [integration]
    data_hdu = scan.hdul[2]
    integration.frames.initialize(
        integration, int(data_hdu.header.get('NAXIS2', 0)))
    return scan


def test_init():
    frames = HawcPlusFrames()
    assert frames.mce_serial is None
    assert frames.hwp_angle is None
    assert frames.los is None
    assert frames.roll is None
    assert frames.status is None
    assert frames.jump_counter is None


def test_copy():
    frames = HawcPlusFrames()
    frames.roll = np.arange(10)
    frames2 = frames.copy()
    assert frames2.roll is not frames.roll
    assert np.allclose(frames2.roll, frames.roll)


def test_default_field_types():
    fields = HawcPlusFrames().default_field_types
    expected = {'mce_serial': 0,
                'hwp_angle': 0.0 * units.Unit('deg'),
                'los': 0.0 * units.Unit('deg'),
                'roll': 0.0 * units.Unit('deg'),
                'status': 0}
    for key, value in expected.items():
        assert fields[key] == value


def test_default_channel_fields():
    fields = HawcPlusFrames().default_channel_fields
    assert fields['jump_counter'] == 0


def test_readout_attributes():
    fields = HawcPlusFrames().readout_attributes
    for key in ['jump_counter', 'chopper_position', 'hwp_angle', 'mjd',
                'mce_serial']:
        assert key in fields


def test_info(no_data_scan):
    scan = no_data_scan.copy()
    frames = scan[0].frames
    assert frames.info is scan.info


def test_site(no_data_frames):
    assert isinstance(no_data_frames.site, GeodeticCoordinates)


def test_configure_hdu_columns(scan_before_frame_read):
    scan = scan_before_frame_read
    frames = scan[0].frames
    hdu = scan.hdul[2]
    table = Table(hdu.data)
    table0 = deepcopy(table)
    scan.close_fits()
    hdu = fits.BinTableHDU(table)

    frames.info.astrometry.equatorial = None
    columns = frames.configure_hdu_columns(hdu)
    expected = {
        'ts': 'Timestamp', 'sn': 'FrameCounter', 'jump': 'FluxJumps',
        'dac': 'SQ1Feedback', 'hwp': 'hwpCounts', 'stat': 'Flag', 'az': 'AZ',
        'el': 'EL', 'ra': 'RA', 'dec': 'DEC', 'ora': 'NonSiderealRA',
        'odec': 'NonSiderealDec', 'lst': 'LST', 'avpa': 'SIBS_VPA',
        'tvpa': 'TABS_VPA', 'cvpa': 'Chop_VPA', 'lon': 'LON', 'lat': 'LAT',
        'chopr': 'sofiaChopR', 'chops': 'sofiaChopS', 'pwv': 'PWV',
        'los': 'LOS', 'roll': 'ROLL'}
    for key, value in expected.items():
        assert columns[key] == value

    assert frames.info.astrometry.equatorial is not None

    table = deepcopy(table0)
    del table['Timestamp']
    del table['NonSiderealRA']
    hdu = fits.BinTableHDU(table)
    assert frames.info.astrometry.is_nonsidereal
    columns = frames.configure_hdu_columns(hdu)
    assert not frames.info.astrometry.is_nonsidereal
    for key, value in expected.items():
        if key in ['ora', 'odec', 'ts']:
            assert columns[key] is None
        else:
            assert columns[key] == value

    table = deepcopy(table0)
    hdu = fits.BinTableHDU(table)
    frames.configuration.parse_key_value('lab', 'True')
    columns = frames.configure_hdu_columns(hdu)
    for key, value in expected.items():
        if key in ['ts', 'sn', 'jump', 'dac', 'hwp']:
            assert columns[key] == value
        else:
            assert columns[key] is None


def test_read_hdus(no_data_frames, full_hdu):
    hdus = [full_hdu]
    frames = no_data_frames.copy()
    frames.configuration.parse_key_value('lab', 'True')
    frames.read_hdus(hdus)
    assert np.allclose(frames.utc, full_hdu.data['TimeStamp'])


def test_apply_hdu(no_data_frames, full_hdu):
    hdu = full_hdu.copy()
    full_table = Table(hdu.data)
    table = full_table.copy()
    frames = no_data_frames.copy()
    frames.scan.is_nonsidereal = True
    frames.configuration.parse_key_value('lab', 'True')
    frames.apply_hdu(hdu)
    assert np.allclose(frames.utc, table['Timestamp'])
    assert np.all(frames.valid)
    assert np.allclose(frames.mce_serial, table['FrameCounter'])
    frames.configuration.parse_key_value('chopper.invert', 'True')
    frames.configuration.parse_key_value('lab', 'False')
    frames.apply_hdu(hdu)
    assert np.allclose(frames.status, 0)
    assert np.allclose(frames.pwv, 6 * units.Unit('um'))
    assert np.allclose(frames.site.lon, 15 * degree)
    assert np.allclose(frames.site.lat, 20 * degree)
    assert np.allclose(frames.lst, table['LST'] * hourangle)
    assert np.allclose(frames.equatorial.ra, 180 * degree)
    assert np.allclose(frames.equatorial.dec, 30 * degree)
    assert frames.object_equatorial == frames.equatorial
    assert np.allclose(frames.instrument_vpa, 1 * degree)
    assert np.allclose(frames.telescope_vpa, 2 * degree)
    assert np.allclose(frames.chop_vpa, 3 * degree)
    assert np.allclose(frames.los, 7 * degree)
    assert np.allclose(frames.roll, 8 * degree)
    assert np.allclose(frames.sin_pa, np.sin(frames.telescope_vpa))
    assert np.all(frames.horizontal_offset.is_null())
    assert np.allclose(frames.chopper_position.x, 164.61334702 * arcsec,
                       atol=1e-6)
    assert np.allclose(frames.chopper_position.y, 136.46968403 * arcsec,
                       atol=1e-6)
    assert np.allclose(
        frames.horizontal.az,
        [22.30553518, 22.30555533, 22.30557547, 22.30559561, 22.30561576,
         22.3056359, 22.30565605, 22.30567619, 22.30569634, 22.30571648
         ] * degree, atol=1e-4)
    assert np.allclose(
        frames.horizontal.el,
        [-36.27121317, -36.27120584, -36.27119851, -36.27119118,
         -36.27118384, -36.27117651, -36.27116918, -36.27116185,
         -36.27115452, -36.27114719] * degree, atol=1e-4)

    frames.scan.is_nonsidereal = False
    del table['LAT']
    del table['LON']
    del table['LST']
    del table['sofiaChopR']
    del table['sofiaChopS']
    hdu = fits.BinTableHDU(table)
    frames.apply_hdu(hdu)
    assert np.all(frames.horizontal_offset.x < -158 * degree)
    assert np.all(frames.chopper_position.is_null())
    assert np.allclose(frames.horizontal.az, table['AZ'] * degree)
    assert np.allclose(frames.horizontal.el, table['EL'] * degree)


def test_add_data_from(valid_frames):
    f1 = valid_frames.copy()
    d1 = f1.data.copy()
    h1 = f1.hwp_angle.copy()
    f2 = valid_frames.copy()
    f1.add_data_from(f2, scaling=0)
    assert np.allclose(f1.data, d1)
    assert np.allclose(f1.hwp_angle, h1)
    f1.add_data_from(f2)
    assert np.allclose(f1.data, d1 * 2)
    assert np.allclose(f1.hwp_angle, h1 * 2)
    f1 = valid_frames.copy()
    indices = np.arange(5)
    f1.add_data_from(f2[:5], indices=indices)
    assert np.allclose(f1.data[:5], d1[:5] * 2)
    assert np.allclose(f1.data[5:], d1[5:])
    assert np.allclose(f1.hwp_angle[:5], h1[:5] * 2)
    assert np.allclose(f1.hwp_angle[5:], h1[5:])
    f1 = valid_frames[0].copy()
    f2 = f1.copy()
    f1.add_data_from(f2, indices=indices)
    assert np.allclose(f1.data, d1[0] * 2)
    assert np.isclose(f1.hwp_angle, h1[0] * 2)


def test_validate(valid_frames):
    frames = valid_frames.copy()
    frames.configuration.parse_key_value('chopped', 'True')
    frames.equatorial[0].nan()
    frames.info.detector_array.dark_squid_correction = True
    frames.channels.create_dark_squid_lookup()
    frames.info.detector_array.dark_squid_lookup = (
        frames.channels.dark_squid_lookup)
    frames.data.fill(1)
    frames.validate()
    assert not frames.valid[0] and np.all(frames.valid[1:])
    # Check the dark squid correction has been applied
    assert not np.allclose(frames.data, 1)
    del frames.configuration['chopped']
    frames.configuration.blacklist('chopped')
    frames.validate()
    assert not frames.valid[0] and np.all(frames.valid[1:])


def test_dark_correct(valid_frames):
    frames = valid_frames.copy()
    frames.channels.create_dark_squid_lookup()
    frames.info.detector_array.dark_squid_lookup = (
        frames.channels.dark_squid_lookup)
    frames.data.fill(1.0)
    frames.dark_correct()
    assert np.allclose(np.unique(frames.data), [0, 1])
    blind = frames.channels.data.is_flagged('BLIND')
    assert np.allclose(frames.data[:, blind], 1)
    assert np.allclose(frames.data[:, ~blind], 0)


def test_set_from_downsampled(valid_frames):
    frames = valid_frames.copy()
    f1 = frames[:5].copy()
    start_indices = np.arange(0, 10, 2)
    valid = f1.valid
    window = np.full(2, 1.0)
    f1.set_from_downsampled(frames, start_indices, valid, window)
    assert np.allclose(f1.hwp_angle, 4.5 * degree)
    frames.hwp_angle = frames.hwp_angle.value
    f1.set_from_downsampled(frames, start_indices, valid, window)
    assert np.allclose(f1.hwp_angle, 4.5)
