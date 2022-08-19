# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.fifi_ls.info.detector_array import \
    FifiLsDetectorArrayInfo
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
mm = units.Unit('mm')
um = units.Unit('um')


class DummyChannels(object):
    def __init__(self):
        self.fixed_index = None
        self.spexel = None
        self.spaxel = None
        self.flag = None
        self.col = None
        self.row = None
        self.channel_id = None
        self.default_set = False
        self.position = None
        self.wavelength = None

    def set_default_values(self):
        self.default_set = True


@pytest.fixture
def fifi_header():
    h = fits.Header()
    h['PLATSCAL'] = 4.5
    h['DATE-OBS'] = '2014-02-01T12:00:00'
    h['OBSLAM'] = 12.1
    h['OBSBET'] = 20.1
    h['SKY_ANGL'] = 0.0
    h['DET_ANGL'] = 70.0
    h['DLAM_MAP'] = 1.0
    h['DBET_MAP'] = 2.0
    h['CHANNEL'] = 'BLUE'
    h['PRIMARAY'] = 'Red'
    h['DICHROIC'] = 105
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
    info = FifiLsDetectorArrayInfo()
    info.configuration = fifi_configuration
    return info


@pytest.fixture
def initialized_fifi_info(fifi_info):
    info = fifi_info
    info.apply_configuration()
    info.calculate_pixel_offsets()
    info.boresight_equatorial = EquatorialCoordinates([10, 20])
    return info


def test_class():
    det = FifiLsDetectorArrayInfo
    assert det.tel_sim_to_detector == 0.842
    assert det.n_spaxel == 25
    assert det.n_spexel == 16
    assert det.pixels == 400
    assert det.spaxel_rows == 5
    assert det.spaxel_cols == 5
    assert det.default_boresight_offset.x == 0
    assert det.default_boresight_offset.y == 0


def test_init():
    info = FifiLsDetectorArrayInfo()
    assert np.isnan(info.plate_scale) and info.plate_scale.unit == 'arcsec/mm'
    assert info.pixel_sizes.is_nan() and info.pixel_sizes.unit == arcsec
    assert info.obs_equatorial.is_nan() and info.obs_equatorial.unit == degree
    assert info.delta_map.is_null() and info.delta_map.unit == arcsec
    assert np.isclose(info.sky_angle, np.nan * degree, equal_nan=True)
    assert np.isclose(info.detector_angle, np.nan * degree, equal_nan=True)
    assert info.spatial_file is None
    assert info.pixel_positions is None
    assert info.pixel_offsets is None
    assert info.channel == 'UNKNOWN'
    assert info.prime_array == 'UNKNOWN'
    assert np.isclose(info.dichroic, np.nan * um, equal_nan=True)
    assert info.coefficients_1 is None
    assert info.coefficients_2 is None


def test_ch():
    info = FifiLsDetectorArrayInfo()
    info.channel = 'RED'
    assert info.ch == 'r'
    info.channel = 'BLUE'
    assert info.ch == 'b'


def test_int_date():
    info = FifiLsDetectorArrayInfo()
    assert info.int_date == -1
    info.date_obs = Time('2022-03-02')
    assert info.int_date == 20220302


def test_apply_configuration(fifi_info):
    info = FifiLsDetectorArrayInfo()
    info.apply_configuration()
    assert info.obs_equatorial.is_nan()
    info = fifi_info.copy()
    info.apply_configuration()
    assert info.plate_scale == 4.5 * arcsec / mm
    assert info.date_obs.isot == '2014-02-01T12:00:00.000'
    assert info.obs_equatorial.ra == 12.1 * degree
    assert info.obs_equatorial.dec == 20.1 * degree
    assert info.delta_map.x == 1 * arcsec
    assert info.delta_map.y == 2 * arcsec
    assert info.channel == 'BLUE'
    assert info.pixel_size == 6.75 * arcsec
    assert np.allclose(info.pixel_sizes.coordinates, info.pixel_size)
    assert info.dichroic == 105 * um
    assert info.prime_array == 'RED'
    assert info.pixel_offsets is not None
    assert info.coefficients_1 is not None
    assert info.coefficients_2 is not None
    assert info.sky_angle == 0 * degree

    info = fifi_info.copy()
    info.configuration.fits.options['CHANNEL'] = 'RED'
    info.configuration.fits.options['DATE-OBS'] = Time.now().isot
    info.configuration.fits.options['SKY_ANGL'] = 45.0
    info.apply_configuration()
    assert info.delta_map.x == 1 * arcsec  # Coordinates not reversed
    assert info.delta_map.y == 2 * arcsec
    assert info.channel == 'RED'
    assert info.coefficients_1 is not None
    assert info.coefficients_2 is None  # N/A if channel is the prime array
    assert info.sky_angle == 45 * degree


def test_calculate_pixel_offsets(fifi_info):
    info = fifi_info.copy()
    info.channel = 'BLUE'
    info.configuration.instrument_name = 'foo'
    with pytest.raises(ValueError) as err:
        info.calculate_pixel_offsets()
    assert 'Could not locate default date' in str(err.value)

    info.configuration.instrument_name = 'fifi_ls'
    date_obs = info.options['DATE-OBS']
    info.date_obs = Time(date_obs)
    info.plate_scale = 2 * arcsec / mm
    info.calculate_pixel_offsets()
    assert info.pixel_positions is not None and info.pixel_positions.unit == mm
    assert info.pixel_positions.size == 25
    assert info.pixel_offsets is not None and info.pixel_offsets.unit == arcsec
    assert info.pixel_offsets.size == 25
    assert np.allclose(info.pixel_offsets.coordinates.value,
                       info.pixel_positions.coordinates.value * 2)


def test_get_boresight_equatorial(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    x = np.zeros((5, info.pixels)) * arcsec
    y = np.zeros((5, info.pixels)) * arcsec

    equatorial = info.get_boresight_equatorial(x, y)
    assert equatorial.shape == (5,)
    assert np.allclose(equatorial.ra, 10 * degree)
    assert np.allclose(equatorial.dec, 20 * degree)


def test_get_boresight_trajectory(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    xs = np.zeros((1, 1, 1, 1))
    ys = xs
    with pytest.raises(ValueError) as err:
        info.get_boresight_trajectory(xs, ys)
    assert 'Incorrect xs and ys input shape' in str(err.value)

    # 3-D
    xs = np.zeros((5, 2, 30)) * arcsec
    xs[:, 0, info.center_spaxel] = 1 * arcsec
    ys = xs.copy()
    trajectory = info.get_boresight_trajectory(xs, ys)
    assert trajectory.shape == (5,)
    assert np.allclose(trajectory.coordinates, 1 * arcsec)

    # Single value 2-D
    xs2 = xs[:, 0]
    ys2 = ys[:, 0]
    trajectory = info.get_boresight_trajectory(xs2, ys2)
    assert trajectory.shape == ()
    assert np.allclose(trajectory.coordinates, 1 * arcsec)

    # Multi value 2-D
    xs = np.zeros((5, info.pixels)) * arcsec
    xs[:, info.center_spaxel] = 1 * arcsec
    ys = xs.copy()
    trajectory = info.get_boresight_trajectory(xs, ys)
    assert trajectory.shape == (5,)
    assert np.allclose(trajectory.coordinates, 1 * arcsec)

    # Single value 1-D
    xs = np.zeros(info.pixels) * arcsec
    xs[info.center_spaxel] = 2 * arcsec
    ys = xs.copy()
    trajectory = info.get_boresight_trajectory(xs, ys)
    assert trajectory.shape == ()
    assert np.allclose(trajectory.coordinates, 2 * arcsec)

    info.rotation = 30 * degree
    info.flip_sign = True
    info.pixel_offsets[info.center_spaxel] = Coordinate2D([1, 1], unit=arcsec)
    trajectory = info.get_boresight_trajectory(xs, ys)
    assert np.isclose(trajectory.x, 1.6339746 * arcsec, atol=1e-3)
    assert np.isclose(trajectory.y, 0.6339746 * arcsec, atol=1e-3)

    info.flip_sign = False
    trajectory = info.get_boresight_trajectory(xs, ys)
    assert np.isclose(trajectory.x, 2.3660254 * arcsec, atol=1e-3)
    assert np.isclose(trajectory.y, 3.3660254 * arcsec, atol=1e-3)


def test_initialize_channel_data(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    data = DummyChannels()
    info.initialize_channel_data(data)
    assert np.allclose(data.fixed_index, np.arange(400))
    assert data.default_set
    fi = data.fixed_index.copy()
    assert np.allclose(data.spexel, fi // 25)
    assert np.allclose(data.spaxel, fi % 25)
    assert np.allclose(data.flag, 0)
    assert np.allclose(data.col, (fi // 5) % 5)
    assert np.allclose(data.row, fi % 5)
    assert data.channel_id[123] == 'B[4,23]'


def test_detector_coordinates_to_equatorial_offsets(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    info.sky_angle = 0 * degree
    offset = Coordinate2D([1, 1], unit='arcsec')
    equatorial_offset = info.detector_coordinates_to_equatorial_offsets(offset)
    assert equatorial_offset.x == -1 * arcsec
    assert equatorial_offset.y == 1 * arcsec
    info.sky_angle = 30 * degree
    equatorial_offset = info.detector_coordinates_to_equatorial_offsets(offset)
    assert np.isclose(equatorial_offset.x, -0.3660254 * arcsec, atol=1e-3)
    assert np.isclose(equatorial_offset.y, 1.3660254 * arcsec, atol=1e-3)


def test_equatorial_offsets_to_detector_coordinates(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    info.sky_angle = 0 * degree
    offset = Coordinate2D([1, 1], unit='arcsec')
    coordinates = info.equatorial_offsets_to_detector_coordinates(offset)
    assert coordinates.x == -1 * arcsec
    assert coordinates.y == 1 * arcsec
    info.sky_angle = 30 * degree
    coordinates = info.equatorial_offsets_to_detector_coordinates(coordinates)
    assert np.isclose(coordinates.x, 1.3660254 * arcsec, atol=1e-3)
    assert np.isclose(coordinates.y, 0.3660254 * arcsec, atol=1e-3)


def test_detector_coordinates_to_equatorial(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    offset = Coordinate2D([60, 60], unit='arcsec')
    equatorial = info.detector_coordinates_to_equatorial(offset)
    assert np.isclose(equatorial.ra, 9.9822637 * degree, atol=1e-3)
    assert np.isclose(equatorial.dec, 20.01666667 * degree, atol=1e-3)


def test_equatorial_to_detector_coordinates(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    equatorial = EquatorialCoordinates([10, 20])
    coordinates = info.equatorial_to_detector_coordinates(equatorial)
    assert coordinates.is_null()


def test_calculate_delta_coefficients(fifi_info):
    info = fifi_info.copy()
    info.date_obs = Time(info.options['DATE-OBS'])
    info.dichroic = -1 * um

    info.configuration.instrument_name = 'foo'
    with pytest.raises(ValueError) as err:
        info.calculate_delta_coefficients()
    assert 'Could not find file' in str(err.value)

    info.configuration.instrument_name = 'fifi_ls'
    with pytest.raises(ValueError) as err:
        info.calculate_delta_coefficients()
    assert "No boresight offsets" in str(err.value)

    info.dichroic = 105 * um
    info.channel = 'BLUE'
    info.prime_array = 'RED'
    info.calculate_delta_coefficients()
    assert np.allclose(info.coefficients_1,
                       [[-4.2640564e-08, -6.9454054e-01, -7.2254000e-01],
                        [-8.4200000e-09, 6.7123398e-01, -6.8383000e-01]],
                       rtol=1e-3)
    assert np.allclose(info.coefficients_2,
                       [[4.6713318e-07, 8.0832000e-01, 1.3700000e+00],
                        [-7.8268110e-09, 5.2920542e-01, -5.5000000e-01]],
                       rtol=1e-3)
    expected = info.coefficients_2.copy()
    info.prime_array = 'BLUE'
    info.calculate_delta_coefficients()
    assert np.allclose(info.coefficients_1, expected)
    assert info.coefficients_2 is None


def test_find_pixel_positions(initialized_fifi_info):
    info = initialized_fifi_info.copy()

    x, y = np.meshgrid(np.arange(25), np.arange(16))
    x = x[None] + np.arange(1, 4)[:, None, None] * 0.25
    y = y[None] + np.arange(1, 4)[:, None, None] * 0.25
    detector_xy = Coordinate2D([x, y], unit='arcsec')
    positions = info.find_pixel_positions(detector_xy)
    assert positions.shape == (400,)
    assert np.isclose(positions[0].x, -4.104242 * arcsec, atol=1e-3)
    assert np.isclose(positions[0].y, 11.276311 * arcsec, atol=1e-3)
    assert np.isclose(positions[100].x, -0.345471 * arcsec, atol=1e-3)
    assert np.isclose(positions[100].y, 12.644392 * arcsec, atol=1e-3)
    info.rotation = np.pi / 2
    positions = info.find_pixel_positions(detector_xy)
    assert np.isclose(positions[100].x, 4 * arcsec)
    assert np.isclose(positions[100].y, 12 * arcsec)


def test_edit_header(initialized_fifi_info):
    info = initialized_fifi_info.copy()
    h = fits.Header()
    info.sky_angle = 30 * degree
    info.edit_header(h)
    assert h['CTYPE3'] == 'WAVE'
    assert h['CROTA2'] == -30
    assert h['SPECSYS'] == 'BARYCENT'
