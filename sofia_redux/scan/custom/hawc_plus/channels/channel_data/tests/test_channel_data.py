# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.custom.hawc_plus.channels.channel_data.channel_data \
    import HawcPlusChannelData
from sofia_redux.scan.custom.hawc_plus.simulation.simulation import \
    HawcPlusSimulation


arcsec = units.Unit('arcsec')


@pytest.fixture
def hawc_channels():
    info = HawcPlusInfo()
    info.configuration.parse_key_value('subarray', 'R0,T0,R1')
    info.configuration.lock('subarray')
    info.read_configuration()
    header = fits.Header(HawcPlusSimulation.default_values)
    info.configuration.read_fits(header)
    info.apply_configuration()
    return info.get_channels_instance()


@pytest.fixture
def hawc_data(hawc_channels):
    return HawcPlusChannelData(channels=hawc_channels)


@pytest.fixture
def initialized_data(hawc_data):
    data = hawc_data.copy()
    data.configuration.parse_key_value('pixelsize', '5.0')
    data.info.detector_array.load_detector_configuration()
    data.info.detector_array.initialize_channel_data(data)
    data.info.detector_array.set_boresight()
    data.channels.subarray_gain_renorm = np.full(4, 1.0)
    return data


@pytest.fixture
def positioned_data(initialized_data):
    data = initialized_data.copy()
    data.calculate_sibs_position()
    center = data.info.detector_array.get_sibs_position(
        sub=0,
        row=39 - data.info.detector_array.boresight_index.y,
        col=data.info.detector_array.boresight_index.x)
    data.position.subtract(center)
    return data


def test_init(hawc_channels):
    data = HawcPlusChannelData(channels=hawc_channels)
    assert data.channels is hawc_channels
    for attr in ['sub', 'pol', 'subrow', 'bias_line', 'series_array', 'mux',
                 'fits_row', 'fits_col', 'fits_index', 'jump', 'has_jumps',
                 'sub_gain', 'mux_gain', 'pin_gain', 'bias_gain',
                 'series_gain', 'los_gain', 'roll_gain']:
        assert hasattr(data, attr)
        assert getattr(data, attr) is None


def test_default_field_types(hawc_data):
    data = hawc_data
    expected = {'jump': 0.0, 'has_jumps': False, 'sub_gain': 1.0,
                'mux_gain': 1.0, 'pin_gain': 1.0, 'bias_gain': 1.0,
                'series_gain': 1.0, 'los_gain': 1.0, 'roll_gain': 1.0}
    defaults = data.default_field_types
    for key, value in expected.items():
        assert defaults[key] == value


def test_info(hawc_data):
    data = hawc_data
    assert data.info is data.channels.info
    assert isinstance(data.info, HawcPlusInfo)


def test_calculate_sibs_position(initialized_data):
    data = initialized_data.copy()
    data.position.nan()
    data.calculate_sibs_position()
    ok = data.is_unflagged('BLIND')
    assert not np.any(data.position[ok].is_nan())
    assert np.all(data.position[~ok].is_nan())
    # Sporadic coordinate check
    rand = np.random.RandomState(2)
    idx = rand.randint(0, data.size - 1, 10)
    assert np.allclose(
        data.position[idx].coordinates,
        [[41.19866716, 260.15, 270.15, 295.15, 112.14788818, 280.15, 245.15,
          55, 295.15, 36.46121478],
         [-97.93286577, -195, -190, -45, -44.16291099, -160, -185, -25,
          -170, -82.84788831]] * arcsec,
        atol=1e-3)


def test_set_uniform_gains(positioned_data):
    data = positioned_data.copy()
    data.gain.fill(np.nan)
    data.coupling.fill(np.nan)
    data.mux_gain.fill(np.nan)
    data.set_uniform_gains()
    assert np.allclose(data.gain, 1)
    assert np.allclose(data.coupling, 1)
    assert np.allclose(data.mux_gain, 1)
    data.mux_gain.fill(np.nan)
    data.gain.fill(np.nan)
    data.set_uniform_gains(field='mux_gain')
    assert np.all(np.isnan(data.gain))
    assert np.allclose(data.mux_gain, 1)


def test_to_string(positioned_data):
    data = positioned_data.copy()
    df = data.to_string(frame=True)
    assert data.to_string(frame=False) == df.to_csv(sep='\t', index=False)
    row = dict(df.iloc[2000])
    expected = {'ch': 'R1[21,16]',
                'gain': '1.000',
                'weight': '1.000e+00',
                'flag': '-',
                'eff': '1.000',
                'Gsub': '1.000',
                'Gmux': '1.000',
                'idx': '2000',
                'sub': '1',
                'row': '21',
                'col': '16'}

    for key, value in expected.items():
        assert row[key] == value


def test_validate_pixel_data(positioned_data):
    data = positioned_data.copy()
    data.gain.fill(0.5)
    data.coupling.fill(0.5)
    data.gain[0] = 10.0
    data.gain[1] = 0.1
    data.gain[2] = 0.6
    data.coupling[3] = 1.0
    data.configuration.parse_key_value('pixels.gain.range', '0.3:3')
    data.configuration.parse_key_value('pixels.coupling.range', '0.3:3')
    data.configuration.parse_key_value('pixels.gain.exclude', '0.6')
    data.configuration.parse_key_value('pixels.coupling.exclude', '1')
    data.validate_pixel_data()
    assert np.allclose(data.flag[:4], data.flagspace.flags.DEAD.value)
    assert np.allclose(data.flag[4:10], 0)


def test_initialize_from_detector(positioned_data):
    data = positioned_data.copy()
    for attr in ['col', 'row', 'sub', 'pol', 'fits_row', 'bias_line', 'mux',
                 'fits_col', 'series_array', 'fits_index', 'flag']:
        getattr(data, attr).fill(-1)
    data.channel_id.fill('')

    data.initialize_from_detector(data.info.detector_array)
    assert data.size == 3936
    assert np.allclose(data.col[:10], np.arange(10))
    assert np.allclose(np.unique(data.col), np.arange(32))
    assert np.allclose(data.row[:32], 0) and data.row[32] == 1
    assert np.allclose(data.sub[:1312], 0)
    assert np.allclose(data.sub[1312:2624], 1)
    assert np.allclose(data.sub[2624:], 2)
    assert np.allclose(data.pol, data.sub // 2)
    assert np.allclose(data.mux, (data.sub * 32) + data.col)
    assert np.allclose(data.fits_col, data.mux)
    assert np.allclose(data.series_array, data.mux // 4)
    assert np.allclose(data.fits_index, (data.fits_row * 128) + data.fits_col)
    ok = data.subrow != 40
    assert np.allclose(data.flag[ok], 0)
    assert not np.any(data.flag[~ok] == 0)
    assert data.channel_id[2000] == 'R1[21,16]'


def test_apply_info(positioned_data):
    data = positioned_data.copy()
    expected = data.position.copy()
    data.position.nan()
    data.apply_info(data.info)
    idx = np.isfinite(expected.x)
    assert data.position[idx] == expected[idx]


def test_set_sibs_position(positioned_data):
    data = positioned_data.copy()
    blind = data.flagspace.flags.BLIND.value
    data.flag[0] = blind
    data.set_sibs_positions(data.info.detector_array)
    assert data.position[0].is_nan()
    assert not data.position[1].is_nan()


def test_set_reference_position(positioned_data):
    data = positioned_data.copy()
    x, y = data.position.coordinates.copy()
    dc = Coordinate2D([1, 2], unit='arcsec')
    data.set_reference_position(dc)
    assert np.allclose(data.position.x, x - 1 * arcsec, equal_nan=True)
    assert np.allclose(data.position.y, y - 2 * arcsec, equal_nan=True)


def test_read_channel_data_file(tmpdir):
    filename = str(tmpdir.mkdir('test_read_channel_data_file').join('pix.dat'))
    header = '# gain wt flag coupling subg muxg id sub subrow col'
    line1 = '   0.5   1    -     0.6   0.7  0.8  1   0      1   2'
    line2 = '   0.6   2    g     1.6   1.7  1.8  2   1      10 11'
    with open(filename, 'w') as f:
        for line in [header, line1, line2]:
            print(line, file=f)

    info = HawcPlusChannelData.read_channel_data_file(filename)
    assert info[0] == {'gain': 0.5, 'weight': 1.0, 'flag': 0,
                       'coupling': 0.6, 'sub_gain': 0.7, 'mux_gain': 0.8,
                       'fixed_id': 1, 'sub': 0, 'subrow': 1, 'col': 2}
    assert info[1] == {'gain': 0.6, 'weight': 2.0, 'flag': 8, 'coupling': 1.6,
                       'sub_gain': 1.7, 'mux_gain': 1.8, 'fixed_id': 2,
                       'sub': 1, 'subrow': 10, 'col': 11}


def test_set_channel_data(positioned_data):
    data = positioned_data.copy()
    channel_info = {'gain': 0.6, 'weight': 2.0, 'flag': 8, 'coupling': 1.6,
                    'sub_gain': 1.7, 'mux_gain': 1.8, 'fixed_id': 2,
                    'sub': 1, 'subrow': 10, 'col': 11}
    data.set_channel_data(1, channel_info)
    assert data.gain[1] == 0.6
    assert data.weight[1] == 2.0
    assert data.flag[1] == 8
    assert data.coupling[1] == 1.6
    assert data.sub_gain[1] == 1  # Set dynamically
    assert data.mux_gain[1] == 1.8
    assert data.fixed_index[1] == 1  # Set programmatically
    assert data.sub[1] == 0  # Set programmatically
    assert data.subrow[1] == 0  # Set programmatically
    assert data.col[1] == 1  # Set programmatically


def test_geometric_rows(initialized_data):
    assert initialized_data.geometric_rows() == 41


def test_geometric_cols(initialized_data):
    assert initialized_data.geometric_cols() == 32


def test_get_geometric_overlap_indices(positioned_data):
    data = positioned_data.copy()
    overlaps = data.get_geometric_overlap_indices(11 * arcsec)
    pixel_index = 1000
    overlapping_indices = overlaps[pixel_index].nonzero()[1]
    # Only check the same subarray, as others are far offset
    pixel = data[pixel_index]
    overlapping = data[overlapping_indices]
    same_sub = overlapping.sub == pixel.sub
    assert pixel.sub == 0

    offset = overlapping[same_sub].position.copy()
    offset.subtract(pixel.position)
    distance = offset.length
    assert np.isclose(distance.max(), 10 * arcsec)

    # Check the other arrays (offsets should range from +-10 arcsec
    for sub in [1, 2]:
        span = overlapping[overlapping.sub == sub].position.span
        assert np.isclose(span.x, 20 * arcsec, atol=0.1)
        assert np.isclose(span.y, 20 * arcsec, atol=0.1)


def test_read_jump_hdu(initialized_data):
    data = initialized_data.copy()
    data.col = np.arange(10)
    data.row = np.zeros(10, dtype=int)

    jump = np.zeros((10, 10), dtype=int)
    jump[:, 0] = np.arange(30, 40)
    hdu = fits.PrimaryHDU(data=jump)
    data.read_jump_hdu(hdu)
    assert np.allclose(data.jump, np.arange(30, 40))
