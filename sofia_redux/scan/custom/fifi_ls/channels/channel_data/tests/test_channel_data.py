# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.fifi_ls.info.info import FifiLsInfo
from sofia_redux.scan.custom.fifi_ls.channels.channel_data.channel_data \
    import FifiLsChannelData
from sofia_redux.scan.reduction.reduction import Reduction


arcsec = units.Unit('arcsec')
um = units.Unit('um')


@pytest.fixture
def full_data(fifi_simulated_hdul):
    reduction = Reduction('fifi_ls')
    reduction.read_scans([fifi_simulated_hdul])
    return reduction.scans[0][0].channels.data


def test_init(fifi_channels):
    data = FifiLsChannelData(channels=fifi_channels)
    for attr in ['spexel', 'spaxel', 'wavelength', 'uncorrected_wavelength',
                 'response', 'atran', 'spexel_gain', 'spaxel_gain',
                 'col_gain', 'row_gain']:
        assert getattr(data, attr) is None
    assert data.channels is fifi_channels


def test_copy(fifi_channel_data):
    data = fifi_channel_data.copy()
    assert isinstance(data, FifiLsChannelData)
    assert data is not fifi_channel_data


def test_default_field_types():
    data = FifiLsChannelData()
    fields = data.default_field_types
    assert fields['spexel'] == -1
    assert fields['spaxel'] == -1
    for key in ['spexel_gain', 'spaxel_gain', 'col_gain', 'row_gain']:
        assert fields[key] == 1.0


def test_info(fifi_channel_data):
    assert FifiLsChannelData().info is None
    assert isinstance(fifi_channel_data.info, FifiLsInfo)


def test_central_wavelength():
    data = FifiLsChannelData()
    assert np.isclose(data.central_wavelength,
                      np.nan * um, equal_nan=True)

    data.wavelength = np.arange(10) * um
    assert np.isclose(data.central_wavelength,
                      np.nan * um, equal_nan=True)

    data.fixed_index = np.arange(10)  # sets size
    assert data.central_wavelength == 4.5 * um


def test_spectral_fwhm(fifi_channel_data):
    data = fifi_channel_data.copy()
    data.wavelength = np.arange(10) * um
    data.fixed_index = np.arange(10)
    assert np.isclose(data.spectral_fwhm, 0.0048913 * um, atol=1e-5)


def test_set_uniform_gains(fifi_channel_data):
    data = fifi_channel_data.copy()
    data.fixed_index = np.arange(10)
    data.set_uniform_gains()
    expected = np.ones(10)
    assert np.allclose(data.spexel_gain, expected)
    assert np.allclose(data.spaxel_gain, expected)
    assert np.allclose(data.col_gain, expected)
    assert np.allclose(data.row_gain, expected)


def test_to_string(fifi_initialized_channel_data):
    data = fifi_initialized_channel_data.copy()
    s = data.to_string()
    lines = s.split('\n')
    assert lines[0].startswith(
        'ch\tgain\tweight\tflag\teff\tGspex\tGspax\tidx\tspex\tspax')
    assert lines[1].startswith(
        'B[0,0]\t1.000\t1.000e+00\t-\t1.000\t1.000\t1.000\t0\t0\t0')
    df = data.to_string(frame=True)
    assert df.columns.to_list() == ['ch', 'gain', 'weight', 'flag', 'eff',
                                    'Gspex', 'Gspax', 'idx', 'spex', 'spax']


def test_validate_pixel_data(fifi_initialized_channel_data):
    data = fifi_initialized_channel_data.copy()
    gain = data.gain
    coupling = data.coupling
    gain[:2] = 4
    coupling[2:4] = 5
    coupling[4] = 1.5
    gain[5] = 2.5
    data.configuration.parse_key_value('pixels.gain.range', '0.3,3.0')
    data.configuration.parse_key_value('pixels.gain.exclude', '0.3,3.0')
    data.configuration.parse_key_value('pixels.gain.exclude', '2.5')
    data.configuration.parse_key_value('pixels.coupling.exclude', '1.5')
    data.coupling = coupling
    data.gain = gain
    data.validate_pixel_data()
    assert np.allclose(data.flag[:6], 1)
    assert np.allclose(data.flag[6:], 0)
    assert np.allclose(data.coupling[:6], 0)
    assert np.allclose(data.coupling[6:], 1)


def test_initialize_from_detector(fifi_channel_data):
    data = fifi_channel_data.copy()
    assert data.size == 0
    detector = data.info.detector_array
    data.initialize_from_detector(detector)
    assert data.size == 400


def test_read_channel_data_file(tmpdir):
    filename = str(tmpdir.mkdir('test_read_channel_data_file').join(
        'channel_file.dat'))
    column_names = ['gain', 'weight', 'flag', 'coupling', 'spexel_gain',
                    'spaxel_gain', 'row_gain', 'col_gain', 'fixed_id',
                    'spexel', 'spaxel']
    contents = '#' + ' '.join(column_names) + '\n'
    contents += ' '.join(list(map(str, range(1, len(column_names) + 1))))
    with open(filename, 'w') as f:
        f.write(contents)
    info = FifiLsChannelData.read_channel_data_file(filename)[0]
    for i, key in enumerate(column_names):
        if key == 'flag':
            assert info[key] == 0
        else:
            assert info[key] == i + 1


def test_set_channel_data(fifi_initialized_channel_data):
    data = fifi_initialized_channel_data.copy()
    data.set_channel_data(0, None)
    assert data.coupling[0] == 1

    column_names = ['gain', 'weight', 'flag', 'coupling', 'spexel_gain',
                    'spaxel_gain', 'row_gain', 'col_gain', 'fixed_id',
                    'spexel', 'spaxel']
    channel_info = {}
    for i, key in enumerate(column_names):
        channel_info[key] = 1 + i
    data.set_channel_data(0, channel_info)
    assert data.coupling[0] == 4
    assert data.spexel_gain[0] == 5
    assert data.spaxel_gain[0] == 6
    assert data.row_gain[0] == 7
    assert data.col_gain[0] == 8


def test_calculate_overlaps(full_data):
    data = full_data.copy()
    data.overlaps = None
    point_size = data.info.instrument.resolution
    data.calculate_overlaps(point_size)
    assert isinstance(data.overlaps, csr_matrix)
    row, col = data.overlaps.nonzero()
    assert row.size != 0 and col.size != 0


def test_read_hdul(fifi_simulated_hdul, fifi_initialized_channel_data):
    data = fifi_initialized_channel_data.copy()
    data.read_hdul(fifi_simulated_hdul)
    assert np.allclose(data.response, 4e-12)
    assert np.isclose(data.wavelength[0], 51.57747408 * um, atol=1e-5)
    data.configuration.parse_key_value('fifi_ls.uncorrected', 'True')
    data.read_hdul(fifi_simulated_hdul)
    assert np.isclose(data.wavelength[0], 51.57747408 * um, atol=1e-5)


def test_apply_hdul_weights(fifi_initialized_channel_data):
    data = fifi_initialized_channel_data.copy()
    detector = data.info.detector_array
    n_spex = detector.n_spexel
    n_spax = detector.n_spaxel
    stddev = np.full((20, n_spex, n_spax), 2.0)
    data.apply_hdul_weights(stddev)
    assert np.allclose(data.weight, 1)


def test_populate_positions(fifi_initialized_channel_data,
                            fifi_simulated_hdul):
    data = fifi_initialized_channel_data.copy()
    xs = fifi_simulated_hdul['XS'].data.copy()
    ys = fifi_simulated_hdul['YS'].data.copy()
    data.position = None
    data.populate_positions(xs, ys)
    assert isinstance(data.position, Coordinate2D)
    assert data.position.shape == (400,)


def test_geometric_rows(fifi_initialized_channel_data):
    assert fifi_initialized_channel_data.geometric_rows() == 5


def test_geometric_cols(fifi_initialized_channel_data):
    assert fifi_initialized_channel_data.geometric_cols() == 5
