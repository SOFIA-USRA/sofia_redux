# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
from pandas import DataFrame
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.example.channels.channel_data.channel_data \
    import ExampleChannelData
from sofia_redux.scan.custom.example.flags.channel_flags import \
    ExampleChannelFlags


@pytest.fixture
def example_channels(populated_integration):
    return populated_integration.channels


@pytest.fixture
def example_data(example_channels):
    return example_channels.data


def test_class():
    assert ExampleChannelData.flagspace == ExampleChannelFlags


def test_init(example_channels):
    channels = example_channels.copy()
    data = ExampleChannelData(channels=channels)
    assert data.channels is channels
    assert data.bias_line is None
    assert data.mux_gain is None
    assert data.bias_gain is None
    assert data.default_info is None


def test_default_field_types():
    data = ExampleChannelData()
    defaults = data.default_field_types
    assert defaults['mux_gain'] == 1.0
    assert defaults['bias_gain'] == 1.0


def test_info(example_data):
    data = example_data.copy()
    assert data.info.name == 'example'
    data.channels = None
    assert data.info is None


def test_calculate_and_set_sibs_position(example_data):
    data = example_data.copy()
    data.position = None
    assert np.allclose(data.flag, 0)
    blind = data.flagspace.convert_flag('BLIND').value
    data.flag[0:2] = blind
    data.position = None
    data.calculate_sibs_position()
    assert np.all(data.position[:2].is_nan())
    assert not np.any(data.position[2:].is_nan())


def test_set_uniform_gains(example_data):
    data = example_data.copy()
    data.mux_gain.fill(2)
    data.bias_gain.fill(2)
    data.set_uniform_gains()
    assert np.allclose(data.mux_gain, 1)
    assert np.allclose(data.bias_gain, 1)


def test_to_string(example_data):
    data = example_data.copy()
    df = data.to_string(frame=True)
    assert isinstance(df, DataFrame)
    assert df.size == 1210
    for column in ['eff', 'Gmux', 'Gbias', 'idx', 'row', 'col']:
        assert column in df.columns
    s = data.to_string(indices=np.arange(2), frame=False)
    lines = s.splitlines()
    lines = [x.split() for x in lines]
    assert lines[0] == ['ch', 'gain', 'weight', 'flag', 'eff', 'Gmux',
                        'Gbias', 'idx', 'row', 'col']
    assert lines[1] == ['0,0', '1.000', '1.000e+00', '-', '1.000', '1.000',
                        '1.000', '0', '0', '0']
    assert lines[2] == ['0,1', '1.000', '1.000e+00', '-', '1.000', '1.000',
                        '1.000', '1', '0', '1']


def test_initialize_from_detector(example_data):
    data = example_data.copy()
    data.row = None
    detector = data.info.detector_array
    data.initialize_from_detector(detector)
    assert np.allclose(data.row, np.arange(121) // 11)


def test_apply_info(example_data):
    data = example_data.copy()
    data.position = None
    data.apply_info(data.info)
    assert isinstance(data.position, Coordinate2D)


def test_read_channel_data_file(tmpdir):
    filename = str(
        tmpdir.mkdir('test_read_channel_data_file').join('data.dat'))
    header = '# This is irrelevant'
    l1 = '1 2 g 3 4 5 6 7 8'
    l2 = '2 3 - 4 5 6 7 8 9'
    lines = [header, l1, l2]
    with open(filename, 'w') as f:
        for line in lines:
            print(line, file=f)
    info = ExampleChannelData.read_channel_data_file(filename)
    assert info == {
        '7,8': {'gain': 1.0, 'weight': 2.0, 'flag': 8, 'coupling': 3.0,
                'mux_gain': 4.0, 'bias_gain': 5.0, 'fixed_id': 6, 'row': 7,
                'col': 8},
        '8,9': {'gain': 2.0, 'weight': 3.0, 'flag': 0, 'coupling': 4.0,
                'mux_gain': 5.0, 'bias_gain': 6.0, 'fixed_id': 7, 'row': 8,
                'col': 9}}


def test_set_channel_data(example_data):
    data = example_data.copy()
    info = {'gain': 0.5, 'weight': 2.0, 'flag': 8, 'coupling': 3.0,
            'mux_gain': 4.0, 'bias_gain': 5.0, 'fixed_id': 6, 'row': 7,
            'col': 8}
    data.set_channel_data(0, info)
    assert np.allclose(data.gain[:2], [0.5, 1])
    assert np.allclose(data.weight[:2], [2, 1])
    assert np.allclose(data.flag[:2], [8, 0])
    assert np.allclose(data.coupling[:2], [3, 1])
    assert np.allclose(data.mux_gain[:2], [4, 1])
    assert np.allclose(data.bias_gain[:2], [5, 1])
    g1 = data.gain.copy()
    data.set_channel_data(0, None)
    assert np.allclose(data.gain, g1)


def test_geometric_rows_and_cols(example_data):
    assert example_data.geometric_rows() == 11
    assert example_data.geometric_cols() == 11


def test_get_geometric_overlap_indices(example_data):
    data = example_data.copy()
    overlaps = data.get_geometric_overlap_indices(4 * units.Unit('arcsec'))
    inds = overlaps.nonzero()
    # Check middle
    i50 = inds[0] == 50
    assert np.allclose(inds[1][i50],
                       [28, 38, 39, 40, 48, 49, 51, 52, 60, 61, 62, 72])
    p50 = data.position[50].copy()
    p = data.position[inds[1][i50]].copy()
    p.subtract(p50)
    distance = p.length
    assert np.all(distance <= 4 * units.Unit('arcsec'))
