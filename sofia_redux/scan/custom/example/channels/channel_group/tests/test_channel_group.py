# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


@pytest.fixture
def example_data(populated_integration):
    return populated_integration.channels.data


@pytest.fixture
def example_group(example_data):
    data = example_data.copy()
    group = ExampleChannelGroup(data, indices=np.arange(10, 15))
    return group


def test_init(example_data):
    data = example_data.copy()
    group = ExampleChannelGroup(data, name='foo')
    assert group.name == 'foo'
    assert group.data is data
    assert np.allclose(group.indices, np.arange(data.size))

    group = ExampleChannelGroup(data, indices=np.arange(10, 15))
    assert group.size == 5
    assert np.allclose(group.col, [10, 0, 1, 2, 3])


def test_set_channel_data(example_group):
    group = example_group.copy()
    group.set_channel_data(0, None)  # Does nothing
    info = {'coupling': 2.0, 'mux_gain': 3.0, 'bias_gain': 4.0}
    for column in ['gain', 'weight', 'flag', 'fixed_id', 'row', 'col']:
        info[column] = 0
    group.set_channel_data(0, info)
    assert np.allclose(group.coupling[:2], [2, 1])
    assert np.allclose(group.mux_gain[:2], [3, 1])
    assert np.allclose(group.bias_gain[:2], [4, 1])
    assert np.allclose(group.gain[:2], [0, 1])


def test_validate_pixel_data(example_group):
    with pytest.raises(NotImplementedError) as err:
        example_group.validate_pixel_data()
    assert 'Not implemented' in str(err.value)


def test_validate_weights(example_group):
    with pytest.raises(NotImplementedError) as err:
        example_group.validate_weights()
    assert 'Not implemented' in str(err.value)


def test_read_pixel_data(example_group):
    with pytest.raises(NotImplementedError) as err:
        example_group.read_pixel_data('foo')
    assert 'Not implemented' in str(err.value)
