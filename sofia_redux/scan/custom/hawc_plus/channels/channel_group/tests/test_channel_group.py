# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.custom.hawc_plus.channels.channel_group.channel_group \
    import HawcPlusChannelGroup


@pytest.fixture
def channel_group(hawc_plus_channel_data):
    return HawcPlusChannelGroup(
        hawc_plus_channel_data.copy(), indices=np.arange(10),
        name='for_unit_tests')


def test_init(hawc_plus_channel_data):
    data = hawc_plus_channel_data.copy()
    group = HawcPlusChannelGroup(data, indices=np.arange(10, 20), name='foo')
    assert group.name == 'foo'
    assert group.size == 10
    assert np.allclose(group.indices, np.arange(10, 20))
    # Also test basic functionality
    assert list(group.channel_id) == [
        'R0[0,10]', 'R0[0,11]', 'R0[0,12]', 'R0[0,13]', 'R0[0,14]', 'R0[0,15]',
        'R0[0,16]', 'R0[0,17]', 'R0[0,18]', 'R0[0,19]']
    group.gain *= 2
    assert np.allclose(group.gain, 2)
    assert np.allclose(data.gain[:10], 1)
    assert np.allclose(data.gain[10:20], 2)


def test_set_channel_data(channel_group):
    group = channel_group
    gain = group.gain.copy()
    group.set_channel_data(0, None)
    assert np.allclose(group.gain, gain)  # No change

    channel_info = {'gain': 0.6, 'weight': 2.0, 'flag': 8, 'coupling': 1.6,
                    'sub_gain': 1.7, 'mux_gain': 1.8, 'fixed_id': 2,
                    'sub': 1, 'subrow': 10, 'col': 11}

    group.set_channel_data(0, channel_info)
    assert group.gain[0] == 0.6
    assert group.weight[0] == 2.0
    assert group.flag[0] == 8
    assert group.coupling[0] == 1.6
    assert group.sub_gain[0] == 1  # Empirically determined
    assert group.mux_gain[0] == 1.8
    # All these are programmatically assigned...
    assert group.indices[0] == 0
    assert group.sub[0] == 0
    assert group.subrow[0] == 0
    assert group.col[0] == 0


def test_validate_pixel_data(channel_group):
    with pytest.raises(NotImplementedError) as err:
        channel_group.validate_pixel_data()
    assert 'Not implemented' in str(err.value)


def test_validate_weights(channel_group):
    with pytest.raises(NotImplementedError) as err:
        channel_group.validate_weights()
    assert 'Not implemented' in str(err.value)


def test_read_pixel_data(channel_group):
    with pytest.raises(NotImplementedError) as err:
        channel_group.read_pixel_data('foo')
    assert 'Not implemented' in str(err.value)
