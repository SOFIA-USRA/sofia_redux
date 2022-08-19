# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.custom.fifi_ls.channels.channel_group.channel_group \
    import FifiLsChannelGroup


@pytest.fixture
def fifi_channel_group(fifi_initialized_channel_data):
    return FifiLsChannelGroup(fifi_initialized_channel_data.copy())


def test_init(fifi_initialized_channel_data):
    data = fifi_initialized_channel_data.copy()
    group = FifiLsChannelGroup(data)
    assert group.data is data


def test_set_channel_data(fifi_channel_group):
    group = fifi_channel_group.copy()
    assert np.allclose(group.gain, 1)
    group.set_channel_data(0, None)
    assert np.allclose(group.gain, 1)
    column_names = ['gain', 'weight', 'flag', 'coupling', 'spexel_gain',
                    'spaxel_gain', 'row_gain', 'col_gain', 'fixed_id',
                    'spexel', 'spaxel']
    channel_info = {}
    for i, key in enumerate(column_names):
        channel_info[key] = 2 + i
    group.set_channel_data(0, channel_info)
    assert group.gain[0] == 2
    assert group.coupling[0] == 5
    assert group.spexel_gain[0] == 6
    assert group.spaxel_gain[0] == 7
    assert group.row_gain[0] == 8
    assert group.col_gain[0] == 9


def test_validate_pixel_data(fifi_channel_group):
    with pytest.raises(NotImplementedError) as err:
        fifi_channel_group.validate_pixel_data()
    assert 'Not implemented for' in str(err.value)


def test_validate_weights(fifi_channel_group):
    with pytest.raises(NotImplementedError) as err:
        fifi_channel_group.validate_weights()
    assert 'Not implemented for' in str(err.value)


def test_read_pixel_data(fifi_channel_group):
    with pytest.raises(NotImplementedError) as err:
        fifi_channel_group.read_pixel_data('foo')
    assert 'Not implemented for' in str(err.value)
