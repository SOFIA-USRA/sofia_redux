# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from sofia_redux.scan.channels.channel_group.channel_group import ChannelGroup


class ChannelGroupCheck(ChannelGroup):
    """An un-abstracted class for testing"""

    def __init__(self, channel_data, indices=None, name=None):
        super().__init__(channel_data, indices=indices, name=name)

    @property
    def special_fields(self):
        # set a special field for testing
        return {'test', 'overlaps'}

    def read_channel_data_file(self, filename):
        pass

    def get_overlap_indices(self, radius):
        pass

    def get_overlap_distances(self, overlap_indices):
        pass

    def get_pixel_count(self):
        pass

    def get_pixels(self):
        pass

    def get_mapping_pixels(self, indices=None, name=None, keep_flag=None,
                           discard_flag=None, match_flag=None):
        pass


@pytest.fixture
def populated_group(populated_data):
    return ChannelGroupCheck(populated_data)


class TestChannelGroup(object):

    def test_init(self, populated_data):
        # can't init abstract class
        with pytest.raises(TypeError):
            ChannelGroup(populated_data)

        # okay with abstract functions implemented
        ChannelGroupCheck(populated_data)

    def test_get_set_special(self, populated_group):
        # special field, non-overlaps
        assert 'test' in populated_group.special_fields
        populated_group.indices = np.arange(5)
        populated_group.data.test = np.arange(10)

        # applies indices to test field in get
        assert np.all(populated_group.test == np.arange(5))

        # set is not applied for special fields
        populated_group.test += 10
        assert np.all(populated_group.test == np.arange(5))
        assert np.all(populated_group.data.test == np.arange(10))

        # set to None is allowed
        populated_group.test = None
        assert populated_group.test is None

        # if parent is already None, set is ignored
        populated_group.data.test = None
        populated_group.test = np.arange(10)
        assert populated_group.test is None
        assert populated_group.data.test is None

    def test_get_set_overlaps(self, populated_group, overlaps):
        populated_group._indices = None
        assert populated_group.overlaps is None

        idx = np.arange(3)
        populated_group.data.overlaps = overlaps.copy()
        group_overlaps = overlaps[idx[:, None], idx[None]]

        # with None indices, return value is still None
        assert populated_group.indices is None
        assert populated_group.overlaps is None

        # applies indices to overlaps
        populated_group.indices = idx
        assert np.all(populated_group.indices == idx)
        assert np.all(populated_group.overlaps.data == group_overlaps.data)

        group_overlaps.data += 10
        populated_group.overlaps = group_overlaps.copy()
        assert np.all(populated_group.overlaps.data
                      == overlaps[idx[:, None], idx[None]].data + 10)

        # set to None is allowed
        populated_group.overlaps = None
        assert populated_group.overlaps is None

        # if parent is already None, set is ignored
        populated_group.data.overlaps = None
        populated_group.overlaps = group_overlaps
        assert populated_group.overlaps is None
        assert populated_group.data.overlaps is None

    def test_str(self, populated_group):
        assert str(populated_group) == 'ChannelGroupCheck (None): 121 channels'

    def test_copy(self, populated_group):
        new = populated_group.copy()
        assert new is not populated_group
        assert np.all(new.data == populated_group.data)
        assert np.all(new.indices == populated_group.indices)
        assert np.all(new.name == populated_group.name)

    def test_set_channel_data(self, populated_group):
        data = populated_group
        nchannel = data.size
        data.gain = np.full(nchannel, 1.0)
        data.weight = np.full(nchannel, 1.0)
        data.flag = np.full(nchannel, 0)

        # no change for None info
        data.set_channel_data(5, None)
        assert data.gain[5] == 1.0
        assert data.weight[5] == 1.0
        assert data.flag[5] == 0

        # pass info to set
        info = {'gain': 2.0, 'weight': 0.0, 'flag': 1}
        data.set_channel_data(5, info)
        assert data.gain[5] == 2.0
        assert data.weight[5] == 0.0
        assert data.flag[5] == 1

    def test_add_dependents(self, populated_group):
        data = populated_group
        nchannel = data.size
        data.dependents = np.arange(nchannel)
        data.add_dependents(np.arange(nchannel))
        assert np.allclose(data.dependents, 2 * np.arange(nchannel))
        data.remove_dependents(np.arange(nchannel))
        assert np.allclose(data.dependents, np.arange(nchannel))
        data.remove_dependents(np.arange(nchannel))
        assert np.allclose(data.dependents, 0)

    def test_set_flag_defaults(self, populated_group):
        # set some flags and data
        populated_group.data.flag[:4] = 1
        populated_group.data.flag[4:8] = 2
        populated_group.data.coupling[:] = 2.0
        populated_group.data.gain[:] = 2.0
        populated_group.data.weight[:] = 2.0
        populated_group.data.variance[:] = 2.0

        populated_group.set_flag_defaults()
        assert np.all(populated_group.coupling[:8] == 0)
        assert np.all(populated_group.coupling[8:] == 2)
        assert np.all(populated_group.gain[:4] == 0)
        assert np.all(populated_group.gain[4:] == 2)
        assert np.all(populated_group.weight[:4] == 0)
        assert np.all(populated_group.weight[4:] == 2)
        assert np.all(populated_group.variance[:4] == 0)
        assert np.all(populated_group.variance[4:] == 2)

    def test_create_group(self, populated_group):
        group = populated_group.create_group(name='test', indices=[1, 2, 3])
        assert isinstance(group, ChannelGroupCheck)
        assert np.all(group.name == 'test')
        assert np.allclose(group.indices, [1, 2, 3])
