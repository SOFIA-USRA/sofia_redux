# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.channels.division.division import ChannelDivision
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


@pytest.fixture
def populated_division(populated_data):
    g1 = ExampleChannelGroup(populated_data)
    g2 = ExampleChannelGroup(populated_data)
    division = ChannelDivision('test', groups=[g1, g2])
    return division


class TestChannelDivision(object):

    def test_init(self, populated_division):
        division = ChannelDivision('test')
        assert division.groups == []
        assert division.name == 'test'

        assert len(populated_division.groups) == 2
        assert populated_division.name == 'test'
        assert populated_division.groups[0].name == 'test-1'
        assert populated_division.groups[1].name == 'test-2'

    def test_size_fields(self, populated_division):
        division = ChannelDivision('test')
        division.groups = None
        assert division.size == 0
        assert len(division.fields) == 0

        assert populated_division.size == 2
        assert populated_division.fields == populated_division.groups[0].fields

    def test_validate_group_index(self, populated_division):
        division = ChannelDivision('test')
        with pytest.raises(KeyError) as err:
            division.validate_group_index(0)
        assert 'No channel groups' in str(err)

        assert populated_division.validate_group_index(0) == 0
        assert populated_division.validate_group_index(1) == 1
        assert populated_division.validate_group_index(-1) == 1
        assert populated_division.validate_group_index(-2) == 0
        assert populated_division.validate_group_index('test-1') == 0

        with pytest.raises(KeyError) as err:
            populated_division.validate_group_index('bad-1')
        assert 'Group bad-1 does not exist' in str(err)

        with pytest.raises(ValueError) as err:
            populated_division.validate_group_index(None)
        assert 'Invalid index type' in str(err)

        with pytest.raises(IndexError) as err:
            populated_division.validate_group_index(-3)
        assert 'Cannot use index -3' in str(err)

        with pytest.raises(IndexError) as err:
            populated_division.validate_group_index(3)
        assert 'Group 3 out of range' in str(err)

    def test_get_set(self, populated_division):
        assert populated_division[0].name == 'test-1'
        assert populated_division['test-2'].name == 'test-2'

        g3 = ExampleChannelGroup(populated_division[0].data)
        g3.name = 'new-group'
        populated_division[0] = g3
        assert populated_division[0].name == 'new-group'

        with pytest.raises(ValueError) as err:
            populated_division[0] = 'test'
        assert 'Group must be' in str(err)

    def test_str(self, populated_division):
        division = ChannelDivision('test')
        assert str(division) == 'ChannelDivision (test): 0 group(s)'

        expected = ('ChannelDivision (test): 2 group(s)\n'
                    'ExampleChannelGroup (test-1): 121 channels\n'
                    'ExampleChannelGroup (test-2): 121 channels')
        assert str(populated_division) == expected
