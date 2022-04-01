# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


@pytest.fixture
def example_mode(populated_data):
    group = ExampleChannelGroup(populated_data, name='test_group')
    mode = CorrelatedMode(channel_group=group, gain_provider='gain',
                          name='test')
    return mode


class TestCorrelatedMode(object):

    def test_init(self, example_mode):
        assert isinstance(example_mode, CorrelatedMode)
        assert example_mode.size == 121
        assert example_mode.skip_flags is not None

    def test_set_channel_group(self, populated_data):
        mode = CorrelatedMode()
        group = ExampleChannelGroup(populated_data)

        mode.set_channel_group(group)
        assert mode.channel_group is group
        assert mode.gain_flag == group.flagspace.flags(0)
        assert mode.skip_flags == group.flagspace.all_flags()

    def test_get_gains(self, example_mode):
        group = example_mode.channel_group
        group.data.gain = np.arange(group.size)
        gain = example_mode.get_gains()
        assert gain.size == group.size
        assert gain is example_mode.gain

        # gains are normalized
        assert not np.allclose(gain, group.gain)
        assert np.isclose(np.mean(gain), 1, atol=.2)
        norm_gain = gain.copy()

        # dimensionless units
        group.data.gain = units.Quantity(group.data.gain,
                                         unit=units.dimensionless_unscaled)
        gain = example_mode.get_gains()
        assert gain.size == example_mode.channel_group.size
        assert np.allclose(gain, norm_gain)

    def test_set_gains(self, example_mode):
        nchannel = example_mode.size

        gain = np.arange(nchannel, dtype=float)
        flagged = example_mode.set_gains(gain, flag_normalized=True)
        assert np.allclose(example_mode.channel_group.gain, gain)
        assert example_mode.gain is None
        assert not flagged

        flagged = example_mode.set_gains(gain, flag_normalized=False)
        assert not flagged

        example_mode.channel_group.data.gain[:] = 1
        example_mode.gain_provider = None
        gain *= units.dimensionless_unscaled
        example_mode.set_gains(gain)
        assert np.allclose(example_mode.gain, gain)
        assert np.allclose(example_mode.channel_group.data.gain, 1)

    def test_normalize_gains(self, example_mode):
        nchannel = example_mode.size
        assert np.isclose(example_mode.normalize_gains(), 1)
        gain = np.arange(nchannel, dtype=float)
        assert np.isclose(example_mode.normalize_gains(gain=gain), 52.28311)
        assert np.isclose(np.mean(gain), 1, atol=.2)

    def test_get_valid_channels(self, example_mode):
        group = example_mode.get_valid_channels()
        assert group.name == 'test-valid'
        assert group.size == example_mode.size

    def test_update_signals(self, mocker, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = CorrelatedMode(channel_group=group, name='test')

        # makes a signal and calls its update function
        m1 = mocker.patch('sofia_redux.scan.signal.correlated_signal.'
                          'CorrelatedSignal.update')
        mode.update_signals(integ)
        m1.assert_called_once()
