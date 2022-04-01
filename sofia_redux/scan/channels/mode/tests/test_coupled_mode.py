# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from sofia_redux.scan.channels.mode.coupled_mode import CoupledMode
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


@pytest.fixture
def example_mode(populated_data):
    group = ExampleChannelGroup(populated_data, name='test_group')
    base = CorrelatedMode(channel_group=group, gain_provider='gain',
                          name='test')
    mode = CoupledMode(base)
    return mode


class TestCoupledMode(object):

    def test_init(self, example_mode):
        assert isinstance(example_mode, CoupledMode)
        assert example_mode.size == 121
        assert isinstance(example_mode.parent_mode, CorrelatedMode)
        assert example_mode.parent_mode.coupled_modes == [example_mode]

        # set with initial gains
        gain = np.arange(example_mode.size, dtype=float)
        mode = CoupledMode(example_mode.parent_mode,
                           gain_provider=gain)
        assert np.all(mode.gain == gain)

    def test_get_gains(self, mocker, example_mode):
        group = example_mode.channel_group
        test_gains = np.full(group.size, 0.1)
        group.data.gain = test_gains.copy()

        gain = example_mode.get_gains()
        assert gain.size == group.size
        assert gain is example_mode.gain
        assert np.allclose(gain, 1)

        # dimensionless units
        group.data.gain = units.Quantity(group.data.gain,
                                         unit=units.dimensionless_unscaled)
        gain = example_mode.get_gains()
        assert gain.size == example_mode.channel_group.size
        assert np.allclose(gain, 1)

    def test_resync_gains(self, mocker, populated_integration, example_mode):
        # no effect if no signal in integration
        example_mode.resync_gains(populated_integration)

        sig = populated_integration.get_acceleration_signal(
            'x', mode=example_mode)
        populated_integration.add_signal(sig)
        m1 = mocker.patch.object(sig, 'resync_gains')
        example_mode.resync_gains(populated_integration)
        assert m1.call_count == 1

        # add a secondary coupled mode: its resync is also called
        new_mode = CoupledMode(example_mode)
        assert new_mode in example_mode.coupled_modes
        m2 = mocker.patch.object(new_mode, 'resync_gains')
        example_mode.resync_gains(populated_integration)
        assert m1.call_count == 2
        assert m2.call_count == 1
