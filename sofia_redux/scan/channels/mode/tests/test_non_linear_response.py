# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from sofia_redux.scan.channels.mode.field_response import FieldResponse
from sofia_redux.scan.channels.mode.non_linear_response \
    import NonLinearResponse
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.signal.signal import Signal


@pytest.fixture
def example_mode(populated_data):
    group = ExampleChannelGroup(populated_data, name='test_group')
    base = FieldResponse(channel_group=group, gain_provider='gain',
                         name='test', field='transmission')
    mode = NonLinearResponse(base)
    return mode


class TestNonLinearResponse(object):

    def test_init(self, example_mode):
        assert isinstance(example_mode, NonLinearResponse)
        assert example_mode.name == 'NonLinearResponse-test'
        assert isinstance(example_mode.parent_mode, FieldResponse)

    def test_get_signal(self, example_mode, populated_integration):
        integ = populated_integration
        integ.frames.transmission = np.arange(integ.size, dtype=float)

        parent_signal = example_mode.parent_mode.get_signal(integ)
        integ.add_signal(parent_signal)
        signal = example_mode.get_signal(integ)
        assert isinstance(signal, Signal)
        assert np.allclose(signal.value, np.arange(integ.size) ** 2)

        # if drifts present, they are removed at the end of the update
        parent_signal.drifts = np.arange(integ.size, dtype=float)
        parent_signal.drift_n = 1
        example_mode.update_signal(populated_integration)
        assert np.allclose(signal.value, 0)

    def test_derive_gains(self, example_mode, populated_integration):
        gains, weights = example_mode.derive_gains(populated_integration)
        assert gains.size == example_mode.size
        assert weights.size == example_mode.size
        assert np.allclose(gains, 0)
        assert np.allclose(weights, 0)
