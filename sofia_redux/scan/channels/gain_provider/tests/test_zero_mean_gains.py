# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from sofia_redux.scan.channels.gain_provider.zero_mean_gains \
    import ZeroMeanGains
from sofia_redux.scan.channels.mode.mode import Mode
from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


class ZeroMeanGainsCheck(ZeroMeanGains):

    def get_relative_gain(self, channel_data):
        return channel_data.gain

    def set_raw_gain(self, channel_data, gain):
        channel_data.gain = gain


class TestFieldGainProvider(object):

    def test_init(self):
        # can't init abstract class
        with pytest.raises(TypeError):
            ZeroMeanGains()

        provider = ZeroMeanGainsCheck()
        assert provider.ave_g == 0

    def test_get_gain(self, populated_data):
        provider = ZeroMeanGainsCheck()
        populated_data.gain[:4] = np.nan
        populated_data.gain[4:] = 1

        gain = provider.get_gain(populated_data)
        assert gain.size == populated_data.size
        assert np.all(gain[:4] == 0)
        assert np.all(gain[4:] == 1)

    def test_set_gain(self, populated_data):
        nchannel = populated_data.size
        provider = ZeroMeanGainsCheck()

        provider.set_gain(populated_data, np.arange(nchannel))
        assert np.allclose(populated_data.gain, np.arange(nchannel))

    def test_validate(self, populated_data):
        mode = Mode(channel_group=ExampleChannelGroup(populated_data))
        provider = ZeroMeanGainsCheck()

        provider.validate(mode)
        assert np.isclose(provider.ave_g, 1)

        # zero weights
        populated_data.weight[:] = 0
        provider.validate(mode)
        assert np.isclose(provider.ave_g, 0)
        populated_data.weight[:] = 1

        # correlated mode with some flags
        populated_data.flag[:4] = 1
        populated_data.gain[:4] = 500
        populated_data.gain[4:] = 2
        mode = CorrelatedMode(
            channel_group=ExampleChannelGroup(populated_data))
        provider.validate(mode)
        assert np.isclose(provider.ave_g, 2)
