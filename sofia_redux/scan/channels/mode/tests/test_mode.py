# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.gain_provider.field_gain_provider \
    import FieldGainProvider
from sofia_redux.scan.channels.gain_provider.sky_gradient import SkyGradient
from sofia_redux.scan.channels.mode.mode import Mode
from sofia_redux.scan.channels.mode.acceleration_response \
    import AccelerationResponse
from sofia_redux.scan.channels.mode.coupled_mode import CoupledMode
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


@pytest.fixture
def example_mode(populated_data):
    group = ExampleChannelGroup(populated_data, name='test_group')
    provider = SkyGradient.x()
    mode = Mode(channel_group=group, gain_provider=provider, name='test')
    return mode


class TestMode(object):

    def test_init(self, example_mode):
        # bare init allowed
        mode = Mode()
        assert mode.gain is None
        assert mode.channel_group is None
        assert mode.gain_provider is None
        assert mode.size == 0
        assert mode.flagspace is None

        assert isinstance(example_mode, Mode)
        assert example_mode.size == 121

    def test_string(self, example_mode):
        # short description
        assert str(example_mode) == 'Mode (test): 121 channels'

        # long description
        description = example_mode.to_string()
        assert description.startswith('test: 0,0 0,1 0,2')
        assert description.endswith('10,8 10,9 10,10')

        # missing channel ids
        example_mode.channel_group.channel_id = None
        assert example_mode.to_string() == 'test:'

    def test_set_channel_group(self, populated_data):
        mode = Mode()
        group = ExampleChannelGroup(populated_data)

        mode.set_channel_group(group)
        assert mode.channel_group is group
        assert mode.gain_flag == group.flagspace.flags(0)

        # coupled modes
        other = Mode()
        mode.coupled_modes = [other]
        mode.set_channel_group(group)
        assert other.channel_group is group
        assert other.gain_flag == group.flagspace.flags(0)

    def test_set_name(self, example_mode):
        # name from input
        example_mode.set_name('new_name')
        assert example_mode.name == 'new_name'

        # name from channel group
        example_mode.name = None
        example_mode.set_name()
        assert example_mode.name == 'test_group'

    def test_set_gain_provider(self, populated_data):
        mode = Mode()
        provider = SkyGradient.x()

        mode.set_gain_provider(provider)
        assert mode.gain_provider is provider

        mode.set_gain_provider('rows')
        assert isinstance(mode.gain_provider, FieldGainProvider)
        assert mode.gain_provider.field == 'rows'

        mode.set_gain_provider(None)
        assert mode.gain_provider is None

        with pytest.raises(ValueError) as err:
            mode.set_gain_provider(1)
        assert 'Gain must be' in str(err)

    def test_add_coupled_mode(self):
        mode = Mode()
        other = Mode()
        mode.add_coupled_mode(other)
        assert mode.coupled_modes == [other]
        mode.add_coupled_mode(other)
        assert mode.coupled_modes == [other, other]

    def test_get_gains(self, example_mode):
        group = example_mode.channel_group
        gain = example_mode.get_gains()
        assert gain.size == group.size
        assert np.allclose(gain, group.position.x)
        assert gain is example_mode.gain

        # unscaled units
        provider = FieldGainProvider('weight')
        group.data.weight = units.Quantity(group.data.weight,
                                           unit=units.dimensionless_unscaled)

        example_mode.set_gain_provider(provider)
        gain = example_mode.get_gains()
        assert gain.size == example_mode.channel_group.size
        assert np.allclose(gain, example_mode.channel_group.weight.value)

        # mismatch between gain size and mode size
        example_mode.gain = np.arange(10)
        with pytest.raises(ValueError) as err:
            example_mode.get_gains()
        assert 'Gain array size differs' in str(err)

    def test_apply_provider_gains(self, mocker, example_mode):
        m1 = mocker.patch.object(example_mode.gain_provider, 'validate')

        example_mode.apply_provider_gains(False)
        assert m1.call_count == 0

        example_mode.apply_provider_gains(True)
        assert m1.call_count == 1

        # nans and dimensionless units handled
        group = example_mode.channel_group
        group.data.position.x[:4] = np.nan
        group.data.position.x = units.Quantity(
            group.data.position.x.value, unit=units.dimensionless_unscaled)

        example_mode.apply_provider_gains(False)
        assert np.all(example_mode.gain[:4] == 0)
        assert np.all(example_mode.gain[4:]
                      == example_mode.channel_group.position.x[4:])

    def test_set_gains(self, example_mode):
        nchannel = example_mode.size
        example_mode.set_gain_provider('weight')

        gain = np.arange(nchannel)
        flagged = example_mode.set_gains(gain, flag_normalized=True)
        assert np.allclose(example_mode.channel_group.weight, gain)
        assert example_mode.gain is None
        assert not flagged

        flagged = example_mode.set_gains(gain, flag_normalized=False)
        assert not flagged

        example_mode.channel_group.data.weight[:] = 1
        example_mode.gain_provider = None
        gain *= units.dimensionless_unscaled
        example_mode.set_gains(gain)
        assert np.allclose(example_mode.gain, gain)
        assert np.allclose(example_mode.channel_group.data.weight, 1)

    def test_flag_gain(self, example_mode):
        assert example_mode.flag_gains(True) is False

        example_mode.channel_group.data.flag[:] = 1
        example_mode.gain_flag = example_mode.channel_group.flagspace.flags(1)
        assert example_mode.gain_flag.value != 0
        assert example_mode.flag_gains(True) is True
        # data is unflagged
        assert np.all(example_mode.channel_group.flag == 0)

        example_mode.gain_type = 2
        example_mode.channel_group.data.flag[:] = 2
        example_mode.gain_flag = example_mode.channel_group.flagspace.flags(2)
        assert example_mode.flag_gains(True) is True
        # data is not unflagged, since type is not in signed/bidirectional
        assert np.all(example_mode.channel_group.flag == 2)

    def test_uniform_gain(self, example_mode):
        example_mode.set_gain_provider('weight')
        example_mode.uniform_gains()
        assert np.all(example_mode.channel_group.weight == 1)

    def test_derive_gains(self, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = AccelerationResponse(channel_group=group)
        sig = integ.get_acceleration_signal('x', mode=mode)
        integ.add_signal(sig)

        mode.fixed_gains = True
        with pytest.raises(ValueError) as err:
            mode.derive_gains(populated_integration)
        assert 'Cannot solve gains' in str(err)

        mode.fixed_gains = False
        mode.get_gains()
        gains, weights = mode.derive_gains(populated_integration)
        assert gains.size == mode.size
        assert weights.size == mode.size
        assert np.allclose(gains, 1)

    def test_sync_all_gains(self, mocker, populated_integration):
        integ = populated_integration
        group = ExampleChannelGroup(populated_integration.channels.data,
                                    name='test_group')
        mode = AccelerationResponse(channel_group=group, name='test')
        sig = integ.get_acceleration_signal('x', mode=mode)
        integ.add_signal(sig)
        m1 = mocker.patch.object(sig, 'synchronize_gains')

        # add coupled mode
        other = CoupledMode(mode)
        m2 = mocker.patch.object(other, 'resync_gains')

        sum_wc2 = np.arange(mode.size)
        mode.sync_all_gains(integ, sum_wc2)
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m2.called_with(integ)

    def test_frame_resolution(self, populated_integration):
        mode = Mode()
        res = mode.get_frame_resolution(populated_integration)
        assert res == 1

        mode.resolution = 10 * units.s
        res = mode.get_frame_resolution(populated_integration)
        assert res == 128

    def test_signal_length(self, populated_integration):
        mode = Mode()
        length = mode.signal_length(populated_integration)
        assert length == 1100

        mode.resolution = 10 * units.s
        length = mode.signal_length(populated_integration)
        assert length == 9
