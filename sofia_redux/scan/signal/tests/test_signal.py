# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.channels.mode.field_response import FieldResponse
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.integration.integration import Integration
from sofia_redux.scan.signal.signal import Signal


@pytest.fixture
def example_signal(populated_integration):
    integ = populated_integration
    integ.frames.transmission = np.arange(integ.size, dtype=float)
    group = ExampleChannelGroup(populated_integration.channels.data,
                                name='test_group')
    mode = FieldResponse(channel_group=group, name='test',
                         field='transmission', floating=True,
                         derivative_order=0)
    signal = Signal(integ, mode=mode, values=integ.frames.transmission,
                    is_floating=True)
    return signal


class TestSignal(object):
    def test_init(self, example_signal):
        assert isinstance(example_signal, Signal)
        integ = example_signal.integration
        mode = example_signal.mode

        assert isinstance(integ, Integration)
        assert isinstance(mode, FieldResponse)
        assert integ.signals[mode] is example_signal

    def test_string(self, example_signal):
        assert str(example_signal) == 'Signal Simulation.1|1.test'

    def test_copy(self, example_signal):
        new = example_signal.copy()
        assert new is not example_signal
        # referenced
        assert new.mode is example_signal.mode
        assert new.integration is example_signal.integration
        # copied
        assert new.value is not example_signal.value
        assert np.allclose(new.value, example_signal.value)

    def test_blank_properties(self):
        integ = Integration()
        signal = Signal(integ)
        assert signal.size == 0
        assert signal.info is None
        assert signal.configuration is None
        assert signal.get_resolution() is None

    def test_full_properties(self, example_signal):
        assert example_signal.size == example_signal.integration.size
        assert example_signal.info is example_signal.integration.info
        assert (example_signal.configuration
                is example_signal.integration.info.configuration)
        assert example_signal.get_resolution() == 1

    def test_value_at(self, example_signal):
        example_signal.resolution = 1
        assert example_signal.value_at(2) == example_signal.value[2]
        example_signal.resolution = 2
        assert example_signal.value_at(2) == example_signal.value[1]
        assert example_signal.value_at(3) == example_signal.value[1]

    def test_weight_at(self, example_signal):
        assert example_signal.weight_at(0) == 1
        assert example_signal.weight_at(1) == 1
        assert example_signal.weight_at(2) == 1

    def test_scale(self, example_signal):
        value = example_signal.value.copy()
        sync_gains = example_signal.sync_gains.copy()

        example_signal.scale(3)
        assert np.allclose(example_signal.value, value * 3)
        assert np.allclose(example_signal.sync_gains, sync_gains / 3)
        assert example_signal.drifts is None

        example_signal.drifts = np.arange(example_signal.size)
        drifts = example_signal.drifts.copy()
        example_signal.scale(2)
        assert np.allclose(example_signal.drifts, drifts * 2)
        assert np.allclose(example_signal.value, value * 6)
        assert np.allclose(example_signal.sync_gains, sync_gains / 6)

    def test_add(self, example_signal):
        value = example_signal.value.copy()

        example_signal.add(3)
        assert np.allclose(example_signal.value, value + 3)
        assert example_signal.drifts is None

        example_signal.drifts = np.arange(example_signal.size)
        drifts = example_signal.drifts.copy()
        example_signal.add(2)
        assert np.allclose(example_signal.drifts, drifts + 2)
        assert np.allclose(example_signal.value, value + 5)

    def test_subtract(self, example_signal):
        value = example_signal.value.copy()

        example_signal.subtract(3)
        assert np.allclose(example_signal.value, value - 3)
        assert example_signal.drifts is None

        example_signal.drifts = np.arange(example_signal.size)
        drifts = example_signal.drifts.copy()
        example_signal.subtract(2)
        assert np.allclose(example_signal.drifts, drifts - 2)
        assert np.allclose(example_signal.value, value - 5)

    def test_add_drifts(self, example_signal):
        value = example_signal.value.copy()

        # no effect with None drifts
        example_signal.drifts = None
        example_signal.add_drifts()
        assert np.allclose(example_signal.value, value)
        assert example_signal.drifts is None

        # if not None, drifts are added in blocks of drift_n size
        drifts = np.arange(example_signal.size)
        example_signal.drifts = drifts.copy()
        example_signal.drift_n = 1
        example_signal.add_drifts()
        assert np.allclose(example_signal.value, value + drifts)
        assert example_signal.drifts is None

    def test_get_rms_variance(self, example_signal):
        var = example_signal.get_variance()
        assert np.isclose(var, 402783.5)
        assert np.isclose(example_signal.get_rms(), np.sqrt(var))

    def test_remove_drifts(self, example_signal):
        value = example_signal.value.copy()

        # removes mean signal from value
        example_signal.drifts = None
        mean_val = np.mean(value)
        expected = value - mean_val

        example_signal.remove_drifts()
        assert np.allclose(example_signal.value, expected)
        assert np.allclose(example_signal.drifts, mean_val)
        assert len(example_signal.drifts) == 1

        # if not None, drifts are added first, then removed
        example_signal.value = value.copy()
        drifts = np.arange(example_signal.size)
        example_signal.drifts = drifts.copy()
        example_signal.drift_n = 1
        mean_val = np.mean(value + drifts)
        expected = value + drifts - mean_val

        example_signal.remove_drifts()
        assert np.allclose(example_signal.value, expected)
        assert np.allclose(example_signal.drifts, mean_val)
        assert len(example_signal.drifts) == 1

        # remove again: no change
        example_signal.remove_drifts()
        assert np.allclose(example_signal.value, expected)
        assert np.allclose(example_signal.drifts, mean_val)
        assert len(example_signal.drifts) == 1

        # again, but not reconstructable: same result
        example_signal.remove_drifts(is_reconstructable=False)
        assert np.allclose(example_signal.value, expected)
        assert np.allclose(example_signal.drifts, mean_val)
        assert len(example_signal.drifts) == 1

    def test_get_median(self, example_signal):
        # median of value, which is arange(size) for example
        value, weight = example_signal.get_median()
        assert value == (example_signal.size - 1) / 2
        assert np.isinf(weight)

    def test_get_mean(self, example_signal):
        # mean of value, which is arange(size) for example
        value, weight = example_signal.get_mean()
        assert value == (example_signal.size - 1) / 2
        assert np.isinf(weight)

    def test_differentiate(self, example_signal):
        # take derivative with respect to sampling interval:
        # first derivative of arange is constant
        example_signal.differentiate()
        assert np.allclose(example_signal.value, 10)
        # second derivative is 0
        example_signal.differentiate()
        assert np.allclose(example_signal.value, 0)

    def test_integrate(self, example_signal):
        # take integral with respect to sampling interval
        # integral of zero is constant
        example_signal.value = np.full(example_signal.size, 0)
        example_signal.integrate()
        assert np.allclose(example_signal.value, 0)
        # is_floating indicates arbitrary constant
        assert example_signal.is_floating is True

        # integral of constant is line
        example_signal.value = np.full(example_signal.size, 10)
        example_signal.integrate()
        assert np.allclose(example_signal.value,
                           np.arange(example_signal.size))
        assert example_signal.is_floating is True

    def test_get_differential_integral(self, example_signal):
        # get differential signal
        diff = example_signal.get_differential()
        assert isinstance(diff, Signal)
        assert diff is not example_signal

        # value is differentiated
        assert np.allclose(example_signal.value,
                           np.arange(example_signal.size))
        assert np.allclose(diff.value, 10)
        assert diff.is_floating is False

        # get integrated signal
        integ = diff.get_integral()
        assert isinstance(integ, Signal)
        assert integ is not diff
        assert np.allclose(diff.value, 10)
        assert np.allclose(integ.value, np.arange(example_signal.size) + 0.5)
        assert integ.is_floating is True

    def test_level(self, example_signal):
        value = example_signal.value.copy()

        # level all frames
        example_signal.level()
        assert np.allclose(example_signal.value, value - np.mean(value))

        # specify frames to level
        value *= units.s
        example_signal.value = value.copy()
        example_signal.level(start_frame=10, end_frame=20)
        assert np.allclose(example_signal.value[:10], value[:10])
        assert np.allclose(example_signal.value[20:], value[20:])
        assert np.allclose(example_signal.value[10:20],
                           value[10:20] - np.mean(value[10:20]))

    def test_smooth(self, example_signal):
        value = example_signal.value.copy()

        # smoothing fwhm is given in number of frames
        example_signal.smooth(1)
        assert np.allclose(example_signal.value, value, atol=0.1)
        example_signal.smooth(2)
        assert np.allclose(example_signal.value, value, atol=0.5)
        example_signal.smooth(3)
        assert np.allclose(example_signal.value, value, atol=1.0)

    def test_set_sync_gains(self, example_signal):
        gains = np.arange(10)
        example_signal.set_sync_gains(gains)
        assert example_signal.sync_gains is not gains
        assert np.allclose(example_signal.sync_gains, gains)

    def test_get_gain_increment(self, example_signal):
        nchannel = example_signal.mode.size

        gain, weight = example_signal.get_gain_increment()
        assert gain.size == nchannel
        assert weight.size == nchannel
        # small positive gain
        assert np.allclose(gain, 5e-5, atol=4e-5)
        assert np.allclose(weight, 4.4306e+08)

        # robust: uses median instead
        gain, weight = example_signal.get_gain_increment(robust=True)
        assert gain.size == nchannel
        assert weight.size == nchannel
        # no gain
        assert np.allclose(gain, 0)
        assert np.allclose(weight, 4.4306e+08)

        # configure for signal response: adds integration comment
        example_signal.configuration.set_option('signal-response', True)
        covar = example_signal.get_covariance()
        example_signal.get_gain_increment()
        assert example_signal.integration.comments[-1] == f'{{{covar:.2f}}}'

    def test_resync_gains(self, example_signal):
        gains = example_signal.mode.get_gains().copy()
        assert np.allclose(example_signal.sync_gains, 0)

        # if no sync_gains, then sync gains are set
        example_signal.resync_gains()
        assert np.allclose(example_signal.sync_gains, gains)

        # redoing it does nothing - no delta
        example_signal.resync_gains()
        assert np.allclose(example_signal.sync_gains, gains)

    def test_synchronize_gains(self, mocker, example_signal):
        nchannel = example_signal.mode.size
        sum_wc2 = np.arange(nchannel)
        gains = example_signal.mode.get_gains().copy()
        assert np.allclose(example_signal.sync_gains, 0)

        # raises error for fixed gains
        example_signal.mode.fixed_gains = True
        with pytest.raises(ValueError) as err:
            example_signal.synchronize_gains(sum_wc2)
        assert 'Cannot change gains' in str(err)

        # if no wc2, then just sets sync_gains
        example_signal.mode.fixed_gains = False
        example_signal.synchronize_gains(None)
        assert np.allclose(example_signal.sync_gains, gains)

        # again: no change
        example_signal.synchronize_gains(None)
        assert np.allclose(example_signal.sync_gains, gains)

        # now provide wc2: dependents modified, sync gains set
        example_signal.sync_gains[:] = 0
        parms = example_signal.integration.get_dependents('gains-test')
        m1 = mocker.patch.object(parms, 'apply')

        example_signal.synchronize_gains(sum_wc2, is_temp_ready=False)
        assert np.allclose(example_signal.sync_gains, gains)
        assert m1.call_count == 1

    def test_write_signal_values(self, tmpdir, example_signal):
        out_file = str(tmpdir.join('signal.txt'))
        example_signal.write_signal_values(out_file)

        with open(out_file) as fh:
            lines = fh.readlines()

        # starts with 1/(res * int time)
        assert lines[0] == '# 10.0\n'
        # then prints values
        assert lines[1] == '0.000e+00\n'
        assert lines[-1] == '1.099e+03\n'
