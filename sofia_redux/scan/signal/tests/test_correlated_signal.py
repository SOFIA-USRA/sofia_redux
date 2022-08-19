# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup
from sofia_redux.scan.integration.integration import Integration
from sofia_redux.scan.signal.correlated_signal import CorrelatedSignal


@pytest.fixture
def example_signal(populated_integration):
    integ = populated_integration
    integ.channels.calculate_overlaps(integ.channels.overlap_point_size)
    group = ExampleChannelGroup(integ.channels.data,
                                name='test_group')

    mode = CorrelatedMode(channel_group=group, name='test',
                          gain_provider='gain')
    signal = CorrelatedSignal(integ, mode=mode)

    signal.value = np.arange(signal.size, dtype=float)
    signal.weight = np.full(signal.size, 1.0)

    return signal


class TestCorrelatedSignal(object):
    def test_init(self, example_signal):
        assert isinstance(example_signal, CorrelatedSignal)
        integ = example_signal.integration
        mode = example_signal.mode

        assert isinstance(integ, Integration)
        assert isinstance(mode, CorrelatedMode)
        assert integ.signals[mode] is example_signal

    def test_copy(self, example_signal):
        new = example_signal.copy()
        assert new is not example_signal
        # referenced
        assert new.mode is example_signal.mode
        assert new.integration is example_signal.integration
        # copied
        assert new.value is not example_signal.value
        assert np.allclose(new.value, example_signal.value)

    def test_weight_at(self, example_signal):
        example_signal.weight = np.arange(example_signal.size, dtype=float)
        example_signal.resolution = 1
        assert example_signal.weight_at(2) == example_signal.weight[2]
        example_signal.resolution = 2
        assert example_signal.weight_at(2) == example_signal.weight[1]
        assert example_signal.weight_at(3) == example_signal.weight[1]

    def test_get_rms_variance(self, example_signal):
        example_signal.weight[:] = 0
        var = example_signal.get_variance()
        # zero weights => zero var
        assert var == 0

        example_signal.weight = np.arange(example_signal.size, dtype=float)
        var = example_signal.get_variance()
        assert np.isclose(var, 604450)
        assert np.isclose(example_signal.get_rms(), np.sqrt(var))

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

        # non-uniform weights
        example_signal.value = value.copy()
        example_signal.weight = value.copy()
        mval, _ = example_signal.get_mean()
        example_signal.level()
        assert np.allclose(example_signal.value, value - mval * units.s)

    def test_get_median(self, example_signal):
        value = example_signal.value.copy()

        # weighted median of value
        mval, weight = example_signal.get_median()
        assert mval == (example_signal.size - 1) // 2
        assert weight == 1100

        # non uniform weights
        example_signal.weight = value.copy()
        mval, weight = example_signal.get_median()
        assert np.isclose(mval, 778)
        assert weight == 604450

    def test_get_mean(self, example_signal):
        value = example_signal.value.copy()

        # weighted mean of value
        mval, weight = example_signal.get_mean()
        assert mval == (example_signal.size - 1) / 2
        assert weight == 1100

        # non uniform weights
        example_signal.weight = value.copy()
        mval, weight = example_signal.get_mean()
        assert np.isclose(mval, 733)
        assert weight == 604450

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
        assert isinstance(diff, CorrelatedSignal)
        assert diff is not example_signal

        # value is differentiated
        assert np.allclose(example_signal.value,
                           np.arange(example_signal.size))
        assert np.allclose(diff.value, 10)
        assert diff.is_floating is False

        # get integrated signal
        integ = diff.get_integral()
        assert isinstance(integ, CorrelatedSignal)
        assert integ is not diff
        assert np.allclose(diff.value, 10)
        assert np.allclose(integ.value, np.arange(example_signal.size) + 0.5)
        assert integ.is_floating is True

    def test_get_parms(self, example_signal):
        nframe = example_signal.size

        # uniform weights
        assert example_signal.get_parms() == nframe - 1

        # non uniform weights
        example_signal.weight = example_signal.value.copy()
        assert np.isclose(example_signal.get_parms(), nframe - 2)

    def test_update(self, mocker, example_signal):
        # check that update completes, with and without robust set
        example_signal.update()
        assert example_signal.generation == 1

        example_signal.update(robust=True)
        assert example_signal.generation == 2

        # check call sequence for update functions:
        # get gains, clear dependents, resync gains,
        # get gain increments, apply gain increments,
        # apply mode dependents, set sync gains,
        # calculate filtering
        example_signal.sync_gains[:] = 0
        m1 = mocker.patch.object(example_signal.mode, 'get_gains',
                                 return_value=example_signal.mode.gain)
        m2 = mocker.patch.object(example_signal.dependents, 'clear')
        m3 = mocker.patch('sofia_redux.scan.signal.correlated_signal.'
                          'snf.resync_gains')
        m4 = mocker.patch.object(example_signal, 'get_ml_correlated',
                                 return_value=(0.0, 1.0))
        m5 = mocker.patch('sofia_redux.scan.signal.correlated_signal.'
                          'snf.apply_gain_increments')
        m6 = mocker.patch.object(example_signal.dependents, 'apply')
        m7 = mocker.patch.object(example_signal, 'set_sync_gains')
        m8 = mocker.patch.object(example_signal, 'calc_filtering')

        example_signal.update()
        assert example_signal.generation == 3
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m3.call_count == 1
        assert m4.call_count == 1
        assert m5.call_count == 1
        assert m6.call_count == 1
        assert m7.call_count == 1
        assert m8.call_count == 1

    def test_get_frame_data_signal(self, example_signal):
        frame_signal = example_signal.get_frame_data_signal()
        n_frames = example_signal.integration.size
        n_channels = example_signal.integration.channels.size
        assert frame_signal.shape == (n_frames, n_channels)

    def test_get_ml_correlated(self, example_signal):
        # prep temp fields
        group = example_signal.mode.channel_group
        group.temp = np.zeros(group.size, dtype=float)
        group.temp_g = example_signal.mode.get_gains().copy()
        group.temp_wg = group.weight * group.temp_g
        group.temp_wg2 = group.temp_wg * group.temp_g

        gain, weight = example_signal.get_ml_correlated(group)
        assert gain.size == example_signal.size
        assert weight.size == example_signal.size
        assert np.allclose(gain, 0, atol=.5)
        assert np.allclose(weight, group.size)

        # explicitly include all frames: same answer
        modeling_frames = np.full(example_signal.size, False)
        gain, weight = example_signal.get_ml_correlated(
            example_signal.mode.channel_group, modeling_frames)
        assert gain.size == example_signal.size
        assert weight.size == example_signal.size
        assert np.allclose(gain, 0, atol=.5)
        assert np.allclose(weight, group.size)

        # exclude all frames: zeroes returned
        modeling_frames = np.full(example_signal.size, True)
        gain, weight = example_signal.get_ml_correlated(
            example_signal.mode.channel_group, modeling_frames)
        assert gain.size == example_signal.size
        assert weight.size == example_signal.size
        assert np.allclose(gain, 0)
        assert np.allclose(weight, 0)

    def test_get_robust_correlated(self, example_signal):
        # prep temp fields
        group = example_signal.mode.channel_group
        group.temp = np.zeros(group.size, dtype=float)
        group.temp_g = example_signal.mode.get_gains().copy()
        group.temp_wg = group.weight * group.temp_g
        group.temp_wg2 = group.temp_wg * group.temp_g

        gain, weight = example_signal.get_robust_correlated(group)
        assert gain.size == example_signal.size
        assert weight.size == example_signal.size
        assert np.allclose(gain, 0, atol=.5)
        assert np.allclose(weight, group.size)

        # explicitly include all frames: same answer
        modeling_frames = np.full(example_signal.size, False)
        gain, weight = example_signal.get_robust_correlated(
            example_signal.mode.channel_group, modeling_frames)
        assert gain.size == example_signal.size
        assert weight.size == example_signal.size
        assert np.allclose(gain, 0, atol=.5)
        assert np.allclose(weight, group.size)

        # exclude all frames: zeroes returned
        modeling_frames = np.full(example_signal.size, True)
        gain, weight = example_signal.get_robust_correlated(
            example_signal.mode.channel_group, modeling_frames)
        assert gain.size == example_signal.size
        assert weight.size == example_signal.size
        assert np.allclose(gain, 0)
        assert np.allclose(weight, 0)
