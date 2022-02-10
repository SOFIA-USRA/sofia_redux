# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.filters.filter import Filter
from sofia_redux.scan.filters.kill_filter import KillFilter


class NoFilter(Filter):

    def __init__(self, integration=None, data=None):
        super().__init__(integration=integration, data=data)
        self.variable = False

    def get_id(self):
        return 'N'

    def get_config_name(self):
        return 'filter.none'

    def response_at(self, fch):
        if not self.variable:
            return np.ones_like(fch).astype(float)
        else:
            return np.ones((self.channel_size, np.asarray(fch).size))

    def get_mean_point_response(self):
        if not self.variable:
            return 1.0
        else:
            return np.ones(self.channel_size)


class VariedKill(KillFilter):

    def response_at(self, fch):
        response = super().response_at(fch)
        if response.ndim == 2:
            return response.T
        return response


@pytest.fixture
def initialized_filter(populated_integration):
    return NoFilter(integration=populated_integration)


@pytest.fixture
def configured_filter(initialized_filter):
    f = initialized_filter
    f.set_integration(f.integration.copy())
    f.configuration.parse_key_value(f.get_config_name(), 'True')
    f.configuration.parse_key_value(f'{f.get_config_name()}.level', '2.0')
    return f


@pytest.fixture
def kill_filter(populated_integration):
    integration = populated_integration.copy()
    f = VariedKill(integration=integration)
    integration.configuration.parse_key_value('filter.kill', 'True')
    integration.configuration.parse_key_value('filter.kill.bands', '0.5:0.6')
    f.update_config()
    f.make_temp_data()
    f.load_time_streams()
    return f


def test_init(populated_integration):

    f = NoFilter()
    for attr in ['integration', 'channels', 'parms', 'frame_parms', 'data',
                 'points']:
        assert getattr(f, attr) is None

    for attr in ['is_sub_filter', 'dft', 'pedantic', 'enabled']:
        assert not getattr(f, attr)

    for attr in ['nt', 'nf', 'df']:
        assert getattr(f, attr) == 0

    data = np.ones(1024)
    f = NoFilter(data=data)
    assert f.data is data

    integration = populated_integration
    f = NoFilter(integration=integration)
    assert f.integration is integration
    assert f.nt == 2048
    assert f.nf == 1024
    assert np.isclose(f.df, 0.0048828125)
    assert f.channels.data is integration.channels.data


def test_copy(initialized_filter):
    f = initialized_filter
    f.data = np.zeros(1024)
    f2 = f.copy()
    assert f2 is not f
    assert np.allclose(f2.data, f.data)
    assert f2.integration is f.integration
    assert f2.channels is f.channels


def test_referenced_attributes():
    f = NoFilter()
    assert 'integration' in f.referenced_attributes
    assert 'channels' in f.referenced_attributes


def test_channel_dependent_attributes():
    f = NoFilter()
    assert 'points' in f.channel_dependent_attributes


def test_size(initialized_filter):
    assert NoFilter().size == 0
    assert initialized_filter.size > 0


def test_channel_size(initialized_filter):
    assert NoFilter().channel_size == 0
    assert initialized_filter.channel_size > 0


def test_frames(initialized_filter):
    assert NoFilter().frames is None
    assert initialized_filter.frames.__class__.__name__.endswith('Frames')


def test_info(initialized_filter):
    assert NoFilter().info is None
    assert initialized_filter.info.__class__.__name__.endswith('Info')


def test_configuration(initialized_filter):
    assert NoFilter().configuration is None
    assert initialized_filter.configuration.__class__.__name__.endswith(
        'Configuration')


def test_flagspace(initialized_filter):
    assert NoFilter().flagspace is None
    assert initialized_filter.flagspace.__name__.endswith('FrameFlags')


def test_channel_flagspace(initialized_filter):
    assert NoFilter().channel_flagspace is None
    assert initialized_filter.channel_flagspace.__name__.endswith(
        'ChannelFlags')


def test_valid_filtering_frames(initialized_filter):
    f = initialized_filter.copy()
    f.set_integration(f.integration.copy())
    f.integration.frames.set_flags('MODELING_FLAGS', np.arange(5))
    f.integration.frames.valid[5] = False
    v = f.valid_filtering_frames
    assert not v[:6].any()
    assert v[6:].all()
    assert NoFilter().valid_filtering_frames.size == 0


def test_reindex(initialized_filter):
    f = NoFilter()
    f.reindex()
    assert f.channels is None

    f = initialized_filter.copy()
    # Set a copy of the integration
    f.set_integration(f.integration.copy())
    # Set dependents to actual values
    f.parms = f.integration.get_dependents(f.get_config_name())
    f.points = np.arange(f.channels.data.size)

    # Set some dead channels
    f.integration.channels.data.set_flags('DEAD', np.arange(5))
    # Slim the integration channels, but not the filter
    s1 = f.integration.channels.size
    f.integration.slim()
    s2 = f.integration.channels.size
    assert s1 - s2 == 5

    f.reindex()
    assert f.data is None
    assert f.parms.for_channel.size == s2
    assert np.allclose(f.points, np.arange(5, s1))
    assert np.allclose(f.channels.fixed_index, np.arange(5, s1))

    # Try again with no points
    f.points = None
    f.reindex()
    assert f.points is None
    assert f.channels.size == s2


def test_has_option(configured_filter):
    assert not NoFilter().has_option('level')
    assert configured_filter.has_option('level')


def test_option(configured_filter):
    assert NoFilter().option('level') is None
    assert configured_filter.option('level') == '2.0'


def test_make_temp_data(initialized_filter):
    f = initialized_filter
    f.make_temp_data()
    assert f.data.shape == (121, 2048) and np.allclose(f.data, 0)
    assert f.points.shape == (121,) and np.allclose(f.points, 0)
    f.make_temp_data()  # Do again with existing data and points...
    assert f.data.shape == (121, 2048) and np.allclose(f.data, 0)
    assert f.points.shape == (121,) and np.allclose(f.points, 0)


def test_discard_temp_data(initialized_filter):
    f = initialized_filter
    f.make_temp_data()
    f.discard_temp_data()
    assert f.data is None and f.points is None


def test_is_enabled(initialized_filter):
    f = initialized_filter
    assert not f.is_enabled()
    f.enabled = True
    assert f.is_enabled()


def test_get_temp_data(initialized_filter):
    f = initialized_filter
    f.make_temp_data()
    assert f.get_temp_data() is f.data


def test_set_temp_data(initialized_filter):
    f = initialized_filter
    x = np.zeros(20)
    f.set_temp_data(x)
    assert f.data is x


def test_rejection_at():
    f = NoFilter()
    assert np.allclose(f.rejection_at(np.arange(10)), 0)


def test_count_parms(initialized_filter):
    f = initialized_filter
    assert f.count_parms() == 0
    f.variable = True
    assert np.allclose(f.count_parms(), np.zeros(f.channel_size))


def test_get_channels(initialized_filter):
    f = initialized_filter
    assert f.get_channels().data is f.integration.channels.data


def test_set_channels(initialized_filter):
    channels = initialized_filter.integration.channels.copy()
    f = NoFilter()
    assert f.channels is None
    f.set_channels(None)
    assert f.channels is None

    f.set_channels(channels)
    assert f.channels.data is channels.data
    f.set_integration(initialized_filter.integration.copy())

    f.set_channels(channels.data)
    assert f.channels.data is channels.data
    group = f.channels

    f.set_channels(group)
    assert f.channels is not group and f.channels.data is channels.data

    with pytest.raises(ValueError) as err:
        f.set_channels(1)
    assert "Channels must be" in str(err.value)


def test_set_integration(initialized_filter):
    f = NoFilter()
    integration = initialized_filter.integration.copy()
    f.set_integration(integration)
    assert f.integration is integration
    assert f.nt == 2048
    assert f.nf == 1024
    assert f.df == 0.0048828125
    assert f.channels.data is integration.channels.data


def test_update_config(configured_filter):
    f = NoFilter()
    f.update_config()
    assert not f.enabled and not f.pedantic
    f = configured_filter.copy()
    assert not f.enabled
    assert not f.pedantic
    f.update_config()
    assert f.enabled
    assert not f.pedantic
    f.configuration.parse_key_value('filter.mrproper', 'True')
    f.update_config()
    assert f.enabled
    assert f.pedantic


def test_apply(configured_filter):
    f = NoFilter()
    assert not f.apply()
    assert f.frame_parms is None
    f = configured_filter
    assert f.apply(report=True)
    assert f.integration.comments == ['N', '(1.0)']


def test_apply_to_channels(configured_filter):
    integration = configured_filter.integration.copy()
    f = configured_filter.copy()
    f.set_integration(integration.copy())
    d1 = f.integration.frames.data.copy()
    f.apply_to_channels()
    d2 = f.integration.frames.data.copy()
    assert not np.allclose(d1, d2)
    f = configured_filter.copy()
    f.set_integration(integration.copy())
    f.pedantic = True
    f.dft = True
    f.apply_to_channels()
    d3 = f.integration.frames.data.copy()
    assert np.allclose(d3, d2)
    f = configured_filter.copy()
    f.set_integration(integration.copy())
    f.data = np.zeros((0, 0))
    f.apply_to_channels()
    assert f.data.shape == (121, 2048)


def test_pre_filter(configured_filter):
    f = configured_filter.copy()
    integration = f.integration.copy()
    i1 = integration.copy()
    f.set_integration(i1)
    parms = i1.get_dependents(f.get_config_name())
    parms.for_channel[:] += 1
    parms.for_frame[:] += 2
    assert np.allclose(i1.frames.dependents, 0)
    assert np.allclose(i1.channels.data.dependents, 0)
    f.pre_filter()
    assert np.allclose(i1.frames.dependents, -2)
    assert np.allclose(parms.for_frame, 0)
    assert np.allclose(i1.channels.data.dependents, -1)
    assert np.allclose(parms.for_channel, 0)

    i2 = integration.copy()
    f = configured_filter.copy()
    f.set_integration(i2)
    f.is_sub_filter = True
    parms = i2.get_dependents(f.get_config_name())
    parms.for_channel[:] += 1
    parms.for_frame[:] += 2
    assert np.allclose(i2.frames.dependents, 0)
    assert np.allclose(i2.channels.data.dependents, 0)
    f.pre_filter()
    assert np.allclose(i2.frames.dependents, 0)
    assert np.allclose(parms.for_frame, 0)
    assert np.allclose(i2.channels.data.dependents, 0)
    assert np.allclose(parms.for_channel, 0)


def test_post_filter(configured_filter):
    f = configured_filter.copy()
    integration = f.integration.copy()
    f.set_integration(integration)
    parms = integration.get_dependents(f.get_config_name())
    parms.for_frame[:] = 1
    parms.for_channel[:] = 2
    f.parms = parms
    f.is_sub_filter = True
    f.post_filter()
    assert np.allclose(integration.frames.dependents, 0)
    assert np.allclose(integration.channels.data.dependents, 0)
    f.is_sub_filter = False
    f.post_filter()
    assert np.allclose(integration.frames.dependents, 1)
    assert np.allclose(integration.channels.data.dependents, 2)


def test_remove(configured_filter):
    f = configured_filter.copy()
    f.make_temp_data()
    f.data.fill(1.0)
    d0 = f.integration.frames.data.copy()
    f.remove()
    assert np.allclose(d0, f.integration.frames.data + 1)


def test_remove_from_frames(configured_filter):
    f = configured_filter.copy()
    frames = f.integration.frames
    channels = f.channels
    f.make_temp_data()
    f.data.fill(1)
    d0 = frames.data.copy()
    NoFilter.remove_from_frames(
        rejected_signal=f.data,
        frames=frames,
        channels=channels)
    assert np.allclose(d0, frames.data + 1)


def test_report(configured_filter):
    f = configured_filter.copy()
    integration = f.integration.copy()
    assert len(integration.comments) == 0
    i1 = integration.copy()
    f.set_integration(i1)
    f.report()
    assert i1.comments == ['(1.0)']
    i2 = integration.copy()
    i2.channels.n_mapping_channels = 0
    f.set_integration(i2)
    f.report()
    assert i2.comments == ['(---)']


def test_load_time_streams(configured_filter):
    f = configured_filter.copy()
    assert f.data is None
    f.load_time_streams()
    df = f.data[:, :1100].T
    df0 = f.data.copy()
    di = f.integration.frames.data
    diff = di - df
    assert np.allclose(diff, diff[0, None])  # Just the mean
    assert np.allclose(f.points, 1100)  # no invalid points

    f.data = np.zeros((0, 0))
    f.load_time_streams()
    assert np.allclose(f.data, df0)


def test_fft_filter(kill_filter, configured_filter):
    f = configured_filter
    f.load_time_streams()
    d0 = f.integration.frames.data.copy()
    f0 = f.data.copy()
    f.fft_filter()
    # Nothing happens because no rejection...
    assert np.allclose(f.integration.frames.data, d0)
    assert np.allclose(f.data, f0)

    f = kill_filter.copy()
    integration = f.integration.copy()
    d0 = integration.frames.data.copy()
    f0 = f.data.copy()
    f.fft_filter()
    assert not np.allclose(f.data, f0)
    assert np.allclose(f.integration.frames.data, d0)
    assert np.allclose(f.data[f.integration.size:], 0)  # Zeroed
    f1 = f.data.copy()

    reject = f.reject.copy()
    varied_reject = np.empty((reject.size, f0.shape[0]), dtype=bool)
    varied_reject[...] = reject[:, None].copy()
    f.reject = varied_reject
    f.load_time_streams()
    f.fft_filter()
    assert np.allclose(f.data, f1)  # Should be the same, but 2-D processing.


def test_dft_filter(kill_filter, configured_filter):
    f = configured_filter
    f.load_time_streams()
    d0 = f.integration.frames.data.copy()
    f0 = f.data.copy()
    f.dft_filter()
    # Nothing happens because no rejection...
    assert np.allclose(f.integration.frames.data, d0)
    assert np.allclose(f.data, f0)

    f = kill_filter.copy()
    f0 = f.integration.frames.data.copy()
    freq_channels = np.arange(f.nf + 1)
    f.rejection_at(freq_channels)
    f.dft_filter()
    d1 = f.data.copy()
    # Check that the integration data is unmodified
    assert np.allclose(f0, f.integration.frames.data)

    f = kill_filter.copy()
    f.load_time_streams()
    f.fft_filter()
    d2 = f.data.copy()
    # Check that the DFT is consistent with the FFT
    assert np.allclose(d1, d2)


def test_calc_point_response(kill_filter):
    f = kill_filter.copy()
    # Infinite integration filter time scale
    assert np.isclose(f.calc_point_response(), 0.9723746209)

    # About half way through the filter rejection block
    f.integration.filter_time_scale = 0.9 * units.Unit('second')
    assert np.isclose(f.calc_point_response(), 0.9874574061)


def test_get_high_pass_index(kill_filter):
    f = kill_filter.copy()
    ts = f.integration.filter_time_scale
    assert np.isinf(ts) and ts > 0
    assert f.get_high_pass_index() == 0

    f.integration.filter_time_scale = 0.9 * units.Unit('second')
    assert f.get_high_pass_index() == 114

    f.integration.filter_time_scale = np.nan * units.Unit('second')
    assert f.get_high_pass_index() == 1


def test_level_data_for_channels(kill_filter):
    f = kill_filter.copy()
    f.data += 1
    d0 = f.data.copy()
    f.level_data_for_channels()
    d1 = f.data.copy()
    diff = d0 - d1
    n = f.integration.size
    assert np.allclose(diff[:, :n], 1)
    assert np.allclose(diff[:, n:], 0)


def test_level_for_channels(kill_filter):
    f = kill_filter.copy()
    f.data += 1
    d0 = f.data.copy()
    d1 = d0.copy()
    f.level_for_channels(d1, channels=f.channels)
    diff = d0 - d1
    n = f.integration.size
    assert np.allclose(diff[:, :n], 1)
    assert np.allclose(diff[:, n:], 0)

    d1 = d0.copy()
    f.level_for_channels(d1, channels=f.channels.copy())
    diff = d0 - d1
    assert np.allclose(diff[:, :n], 1)
    assert np.allclose(diff[:, n:], 0)


def test_level_data(kill_filter):
    f = kill_filter.copy()
    f.data += 1
    d0 = f.data.copy()
    f.level_data()
    d1 = f.data.copy()
    diff = d0 - d1
    n = f.integration.size
    assert np.allclose(diff[:, :n], 1)
    assert np.allclose(diff[:, n:], 0)


def test_level(kill_filter):
    f = kill_filter.copy()
    signal = np.ones(f.integration.size)
    f.level(signal)
    assert np.allclose(signal, 0)
    signal = (np.arange(3) + 1)[:, None] * np.ones(f.integration.size)[None]
    f.level(signal)
    assert np.allclose(signal, 0) and signal.shape == (3, f.integration.size)


def test_set_dft(kill_filter):
    f = kill_filter.copy()
    assert f.dft
    f.set_dft(False)
    assert not f.dft
