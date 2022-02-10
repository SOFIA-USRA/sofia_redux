# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.varied_filter import VariedFilter


class WorkingFilter(VariedFilter):

    def dft_filter(self, channels=None):
        # Raises an error
        super().dft_filter(channels=channels)

    def get_id(self):
        return 'V'

    def get_config_name(self):
        return 'filter.varied'

    def response_at(self, fch):
        response = np.ones(self.nf + 1)
        response[15:20] = 0.5
        return response[fch]


class ChannelFilter(WorkingFilter):
    def count_parms(self):
        return np.full(self.channels.size, super().count_parms())


@pytest.fixture
def configured_integration(populated_integration):
    integration = populated_integration
    integration.configuration.parse_key_value('filter.varied', 'True')
    return integration


@pytest.fixture
def configured_filter(configured_integration):
    f = WorkingFilter(integration=configured_integration)
    return f


@pytest.fixture
def channel_filter(configured_integration):
    f = ChannelFilter(integration=configured_integration)
    return f


def test_init(configured_integration):
    f = WorkingFilter()
    assert f.source_profile is None
    assert f.point_response is None
    assert f.dp is None
    assert f.source_norm == 0
    assert f.integration is None
    f = WorkingFilter(integration=configured_integration)
    assert f.integration is not None


def test_channel_dependent_attributes():
    f = WorkingFilter()
    assert 'dp' in f.channel_dependent_attributes
    assert 'point_response' in f.channel_dependent_attributes


def test_get_source_profile():
    f = WorkingFilter()
    x = np.arange(10)
    f.source_profile = x
    assert f.get_source_profile() is x


def test_set_integration(configured_integration):
    f = WorkingFilter()
    f.set_integration(configured_integration)
    assert isinstance(f.point_response, np.ndarray)
    assert np.allclose(f.point_response, 1)
    assert f.point_response.size == f.channels.size
    assert isinstance(f.dp, np.ndarray) and np.allclose(f.dp, 0)
    assert np.isclose(f.source_norm, 787.853317157, atol=1e-6)


def test_pre_filter_channels(configured_filter):
    f = configured_filter

    df = f.channels.direct_filtering.copy()
    sf = f.channels.source_filtering.copy()
    assert np.allclose(df, 1) and np.allclose(sf, 1)
    cg = f.channels.create_group(np.arange(10))
    f.pre_filter()
    f.point_response[:] = 0.5

    f.pre_filter_channels()
    assert np.allclose(f.channels.source_filtering, 2)
    assert np.allclose(f.channels.direct_filtering, 2)

    f.pre_filter_channels(channels=cg)
    assert np.allclose(f.channels.source_filtering[:10], 4)
    assert np.allclose(f.channels.direct_filtering[:10], 4)
    assert np.allclose(f.channels.source_filtering[10:], 2)
    assert np.allclose(f.channels.direct_filtering[10:], 2)


def test_post_filter_channels(configured_filter, channel_filter):
    f = configured_filter
    f.pre_filter()  # Set up parms
    f.make_temp_data()  # Set up points
    assert np.allclose(f.channels.source_filtering, 1.0)
    assert np.allclose(f.channels.direct_filtering, 1.0)
    assert np.allclose(f.parms.for_channel, 0)
    assert np.allclose(f.parms.for_frame, 0)
    assert np.allclose(f.dp, 0)

    f.points[0] = 1.0

    f.post_filter_channels()
    assert np.allclose(f.parms.for_channel, 2.5)
    assert np.allclose(f.parms.for_frame, 0)
    assert f.dp[0] == 2.5
    assert np.allclose(f.dp[1:], 0)

    assert np.allclose(f.channels.source_filtering, 0.9968276, atol=1e-6)
    assert np.allclose(f.channels.direct_filtering, 0.9968276, atol=1e-6)

    f.points = 1.0
    cg = f.channels.create_group(np.arange(10))
    f.post_filter_channels(channels=cg)
    assert np.allclose(f.dp[:10], 2.5) and np.allclose(f.dp[10:], 0)
    assert np.allclose(f.parms.for_channel[:10], 5)
    assert np.allclose(f.parms.for_channel[10:], 2.5)
    assert np.allclose(f.channels.direct_filtering[:10], 0.9936653, atol=1e-6)
    assert np.allclose(f.channels.direct_filtering[10:], 0.9968276, atol=1e-6)
    assert np.allclose(f.channels.source_filtering,
                       f.channels.direct_filtering)

    f.points = -1
    f.post_filter_channels()
    assert np.allclose(f.dp, 0)
    assert np.allclose(f.parms.for_channel[:10], 7.5)
    assert np.allclose(f.parms.for_channel[10:], 5.0)
    assert np.allclose(f.channels.direct_filtering[:10], 0.99051297, atol=1e-6)
    assert np.allclose(f.channels.direct_filtering[10:], 0.99366527, atol=1e-6)
    assert np.allclose(f.channels.source_filtering,
                       f.channels.direct_filtering)

    f = channel_filter
    f.pre_filter()
    f.make_temp_data()
    f.points[0] = 1.0
    f.post_filter_channels()
    assert f.dp[0] == 2.5 and np.allclose(f.dp[1:], 0)
    assert np.allclose(f.parms.for_channel, 5)


def test_remove_from_frames(configured_filter):
    f = configured_filter
    frames = f.integration.frames
    channels = f.channels
    f.make_temp_data()
    f.pre_filter()
    f.data.fill(1.0)
    d0 = frames.data.copy()
    f.dp.fill(0.5)
    f.frame_parms = np.zeros(frames.size)

    f.remove_from_frames(f.data, frames, channels)
    assert np.allclose(d0 - frames.data, 1)
    assert np.allclose(f.frame_parms, 60.5)


def test_dft_filter():
    f = WorkingFilter()
    with pytest.raises(NotImplementedError) as err:
        f.dft_filter()
    assert "No DFT for varied filters" in str(err.value)


def test_get_point_response(configured_filter):
    f = configured_filter
    f.point_response = np.arange(f.channels.size, dtype=float)
    assert f.get_point_response() is f.point_response
    cg = f.channels.create_group(np.arange(10))
    assert np.allclose(f.get_point_response(channels=cg), np.arange(10))


def test_get_mean_point_response(configured_filter):
    f = configured_filter
    f.point_response.fill(2.0)
    assert f.get_mean_point_response() == 2


def test_update_source_profile(configured_filter):
    f = configured_filter
    f.update_source_profile()
    assert np.isclose(f.source_norm, 787.853317158, atol=1e-6)
    assert np.isclose(np.sum(f.source_profile), f.source_norm)
    assert f.source_profile[0] == 1


def test_calc_point_response(configured_filter):
    f = configured_filter
    cg = f.channels.create_group(np.arange(10))
    response = f.calc_point_response()
    assert np.allclose(response, 0.9968276, atol=1e-6)
    assert response.size == f.channels.size
    response = f.calc_point_response(channels=cg)
    assert np.allclose(response, 0.9968276, atol=1e-6)
    assert response.size == 10
