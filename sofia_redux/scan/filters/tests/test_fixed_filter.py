# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.fixed_filter import FixedFilter


class SingleFilter(FixedFilter):

    def __init__(self, integration=None, data=None):
        super().__init__(integration=integration, data=data)

    def get_id(self):
        return 'S'

    def get_config_name(self):
        return 'filter.single'

    def response_at(self, fch):
        response = np.ones(self.nf + 1)
        response[16] = 0.0
        return response[fch]


class ZeroArrayFilter(SingleFilter):
    def calc_point_response(self):
        return np.full(self.channels.size, 0.0)


@pytest.fixture
def initialized_filter(populated_integration):
    return SingleFilter(integration=populated_integration)


@pytest.fixture
def configured_filter(initialized_filter):
    f = initialized_filter
    f.set_integration(f.integration.copy())
    f.configuration.parse_key_value(f.get_config_name(), 'True')
    f.configuration.parse_key_value(f'{f.get_config_name()}.level', '2.0')
    return f


def test_init(populated_integration):
    f = SingleFilter()
    assert f.point_response is None
    assert f.rejected is None
    assert f.integration is None
    f = SingleFilter(integration=populated_integration)
    assert f.integration is not None


def test_channel_dependent_attributes():
    f = SingleFilter()
    assert 'point_response' in f.channel_dependent_attributes
    assert 'rejected' in f.channel_dependent_attributes


def test_set_integration(populated_integration):
    f = SingleFilter()
    f.set_integration(populated_integration)
    assert f.point_response.size == f.integration.channels.size
    assert np.allclose(f.point_response, 1)
    assert f.rejected.size == f.integration.channels.size
    assert np.allclose(f.rejected, 0)


def test_get_point_response(configured_filter):
    f = configured_filter
    assert f.get_point_response() is f.point_response
    channels = f.channels.copy()
    pr = f.get_point_response(channels=channels)
    assert pr is not f.point_response and np.allclose(pr, f.point_response)


def test_get_mean_point_rseponse(configured_filter):
    f = configured_filter
    assert f.get_mean_point_response() is f.point_response
    pr = f.get_mean_point_response(channels=f.channels.copy())
    assert pr is not f.point_response and np.allclose(pr, f.point_response)


def test_pre_filter(configured_filter):
    f = configured_filter
    assert np.allclose(f.rejected, 0)
    f.pre_filter()
    assert np.allclose(f.rejected, 1)


def test_reset_point_response(configured_filter):
    f = configured_filter
    f.point_response.fill(-1)
    f.reset_point_response()
    assert np.allclose(f.point_response, 0.998731, atol=1e-6)
    f.point_response.fill(-1)
    cg = f.channels.create_group(np.arange(10))
    f.reset_point_response(channels=cg)
    assert np.allclose(f.point_response[:10], 0.998731, atol=1e-6)
    assert np.allclose(f.point_response[10:], -1)

    f = ZeroArrayFilter()
    f.point_response = np.ones(10)
    f.channels = cg
    f.reset_point_response()
    assert np.allclose(f.point_response, 0)


def test_pre_filter_channels(configured_filter):
    f = configured_filter
    cg = f.channels.create_group(np.arange(10))
    f.pre_filter_channels()
    assert np.allclose(f.channels.direct_filtering, 1.00127061, atol=1e-6)
    assert np.allclose(f.channels.source_filtering, 1.00127061, atol=1e-6)
    f.pre_filter_channels(channels=cg)
    assert np.allclose(f.channels.direct_filtering[10:], 1.00127061, atol=1e-6)
    assert np.allclose(f.channels.source_filtering[10:], 1.00127061, atol=1e-6)
    assert np.allclose(f.channels.direct_filtering[:10], 1.00254283, atol=1e-6)
    assert np.allclose(f.channels.source_filtering[:10], 1.00254283, atol=1e-6)

    expected = f.channels.direct_filtering.copy()
    f.is_sub_filter = True
    f.pre_filter_channels()
    assert np.allclose(f.channels.direct_filtering, expected)
    assert np.allclose(f.channels.source_filtering, expected)

    f = ZeroArrayFilter()
    f.point_response = np.ones(10)
    f.channels = cg  # Keeps values from above...
    f.pre_filter_channels()
    assert np.allclose(f.channels.direct_filtering, 1.00254283, atol=1e-6)
    assert np.allclose(f.channels.source_filtering, 1.00254283, atol=1e-6)


def test_post_filter_channels(configured_filter):
    f = configured_filter
    f.integration.configuration.parse_key_value('crushbugs', 'False')
    cg = f.channels.create_group(np.arange(10))
    f.make_temp_data()  # Populate points
    f.pre_filter()  # Set up the dependents
    assert np.allclose(f.parms.for_frame, 0)
    assert np.allclose(f.parms.for_channel, 0)
    assert np.allclose(f.channels.direct_filtering, 1)
    assert np.allclose(f.channels.source_filtering, 1)
    assert np.allclose(f.points, 0)

    f.points.fill(2.0)
    assert f.frame_parms is None

    f.post_filter_channels()
    assert isinstance(f.frame_parms, np.ndarray)
    assert np.allclose(f.frame_parms, 60.5)
    assert np.allclose(f.parms.for_channel, 1)
    assert np.allclose(f.parms.for_frame, 0)
    assert np.allclose(f.channels.direct_filtering, 0.998731, atol=1e-6)
    assert np.allclose(f.channels.source_filtering, 0.998731, atol=1e-6)

    f.integration.configuration.parse_key_value('crushbugs', 'True')
    f.post_filter_channels(channels=cg)
    assert np.allclose(f.frame_parms, 65.5)
    assert np.allclose(f.parms.for_channel, 1)  # No change with crush bugs
    assert np.allclose(f.channels.direct_filtering[10:], 0.998731, atol=1e-6)
    assert np.allclose(f.channels.source_filtering[10:], 0.998731, atol=1e-6)
    assert np.allclose(f.channels.direct_filtering[:10], 0.997464, atol=1e-6)
    assert np.allclose(f.channels.source_filtering[:10], 0.997464, atol=1e-6)


def test_add_frame_parms(configured_filter):
    f = configured_filter
    f.make_temp_data()
    f.points.fill(1.0)
    f.rejected.fill(2.0)
    assert f.frame_parms is None
    f.add_frame_parms()
    assert isinstance(f.frame_parms, np.ndarray)
    assert np.allclose(f.frame_parms, 242)
