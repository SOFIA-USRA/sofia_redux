# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.adaptive_filter import AdaptiveFilter


class WorkingFilter(AdaptiveFilter):
    def get_id(self):
        return 'A'

    def get_config_name(self):
        return 'filter.adaptive'

    def dft_filter(self, channels=None):
        # Raises an error
        super().dft_filter(channels=channels)


@pytest.fixture
def configured_integration(populated_integration):
    integration = populated_integration
    integration.configuration.parse_key_value('filter.adaptive', 'True')
    return integration


@pytest.fixture
def configured_filter(configured_integration):
    f = WorkingFilter(integration=configured_integration)
    return f


@pytest.fixture
def sized_filter(configured_filter):
    f = configured_filter
    n = f.integration.frames_for(f.integration.filter_time_scale)
    f.set_size(n)
    return f


@pytest.fixture
def profiled_filter(sized_filter):
    f = sized_filter
    n_channels = f.channels.size
    f.channel_profiles = np.arange(f.nf + 1, dtype=float)[None] + np.arange(
        n_channels)[:, None]
    f.profile = np.full_like(f.channel_profiles, 0.5)
    return f


def test_init(configured_integration):
    f = WorkingFilter()
    assert f.channel_profiles is None
    assert f.profile is None
    assert f.nF == 0
    assert np.isnan(f.dF)
    assert f.integration is None
    f = WorkingFilter(integration=configured_integration)
    assert f.integration is not None


def test_channel_dependent_attributes():
    f = WorkingFilter()
    assert 'channel_profiles' in f.channel_dependent_attributes
    assert 'profile' in f.channel_dependent_attributes


def test_get_profile(configured_filter):
    f = configured_filter
    assert f.get_profile() is f.profile


def test_set_integration(configured_integration):
    f = WorkingFilter()
    integration = configured_integration
    f.set_integration(integration)
    assert f.channel_profiles.shape == (integration.channels.size, 0)


def test_set_size(configured_filter):
    f = configured_filter
    n = f.integration.frames_for(f.integration.filter_time_scale)
    f.set_size(n)
    n_channels = f.channels.size
    assert np.isclose(f.dF, 5 / n)
    assert f.profile.shape == (n_channels, n)
    assert f.channel_profiles.shape == (n_channels, 0)

    f.channel_profiles = (
        np.arange(n_channels)[:, None] + np.arange(n)[None])

    n2 = n // 2
    f.set_size(n2)
    assert f.channel_profiles.shape == (n_channels, n2)

    expected = np.linspace(0.5, 1098.5, n2)[None] + np.arange(
        n_channels)[:, None]
    assert np.allclose(f.channel_profiles, expected)


def test_resample():
    f = WorkingFilter()
    n_frames, n_channels = 10, 5
    old = np.arange(n_frames, dtype=float)[None] + np.arange(
        n_channels)[:, None]
    new = np.empty((n_channels, n_frames // 2))
    f.resample(old, new)
    expected = np.linspace(0.5, 8.5, 5)[None] + np.arange(n_channels)[:, None]
    assert np.allclose(new, expected)


def test_post_filter_channels(sized_filter):
    f = sized_filter
    f.make_temp_data()
    f.pre_filter()
    f.profile.fill(0.5)
    f.channel_profiles = np.ones_like(f.profile)
    f.post_filter_channels()
    assert np.allclose(f.profile, f.channel_profiles)


def test_accumulate_profiles(sized_filter):
    f = sized_filter
    f.profile.fill(0.5)
    f.channel_profiles = np.ones_like(f.profile)
    f.accumulate_profiles()
    assert np.allclose(f.profile, f.channel_profiles)
    assert np.allclose(f.profile, 0.5)

    cg = f.channels.create_group(np.arange(10))
    f.accumulate_profiles(channels=cg)
    assert np.allclose(f.profile, f.channel_profiles)
    assert np.allclose(f.profile[:10], 0.25)
    assert np.allclose(f.profile[10:], 0.5)


def test_response_at(sized_filter):
    f = sized_filter
    f.profile = None
    n_channels = f.channels.size
    expected = np.ones(n_channels)
    assert np.allclose(f.response_at(0), expected)
    response = f.response_at(np.arange(3))
    assert response.shape == (n_channels, 3)
    assert np.allclose(response, 1)

    f.profile = np.arange(f.nf + 1, dtype=float)[None] + np.zeros(
        n_channels)[:, None]

    response = f.response_at(1)
    assert np.allclose(response, f.profile[:, 1])


def test_get_valid_profiles(sized_filter):
    f = sized_filter
    f.channel_profiles = None
    f.profile = None
    cg = f.channels.create_group(np.arange(10))
    assert f.get_valid_profiles() is None

    n_channels = f.channels.size
    f.profile = np.arange(f.nf + 1, dtype=float)[None] + np.arange(
        n_channels)[:, None]
    valid = f.get_valid_profiles()
    assert valid.shape == (n_channels, f.nf + 1)
    assert np.allclose(valid, 1)

    f.channel_profiles = f.profile.copy()
    valid = f.get_valid_profiles()
    assert np.allclose(valid, f.channel_profiles)

    valid = f.get_valid_profiles(channels=cg)
    assert np.allclose(valid, f.channel_profiles[:10])


def test_count_parms(profiled_filter):
    f = profiled_filter
    cp = f.count_parms()
    assert cp.shape == (f.channels.size,)
    assert np.allclose(cp, 768.75)

    cg = f.channels.create_group(np.arange(10))
    cp = f.count_parms(channels=cg)
    assert cp.shape == (10,) and np.allclose(cp, 768.75)

    f.profile = None
    cp = f.count_parms(channels=cg)
    assert cp.shape == (10,) and np.allclose(cp, 0)


def test_update_source_profile(profiled_filter):
    f = profiled_filter
    source_profile_prior = f.source_profile.copy()
    assert np.isclose(f.source_norm, 845.862938, atol=1e-6)
    f.update_source_profile()
    assert f.source_profile.shape != source_profile_prior.shape
    source_profile = f.source_profile.copy()
    assert np.isclose(f.source_norm, 813.075419, atol=1e-6)

    # No update if no change in size to source profile
    f.update_source_profile()
    assert np.allclose(f.source_profile, source_profile)

    # No update without the profile
    f.profile = None
    f.source_profile = None
    f.update_source_profile()
    assert f.source_profile is None


def test_calc_point_response(profiled_filter):
    f = profiled_filter
    f.update_source_profile()
    cg = f.channels.create_group(np.arange(10))
    pr = f.calc_point_response()
    assert pr.shape == (f.channels.size,)
    assert np.allclose(pr, 0.5)
    pr = f.calc_point_response(channels=cg)
    assert pr.shape == (10,) and np.allclose(pr, 0.5)
