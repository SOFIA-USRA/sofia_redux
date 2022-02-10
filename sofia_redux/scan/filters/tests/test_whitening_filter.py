# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.whitening_filter import WhiteningFilter


@pytest.fixture
def configured_integration(populated_integration):
    integration = populated_integration
    config = integration.configuration
    name = 'filter.whiten'
    config.parse_key_value(name, 'True')
    config.parse_key_value(f'{name}.level', '2.0')
    config.parse_key_value(f'{name}.proberange', 'auto')
    return integration


@pytest.fixture
def configured_filter(configured_integration):
    f = WhiteningFilter(integration=configured_integration)
    return f


@pytest.fixture
def prepared_filter(configured_filter):
    f = configured_filter
    f.update_config()
    f.load_time_streams()
    f.pre_filter()
    return f


def test_init(configured_integration):
    f = WhiteningFilter()
    assert f.level == 1.2
    assert f.significance == 2.0
    assert f.windows == -1
    assert f.white_from == -1
    assert f.white_to == -1
    assert f.one_over_f_bin == -1
    assert f.white_noise_bin == -1
    assert f.amplitudes is None
    assert f.amplitude_weights is None
    assert f.integration is None
    f = WhiteningFilter(integration=configured_integration)
    assert f.integration is not None


def test_channel_dependent_attributes():
    f = WhiteningFilter()
    assert 'amplitudes' in f.channel_dependent_attributes
    assert 'amplitude_weights' in f.channel_dependent_attributes


def test_update_config(configured_filter):
    f = configured_filter
    config = f.integration.configuration
    name = 'filter.whiten'
    f.update_config()
    assert np.isclose(f.level, np.sqrt(2))
    assert f.windows == 1
    assert f.white_from == 193
    assert f.white_to == 1100
    assert f.one_over_f_bin == 2
    assert f.white_noise_bin == 550

    del config[f'{name}.proberange']
    bin_from = 5 * f.dF
    bin_to = 200 * f.dF
    config.parse_key_value(f'{name}.1overf.freq', f'{bin_from}')
    config.parse_key_value(f'{name}.1overf.ref', f'{bin_to}')
    f.update_config()

    assert f.white_from == 1
    assert f.white_to == 1100
    assert f.one_over_f_bin == 5
    assert f.white_noise_bin == 200

    config.parse_key_value(f'{name}.proberange', '0.2:0.5')
    f.update_config()
    assert f.white_from == 44
    assert f.white_to == 111

    # Checking min channels
    config.parse_key_value(f'{name}.proberange', '100:200')
    f.update_config()
    assert f.white_from == 1085
    assert f.white_to == 1100

    config.parse_key_value(f'{name}.minchannels', '100')
    f.update_config()
    assert f.white_from == 1001
    assert f.white_to == 1100

    config.parse_key_value(f'{name}.minchannels', '10000')
    f.update_config()
    assert f.white_from == 0
    assert f.white_to == 1100


def test_set_size(configured_filter):
    f = configured_filter
    f.set_size(1000)
    assert f.nF == 1000
    n_channels = f.channels.size
    assert f.amplitudes.shape == (n_channels, 1000)
    assert f.amplitudes.shape == (n_channels, 1000)

    a = f.amplitudes
    aw = f.amplitude_weights
    # No change if the same size.
    f.set_size(1000)
    assert f.amplitudes is a and f.amplitude_weights is aw


def test_update_profile(prepared_filter):
    f = prepared_filter
    assert np.isnan(f.channels.one_over_f_stat).all()
    f.update_profile()
    assert np.allclose(f.channels.one_over_f_stat[:3], 1)
    assert np.isclose(f.channels.one_over_f_stat[3], 0.32716349, atol=1e-6)


def test_calc_mean_amplitudes(prepared_filter):
    f = prepared_filter
    cg = f.channels.create_group(np.arange(10))
    n_channels = f.channels.size
    assert f.channel_profiles.size == 0
    f.calc_mean_amplitudes()
    assert f.channel_profiles.shape == (n_channels, 1100)
    assert np.allclose(f.channel_profiles, 1)

    # Just pick a single value for validity
    assert np.isclose(f.amplitudes[3, 4], 0.0139822229, atol=1e-6)
    assert np.allclose(f.amplitude_weights, 1)
    expect_a = f.amplitudes.copy()
    expect_w = f.amplitude_weights.copy()

    f.amplitudes.fill(0)
    f.amplitude_weights.fill(0)
    f.calc_mean_amplitudes(channels=cg)

    assert np.allclose(expect_a[:10], f.amplitudes[:10])
    assert np.allclose(expect_w[:10], f.amplitude_weights[:10])
    assert np.allclose(f.amplitudes[10:], 0)
    assert np.allclose(f.amplitude_weights[10:], 0)


def test_whiten_profile(prepared_filter):
    f = prepared_filter
    f.calc_mean_amplitudes()
    p0 = f.profile.copy()
    f.whiten_profile()
    assert not np.allclose(p0, f.profile, equal_nan=True)
    assert np.allclose(f.channels.one_over_f_stat[:6],
                       [1, 1, 1, 0.32716349, 0.20908613, 0.17847046],
                       atol=1e-6)

    p1 = f.profile.copy()
    f1 = f.channels.one_over_f_stat.copy()
    cg = f.channels.create_group(np.arange(10))
    f.whiten_profile(channels=cg)
    assert not np.allclose(p1, f.profile)
    assert np.allclose(p1[10:], f.profile[10:])
    assert not np.allclose(f1, f.channels.one_over_f_stat)
    assert np.allclose(f1[10:], f.channels.one_over_f_stat[10:])


def test_get_id():
    f = WhiteningFilter()
    assert f.get_id() == 'wh'


def test_get_config_name():
    f = WhiteningFilter()
    assert f.get_config_name() == 'filter.whiten'


def test_dft_filter():
    f = WhiteningFilter()
    with pytest.raises(NotImplementedError) as err:
        f.dft_filter()
    assert "No DFT for varied filters" in str(err.value)
