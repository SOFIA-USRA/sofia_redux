# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.motion_filter import MotionFilter


@pytest.fixture
def configured_integration(populated_integration):
    integration = populated_integration
    config = integration.configuration
    name = 'filter.motion'
    config.parse_key_value(name, 'True')
    config.parse_key_value(f'{name}.s2n', '5.0')
    config.parse_key_value(f'{name}.stability', '30.0')
    config.parse_key_value(f'{name}.harmonics', '1')
    config.parse_key_value(f'{name}.odd', 'True')
    config.parse_key_value(f'{name}.above', '0.3')
    config.parse_key_value(f'{name}.range', '0.01:1.0')
    return integration


@pytest.fixture
def configured_filter(configured_integration):
    f = MotionFilter(integration=configured_integration)
    return f


@pytest.fixture
def reject_indices():
    return np.array([2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                     50, 51, 52, 53, 54, 55, 56, 57, 58])


def test_init(populated_integration):
    f = MotionFilter()
    assert f.critical == 10
    assert f.half_width == 0
    assert f.harmonics == 1
    assert not f.odd_harmonics_only
    assert f.integration is None
    f = MotionFilter(integration=populated_integration)
    assert f.integration is not None


def test_set_integration(configured_integration, reject_indices):
    f = MotionFilter()
    integration = configured_integration
    f.set_integration(integration)
    assert f.critical == 5
    assert f.harmonics == 1
    assert f.odd_harmonics_only
    assert np.allclose(np.nonzero(f.reject)[0], reject_indices)

    f = MotionFilter()
    integration.configuration.parse_key_value('lab', 'True')
    f.set_integration(integration)
    assert not np.any(f.reject)


def test_range_check(configured_filter, reject_indices):
    f = configured_filter
    del f.integration.configuration['filter.motion.range']
    f.range_check()
    assert np.allclose(np.nonzero(f.reject)[0], reject_indices)
    max_index = 50
    max_freq = max_index * f.df
    f.integration.configuration.parse_key_value('filter.motion.range',
                                                f'0:{max_freq}')
    f.range_check()
    assert not np.any(f.reject[50:])


def test_add_filter(configured_filter):
    f = configured_filter
    positions = f.integration.get_smooth_positions('SCANNING').copy()
    positions.zero()
    f.reject.fill(False)

    df = f.df
    x = np.sin(df * np.arange(positions.size) * 16 * np.pi)
    positions.x = x
    f.add_filter(positions, 'x')
    assert np.allclose(np.nonzero(f.reject)[0], [0, 79, 80, 81])


def test_get_fft_rms():
    fft_signal = np.empty(10, dtype=complex)
    fft_signal.real = np.arange(10)
    fft_signal.imag = np.arange(10)
    rms = MotionFilter.get_fft_rms(fft_signal)
    assert np.isclose(rms, 12.43436838, atol=1e-6)


def test_expand_filter():
    f = MotionFilter()
    f.reject = np.full(100, False)
    f.reject[50] = True
    f.df = 0.5
    f.half_width = 3.5
    f.expand_filter()
    assert np.allclose(np.nonzero(f.reject)[0], np.arange(43, 58))


def test_harmonize():
    f = MotionFilter()
    f.harmonics = 1
    f.odd_harmonics_only = False

    reject = np.full(128, False)
    reject[16] = True
    f.reject = reject.copy()

    f.harmonize()
    assert np.allclose(f.reject, reject)

    f.harmonics = 4
    f.harmonize()
    assert np.allclose(np.nonzero(f.reject)[0], [16, 32, 48, 64])
    f.reject = reject.copy()
    f.odd_harmonics_only = True
    f.harmonize()
    assert np.allclose(np.nonzero(f.reject)[0], [16, 48])


def test_pre_filter(configured_filter):
    f = configured_filter
    integration = f.integration
    fake_group = integration.channels.create_channel_group(np.arange(10))
    integration.channels.groups['obs-channels'] = fake_group
    f.pre_filter()
    assert np.allclose(f.channels.indices, np.arange(10))


def test_get_id():
    f = MotionFilter()
    assert f.get_id() == 'Mf'


def test_get_config_name():
    f = MotionFilter()
    assert f.get_config_name() == 'filter.motion'
