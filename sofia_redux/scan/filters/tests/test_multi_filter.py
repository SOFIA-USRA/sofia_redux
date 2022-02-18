# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.filters.filter import Filter
from sofia_redux.scan.filters.multi_filter import MultiFilter


@pytest.fixture
def configured_integration(populated_integration):
    integration = populated_integration
    config = integration.configuration
    config.parse_key_value('filter', 'True')
    config.parse_key_value('ordering', ['motion', 'kill', 'whiten'])

    # Motion filter first
    name = 'filter.motion'
    if name in config:  # pragma: no cover
        del config[name]
    config.parse_key_value(name, 'True')
    config.parse_key_value(f'{name}.range', '0.01:1.0')
    config.parse_key_value(f'{name}.s2n', '5.0')
    config.parse_key_value(f'{name}.above', '0.3')

    # Kill filter
    name = 'filter.kill'

    if name in config:  # pragma: no cover
        del config[name]
    config.parse_key_value(name, 'True')

    nt = 2048  # pow2ceil(1100)
    dt = 0.1
    df = 1 / (dt * nt)  # 5 / 1100
    # Just kill a couple of frequencies
    f1 = 100 * df
    f2 = 101 * df
    config.parse_key_value(f'{name}.bands', f'{f1}:{f2}')

    # Whitening filter
    name = 'filter.whiten'
    if name in config:  # pragma: no cover
        del config[name]
    config.parse_key_value(name, 'True')
    config.parse_key_value(f'{name}.level', '2.0')
    config.parse_key_value(f'{name}.proberange', 'auto')

    return integration


@pytest.fixture
def configured_filter(configured_integration):
    f = MultiFilter(integration=configured_integration)
    return f


@pytest.fixture
def prepared_filter(configured_filter):
    f = configured_filter
    filter_ordering = f.integration.configuration.get_list('filter.ordering')
    for name in filter_ordering:
        sub_filter = f.integration.get_filter(name)
        f.add_filter(sub_filter)

    f.update_config()
    f.load_time_streams()
    f.pre_filter()
    return f


def test_init(configured_integration):
    f = MultiFilter()
    assert f.filters == []
    assert f.n_enabled == 0
    assert f.integration is None
    integration = configured_integration
    f = MultiFilter(integration=integration)
    assert f.integration is integration


def test_copy(prepared_filter):
    f = prepared_filter
    f2 = f.copy()
    for i, sub_filter in enumerate(f.filters):
        sub_filter2 = f2.filters[i]
        assert sub_filter2 is not sub_filter
        assert sub_filter.__class__ == sub_filter2.__class__
        # Check referenced attributes
        assert sub_filter.integration is sub_filter2.integration
        assert sub_filter.channels is sub_filter2.channels

    f.filters = None
    f2 = f.copy()
    assert f2.filters is None and f2 is not f


def test_size(prepared_filter):
    f = prepared_filter
    assert f.size == 3
    f.filters = None
    assert f.size == 0


def test_contains(prepared_filter):
    f = prepared_filter
    assert 'kill' in f
    assert 'foo' not in f


def test_getitem(prepared_filter):
    f = prepared_filter
    for sub_filter in f.filters:
        assert isinstance(f[sub_filter], Filter)
    for sub_filter in f.filters:
        assert isinstance(f[sub_filter.__class__], Filter)
    for name in ['kill', 'motion', 'whiten']:
        assert isinstance(f[name], Filter)
    for label in ['K', 'Mf', 'wh']:
        assert isinstance(f[label], Filter)
    for i in range(3):
        assert isinstance(f[i], Filter)

    assert f['foo'] is None
    f.filters = None
    assert f['kill'] is None


def test_reindex(prepared_filter):
    f = prepared_filter
    expected = f.channels.fixed_index[5:].copy()
    f.integration.channels.data.set_flags('DEAD', np.arange(5))
    f.integration.slim()
    f.reindex()
    assert np.allclose(f.channels.fixed_index, expected)
    for sub_filter in f.filters:
        assert np.allclose(sub_filter.channels.fixed_index, expected)


def test_get_filters(prepared_filter):
    f = prepared_filter
    assert f.get_filters() is f.filters


def test_set_integration(configured_integration):
    integration = configured_integration
    f = MultiFilter(integration=integration)
    filter_ordering = integration.configuration.get_list('filter.ordering')
    for name in filter_ordering:
        sub_filter = integration.get_filter(name)
        f.add_filter(sub_filter)
    f.set_integration(integration)
    for sub_filter in f.filters:
        assert sub_filter.integration is integration
    assert np.isclose(f.source_norm, 787.8533172, atol=1e-6)


def test_set_channels(prepared_filter):
    f = prepared_filter
    cg = f.channels.create_group(np.arange(10))
    f.set_channels(cg)
    assert np.allclose(f.channels.fixed_index, cg.fixed_index)
    for sub_filter in f.filters:
        assert np.allclose(sub_filter.channels.fixed_index, cg.fixed_index)


def test_add_filter(prepared_filter):
    f = prepared_filter
    sub_filter = f.filters[0].copy()
    sub_filter.integration = sub_filter.integration.copy()
    with pytest.raises(ValueError) as err:
        f.add_filter(sub_filter)
    assert "Cannot compound filter from a different integration" in str(
        err.value)

    sub_filter = f.filters[0].copy()
    sub_filter.integration = None
    f.add_filter(sub_filter)
    assert sub_filter.integration is f.integration
    assert f.size == 4

    f.filters = None
    f.add_filter(sub_filter)
    assert f.size == 1


def test_set_filter(prepared_filter):
    f = prepared_filter
    sub_filter = f.filters[1].copy()
    sub_filter.integration = sub_filter.integration.copy()
    with pytest.raises(ValueError) as err:
        f.set_filter(1, sub_filter)
    assert "Cannot compound filter from a different integration" in str(
        err.value)

    sub_filter = f.filters[0]
    f.filters = None
    sub_filter.integration = None
    f.set_filter(2, sub_filter)
    assert f.filters == [None, None, sub_filter]
    assert sub_filter.integration is f.integration
    f.set_filter(2, sub_filter)  # Again
    assert f.filters == [None, None, sub_filter]
    f.set_filter(1, sub_filter)
    assert f.filters == [None, sub_filter, sub_filter]


def test_remove_filter(prepared_filter):
    f = prepared_filter
    filters = f.filters.copy()
    f.remove_filter(0)
    assert f.filters == filters[1:]
    f.remove_filter('whiten')
    assert f.filters == filters[1:2]


def test_update_config(prepared_filter):
    f = prepared_filter
    for sub_filter in f.filters:
        sub_filter.is_sub_filter = False
        sub_filter.enabled = False
    f.update_config()
    for sub_filter in f.filters:
        assert sub_filter.is_sub_filter
        assert sub_filter.enabled
    assert not f.is_sub_filter
    assert f.is_enabled()


def test_is_enabled(prepared_filter):
    f = prepared_filter
    assert f.is_enabled()
    for sub_filter in f.filters:
        sub_filter.enabled = False
    assert not f.is_enabled()
    f.update_config()
    assert f.is_enabled()
    f.enabled = False
    assert not f.is_enabled()


def test_pre_filter(prepared_filter):
    f = prepared_filter
    f.parms = None
    for sub_filter in f.filters:
        sub_filter.parms = None
    f.pre_filter()
    assert f.parms is not None
    for sub_filter in f.filters:
        assert sub_filter.parms is not None


def test_post_filter(prepared_filter):
    f = prepared_filter
    integration = f.integration
    assert np.allclose(integration.frames.dependents, 0)
    assert np.allclose(integration.channels.data.dependents, 0)
    assert np.allclose(f.parms.for_frame, 0)
    assert np.allclose(f.parms.for_channel, 0)
    for sub_filter in f.filters:
        assert np.allclose(sub_filter.parms.for_frame, 0)
        assert np.allclose(sub_filter.parms.for_channel, 0)

    for i, flt in enumerate([f] + f.filters):
        parms = flt.parms
        frame_dp = (i + 1.0)
        channel_dp = frame_dp / 10
        parms.for_frame.fill(frame_dp)
        parms.for_channel.fill(channel_dp)

    f.post_filter()
    # Only the main filter dependents should update integration dependents
    assert np.allclose(integration.frames.dependents, 1)
    assert np.allclose(integration.channels.data.dependents, 0.1)


def test_fft_filter(prepared_filter):
    f = prepared_filter
    d0 = f.data.copy()
    f0 = f.integration.frames.data.copy()
    assert np.allclose(prepared_filter.points, 1100)
    for sub_filter in f.filters:
        sub_filter.enabled = False
        assert sub_filter.points is None
    f.fft_filter()
    assert np.allclose(f.data, 0)  # No filtered signal
    f.data = d0.copy()  # Reload the original timestream
    for sub_filter in f.filters:
        sub_filter.enabled = True
        sub_filter.data = None  # Gets reset to the main filter data

    f.fft_filter()
    # Check the filtered signal exists and is non-zero
    assert not np.allclose(f.data, d0) and not np.allclose(f.data, 0)

    for sub_filter in f.filters:
        assert isinstance(sub_filter.data, np.ndarray)
        assert sub_filter.data.dtype == complex  # Is still in FFT form
        assert sub_filter.points is f.points

    # Check the integration data has not been updated at this point
    assert np.allclose(f.integration.frames.data, f0)


def test_response_at(prepared_filter):
    f = prepared_filter
    response = f.response_at(np.arange(f.nf))
    assert np.allclose(response, 0)
    assert response.shape == (f.channels.size, f.nf)
    response = f.response_at(0)
    assert np.allclose(response, 0) and response.shape == (f.channels.size,)

    # Load in the responses
    f.fft_filter()
    response = f.response_at(np.arange(f.nf))
    assert not np.allclose(response, 0)

    # Just leave the kill filter
    for i in [0, 2]:
        f.filters[i].enabled = False

    response = f.response_at(np.arange(f.nf))
    assert np.allclose(np.nonzero(response == 0)[0], [100, 101])

    f.filters = None
    response = f.response_at(np.arange(f.nf))
    assert np.allclose(response, 1) and response.shape == (f.nf,)


def test_get_id(prepared_filter):
    f = prepared_filter
    assert f.get_id() == 'Mf:K:wh'
    f.filters = None
    assert f.get_id() == ''


def test_get_config_name():
    f = MultiFilter()
    assert f.get_config_name() == 'filter'


def test_dft_filter():
    f = MultiFilter()
    with pytest.raises(NotImplementedError) as err:
        f.dft_filter()
    assert "No DFT for varied filters" in str(err.value)
