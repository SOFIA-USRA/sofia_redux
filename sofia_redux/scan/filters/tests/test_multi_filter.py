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
