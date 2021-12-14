# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_base import ResampleBase
from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
from sofia_redux.toolkit.resampling.grid.base_grid import BaseGrid

import numpy as np
import pytest


@pytest.fixture
def inputs():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    return coordinates, data


def test_features(inputs):
    coordinates, data = inputs
    r = ResampleBase(coordinates, data)
    assert r.features == 2
    with pytest.raises(AttributeError) as err:
        r.features = 1
    assert "can't set attribute" in str(err.value)


def test_multiset(inputs):
    coordinates, data = inputs
    assert not ResampleBase(coordinates, data).multi_set

    d2 = np.stack([data, data])
    r = ResampleBase(coordinates, d2)
    assert r.multi_set

    with pytest.raises(AttributeError) as err:
        r.multi_set = False
    assert "can't set attribute" in str(err.value)


def test_n_sets(inputs):
    coordinates, data = inputs
    assert ResampleBase(coordinates, data).n_sets == 1

    d2 = np.stack([data, data])
    r = ResampleBase(coordinates, d2)
    assert r.n_sets == 2
    with pytest.raises(AttributeError) as err:
        r.n_sets = 3
    assert "can't set attribute" in str(err.value)


def test_n_samples(inputs):
    coordinates, data = inputs
    r = ResampleBase(coordinates, data)
    assert r.n_samples == 100
    with pytest.raises(AttributeError) as err:
        r.n_samples = 101
    assert "can't set attribute" in str(err.value)


def test_window(inputs):
    coordinates, data = inputs
    r = ResampleBase(coordinates, data, window=2)
    w = r.window
    assert w.shape == (2,)
    assert np.allclose(w, 2.0)
    with pytest.raises(AttributeError) as err:
        r.window = 3
    assert "can't set attribute" in str(err.value)


def test_fit_settings(inputs):
    coordinates, data = inputs
    r = ResampleBase(coordinates, data)
    assert r.fit_settings is None

    settings = {'test', 'foo'}
    r._fit_settings = settings
    assert r.fit_settings is settings

    with pytest.raises(AttributeError) as err:
        r.fit_settings = settings
    assert "can't set attribute" in str(err.value)


def test_fit_tree(inputs):
    coordinates, data = inputs
    r = ResampleBase(coordinates, data)
    assert r.fit_tree is None
    r.pre_fit({}, r.coordinates)
    assert isinstance(r.fit_tree, BaseTree)

    with pytest.raises(AttributeError) as err:
        r.fit_tree = None
    assert "can't set attribute" in str(err.value)


def test_grid_class(inputs):
    coordinates, data = inputs
    r = ResampleBase(coordinates, data)
    assert r.grid_class == BaseGrid
