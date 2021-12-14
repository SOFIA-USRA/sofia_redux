# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_base import ResampleBase
from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
from sofia_redux.toolkit.resampling.grid.base_grid import BaseGrid

import numpy as np
import pytest


@pytest.fixture
def test_resampler():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    return ResampleBase(coordinates, data)


def test_init():
    r = ResampleBase(np.arange(10), np.arange(10))
    assert r.data is not None
    assert r.coordinates is not None
    assert isinstance(r.sample_tree, BaseTree)


def test_global_resampling_values(test_resampler):
    r = test_resampler
    g = r.global_resampling_values()
    assert isinstance(g, dict)
    g['foo'] = 'bar'
    r2 = test_resampler
    g = r2.global_resampling_values()
    assert g.get('foo') == 'bar'


def test_get_grid_class(test_resampler):
    r = test_resampler
    assert r.get_grid_class() == BaseGrid


def test_check_input_arrays():
    n_features = 3
    n_samples = 100
    coordinates = tuple(x for x in np.random.random((n_features, n_samples)))
    data = np.random.random(n_samples)
    c, d, e, m = ResampleBase._check_input_arrays(coordinates, data)
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert e is None
    assert m is None

    c, d, e, m = ResampleBase._check_input_arrays(coordinates, data,
                                                  error=data.copy())
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d, e)
    assert m is None

    c, d, e, m = ResampleBase._check_input_arrays(coordinates, data,
                                                  mask=data > 0.5)
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d > 0.5, m)
    assert e is None

    c, d, e, m = ResampleBase._check_input_arrays(coordinates, data,
                                                  mask=data > 0.5,
                                                  error=data.copy())
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d > 0.5, m)
    assert np.allclose(d, e)

    c, d, e, m = ResampleBase._check_input_arrays(coordinates, data,
                                                  mask=data > 0.5,
                                                  error=1.0)
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d > 0.5, m)
    assert np.allclose(e, 1) and e.size == 1

    c, d, e, m = ResampleBase._check_input_arrays(
        coordinates, np.vstack([data, data]))
    assert d.shape == (2, n_samples)

    c, d, e, m = ResampleBase._check_input_arrays(
        coordinates, np.vstack([data, data]), error=[1, 2])
    assert np.allclose(e, [[1], [2]])

    f = ResampleBase._check_input_arrays
    r = f(np.arange(10), np.arange(10))
    assert np.allclose(r[0], np.arange(10))
    assert np.allclose(r[1], r[0])
    assert r[2] is None
    assert r[3] is None

    with pytest.raises(ValueError) as err:
        f(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
    assert 'or 2 (n_features, n_samples) axes' in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        f(np.arange(10), np.zeros((2, 2, 10)))
    assert 'data must have 1 or 2 (multi-set) dimensions' in str(
        err.value).lower()

    with pytest.raises(ValueError) as err:
        f(np.arange(10), np.arange(9))
    assert "data sample size does not match coordinates" in str(
        err.value).lower()

    with pytest.raises(ValueError) as err:
        f(np.arange(10), np.arange(10), error=np.arange(9))
    assert "Error must be a single value, an array" in str(err.value)

    with pytest.raises(ValueError) as err:
        f(np.arange(10), np.arange(10), mask=np.arange(9))
    assert "mask shape does not match data" in str(err.value).lower()


def test_set_sample_tree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    r = ResampleBase(coordinates, data)
    r.sample_tree = None
    r.set_sample_tree(coordinates)
    assert np.allclose(r.window, [1.1680961, 1.1680961])
    assert isinstance(r.sample_tree, BaseTree)

    r.set_sample_tree(coordinates, radius=2.0)
    assert np.allclose(r.window, [2, 2])

    r.set_sample_tree(coordinates, window_estimate_bins=20)
    assert np.allclose(r.window, [0.58404805, 0.58404805])

    r.set_sample_tree(coordinates, window_estimate_oversample=3.0)
    assert np.allclose(r.window, [1.32948452, 1.32948452])


def test_scale_to_window():
    rand = np.random.RandomState(1)
    coordinates = rand.random((2, 101))
    data = np.ones(coordinates.shape[1])
    r = ResampleBase(coordinates, data)
    c = r._scale_to_window(coordinates, radius=None)
    assert np.allclose(r.window, [0.09074154, 0.09037629])
    assert c.shape == coordinates.shape
    r._scale_to_window(coordinates, radius=None, feature_bins=20)
    assert np.allclose(r.window, [0.06416395, 0.06390569])
    r._scale_to_window(coordinates, radius=None, percentile=10)
    assert np.allclose(r.window, [0.12832791, 0.12781137])
    r._scale_to_window(coordinates, radius=None, oversample=3)
    assert np.allclose(r.window, [0.10327872, 0.10286301])
    r._scale_to_window(coordinates, radius=0.2)
    assert np.allclose(r.window, [0.2, 0.2])


def test_calculate_minimum_points(test_resampler):
    r = test_resampler
    assert r.calculate_minimum_points() == 1


def test_check_call_arguments(test_resampler):
    r = test_resampler
    r._check_call_arguments(1, 2)
    with pytest.raises(ValueError) as err:
        r._check_call_arguments(1, 2, 3)
    assert "3-feature coordinates passed to 2-feature resampler" in str(
        err.value)


def test_process_block(test_resampler):
    r = test_resampler
    filename = None
    iteration = 1
    assert r.process_block((filename, iteration), 0) is None
