# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample import Resample

import numpy as np
import pytest


def test_check_input_arrays():
    n_features = 3
    n_samples = 100
    coordinates = tuple(x for x in np.random.random((n_features, n_samples)))
    data = np.random.random(n_samples)
    c, d, e, m = Resample._check_input_arrays(coordinates, data)
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert e is None
    assert m is None

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              error=data.copy())
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d, e)
    assert m is None

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              mask=data > 0.5)
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d > 0.5, m)
    assert e is None

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              mask=data > 0.5,
                                              error=data.copy())
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d > 0.5, m)
    assert np.allclose(d, e)

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              mask=data > 0.5,
                                              error=1.0)
    assert c.shape == (n_features, n_samples)
    assert d.size == n_samples
    assert np.allclose(d > 0.5, m)
    assert np.allclose(e, 1) and e.size == d.size

    c, d, e, m = Resample._check_input_arrays(
        coordinates, np.vstack([data, data]))
    assert d.shape == (2, n_samples)

    f = Resample._check_input_arrays
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
    assert "error shape does not match data" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        f(np.arange(10), np.arange(10), mask=np.arange(9))
    assert "mask shape does not match data" in str(err.value).lower()
