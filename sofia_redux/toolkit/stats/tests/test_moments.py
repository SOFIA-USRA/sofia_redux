# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from scipy.stats import skewnorm

from sofia_redux.toolkit.stats.stats import moments


def test_invalid_input():
    with pytest.raises(ValueError):
        moments([1, 2, 3], mask=[True, True])


def test_expected_results():
    a = 4
    mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
    gen = skewnorm(a)
    data = gen.rvs(1000000, random_state=42)
    result = moments(data)
    assert np.isclose(mean, result['mean'], atol=0.03)
    assert np.isclose(var, result['var'], atol=0.03)
    assert np.isclose(skew, result['skewness'], atol=0.03)
    assert np.isclose(kurt, result['kurtosis'], atol=0.03)
    assert result['mask'].all()

    rand = np.random.RandomState(42)
    data = rand.rand(100)
    data[20:30] = np.nan
    result = moments(data)
    assert result['mask'].sum() == 90

    data[0] = result['mean'] * 10
    assert result['mask'][0]
    result = moments(data, threshold=5)
    assert not result['mask'][0]
    data[:-1] = np.nan
    result = moments(data)
    assert result['mean'] == data[-1]
    assert result['var'] == 0
    assert result['stddev'] == 0
    assert result['skewness'] == 0
    assert result['kurtosis'] == -3


def test_mask():
    rand = np.random.RandomState(42)
    data = rand.rand(10)
    mask = np.full(10, True)
    mask[0] = False
    assert moments(data, mask=mask)['mask'].sum() == 9
    m = moments(data, mask=mask, get_mask=True)
    assert m[1:].all()
    assert not m[0]


def test_axis():
    y, x = np.mgrid[:5, :5]
    result = moments(x, axis=0)
    assert np.allclose(result['mean'], [0, 1, 2, 3, 4])
    assert np.allclose(result['var'], 0)
    result = moments(x, axis=1)
    assert np.allclose(result['mean'], 2)
    assert np.allclose(result['var'], 2.5)
    result = moments(x, axis=0, threshold=1.0)
    assert not result['mask'].any()
