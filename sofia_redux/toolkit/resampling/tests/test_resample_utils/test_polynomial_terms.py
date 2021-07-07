# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import (
    polynomial_terms, single_polynomial_terms, multiple_polynomial_terms,
    polynomial_exponents)


def test_polynomial_terms():
    random = np.random.RandomState(41)
    exponents = polynomial_exponents(2, ndim=2)
    datasets = random.rand(2, 100)  # 2-D, 100 points

    # test multiple_polynomial_terms
    x = polynomial_terms(datasets, exponents)
    assert x.shape == (6, 100)
    assert np.allclose(x[5, 0], datasets[1, 0] ** 2)

    # test single_polynomial_terms
    x = polynomial_terms(datasets[:, 0], exponents)
    assert x.shape == (6,)
    assert np.allclose(x[4], datasets[0, 0] * datasets[1, 0])


def test_single_polynomial_terms():
    vector = np.arange(3, dtype=np.float64) + 1
    exponents = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [1, 2, 3]
    ])

    result = single_polynomial_terms(vector, exponents)
    assert result.shape == (4,)
    assert result[0] == 1  # 1^0.2^0.3^0
    assert result[1] == 6  # 1^1.2^1.3^1
    assert result[2] == 36  # 1^2.2^2.3^2
    assert result[3] == 108  # 1^1.2^2.3^3


def test_multiple_polynomial_terms():
    vectors = np.repeat(np.arange(3, dtype=np.float64)[:, None] + 1, 3, axis=1)
    exponents = np.array([
        [0, 0, 0],
        [2, 2, 2],
        [1, 2, 3],
    ])
    result = multiple_polynomial_terms(vectors, exponents)
    assert np.allclose(result, [[1], [36], [108]])
    assert result.shape == (3, 3)
