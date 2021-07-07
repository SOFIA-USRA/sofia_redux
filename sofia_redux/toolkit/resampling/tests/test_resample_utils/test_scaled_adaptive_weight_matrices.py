# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    scaled_adaptive_weight_matrices)

import numpy as np


def test_rchi2_scaling_and_shape():
    sigma = np.full(3, 0.5)  # 0.5 also equals alpha = 0.5, inverse_alpha = 2

    # Check rchi2 scaling
    # 2 data sets of 5x10 output fit array
    rchi2 = np.linspace(0, 2, 100).reshape((2, 5, 10))

    inverse_alpha = scaled_adaptive_weight_matrices(sigma, rchi2)
    assert inverse_alpha.shape == (2, 5, 10, 1, 3)

    # Check the zero value
    assert np.allclose(inverse_alpha[0, 0, 0, :], 2)

    # Check the rest of the values
    scaling_factor = rchi2 ** (1 / 6)  # sqrt(rchi2) ^ (1 / n_dimensions)
    expected = 2 * scaling_factor
    expected[0, 0, 0] = 2.0  # Already checked the zero value above.

    assert np.allclose(inverse_alpha, expected[..., None, None])


def test_fixed_dimensions():
    sigma = np.full(3, 0.5)
    fixed = np.array([True, True, False])
    rchi2 = np.random.random((2, 100))

    # Just check we get different results
    standard_result = scaled_adaptive_weight_matrices(sigma, rchi2)
    fixed_result = scaled_adaptive_weight_matrices(sigma, rchi2, fixed=fixed)
    assert not np.allclose(standard_result, fixed_result)
