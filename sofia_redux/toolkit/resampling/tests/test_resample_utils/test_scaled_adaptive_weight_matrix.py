# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils \
    import scaled_adaptive_weight_matrix

import numpy as np


def test_rchi2_scaling():
    sigma = np.full(3, 0.5)  # 0.5 also equals alpha = 0.5, inverse_alpha = 2

    # Check rchi2 scaling
    for rchi2 in np.linspace(0.5, 1.5, 10):
        scaling = rchi2 ** (1 / 6)  # sqrt(rchi2)^(1/dimensions)
        assert np.allclose(scaled_adaptive_weight_matrix(sigma, rchi2),
                           2 * scaling)


def test_fixed_dimensions():
    sigma = np.full(3, 0.5)
    fixed = np.full(3, False)
    fixed[0] = True

    rchi2 = 4.0  # rchi = 2
    inverse_alpha = scaled_adaptive_weight_matrix(sigma, rchi2, fixed=fixed)
    assert inverse_alpha[0] == 2
    assert np.allclose(inverse_alpha[1:], 2 * np.sqrt(2))


def test_bad_values():
    sigma = np.full(3, 0.5)

    # Bad rchi2 values
    assert np.allclose(scaled_adaptive_weight_matrix(sigma, 0), 2)
    assert np.allclose(scaled_adaptive_weight_matrix(sigma, -1.0), 2)
    assert np.isnan(scaled_adaptive_weight_matrix(sigma, np.nan)[0])

    # All dimensions fixed
    assert np.allclose(
        scaled_adaptive_weight_matrix(sigma, 100, fixed=np.full(3, True)), 2)
