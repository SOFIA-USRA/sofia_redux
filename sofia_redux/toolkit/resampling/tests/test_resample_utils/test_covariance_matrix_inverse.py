# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils \
    import covariance_matrix_inverse

import numpy as np


def test_supplied_amat():
    amat = np.array([[2.0, 0], [0, 3]])
    cov = covariance_matrix_inverse(amat, np.empty((0, 0)),
                                    np.empty(0), np.empty(0))
    assert np.allclose(cov, [[1 / 2, 0], [0, 1 / 3]])

    cov_ranked = covariance_matrix_inverse(
        amat, np.empty((0, 10)), np.empty(10), np.empty(10))

    assert np.allclose(cov, cov_ranked * (8 / 10))  # * (n - rank) / n


def test_covariance_matrix():
    w = np.diag(np.ones(10) * 0.5)
    e = np.ones(10) * 3
    var_y = np.diag(1 / (e ** 2))
    x = np.arange(10, dtype=float)
    phi = np.stack(((x ** 0), (x ** 1)))
    amat = np.empty((0, 0))

    expected = np.linalg.pinv(phi @ w @ var_y @ phi.T)
    cov = covariance_matrix_inverse(amat, phi, e, np.diag(w), rank=0)
    assert np.allclose(cov, expected)
