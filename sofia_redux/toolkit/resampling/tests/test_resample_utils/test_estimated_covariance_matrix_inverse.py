# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    estimated_covariance_matrix_inverse)

import numpy as np


def test_estimated_covariance_matrix():

    x = np.linspace(-1, 1, 100)
    phi = np.stack(((x ** 0), (x ** 1)))  # y = mx + c
    weights = np.ones(100) * 0.5
    error = np.ones(100) * 2
    rank = 2

    av_error_squared = np.sum(weights * error ** 2) / np.sum(weights)
    const = av_error_squared * (100 / (100 - rank))

    cov = estimated_covariance_matrix_inverse(phi, error, weights, rank=rank)
    assert np.allclose([cov[0, 1], cov[1, 0]], 0)  # because sum(wx) = 0
    assert np.allclose(cov[0, 0], const / np.sum(weights))
    assert np.allclose(cov[1, 1], const / np.sum(weights * x ** 2))

    cov_find_rank = estimated_covariance_matrix_inverse(phi, error, weights)
    assert np.allclose(cov_find_rank, cov)

    cov0 = estimated_covariance_matrix_inverse(phi, error, weights, rank=0)
    assert np.allclose(cov0, [[0.08, 0], [0, 0.235247525]])

    cov_bad_rank = estimated_covariance_matrix_inverse(phi, error, weights,
                                                       rank=1000)
    assert np.allclose(cov_bad_rank, cov0)
