# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    solve_mean_fit, solve_rchi2_from_variance)

import numpy as np


def test_solve_mean_fit():
    data = np.array([1, 2, 3, 3, 2, 1], dtype=float)
    error = np.full(6, 2.0)
    weights = np.ones(6)

    for calculate_variance in [True, False]:
        for calculate_rchi2 in [True, False]:
            mean, variance, rchi2 = solve_mean_fit(
                data, error, weights,
                calculate_variance=calculate_variance,
                calculate_rchi2=calculate_rchi2)
            assert mean == 2
            assert variance == (2 / 3 if calculate_variance else 0)
            assert rchi2 == (0.2 if calculate_rchi2 else 0)

    no_error = np.empty(0)
    mean, variance, rchi2 = solve_mean_fit(data, no_error, weights)
    assert mean == 2
    assert rchi2 == 1
    assert variance == 12 / 90

    residuals = data - mean
    assert solve_rchi2_from_variance(residuals, weights, variance, rank=1) == 1
