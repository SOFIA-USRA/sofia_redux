# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    solve_rchi2_from_variance)

import numpy as np


def test_solve_rchi2_from_variance():
    residuals = np.full(100, -1.0) ** np.arange(100)
    weights = np.full(100, 1 / 100)
    variance = 2.0
    weightsum = float(np.sum(weights))

    rchi2 = solve_rchi2_from_variance(residuals, weights, variance,
                                      weightsum=weightsum, rank=1)

    expected = np.sum(weights * residuals ** 2) / weightsum / variance
    scaled_expected = expected / 99
    assert np.allclose(scaled_expected, rchi2)

    # test no weightsum
    rchi2 = solve_rchi2_from_variance(residuals, weights, variance, rank=1)
    assert np.allclose(scaled_expected, rchi2)

    # test bad rank
    rchi2 = solve_rchi2_from_variance(residuals, weights, variance, rank=1e6)
    assert np.allclose(rchi2, expected)
