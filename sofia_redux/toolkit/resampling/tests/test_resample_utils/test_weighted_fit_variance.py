# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import weighted_fit_variance


def test_weighted_fit_variance():

    residuals = np.ones(100)
    weights = np.ones(100)
    variance = weighted_fit_variance(residuals, weights, rank=0)
    assert np.isclose(variance, 1 / 100)

    # Test rank
    variance = weighted_fit_variance(residuals, weights, rank=1)
    assert np.isclose(variance, 1 / 99)

    residuals[50] = 2
    variance = weighted_fit_variance(residuals, weights, rank=0)
    assert np.isclose(variance, 0.0103)

    weights[50] = 0
    variance = weighted_fit_variance(residuals, weights, rank=0)
    assert np.isclose(variance, 1 / 100)


def test_rank():
    residuals = np.ones(100)
    weights = np.ones(100)
    for rank in range(99, 102):
        assert np.isclose(weighted_fit_variance(residuals, weights, rank=rank),
                          1)

    assert np.isclose(weighted_fit_variance(residuals, weights, rank=98), 0.5)
