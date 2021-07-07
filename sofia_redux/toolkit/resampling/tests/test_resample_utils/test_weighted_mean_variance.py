# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils \
    import weighted_mean_variance


def test_weighted_mean_variance():

    variance = np.full(10, 4.0)
    weights = np.full(10, 1.0)
    assert np.isclose(weighted_mean_variance(variance, weights), 4 / 10)

    weights /= 10
    assert np.isclose(weighted_mean_variance(variance, weights), 4 / 10)

    variance[5] = 16
    assert np.isclose(weighted_mean_variance(variance, weights), 0.52)

    weights[5] = 0
    assert np.isclose(weighted_mean_variance(variance, weights), 4 / 9)


def test_weightsum():

    variance = np.ones(10) * 4
    weights = np.ones(10)
    assert np.isclose(weighted_mean_variance(variance, weights, weightsum=1.0),
                      40)


def test_numba_poison_values():

    variance = np.ones(10)
    weights = np.ones(10)
    variance[5] = np.nan
    assert np.isnan(weighted_mean_variance(variance, weights))

    variance[5] = np.inf
    assert weighted_mean_variance(variance, weights) == np.inf

    variance[5] = -np.inf
    assert weighted_mean_variance(variance, weights) == -np.inf

    weights[5] = 0.0
    assert np.isnan(weighted_mean_variance(variance, weights))
