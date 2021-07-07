# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import weighted_variance


def test_weighted_variance():

    error = np.full(10, 2.0)
    weights = np.full(10, 1.0)
    assert np.isclose(weighted_variance(error, weights), 4 / 10)

    weights /= 10
    assert np.isclose(weighted_variance(error, weights), 4 / 10)

    error[5] = 4
    assert np.isclose(weighted_variance(error, weights), 0.52)

    weights[5] = 0
    assert np.isclose(weighted_variance(error, weights), 4 / 9)


def test_weightsum():

    error = np.ones(10) * 2
    weights = np.ones(10)
    assert np.isclose(weighted_variance(error, weights, weightsum=1.0), 40)


def test_numba_poison_values():

    error = np.ones(10)
    weights = np.ones(10)
    error[5] = np.nan
    assert np.isnan(weighted_variance(error, weights))

    error[5] = np.inf
    assert weighted_variance(error, weights) == np.inf

    error[5] = -np.inf
    assert weighted_variance(error, weights) == np.inf

    weights[5] = 0.0
    assert np.isnan(weighted_variance(error, weights))
