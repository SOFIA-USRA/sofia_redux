# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import weighted_mean


def test_weighted_mean():

    data = np.ones(10)
    weights = np.ones(10)
    assert np.isclose(weighted_mean(data, weights), 1)

    data[5] = 2
    assert np.isclose(weighted_mean(data, weights), 1.1)

    weights[5] = 0
    assert np.isclose(weighted_mean(data, weights), 1)


def test_weightsum():

    data = np.ones(10)
    weights = np.ones(10)
    assert np.isclose(weighted_mean(data, weights, weightsum=1.0), 10)


def test_numba_poison_values():

    data = np.ones(10)
    weights = np.ones(10)
    data[5] = np.nan
    assert np.isnan(weighted_mean(data, weights))

    data[5] = np.inf
    assert weighted_mean(data, weights) == np.inf

    data[5] = -np.inf
    assert weighted_mean(data, weights) == -np.inf

    weights[5] = 0.0
    assert np.isnan(weighted_mean(data, weights))
