# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import variance_from_offsets

import numpy as np


def test_variance_from_offsets():
    rand = np.random.RandomState(0)
    covariance = rand.random((2, 2))
    covariance[0, 1] = covariance[1, 0]
    offsets = rand.random(2)

    sigma_inv = np.linalg.pinv(covariance)

    variance = variance_from_offsets(offsets, covariance)
    expected = offsets @ sigma_inv @ offsets
    assert np.isclose(variance, expected)

    # Test sigma_inv is passed in correctly
    variance = variance_from_offsets(offsets, np.zeros((2, 2)),
                                     sigma_inv=sigma_inv)
    assert np.isclose(variance, expected)
