# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    calculate_distance_weights_from_matrix)

import numpy as np


def test_calculate_distance_weights_from_matrix():
    # spacing is 0.2 in both dimensions
    x = np.stack([x.ravel() for x in
                 np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))])
    alpha = np.eye(2) * 0.5
    reference = np.zeros(2)

    weights = calculate_distance_weights_from_matrix(x, reference, alpha)
    weights = weights.reshape((11, 11))
    assert weights[5, 5] == 1
    assert weights[5, 6] == np.exp(-(0.2 * 0.2 / 2))
    assert weights[10, 10] == np.exp(-1)
