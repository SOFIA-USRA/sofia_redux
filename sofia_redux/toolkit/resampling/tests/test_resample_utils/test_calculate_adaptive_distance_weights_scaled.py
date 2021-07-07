# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    calculate_adaptive_distance_weights_scaled)

import numpy as np


def test_calculate_adaptive_distance_weights_scaled():
    # alpha contains 9 samples, 2 sets, 1 dummy, 2 dimensions
    alpha = np.zeros((9, 2, 1, 2))
    reference = np.zeros(2)
    alpha[:, :, :, 0] = 1.0
    alpha[:, :, :, 1] = 0.5
    alpha[:, 1] /= 4  # different alpha values for set 2
    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
    coordinates = np.stack((x.ravel(), y.ravel()))  # 9 coordinates

    weights = calculate_adaptive_distance_weights_scaled(
        coordinates, reference, alpha)

    w0 = weights[0].reshape((3, 3))
    w1 = weights[1].reshape((3, 3))

    dr2_set0 = (x ** 2) + (0.5 * y ** 2)
    assert np.allclose(np.exp(-dr2_set0), w0)

    dr2_set1 = (0.25 * x ** 2) + (0.125 * y ** 2)
    assert np.allclose(np.exp(-dr2_set1), w1)
