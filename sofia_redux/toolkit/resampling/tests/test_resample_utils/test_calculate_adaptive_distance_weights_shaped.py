# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    calculate_adaptive_distance_weights_shaped)

import numpy as np


def test_calculate_adaptive_distance_weights_shaped():

    alpha_base = np.zeros((2, 2))
    alpha_base[0, 0] = 1.0  # elongated in first dimension (zero rotation)
    alpha_base[1, 1] = 0.5
    # normalize
    alpha_base /= np.sqrt(np.linalg.det(alpha_base))

    # alpha contains 9 samples, 1 set, 2 dimensions (*2)
    alpha = np.zeros((9, 1, 2, 2))
    alpha[:, :] = alpha_base.copy()

    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
    coordinates = np.stack((x.ravel(), y.ravel()))  # 9 coordinates
    reference = np.ones(2)  # reference position is center of array

    weights = calculate_adaptive_distance_weights_shaped(
        coordinates, reference, alpha)

    w_no_rotation = weights[0].reshape((3, 3))

    # Test no rotation occurred
    assert w_no_rotation[0, 0] == w_no_rotation[0, 2]

    # Add rotation and normalize
    alpha_base[0, 1] = 0.1
    alpha_base[1, 0] = 0.1
    alpha_base /= np.sqrt(np.linalg.det(alpha_base))
    alpha[:, :] = alpha_base.copy()

    weights = calculate_adaptive_distance_weights_shaped(
        coordinates, reference, alpha)

    w_rotated = weights[0].reshape((3, 3))
    # Test rotation occurred
    assert w_rotated[0, 0] != w_rotated[0, 2]

    # Should be the same shape/size, just rotated
    # atol accounts for discretization.
    assert np.allclose(w_no_rotation.sum(), w_rotated.sum(), atol=1 / 9)
