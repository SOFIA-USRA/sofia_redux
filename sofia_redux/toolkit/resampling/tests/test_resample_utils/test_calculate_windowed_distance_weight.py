# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    calculate_windowed_distance_weight)

import numpy as np


def test_calculate_windowed_distance_weight():

    coordinates = np.full(3, 1 / np.sqrt(3))
    center = np.zeros(3)
    alpha = np.full(3, 0.5)

    # Check weights are clipped outside the unity window
    w = calculate_windowed_distance_weight(
        coordinates + 1e-6, center, alpha)
    assert w == 0

    # Check weighting is appropriate for coordinates inside the window.
    w = calculate_windowed_distance_weight(
        coordinates - 1e-6, center, alpha)
    assert np.isclose(w, 0.13533622086939517)

    assert np.isclose(calculate_windowed_distance_weight(
        np.zeros(3), center, alpha), 1)
