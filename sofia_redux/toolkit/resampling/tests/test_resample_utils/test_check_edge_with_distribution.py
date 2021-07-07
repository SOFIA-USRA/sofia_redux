# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    check_edge_with_distribution)

import numpy as np


def test_check_edge_by_distribution():
    rand = np.random.RandomState(0)
    # 2-D normal distribution with deviation = 1, mean = 0
    coordinates = rand.multivariate_normal(np.zeros(2), np.eye(2), 100000).T

    one_sigma_edge = np.full(2, 1 / np.sqrt(2))
    mask = np.full(100000, True)

    # Check a reference coordinate inside the "edge"
    assert check_edge_with_distribution(coordinates, one_sigma_edge * 0.95,
                                        mask, 1.0)

    # Check a reference coordinate outside the "edge"
    assert not check_edge_with_distribution(coordinates, one_sigma_edge * 1.05,
                                            mask, 1.0)

    # Now define a new edge and do the same
    assert check_edge_with_distribution(coordinates, one_sigma_edge,
                                        mask, 0.95)

    assert not check_edge_with_distribution(coordinates, one_sigma_edge,
                                            mask, 1.05)
