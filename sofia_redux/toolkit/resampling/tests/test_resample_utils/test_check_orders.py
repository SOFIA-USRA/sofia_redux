# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import check_orders

import numpy as np


def test_check_orders():
    rand = np.random.RandomState(0)
    coordinates = rand.normal(loc=0.0, scale=1.0, size=(2, 1000))
    reference = np.zeros(2)
    mask = np.full(1000, True)
    orders = np.array([2, 2])
    minimum_points = None
    required = False
    counts = mask.sum()

    for algorithm in range(5):
        max_orders = check_orders(orders, coordinates, reference, algorithm,
                                  mask=mask, minimum_points=minimum_points,
                                  required=required, counts=counts)
        assert max_orders.shape == (2,)
        if algorithm in [1, 2, 3]:  # The currently available algorithms
            assert np.allclose(max_orders, [2, 2])
        else:
            assert max_orders[0] == -1
