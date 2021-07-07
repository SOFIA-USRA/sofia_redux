# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    check_orders_without_bounds)

import numpy as np


def test_check_orders_without_bounds_1d():
    coordinates = np.hstack([np.arange(5), np.arange(5)])[None]
    max_order = 4
    for order in range(10):
        orders = np.atleast_1d(order)
        max_orders = check_orders_without_bounds(orders, coordinates)
        assert max_orders.shape == (1,)
        if order <= max_order:
            assert order == max_orders[0]
        else:
            assert max_orders[0] == max_order

        # Check required keyword sets max_orders to -1 if order is not possible
        max_orders = check_orders_without_bounds(orders, coordinates,
                                                 required=True)
        assert max_orders.shape == (1,)
        if order <= max_order:
            assert order == max_orders[0]
        else:
            assert max_orders[0] == -1


def test_mask():
    coordinates = np.arange(5, dtype=float)[None]
    max_order = check_orders_without_bounds(np.array([10]), coordinates,
                                            mask=None)
    assert max_order.size == 1 and max_order[0] == 4

    max_order = check_orders_without_bounds(np.array([10]), coordinates,
                                            mask=np.arange(5) < 3)
    assert max_order.size == 1 and max_order[0] == 2


def test_check_orders_without_bounds_2d():
    coordinates = np.array([[0.0, 0, 1, 2, 3, 4, 4],  # 5 unique values
                            [0.0, 1, 2, 3, 4, 5, 5]])  # 6 unique values

    # Test "symmetric orders"
    max_order = check_orders_without_bounds(np.array([10]), coordinates)

    # minimum order (5 - 1 = 4)
    assert max_order.size == 1 and max_order[0] == 4

    # Test orders for each dimension
    max_order = check_orders_without_bounds(np.array([10, 10]), coordinates)
    assert np.allclose(max_order, [4, 5])

    # Test order clipping
    max_order = check_orders_without_bounds(np.array([1, 2]), coordinates)
    assert np.allclose(max_order, [1, 2])

    # Test required
    max_order = check_orders_without_bounds(np.array([10, 10]), coordinates,
                                            required=True)
    assert max_order[0] == -1

    # Test mask
    mask = np.full(7, True)
    mask[4] = False
    max_order = check_orders_without_bounds(np.array([10, 10]), coordinates,
                                            mask=mask)
    assert np.allclose(max_order, [3, 4])
