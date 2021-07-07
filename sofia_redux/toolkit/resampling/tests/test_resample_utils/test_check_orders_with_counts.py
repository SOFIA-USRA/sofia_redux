# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    check_orders_with_counts)

import numpy as np


def test_check_symmetric_orders_with_counts():
    orders = np.array([2])
    for counts in range(0, 6):
        # Assumes 1-dimension
        o_max = check_orders_with_counts(orders, counts)
        assert o_max.shape == (1,)
        if counts <= 2:
            assert o_max[0] == counts - 1
        else:
            assert o_max[0] == 2

    # Test 2-dimensions - for order 2, requires 3 points for each dimension (9)

    for counts in range(15):
        o_max = check_orders_with_counts(orders, counts, n_dimensions=2)
        assert o_max.shape == (1,)
        if counts == 0:
            assert o_max[0] == -1
        elif counts <= 3:
            assert o_max[0] == 0
        elif counts <= 8:
            assert o_max[0] == 1
        else:
            assert o_max[0] == 2

    # Test symmetric orders supplied for each dimension
    orders = np.full(2, 2)
    for counts in range(15):
        o_max = check_orders_with_counts(orders, counts, n_dimensions=2)
        assert o_max.shape == (2,)
        if counts == 0:
            assert o_max[0] == -1
        elif counts <= 3:
            assert np.allclose(o_max, 0)
        elif counts <= 8:
            assert np.allclose(o_max, 1)
        else:
            assert np.allclose(o_max, 2)


def test_mask():
    counts = -1
    mask = np.full(10, True)  # counts = 10
    orders = np.array([2])
    o_max = check_orders_with_counts(orders, counts, mask=mask, n_dimensions=2)
    assert o_max.shape == (1,)
    assert o_max[0] == 2
    mask[5:] = False
    o_max = check_orders_with_counts(orders, counts, mask=mask, n_dimensions=2)
    assert o_max[0] == 1
    mask[2:] = False
    o_max = check_orders_with_counts(orders, counts, mask=mask, n_dimensions=2)
    assert o_max[0] == 0
    mask[:] = False
    o_max = check_orders_with_counts(orders, counts, mask=mask, n_dimensions=2)
    assert o_max[0] == -1


def test_asymmetric_orders():
    orders = np.array([1, 2])

    # Required is always True for asymmetric orders
    for counts in range(10):
        o_max = check_orders_with_counts(orders, counts, required=False)
        assert o_max.shape == (2,)
        if counts <= 5:
            assert o_max[0] == -1
        else:
            assert np.allclose(o_max, orders)


def test_required():

    orders = np.full(2, 2)
    for counts in range(10):
        o_max = check_orders_with_counts(orders, counts, required=True)
        assert o_max.shape == (2,)
        if counts < 9:
            assert o_max[0] == -1
        else:
            assert np.allclose(o_max, orders)


def test_minimum_points():

    orders = np.full(2, 2)
    for counts in range(10):
        # Actually requires 9 points
        o_max = check_orders_with_counts(orders, counts, required=True,
                                         minimum_points=4)
        if counts < 4:
            assert o_max[0] == -1
        else:
            assert np.allclose(o_max, orders)
