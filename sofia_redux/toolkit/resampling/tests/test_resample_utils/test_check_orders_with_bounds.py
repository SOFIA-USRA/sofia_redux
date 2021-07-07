# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    check_orders_with_bounds)

import numpy as np
import pytest


@pytest.fixture
def coordinates():
    # All coordinates should range between 0 and 11 in both dimensions, i.e.,
    # There are only 11 unique coordinates for each dimension (121 unique
    # values altogether).  Centered around (5, 5)
    rand = np.random.RandomState(0)
    coordinates = (rand.random((2, 10000)) * 12).astype(int)
    return coordinates


def test_check_orders_with_bounds(coordinates):

    # At a reference position of (5, 5) there are
    # 5 values +/- in each dimension
    o_max = check_orders_with_bounds(
        np.full(2, 10), coordinates, np.full(2, 5.0))
    assert np.allclose(o_max, [5, 5])

    o_max = check_orders_with_bounds(
        np.full(2, 4), coordinates, np.full(2, 5.0))
    assert np.allclose(o_max, [4, 4])

    # Check left
    o_max = check_orders_with_bounds(
        np.array([5, 5]), coordinates, np.array([2, 5]))
    assert np.allclose(o_max, [2, 5])

    # Check right
    o_max = check_orders_with_bounds(
        np.array([5, 5]), coordinates, np.array([9, 5]))
    assert np.allclose(o_max, [2, 5])

    # Check other dimension
    o_max = check_orders_with_bounds(
        np.array([5, 5]), coordinates, np.array([5, 9]))
    assert np.allclose(o_max, [5, 2])


def test_symmetric_order(coordinates):

    o_max = check_orders_with_bounds(
        np.array([4]), coordinates, np.full(2, 5.0))
    assert np.allclose(o_max, [4])

    # Check hitting limit
    o_max = check_orders_with_bounds(
        np.array([5]), coordinates, np.array([5, 9]))
    assert np.allclose(o_max, [2])


def test_check_mask(coordinates):
    mask = np.all(coordinates > 2, axis=0)
    o_max = check_orders_with_bounds(
        np.array([5, 5]), coordinates, np.array([5, 6]), mask=mask)
    assert np.allclose(o_max, [2, 3])


def test_check_required(coordinates):
    o_max = check_orders_with_bounds(
        np.array([5, 5]), coordinates, np.array([5, 9]), required=True)

    assert o_max.size == 2 and o_max[0] == -1

    # Check symmetric order
    o_max = check_orders_with_bounds(
        np.array([5]), coordinates, np.array([5, 9]), required=True)
    assert o_max.size == 1 and o_max[0] == -1
