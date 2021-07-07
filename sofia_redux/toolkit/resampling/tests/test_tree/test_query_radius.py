# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np
import pytest


def test_errors():
    tree = Rtree((0, 0))
    with pytest.raises(RuntimeError) as err:
        tree.query_radius(np.array([[0.0], [0.0]]))
    assert "ball tree not initialized" in str(err.value).lower()


def test_single():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = Rtree(coordinates)
    ind, distances = tree.query_radius(np.array([2.5, 2.5]), radius=2.0,
                                       return_distance=True)
    c0 = coordinates[:, ind[0]] - np.full(2, 2.5)[:, None]
    r = np.hypot(*c0)
    assert np.allclose(r, distances[0])


def test_multiple():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = Rtree(coordinates)
    points = np.empty((2, 2))
    points[:, 0] = 2  # (x, y) = (2, 2)
    points[:, 1] = 4  # (x, y) = (4, 4)
    # default radius = 1
    ind, distances = tree.query_radius(points, return_distance=True)
    r0 = np.hypot(*(coordinates[:, ind[0]] - points[:, 0, None]))
    assert np.allclose(r0, distances[0])
    r1 = np.hypot(*(coordinates[:, ind[1]] - points[:, 1, None]))
    assert np.allclose(r1, distances[1])
