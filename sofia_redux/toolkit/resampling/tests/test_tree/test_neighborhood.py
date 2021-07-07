# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np


def test_neighbor():
    tree = Rtree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    block = tree.to_index([3, 3])
    hood = tree.neighborhood(block)

    lower_coords = tree.from_index(hood)
    assert np.allclose(lower_coords,
                       [[2, 2, 2, 3, 3, 3, 4, 4, 4],  # x
                        [2, 3, 4, 2, 3, 4, 2, 3, 4]])  # y


def test_culling():
    tree = Rtree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    block = tree.to_index([0, 0])  # corner block
    hood = tree.neighborhood(block)
    assert np.sum(hood != -1) == 4  # should only be 4 valid neighboring blocks

    hood, valid = tree.neighborhood(block, valid_neighbors=True)
    assert np.all(hood[valid] >= 0)

    hood = tree.neighborhood(block, cull=True)
    assert hood.size == 4
    assert np.all(hood >= 0)
