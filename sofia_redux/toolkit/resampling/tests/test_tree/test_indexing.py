# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np


def test_to_index():
    tree = Rtree((3, 7))
    index = tree.to_index([2.2, 2.2])
    assert index.shape == () and index == 16  # (7 * 2) + 2
    index = tree.to_index([[2.2], [2.2]])
    assert index.shape == (1,) and np.allclose(index, 16)


def test_from_index():
    tree = Rtree((3, 7))
    coordinates = tree.from_index(16)
    assert np.allclose(coordinates, 2) and coordinates.shape == (2,)
    coordinates = tree.from_index([16])
    assert np.allclose(coordinates, 2) and coordinates.shape == (2, 1)
