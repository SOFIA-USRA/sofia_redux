# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np


def test_shape():
    tree = Rtree([1])
    tree._set_shape((3, 3))
    assert tree.shape == (3, 3)
    assert tree.features == 2
    assert tree.n_blocks == 9
    assert np.allclose(tree.search,
                       [[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                        [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
