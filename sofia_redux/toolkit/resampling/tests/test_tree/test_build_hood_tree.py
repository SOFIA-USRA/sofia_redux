# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np


def test_build_hood_tree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = Rtree((0, 0))
    tree.coordinates = coordinates
    tree._set_shape((5, 6))
    assert not tree.hood_initialized
    tree._build_hood_tree()
    assert tree.hood_initialized
    assert tree.block_population.shape == (30,)
    assert np.allclose(tree.block_population, 1)
    assert tree.max_in_hood.shape == (30,)
    assert np.allclose(tree.max_in_hood, 1)
    assert tree.hood_population.shape == (30,)

    # Corners, edges, and completely surrounded
    assert np.allclose(np.unique(tree.hood_population), [4, 6, 9])
