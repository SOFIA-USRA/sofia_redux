# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np
import pytest


def test_build_tree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = Rtree((0, 0))
    tree.build_tree(coordinates, method='all')
    assert tree.shape == (5, 6)
    assert tree.balltree_initialized
    assert tree.hood_initialized

    tree.build_tree(coordinates, shape=(10, 12))
    assert tree.shape == (10, 12)

    tree = Rtree((0, 0))
    tree.build_tree(coordinates, method='hood')
    assert not tree.balltree_initialized
    assert tree.hood_initialized

    tree = Rtree((0, 0))
    tree.build_tree(coordinates, method='balltree', leaf_size=40)
    assert tree.balltree_initialized
    assert not tree.hood_initialized

    tree = Rtree((0, 0))
    tree.build_tree(coordinates, method=None)
    assert not tree.balltree_initialized
    assert not tree.hood_initialized

    tree = Rtree((0, 0))
    with pytest.raises(ValueError) as err:
        tree.build_tree(coordinates, method='foo')
    assert "unknown tree building method" in str(err.value).lower()
