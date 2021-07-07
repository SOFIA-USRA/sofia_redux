# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np
import pytest
from sklearn.neighbors import BallTree


def test_build_balltree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = Rtree((0, 0))
    tree.coordinates = coordinates
    assert not tree.balltree_initialized
    tree._build_ball_tree()
    assert tree.balltree_initialized
    assert isinstance(tree._balltree, BallTree)

    tree = Rtree((0, 0))
    tree.coordinates = coordinates

    # Test options are getting through (indirectly)
    with pytest.raises(TypeError):
        tree._build_ball_tree(leaf_size='a')

    with pytest.raises(ValueError) as err:
        tree._build_ball_tree(metric='foo')

    assert 'unrecognized metric' in str(err.value).lower()
