# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np


def test_call():
    tree = Rtree((3, 3))
    assert tree((1, 1)) == 4
    assert np.allclose(tree(4, reverse=True), [1, 1])
