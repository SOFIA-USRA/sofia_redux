# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.grid.kernel_grid import KernelGrid
from sofia_redux.toolkit.resampling.tree.kernel_tree import KernelTree


def test_tree_class():
    grid = KernelGrid(np.arange(10), np.arange(10))
    assert grid.tree_class == KernelTree
    with pytest.raises(AttributeError) as err:
        grid.tree_class = None
    assert "can't set attribute" in str(err.value)
