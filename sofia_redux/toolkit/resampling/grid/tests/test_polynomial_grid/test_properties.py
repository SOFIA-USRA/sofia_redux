# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.grid.polynomial_grid import PolynomialGrid
from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree


def test_tree_class():
    grid = PolynomialGrid(np.arange(10), np.arange(10))
    assert grid.tree_class == PolynomialTree
    with pytest.raises(AttributeError) as err:
        grid.tree_class = None
    assert "can't set attribute" in str(err.value)
