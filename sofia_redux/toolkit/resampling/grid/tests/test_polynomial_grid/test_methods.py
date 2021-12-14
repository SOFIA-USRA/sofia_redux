# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.grid.polynomial_grid import PolynomialGrid
from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree


def test_init():
    # The only differences are to do with the construction of the tree.
    coords = np.arange(10), np.arange(10)
    grid = PolynomialGrid(*coords)
    assert grid.tree.order is None
    grid = PolynomialGrid(*coords, order=[2, 3])
    assert isinstance(grid.tree, PolynomialTree)
    assert np.allclose(grid.tree.order, [2, 3])
    assert grid.tree.hood_population is None
    grid = PolynomialGrid(*coords, order=[2, 3], build_tree=True,
                          build_type='hood')
    assert isinstance(grid.tree.hood_population, np.ndarray)
    assert grid.tree.hood_initialized
    assert not grid.tree.balltree_initialized

    grid = PolynomialGrid(*coords, order=[2, 3], build_tree=True,
                          build_type='all')
    assert grid.tree.hood_initialized
    assert grid.tree.balltree_initialized
