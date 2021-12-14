# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.grid.kernel_grid import KernelGrid
from sofia_redux.toolkit.resampling.tree.kernel_tree import KernelTree


def test_init():
    # The only differences are to do with the construction of the tree.
    coords = np.arange(10), np.arange(10)
    grid = KernelGrid(*coords)
    assert isinstance(grid.tree, KernelTree)
    assert grid.tree.kernel is None
    grid = KernelGrid(*coords, kernel=np.ones((7, 7)),
                      kernel_spacing=[0.5, 0.25], degrees=1)
    assert np.allclose(grid.tree.kernel, 1)
    assert np.allclose(grid.tree.kernel_spacing, [0.5, 0.25])
    y, x = np.mgrid[:7, :7]

    assert np.allclose(grid.tree.kernel_coordinates[0], (x.ravel() - 3) / 2)
    assert np.allclose(grid.tree.kernel_coordinates[1], (y.ravel() - 3) / 4)
    assert grid.tree.hood_population is None

    grid = KernelGrid(*coords, kernel=np.ones((7, 7)),
                      kernel_spacing=[0.5, 0.25], build_tree=True,
                      build_type='hood', degrees=1)
    assert isinstance(grid.tree.hood_population, np.ndarray)
    assert grid.tree.hood_initialized
    assert not grid.tree.balltree_initialized

    grid = KernelGrid(*coords, kernel=np.ones((7, 7)),
                      kernel_spacing=[0.5, 0.25], build_tree=True,
                      build_type='all', degrees=1)
    assert grid.tree.hood_initialized
    assert grid.tree.balltree_initialized
