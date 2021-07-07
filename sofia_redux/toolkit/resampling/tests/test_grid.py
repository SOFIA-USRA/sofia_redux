# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.grid import ResampleGrid


def test_regular_grid():
    grid = np.arange(10, dtype=float), np.arange(5) + 10.0
    g = ResampleGrid(*grid)
    assert g.regular
    assert not g.singular
    assert g.shape == (5, 10)
    assert g.size == 50
    assert g.features == 2
    assert g.scale_factor is None
    assert g.scale_offset is None
    assert g.grid[0].min() == 0
    assert g.grid[0].max() == 9
    assert g.grid[1].min() == 10
    assert g.grid[1].max() == 14
    assert np.allclose(g(), g.grid)


def test_irregular_grid():
    grid = np.random.random((3, 100))
    g = ResampleGrid(grid)
    assert not g.regular
    assert not g.singular
    assert g.shape == (100,)
    assert g.size == 100
    assert g.features == 3
    assert g.scale_factor is None
    assert g.scale_offset is None


def test_singular_grid():
    g = ResampleGrid(1, 2, 3)
    assert not g.regular
    assert g.singular
    assert g.shape == (1, 1, 1)
    assert g.size == 1
    assert g.features == 3
    assert g.scale_factor is None
    assert g.scale_offset is None


def test_reshape_data():
    # regular output with multiple datasets
    grid = np.arange(10, dtype=float), np.arange(5) + 10.0
    g = ResampleGrid(*grid)
    data = np.random.random((3, g.size))
    assert g.reshape_data(data).shape == (3, 5, 10)
    data = np.random.random(g.size)
    assert g.reshape_data(data).shape == (5, 10)
    with pytest.raises(ValueError):
        data = np.random.random((3, 3, g.size))
        g.reshape_data(data)

    single_grid = ResampleGrid(1, 1)
    assert single_grid.reshape_data(np.array([1])).shape == ()
    with pytest.raises(ValueError) as err:
        single_grid.reshape_data(np.empty((3, 3, 3)))
    assert "incompatible data dimensions" in str(err.value).lower()

    irregular_grid = ResampleGrid(np.zeros((2, 10)))
    assert irregular_grid.reshape_data(np.empty(20)).shape == (20,)


def test_dtype():
    grid = np.random.random((3, 100)) * 10
    g = ResampleGrid(grid, dtype=int)
    assert np.issubdtype(g.grid.dtype, np.int64) \
           or np.issubdtype(g.grid.dtype, np.int32)


def test_scaling():
    grid = np.arange(10, dtype=float), np.arange(5) + 10.0
    scale_factor = np.array([1.0, 2])
    scale_offset = np.array([100, 200.0])
    g = ResampleGrid(*grid, scale_factor=None, scale_offset=None)
    g.rescale()  # check it does nothing
    g = ResampleGrid(*grid, scale_factor=scale_factor,
                     scale_offset=scale_offset)
    assert np.allclose(g.grid[0, :10], grid[0] - 100)
    assert np.allclose(g.grid[1, ::10], (grid[1] - 200) / 2)
    g.unscale()
    g.unscale()  # second time to check it's only done once
    assert np.allclose(g.grid[0, :10], grid[0])
    assert np.allclose(g.grid[1, ::10], grid[1])
    g.rescale()
    g.rescale()  # second time to check it's only done once
    assert np.allclose(g.grid[0, :10], grid[0] - 100)
    assert np.allclose(g.grid[1, ::10], (grid[1] - 200) / 2)
    g.scale(scale_factor, -scale_offset)
    assert np.allclose(g.grid[0, :10], grid[0] + 100)
    assert np.allclose(g.grid[1, ::10], (grid[1] + 200) / 2)
    g.scale(scale_factor / 2, [0, 0])
    assert np.allclose(g.grid[0, :10], grid[0] * 2)
    assert np.allclose(g.grid[1, ::10], grid[1])
    g.unscale()
    assert np.allclose(g.grid[0, :10], grid[0])
    assert np.allclose(g.grid[1, ::10], grid[1])


def test_set_indexer():
    grid = np.arange(10, dtype=float), np.arange(5) + 10.0
    g = ResampleGrid(*grid, build_tree=True)
    assert g.tree.hood_population is not None
    g = ResampleGrid(*grid, build_tree=False)
    assert g.tree.hood_population is None


def test_error():
    # both scale_factor and scale_offset need to be supplied
    with pytest.raises(ValueError):
        ResampleGrid(1, 1, scale_factor=[1.0, 2.0])

    with pytest.raises(ValueError):
        ResampleGrid(1, 1, scale_offset=[1.0, 2.0])
