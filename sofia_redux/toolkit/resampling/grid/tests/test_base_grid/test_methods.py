# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.grid.base_grid import BaseGrid
from sofia_redux.toolkit.resampling.grid.polynomial_grid import PolynomialGrid
from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree
from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial


@pytest.fixture
def tuple_2d_coordinates():
    return np.arange(3, dtype=float), np.arange(5) + 10.0


@pytest.fixture
def tuple_3d_coordinates():
    return np.arange(4) + 1.0, np.arange(5) + 2.0, np.arange(6) + 3.0


@pytest.fixture
def irregular_2d_coordinates():
    rand = np.random.RandomState(0)
    return rand.random((2, 10)) * 10


def test_init(tuple_2d_coordinates, tuple_3d_coordinates,
              irregular_2d_coordinates):
    coords = tuple_2d_coordinates
    with pytest.raises(ValueError) as err:
        BaseGrid(*coords, scale_factor=[1, 1])
    assert "Specify both factor and offset to scale" in str(err.value)

    with pytest.raises(ValueError) as err:
        BaseGrid(*coords, scale_offset=[1, 1])
    assert "Specify both factor and offset to scale" in str(err.value)

    # Test standard
    grid = BaseGrid(*coords)
    assert grid.shape == (5, 3)
    assert grid.features == 2
    assert np.allclose(
        grid.grid,
        [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
         [10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14]])
    assert grid.scale_factor is None
    assert grid.scale_offset is None
    assert grid.regular
    assert not grid.singular

    # Testing scaling is propagated
    grid = BaseGrid(*coords, scale_factor=[2, 4], scale_offset=[3, 5])
    assert np.allclose(grid.scale_offset, [3, 5])
    assert np.allclose(grid.scale_factor, [2, 4])
    assert np.allclose(
        grid.grid,
        [[-1.5, -1., -0.5, -1.5, -1., -0.5, -1.5, -1., -0.5,
          -1.5, -1., -0.5, -1.5, -1., -0.5],
         [1.25, 1.25, 1.25, 1.5, 1.5, 1.5, 1.75, 1.75, 1.75,
          2., 2., 2., 2.25, 2.25, 2.25]]
    )

    # Test indexing is correct in 3+ dimensions
    grid = BaseGrid(*tuple_3d_coordinates)
    assert np.allclose(grid.grid[0], [1, 2, 3, 4] * 30)
    g1 = []
    for x in range(5):
        g1 += [x + 2] * 4
    assert np.allclose(grid.grid[1], g1 * 6)
    g2 = []
    for x in range(6):
        g2 += [x + 3] * 20
    assert np.allclose(grid.grid[2], g2)

    # Test irregular grids
    grid = BaseGrid(irregular_2d_coordinates)
    assert not grid.regular
    assert grid.shape == (10,)

    # Test singular grid
    grid = BaseGrid(1, 1)
    assert grid.singular
    assert grid.shape == (1, 1)


def test_get_class_for():
    grid_class = BaseGrid.get_class_for(ResamplePolynomial(
        np.arange(10), np.arange(10)))
    assert grid_class == PolynomialGrid
    grid_class = BaseGrid.get_class_for(PolynomialTree((3, 4)))
    assert grid_class == PolynomialGrid
    grid_class = BaseGrid.get_class_for(None)
    assert grid_class == BaseGrid
    grid_class = BaseGrid.get_class_for('polynomial')
    assert grid_class == PolynomialGrid
    grid_class = BaseGrid.get_class_for('foo')
    assert grid_class == BaseGrid


def test_get_class_for_name():
    grid_class = BaseGrid.get_class_for('polynomial')
    assert grid_class == PolynomialGrid
    grid_class = BaseGrid.get_class_for('foo')
    assert grid_class == BaseGrid


def test_get_tree_class():
    grid = BaseGrid(np.arange(3), np.arange(3))
    assert grid.get_tree_class() == BaseTree
    grid = PolynomialGrid(np.arange(3), np.arange(3))
    assert grid.get_tree_class() == PolynomialTree


def test_reshape_data():
    # Test regular grid
    grid = BaseGrid(np.arange(3), np.arange(3))
    data = np.arange(9)
    result = grid.reshape_data(data)
    assert np.allclose(result, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    # Test multiple sets
    data2 = np.stack([data, data])
    result2 = grid.reshape_data(data2)
    assert result2.shape == (2, 3, 3)
    assert np.allclose(result2, result[None])

    with pytest.raises(ValueError) as err:
        grid.reshape_data(np.ones((3, 3, 3)))
    assert "Incompatible data dimensions" in str(err.value)

    # Test singular grid
    grid = BaseGrid(1, 1)
    assert grid.singular
    assert grid.reshape_data(np.arange(10)) == 0
    assert np.allclose(grid.reshape_data(data2), [0, 0])
    with pytest.raises(ValueError) as err:
        grid.reshape_data(np.ones((3, 3, 3)))
    assert "Incompatible data dimensions" in str(err.value)

    # Test irregular grid
    grid = BaseGrid((np.random.random((2, 10))))
    result = grid.reshape_data(np.arange(10))
    assert np.allclose(result, np.arange(10))


def test_unscale():
    grid = BaseGrid(np.arange(3), np.arange(3))
    expected = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2],
                         [0, 0, 0, 1, 1, 1, 2, 2, 2]])
    assert np.allclose(grid.grid, expected)
    grid.unscale()
    assert np.allclose(grid.grid, expected)

    grid.scale(np.array([2, 3]), np.array([3, 4]))
    assert not np.allclose(grid.grid, expected)
    grid.unscale()
    assert np.allclose(grid.grid, expected)


def test_scale():
    grid = BaseGrid(np.arange(3), np.arange(3))
    assert np.allclose(grid.grid,
                       [[0, 1, 2, 0, 1, 2, 0, 1, 2],
                        [0, 0, 0, 1, 1, 1, 2, 2, 2]])
    factor = np.array([1, 2])
    offset = np.array([3, 4])
    grid.scale(factor, offset)
    assert np.allclose(grid.grid,
                       [[-3, -2, -1, -3, -2, -1, -3, -2, -1],
                        [-2, -2, -2, -1.5, -1.5, -1.5, -1, -1, -1]])

    factor = np.array([2, 1])
    offset = np.array([4, 3])
    grid.scale(factor, offset)
    assert np.allclose(grid.grid,
                       [[-2, -1.5, -1, -2, -1.5, -1, -2, -1.5, -1],
                        [-3, -3, -3, -2, -2, -2, -1, -1, -1]])


def test_rescale():
    grid = BaseGrid(np.arange(3), np.arange(3))
    expected = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2],
                         [0, 0, 0, 1, 1, 1, 2, 2, 2]])
    grid.rescale()
    assert np.allclose(grid.grid, expected)

    grid.scale(np.array([1, 2]), np.array([3, 4]))
    coords = grid.grid.copy()
    grid.rescale()
    assert np.allclose(grid.grid, coords)

    assert not np.allclose(coords, expected)
    grid.unscale()
    assert np.allclose(grid.grid, expected)
    grid.rescale()
    assert np.allclose(grid.grid, coords)


def test_set_indexer():
    rand = np.random.RandomState(0)
    coords = rand.random((2, 100)) * 10
    grid = BaseGrid(coords)
    grid.set_indexer(shape=None, build_tree=False, build_type='hood')
    assert grid.tree.shape == (10, 10)
    assert grid.tree.hood_population is None

    grid.set_indexer(shape=(9, 9), build_tree=False, build_type='hood')
    assert grid.tree.shape == (9, 9)
    assert grid.tree.hood_population is None

    grid.set_indexer(shape=None, build_tree=True, build_type='hood')
    assert isinstance(grid.tree.hood_population, np.ndarray)


def test_call():
    grid = BaseGrid(np.arange(3), np.arange(3))
    assert grid() is grid.grid and np.allclose(
        grid.grid,
        [[0, 1, 2, 0, 1, 2, 0, 1, 2],
         [0, 0, 0, 1, 1, 1, 2, 2, 2]])
