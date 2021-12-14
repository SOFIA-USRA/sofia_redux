# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree
from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial
from sofia_redux.toolkit.resampling.grid.polynomial_grid import \
    PolynomialGrid

import numpy as np
import pytest
from sklearn.neighbors import BallTree


def test_init():
    tree = BaseTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    assert tree.shape == (5, 6)
    assert tree.n_members == 30

    tree = BaseTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                    shape=(10, 10))
    assert tree.shape == (10, 10)
    assert tree.n_members == 30

    tree = BaseTree((3, 4))
    assert tree.shape == (3, 4)
    assert tree.n_members == 0


def test_block_members():
    c1 = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    coordinates = np.hstack([c1, c1 + 0.5])  # half pixel offset
    tree = BaseTree(coordinates)

    test_block = tree.to_index([2, 2])  # should contain (2, 2) and (2.5, 2.5)
    members = tree.block_members(test_block)
    retrieved_coordinates = coordinates[:, members]
    assert np.allclose(retrieved_coordinates, [[2, 2.5], [2, 2.5]])

    members, locations = tree.block_members(test_block, get_locations=True)
    assert np.allclose(locations, retrieved_coordinates)

    tree = BaseTree((0, 0))
    with pytest.raises(RuntimeError) as err:
        tree.block_members(0)
    assert "neighborhood tree not initialized" in str(err.value).lower()


def test_get_class_for():
    rand = np.random.random
    resampler = ResamplePolynomial(rand(100), rand(100), rand(100))
    c = BaseTree.get_class_for(resampler)
    assert c == PolynomialTree
    c = BaseTree.get_class_for('polynomial')
    assert c == PolynomialTree
    c = BaseTree.get_class_for('foo')
    assert c == BaseTree
    c = BaseTree.get_class_for('')
    assert c == BaseTree
    c = BaseTree.get_class_for(None)
    assert c == BaseTree
    grid = PolynomialGrid(np.arange(10), np.arange(10))
    c = BaseTree.get_class_for(grid)
    assert c == PolynomialTree


def test_get_class_for_name():
    assert BaseTree.get_class_for_name('polynomial') == PolynomialTree
    assert BaseTree.get_class_for_name('foo') == BaseTree


def test_set_shape():
    tree = BaseTree((3, 4))
    tree._set_shape((4, 5))
    assert tree.shape == (4, 5)
    assert tree.features == 2
    assert tree.n_blocks == 20
    assert np.allclose(tree.search,
                       [[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                        [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
    assert tree._tree is None
    assert tree.block_offsets is None
    assert tree.block_population is None
    assert tree.populated is None
    assert tree.hood_population is None


def test_to_index():
    tree = BaseTree((5, 6))
    assert np.allclose(tree.to_index((2.7, 3.1)), 15)
    assert np.allclose(tree.to_index((-1, -1)), 0)
    assert np.allclose(tree.to_index((10, 10)), 29)
    assert np.allclose(tree.to_index([[3, 4], [3, 4]]), [21, 28])


def test_from_index():
    tree = BaseTree((5, 6))
    assert np.allclose(tree.from_index(15), [2, 3])
    assert np.allclose(tree.from_index(0), [0, 0])
    assert np.allclose(tree.from_index(29), [4, 5])
    assert np.allclose(tree.from_index([21, 28]), [[3, 4], [3, 4]])


def test_build_tree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = BaseTree((0, 0))
    tree.build_tree(coordinates, method='all')
    assert tree.shape == (5, 6)
    assert tree.balltree_initialized
    assert tree.hood_initialized

    tree.build_tree(coordinates, shape=(10, 12))
    assert tree.shape == (10, 12)

    tree = BaseTree((0, 0))
    tree.build_tree(coordinates, method='hood')
    assert not tree.balltree_initialized
    assert tree.hood_initialized

    tree = BaseTree((0, 0))
    tree.build_tree(coordinates, method='balltree', leaf_size=40)
    assert tree.balltree_initialized
    assert not tree.hood_initialized

    tree = BaseTree((0, 0))
    tree.build_tree(coordinates, method=None)
    assert not tree.balltree_initialized
    assert not tree.hood_initialized

    tree = BaseTree((0, 0))
    with pytest.raises(ValueError) as err:
        tree.build_tree(coordinates, method='foo')
    assert "unknown tree building method" in str(err.value).lower()


def test_build_balltree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = BaseTree((0, 0))
    tree.coordinates = coordinates
    assert not tree.balltree_initialized
    tree._build_ball_tree()
    assert tree.balltree_initialized
    assert isinstance(tree._balltree, BallTree)

    tree = BaseTree((0, 0))
    tree.coordinates = coordinates

    # Test options are getting through (indirectly)
    with pytest.raises(TypeError) as err:
        tree._build_ball_tree(leaf_size='a')
    assert "not supported" in str(err.value)

    with pytest.raises(ValueError) as err:
        tree._build_ball_tree(metric='foo')
    assert 'unrecognized metric' in str(err.value).lower()

    tree._build_ball_tree(leaf_size=None)
    assert tree.balltree_initialized


def test_build_hood_tree():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = BaseTree((0, 0))
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


def test_query_radius():
    tree = BaseTree((0, 0))
    with pytest.raises(RuntimeError) as err:
        tree.query_radius(np.array([[0.0], [0.0]]))
    assert "ball tree not initialized" in str(err.value).lower()

    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = BaseTree(coordinates)
    ind, distances = tree.query_radius(np.array([2.5, 2.5]), radius=2.0,
                                       return_distance=True)
    c0 = coordinates[:, ind[0]] - np.full(2, 2.5)[:, None]
    r = np.hypot(*c0)
    assert np.allclose(r, distances[0])

    coordinates = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    tree = BaseTree(coordinates)
    points = np.empty((2, 2))
    points[:, 0] = 2  # (x, y) = (2, 2)
    points[:, 1] = 4  # (x, y) = (4, 4)
    # default radius = 1
    ind, distances = tree.query_radius(points, return_distance=True)
    r0 = np.hypot(*(coordinates[:, ind[0]] - points[:, 0, None]))
    assert np.allclose(r0, distances[0])
    r1 = np.hypot(*(coordinates[:, ind[1]] - points[:, 1, None]))
    assert np.allclose(r1, distances[1])


def test_neighborhood():
    tree = BaseTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    block = tree.to_index([3, 3])
    hood = tree.neighborhood(block)

    lower_coords = tree.from_index(hood)
    assert np.allclose(lower_coords,
                       [[2, 2, 2, 3, 3, 3, 4, 4, 4],  # x
                        [2, 3, 4, 2, 3, 4, 2, 3, 4]])  # y

    # Test culling
    tree = BaseTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    block = tree.to_index([0, 0])  # corner block
    hood = tree.neighborhood(block)
    assert np.sum(hood != -1) == 4  # should only be 4 valid neighboring blocks

    hood, valid = tree.neighborhood(block, valid_neighbors=True)
    assert np.all(hood[valid] >= 0)

    hood = tree.neighborhood(block, cull=True)
    assert hood.size == 4
    assert np.all(hood >= 0)


def test_hood_members():
    tree = BaseTree((3, 3))
    with pytest.raises(RuntimeError) as err:
        tree.hood_members(0)
    assert "neighborhood tree not initialized" in str(err.value).lower()

    tree = BaseTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                    shape=(10, 10))
    populated_block = tree.to_index([3, 3])

    hood_members = tree.hood_members(populated_block)
    hood_coordinates = tree.coordinates[:, hood_members]
    assert np.allclose(hood_coordinates,
                       [[2, 2, 2, 3, 3, 3, 4, 4, 4],
                        [2, 3, 4, 2, 3, 4, 2, 3, 4]])

    members, locations = tree.hood_members(populated_block, get_locations=True)
    assert np.allclose(members, hood_members)
    assert np.allclose(locations, hood_coordinates)


def test_call():
    tree = BaseTree((3, 3))
    assert tree((1, 1)) == 4
    assert np.allclose(tree(4, reverse=True), [1, 1])
