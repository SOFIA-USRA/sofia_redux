# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree

import numpy as np
import pytest


@pytest.fixture
def coords1d():
    rand = np.random.RandomState(3)
    max_shape = 101
    coordinates = rand.random(1000)[None] * max_shape
    return coordinates


@pytest.fixture
def coords2d():
    rand = np.random.RandomState(3)
    max_shape = np.array([9, 11])
    coordinates = rand.random((2, 101)) * max_shape[:, None]
    return coordinates


@pytest.fixture
def coords3d():
    rand = np.random.RandomState(3)
    max_shape = np.array([4, 5, 6])
    coordinates = rand.random((3, 512)) * max_shape[:, None]
    return coordinates


@pytest.fixture
def tree1d(coords1d):
    return BaseTree(coords1d)


@pytest.fixture
def tree2d(coords2d):
    return BaseTree(coords2d)


@pytest.fixture
def tree3d(coords3d):
    return BaseTree(coords3d)


def test_shape(tree2d):
    tree = tree2d
    assert tree.shape == (9, 11)
    with pytest.raises(AttributeError) as err:
        tree.shape = None
    assert "can't set attribute" in str(err.value)


def test_features(tree2d):
    tree = tree2d
    assert tree.features == 2
    with pytest.raises(AttributeError) as err:
        tree.features = 1
    assert "can't set attribute" in str(err.value)


def test_n_blocks(tree2d):
    tree = tree2d
    assert tree.n_blocks == 99
    with pytest.raises(AttributeError) as err:
        tree.n_blocks = 1
    assert "can't set attribute" in str(err.value)


def test_search(tree1d, tree2d, tree3d):
    assert np.allclose(tree1d.search, [-1, 0, 1])
    assert np.allclose(tree2d.search,
                       [[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                        [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
    assert np.allclose(tree3d.search,
                       [([-1] * 9) + ([0] * 9) + ([1] * 9),
                        [-1, -1, -1, 0, 0, 0, 1, 1, 1] * 3,
                        [-1, 0, 1] * 9])
    with pytest.raises(AttributeError) as err:
        tree2d.search = 1
    assert "can't set attribute" in str(err.value)


def test_balltree_initialized(tree2d):
    tree = tree2d
    # Should be initialized on build if coordinates are passed
    assert tree.balltree_initialized
    # Should be blank if a shape was passed
    tree = BaseTree((2, 3, 4))
    assert not tree.balltree_initialized


def test_hood_initialized(tree2d):
    # Should be initialized on build if coordinates are passed
    tree = tree2d
    assert tree.hood_initialized
    # Should be blank if a shape was passed
    tree = BaseTree((3, 4, 5))
    assert not tree.hood_initialized


def test_n_members(tree3d):
    assert tree3d.n_members == 512
    tree = BaseTree((2, 4))
    assert tree.n_members == 0
