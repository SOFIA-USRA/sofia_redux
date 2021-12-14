# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.kernel_tree import KernelTree

import numpy as np
import pytest


@pytest.fixture
def kernel_2d():
    y, x = np.mgrid[:15, :17]
    y2 = (y - 7) ** 2
    x2 = (x - 8) ** 2
    a = -0.1
    b = -0.15
    data = np.exp(a * x2 + b * y2)
    return data


@pytest.fixture
def coordinates_grid_2d():
    y, x = np.mgrid[:19, :21]
    coordinates = np.stack([z.ravel() for z in [x, y]]).astype(float)
    return coordinates


@pytest.fixture
def initialized_tree_2d(kernel_2d, coordinates_grid_2d):
    tree = KernelTree(coordinates_grid_2d, kernel_spacing=0.5,
                      kernel=kernel_2d, degrees=3, smoothing=0.0)
    return tree


@pytest.fixture
def uninitialized_tree_2d(coordinates_grid_2d):
    tree = KernelTree(coordinates_grid_2d)
    return tree


def test_degrees(initialized_tree_2d, uninitialized_tree_2d):
    assert np.allclose(initialized_tree_2d.degrees, [3, 3])
    assert uninitialized_tree_2d.degrees is None


def test_smoothing(initialized_tree_2d, uninitialized_tree_2d):
    assert initialized_tree_2d.smoothing == 0
    assert uninitialized_tree_2d.smoothing is None


def test_exit_code(initialized_tree_2d, uninitialized_tree_2d):
    assert initialized_tree_2d.exit_code == -1
    assert uninitialized_tree_2d.exit_code is None


def test_exit_message(initialized_tree_2d, uninitialized_tree_2d):
    assert "interpolating spline" in initialized_tree_2d.exit_message
    assert "has not been initialized" in uninitialized_tree_2d.exit_message


def test_fit_valid(kernel_2d, uninitialized_tree_2d):
    tree = KernelTree((5, 6), kernel=kernel_2d, degrees=3, smoothing=0,
                      imperfect=False)
    assert tree.fit_valid
    assert not uninitialized_tree_2d.fit_valid
    tree.spline.exit_code = 6
    assert not tree.fit_valid
    tree.spline.exit_code = -4
    assert not tree.fit_valid
    for code in range(-3, 1):
        tree.spline.exit_code = code
        assert tree.fit_valid
    tree.spline.exit_code = -tree.spline.rank
    assert tree.fit_valid
    tree.spline.exit_code += 1
    assert not tree.fit_valid
    tree.imperfect = True
    assert tree.fit_valid


def test_coefficients(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert uninitialized_tree_2d.coefficients is None
    assert isinstance(tree.coefficients, np.ndarray)
    assert tree.coefficients.ndim == 1


def test_knots(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert uninitialized_tree_2d.knots is None
    assert isinstance(tree.knots, np.ndarray)
    assert tree.knots.ndim == 2 and tree.knots.shape[0] == 2


def test_panel_mapping(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert tree.panel_mapping.ndim == 2 and tree.panel_mapping.shape[0] == 2
    assert uninitialized_tree_2d.panel_mapping is None


def test_panel_steps(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert np.allclose(tree.panel_steps, [12, 1])
    assert uninitialized_tree_2d.panel_steps is None


def test_knot_steps(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert np.allclose(tree.knot_steps, [15, 1])
    assert uninitialized_tree_2d.knot_steps is None


def test_nk1(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert np.allclose(tree.nk1, [17, 15])
    assert uninitialized_tree_2d.nk1 is None


def test_spline_mapping(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert tree.spline_mapping.shape == (2, 16)
    assert uninitialized_tree_2d.spline_mapping is None


def test_n_knots(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert np.allclose(tree.n_knots, [21, 19])
    assert uninitialized_tree_2d.n_knots is None


def test_extent(initialized_tree_2d, uninitialized_tree_2d):
    tree = initialized_tree_2d
    assert np.allclose(tree.extent, [[-4, 4], [-3.5, 3.5]])
    assert uninitialized_tree_2d.extent is None


def test_resampling_arguments(initialized_tree_2d, uninitialized_tree_2d):
    args = initialized_tree_2d.resampling_arguments
    assert len(args) == 9
    for arg in args:
        assert isinstance(arg, np.ndarray)
    args = uninitialized_tree_2d.resampling_arguments
    assert len(args) == 9
    for arg in args:
        assert arg is None


def test_setters(uninitialized_tree_2d):
    tree = uninitialized_tree_2d
    for p in ['degrees', 'exit_code', 'exit_message', 'fit_valid',
              'coefficients', 'knots', 'panel_mapping', 'panel_steps',
              'knot_steps', 'nk1', 'spline_mapping', 'n_knots',
              'resampling_arguments']:
        with pytest.raises(AttributeError) as err:
            setattr(tree, p, None)
        assert "can't set attribute" in str(err.value)
