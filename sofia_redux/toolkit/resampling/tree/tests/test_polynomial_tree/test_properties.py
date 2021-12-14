from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree

import numpy as np
import pytest


@pytest.fixture
def tree2d3o():
    return PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                          order=3)


@pytest.fixture
def tree2d2o():
    return PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                          order=2)


def test_order(tree2d3o):
    tree = tree2d3o
    assert tree.order == 3
    tree.set_order([2, 3])
    assert np.allclose(tree.order, [2, 3])
    with pytest.raises(AttributeError) as err:
        tree.order = 2
    assert "can't set attribute" in str(err.value)


def test_exponents(tree2d3o):
    tree = tree2d3o
    assert np.allclose(tree.exponents,
                       [[0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [0, 2],
                        [1, 2],
                        [0, 3]])
    tree._exponents = None
    assert tree.exponents is None
    with pytest.raises(AttributeError) as err:
        tree.exponents = 2
    assert "can't set attribute" in str(err.value)


def test_derivative_term_map(tree2d2o):
    tree = tree2d2o
    assert np.allclose(tree.derivative_term_map,
                       [[[1, 2, 1],
                         [1, 2, 4],
                         [0, 1, 3]],

                        [[1, 1, 2],
                         [3, 4, 5],
                         [0, 1, 3]]]
                       )
    tree._derivative_term_map = None
    assert tree.derivative_term_map is None
    with pytest.raises(AttributeError) as err:
        tree.derivative_term_map = 2
    assert "can't set attribute" in str(err.value)


def test_order_symmetry(tree2d3o):
    tree = tree2d3o
    assert tree.order_symmetry
    tree.set_order([2, 3])
    assert not tree.order_symmetry
    with pytest.raises(AttributeError) as err:
        tree.order_symmetry = False
    assert "can't set attribute" in str(err.value)


def test_order_varies(tree2d3o):
    tree = tree2d3o
    assert not tree.order_varies
    tree.set_order(3, fix_order=False)
    assert tree.order_varies
    with pytest.raises(AttributeError) as err:
        tree.order_varies = False
    assert "can't set attribute" in str(err.value)


def test_phi_terms_precalculated(tree2d3o):
    assert tree2d3o.phi_terms_precalculated
    tree = PolynomialTree((2, 3), order=3)
    assert not tree.phi_terms_precalculated
    with pytest.raises(AttributeError) as err:
        tree.phi_terms_precalculated = False
    assert "can't set attribute" in str(err.value)
