# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree

import numpy as np
import pytest


def test_errors():
    tree = PolynomialTree((1, 2))  # supplying a shape
    with pytest.raises(ValueError) as err:
        tree.precalculate_phi_terms()

    assert "tree has not been populated" in str(err.value).lower()

    tree = PolynomialTree(np.arange(10)[None])
    with pytest.raises(ValueError) as err:
        tree.precalculate_phi_terms()

    assert "order has not been set" in str(err.value).lower()

    assert not tree.phi_terms_precalculated


def test_fix_and_vary():
    coordinates = np.stack((np.arange(5), np.arange(5)))  # supplying coords
    tree = PolynomialTree(coordinates)
    tree.set_order(2)
    tree.precalculate_phi_terms()

    order2_terms = [[1, 1, 1, 1, 1],  # constant
                    [0, 1, 2, 3, 4],  # x
                    [0, 1, 4, 9, 16],  # x^2
                    [0, 1, 2, 3, 4],  # y
                    [0, 1, 4, 9, 16],  # x.y
                    [0, 1, 4, 9, 16]]  # y^2

    assert np.allclose(tree.phi_terms, order2_terms)
    assert np.allclose(tree.derivative_term_map,
                       [[[1, 2, 1],
                         [1, 2, 4],
                         [0, 1, 3]],

                        [[1, 1, 2],
                         [3, 4, 5],
                         [0, 1, 3]]])
    assert tree.derivative_term_indices is None

    assert tree.phi_terms_precalculated

    # Allow order variation
    tree.set_order(2, fix_order=False)
    tree.precalculate_phi_terms()

    assert np.allclose(tree.phi_terms,
                       [[1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1.],
                        [0., 1., 2., 3., 4.],
                        [0., 1., 2., 3., 4.],
                        [1., 1., 1., 1., 1.],
                        [0., 1., 2., 3., 4.],
                        [0., 1., 4., 9., 16.],
                        [0., 1., 2., 3., 4.],
                        [0., 1., 4., 9., 16.],
                        [0., 1., 4., 9., 16.]])

    assert np.allclose(tree.term_indices, [0, 1, 4, 10])
    assert np.allclose(tree.derivative_term_map,
                       [[[1, 1, 2, 1],
                         [1, 1, 2, 4],
                         [0, 0, 1, 3]],

                        [[1, 1, 1, 2],
                         [2, 3, 4, 5],
                         [0, 0, 1, 3]]])

    assert np.allclose(tree.derivative_term_indices, [0, 0, 1, 4])

    assert tree.phi_terms_precalculated


def test_orders():
    coordinates = np.stack((np.arange(5), np.arange(5)))  # supplying coords
    tree = PolynomialTree(coordinates)
    tree.set_order(2)
    tree.precalculate_phi_terms()
    assert tree.order == 2
    assert tree.order_symmetry
    assert not tree.order_varies

    tree = PolynomialTree(coordinates)
    tree.set_order(2, fix_order=False)
    tree.precalculate_phi_terms()
    assert tree.order_varies
