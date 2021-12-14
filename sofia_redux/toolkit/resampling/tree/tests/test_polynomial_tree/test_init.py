# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree

import numpy as np


def test_shape():
    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    assert tree.shape == (5, 6)
    assert tree.n_members == 30

    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                          shape=(10, 10))
    assert tree.shape == (10, 10)
    assert tree.n_members == 30

    tree = PolynomialTree((3, 4))
    assert tree.shape == (3, 4)
    assert tree.n_members == 0


def test_order():
    tree = PolynomialTree((5, 6), order=2)
    assert tree.order == 2
    assert not tree.order_varies

    tree = PolynomialTree((5, 6), order=2, fix_order=False)
    assert tree.order == 2
    assert tree.order_varies


def test_phi():
    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    assert tree.phi_terms is None
    assert tree.derivative_term_map is None
    assert tree.exponents is None

    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                          order=2)
    assert tree.phi_terms.shape == (6, 30)
    assert tree.exponents.shape == (6, 2)
    assert tree.derivative_term_map.shape == (2, 3, 3)

    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
                          order=2,
                          fix_order=False)
    assert tree.phi_terms.shape == (10, 30)
    assert tree.exponents.shape == (6, 2)
    assert tree.derivative_term_map.shape == (2, 3, 4)
