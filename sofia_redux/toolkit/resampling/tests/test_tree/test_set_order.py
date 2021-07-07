# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np
import pytest


def test_symmetrical_order():
    tree = Rtree((5, 6))
    assert tree.order_symmetry is None
    assert tree.order_varies is None

    tree.set_order(2, fix_order=True)
    assert tree.order_symmetry
    assert tree.order == 2
    assert not tree.order_varies

    tree.set_order(2, fix_order=False)
    assert tree.order_symmetry
    assert tree.order == 2
    assert tree.order_varies


def test_asymmetrical_order():

    tree = Rtree((5, 6))

    with pytest.raises(ValueError) as err:
        tree.set_order([1, 2, 3, 4, 5])

    assert "does not match number of features" in str(err.value).lower()

    tree.set_order([2, 3], fix_order=True)
    assert np.allclose(tree.order, [2, 3])
    assert not tree.order_varies
    assert not tree.order_symmetry

    tree.set_order([2, 3], fix_order=False)
    assert np.allclose(tree.order, [2, 3])
    assert not tree.order_varies  # order cannot vary for asymmetrical orders
    assert not tree.order_symmetry
