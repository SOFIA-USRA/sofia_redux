# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np
import pytest


def test_errors():
    tree = Rtree((3, 3))
    with pytest.raises(RuntimeError) as err:
        tree.hood_members(0)

    assert "neighborhood tree not initialized" in str(err.value).lower()

    tree = Rtree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    block = tree.to_index([3, 3])
    with pytest.raises(RuntimeError) as err:
        tree.hood_members(block, get_terms=True)

    assert "phi terms have not been calculated" in str(err.value).lower()


def test_hood_members():
    tree = Rtree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
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

    tree.set_order(1)
    tree.precalculate_phi_terms()

    members, phi = tree.hood_members(populated_block, get_terms=True)
    assert np.allclose(members, hood_members)
    assert np.allclose(phi,
                       [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [2, 2, 2, 3, 3, 3, 4, 4, 4],
                        [2, 3, 4, 2, 3, 4, 2, 3, 4]])

    a, b, c = tree.hood_members(populated_block, get_locations=True,
                                get_terms=True)
    assert np.allclose(a, members)
    assert np.allclose(b, locations)
    assert np.allclose(c, phi)
