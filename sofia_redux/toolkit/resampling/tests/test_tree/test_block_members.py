# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree import Rtree

import numpy as np
import pytest


def test_errors():
    tree = Rtree((0, 0))
    with pytest.raises(RuntimeError) as err:
        tree.block_members(0)
    assert "neighborhood tree not initialized" in str(err.value).lower()

    tree = Rtree(np.random.random((2, 50)))
    with pytest.raises(RuntimeError) as err:
        tree.block_members(0, get_terms=True)
    assert "phi terms have not been calculated" in str(err.value).lower()


def test_block_members():
    c1 = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    coordinates = np.hstack([c1, c1 + 0.5])  # half pixel offset
    tree = Rtree(coordinates)

    test_block = tree.to_index([2, 2])  # should contain (2, 2) and (2.5, 2.5)
    members = tree.block_members(test_block)
    retrieved_coordinates = coordinates[:, members]
    assert np.allclose(retrieved_coordinates, [[2, 2.5], [2, 2.5]])

    members, locations = tree.block_members(test_block, get_locations=True)
    assert np.allclose(locations, retrieved_coordinates)

    tree.set_order(1)
    tree.precalculate_phi_terms()

    members, phi0 = tree.block_members(test_block, get_terms=True)
    assert np.allclose(phi0, [[1, 1], [2, 2.5], [2, 2.5]])

    members, locations1, phi1 = tree.block_members(
        test_block, get_locations=True, get_terms=True)

    assert np.allclose(phi1, phi0)
    assert np.allclose(locations1, locations)
