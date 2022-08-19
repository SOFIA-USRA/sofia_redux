from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree


import numpy as np
import pytest


def test_estimate_max_bytes():
    x = np.linspace(0, 127, 128)
    coordinates = np.array(np.meshgrid(x, x, x))
    window = 10
    n_bytes = PolynomialTree.estimate_max_bytes(
        coordinates, window, leaf_size=40, order=2)
    assert np.isclose(n_bytes, 369326064, rtol=0.1)

    n_bytes = PolynomialTree.estimate_max_bytes(
        coordinates, window, leaf_size=40, order=3)
    assert np.isclose(n_bytes, 537098224, rtol=0.1)


def test_set_shape():
    tree = PolynomialTree((3, 4))
    tree.term_indices = 1
    tree._set_shape((3, 4))
    assert tree.term_indices is None


def test_set_order():
    tree = PolynomialTree((5, 6))
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

    tree = PolynomialTree((5, 6))
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


def test_create_phi_terms_for():
    coordinates = np.stack((np.arange(5), np.arange(5)))
    tree = PolynomialTree(coordinates)
    tree.set_order(2)
    tree.phi_terms = None
    tree.large_data = True
    assert not tree.order_varies
    tree.create_phi_terms_for()
    assert not tree.phi_terms_precalculated
    assert tree.phi_terms is None

    tree.large_data = False
    inds = tree.create_phi_terms_for()
    assert inds == slice(0, 5)
    assert tree.phi_terms_precalculated
    assert isinstance(tree.phi_terms, np.ndarray)
    assert tree.phi_terms.shape == (6, 5)

    tree.set_order(2, fix_order=False)
    tree.phi_terms = None
    assert tree.order_varies
    inds = tree.create_phi_terms_for()
    assert inds == slice(0, 5)
    assert isinstance(tree.phi_terms, np.ndarray)
    assert tree.phi_terms.shape == (10, 5)

    tree.large_data = True
    inds = tree.create_phi_terms_for(1)
    assert np.allclose(inds, [0, 1])
    assert tree.phi_terms.shape == (10, 2)


def test_block_members():
    rand = np.random.RandomState(42)

    tree = PolynomialTree((0, 0))
    with pytest.raises(RuntimeError) as err:
        tree.block_members(0)
    assert "neighborhood tree not initialized" in str(err.value).lower()

    tree = PolynomialTree(rand.random((2, 50)))
    with pytest.raises(RuntimeError) as err:
        tree.block_members(0, get_terms=True)
    assert "phi terms have not been calculated" in str(err.value).lower()

    c1 = np.stack([x.ravel() for x in np.mgrid[:5, :6]])
    coordinates = np.hstack([c1, c1 + 0.5])  # half pixel offset
    tree = PolynomialTree(coordinates)

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

    tree.large_data = True
    members, locations1, phi1 = tree.block_members(
        test_block, get_locations=True, get_terms=True)
    assert np.allclose(members, [14, 44])
    assert phi1 is None


def test_hood_members():
    tree = PolynomialTree((3, 3))
    with pytest.raises(RuntimeError) as err:
        tree.hood_members(0)
    assert "neighborhood tree not initialized" in str(err.value).lower()

    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]))
    block = tree.to_index([3, 3])
    with pytest.raises(RuntimeError) as err:
        tree.hood_members(block, get_terms=True)
    assert "phi terms have not been calculated" in str(err.value).lower()

    tree = PolynomialTree(np.stack([x.ravel() for x in np.mgrid[:5, :6]]),
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

    tree.large_data = True
    a, b, c = tree.hood_members(populated_block, get_locations=True,
                                get_terms=True)
    assert np.allclose(a, members)
    assert np.allclose(b, locations)
    assert c is None
