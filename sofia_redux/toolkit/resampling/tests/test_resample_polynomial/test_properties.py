# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial
from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree
from sofia_redux.toolkit.resampling.grid.polynomial_grid import PolynomialGrid

import numpy as np
import pytest


@pytest.fixture
def inputs():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    return coordinates, data


def test_fit_tree(inputs):
    coordinates, data = inputs
    r = ResamplePolynomial(coordinates, data)
    assert r.fit_tree is None
    settings = r.reduction_settings()
    r.pre_fit(settings, r.coordinates)
    assert isinstance(r.fit_tree, PolynomialTree)

    with pytest.raises(AttributeError) as err:
        r.fit_tree = None
    assert "can't set attribute" in str(err.value)


def test_grid_class(inputs):
    coordinates, data = inputs
    r = ResamplePolynomial(coordinates, data)
    assert r.grid_class == PolynomialGrid
    with pytest.raises(AttributeError) as err:
        r.grid_class = None
    assert "can't set attribute" in str(err.value)


def test_order(inputs):
    coordinates, data = inputs
    r = ResamplePolynomial(coordinates, data, order=[2, 3])
    assert np.allclose(r.order, [2, 3])
    r.sample_tree.set_order([3, 4])
    assert np.allclose(r.order, [3, 4])
    r.sample_tree = None
    assert np.allclose(r.order, [2, 3])
    with pytest.raises(AttributeError) as err:
        r.order = 3
    assert "can't set attribute" in str(err.value)
