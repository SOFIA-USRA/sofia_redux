# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import offset_variance

import numpy as np


def test_offset_variance():

    rand = np.random.RandomState(0)
    coordinates = rand.normal(loc=0.0, scale=1.0, size=(2, 1000000))
    tol = 1e-2

    var0 = offset_variance(coordinates, np.zeros(2))
    assert np.isclose(var0, 0, atol=tol)

    var1 = offset_variance(coordinates, np.array([0.0, 1.0]))
    assert np.isclose(var1, 1, atol=tol)

    var11 = offset_variance(coordinates, np.ones(2) / np.sqrt(2))
    assert np.isclose(var11, 1, atol=tol)

    var2 = offset_variance(coordinates, np.array([2.0, 0.0]))
    assert np.isclose(var2, 4, atol=tol)


def test_scale():
    rand = np.random.RandomState(1)
    coordinates = rand.normal(loc=0.0, scale=1.0, size=(2, 10000))
    v1 = offset_variance(coordinates, np.ones(2))
    v2 = offset_variance(coordinates, np.ones(2), scale=2)
    assert np.isclose(v1, v2 / 4)


def test_sigma_inv():
    rand = np.random.RandomState(2)
    coordinates = rand.normal(loc=0.0, scale=1.0, size=(2, 1000))
    v1 = offset_variance(coordinates, np.ones(2))
    v2 = offset_variance(coordinates, np.ones(2), sigma_inv=np.eye(1) * 0.01)
    assert not np.isclose(v1, v2, atol=1e-2)
