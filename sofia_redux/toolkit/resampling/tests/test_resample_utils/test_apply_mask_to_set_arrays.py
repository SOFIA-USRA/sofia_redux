# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    apply_mask_to_set_arrays)

import numpy as np


def test_apply_mask_to_set_arrays():
    n = 100
    mask = np.random.random(n) > 0.5
    data = np.random.random(n)
    error = np.random.random(n)
    weights = np.random.random(n)
    phi = np.random.random((3, n))

    data_out, phi_out, error_out, weights_out = apply_mask_to_set_arrays(
        mask, data, phi, error, weights)

    assert np.allclose(data[mask], data_out)
    assert np.allclose(phi[:, mask], phi_out)
    assert np.allclose(error[mask], error_out)
    assert np.allclose(weights[mask], weights_out)


def test_1_sized_arrays():
    n = 100
    mask = np.random.random(n) > 0.5
    data = np.empty(n)
    error = np.array([1.0])
    weights = np.array([2.0])
    phi = np.empty((2, n))

    data_out, phi_out, error_out, weights_out = apply_mask_to_set_arrays(
        mask, data, phi, error, weights)

    counts = mask.sum()

    assert error_out.shape == (counts,) and np.allclose(error_out, 1)
    assert weights_out.shape == (counts,) and np.allclose(weights_out, 2)


def test_counts():
    n = 100
    mask = np.random.random(n) > 0.5
    data = np.empty(n)
    error = np.empty(n)
    weights = np.empty(n)
    phi = np.empty((4, n))

    data_out, phi_out, error_out, weights_out = apply_mask_to_set_arrays(
        mask, data, phi, error, weights, counts=5)
    assert data_out.shape == (5,)
    assert phi_out.shape == (4, 5)
    assert error_out.shape == (5,)
    assert weights_out.shape == (5,)
