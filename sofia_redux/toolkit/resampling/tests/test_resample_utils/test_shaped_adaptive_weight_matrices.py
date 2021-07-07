# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    shaped_adaptive_weight_matrices, scaled_adaptive_weight_matrices)

import numpy as np


def test_rchi2_scaling_and_shape():
    sigma = np.full(3, 0.5)  # 0.5 also equals alpha = 0.5, inverse_alpha = 2

    # Check rchi2 scaling
    # 2 data sets of 5x10 output fit array
    rchi2_values = np.linspace(0, 2, 100).reshape((2, 5, 10))
    gradient_mscp = np.zeros(rchi2_values.shape + (3, 3))

    # Test the shaped and scaled solutions are equal for spheroid shape
    gradient_mscp[..., :, :] = np.eye(3)

    inverse_alpha_shaped = shaped_adaptive_weight_matrices(sigma, rchi2_values,
                                                           gradient_mscp)
    assert inverse_alpha_shaped.shape == (2, 5, 10, 3, 3)

    inverse_alpha_scaled = scaled_adaptive_weight_matrices(sigma, rchi2_values)

    reshaped_shape_matrices = np.zeros(rchi2_values.shape + (1, 3))
    diag = np.arange(3)
    reshaped_shape_matrices[..., 0, :] = inverse_alpha_shaped[..., diag, diag]

    assert np.allclose(reshaped_shape_matrices, inverse_alpha_scaled)

    # Now test with shape
    gradient_mscp[..., diag, diag] += np.arange(3)
    inverse_alpha_shaped = shaped_adaptive_weight_matrices(sigma, rchi2_values,
                                                           gradient_mscp)
    reshaped_shape_matrices[..., 0, :] = inverse_alpha_shaped[..., diag, diag]
    assert not np.allclose(reshaped_shape_matrices, inverse_alpha_scaled)

    # Test overall scaling factor is the same
    determinants = np.linalg.det(inverse_alpha_shaped)
    det0 = 8.0
    expected = np.sqrt(rchi2_values) * det0
    expected[rchi2_values == 0] = det0
    assert np.allclose(determinants, expected)


def test_density():
    gradient_mscp = np.zeros((1, 10, 3, 3))
    gradient_mscp[..., :, :] = np.diag(np.arange(3) + 1.0)
    rchi2 = np.full((1, 10), 4.0)
    density = np.full(rchi2.shape, 0.0)
    sigma = np.full(3, 0.5)

    inverse_alpha = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, density=density)
    assert np.allclose(np.linalg.det(inverse_alpha), 16)

    # Test no shape for zero density
    diag = inverse_alpha[..., np.arange(3), np.arange(3)].ravel()
    assert np.allclose(diag, diag[0])

    # Test shape approaches the gradient matrix as density -> infinity
    density.fill(100.0)
    inverse_alpha = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, density=density)
    assert np.allclose(np.linalg.det(inverse_alpha), 16)

    diag = inverse_alpha[..., np.arange(3), np.arange(3)]
    diag /= diag[0, 0, 0]
    assert np.allclose(diag, np.arange(3) + 1)


def test_offset():
    gradient_mscp = np.zeros((1, 10, 3, 3))
    gradient_mscp[..., :, :] = np.diag(np.arange(3) + 1.0)
    rchi2 = np.full((1, 10), 4.0)
    sigma = np.full(3, 0.5)
    offsets = np.full(rchi2.shape, 50)

    inverse_alpha = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, variance_offsets=offsets)
    assert np.allclose(np.linalg.det(inverse_alpha), 16)

    # Test no shape for high offset variance
    diag = inverse_alpha[..., np.arange(3), np.arange(3)].ravel()
    assert np.allclose(diag, diag[0])

    # Test shape is not impeded by low variance offsets
    offsets = np.full(rchi2.shape, 1e-6)
    inverse_alpha = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, variance_offsets=offsets)
    assert np.allclose(np.linalg.det(inverse_alpha), 16)

    diag = inverse_alpha[0, 0, np.arange(3), np.arange(3)]
    assert diag[0] < diag[1] < diag[2]


def test_fixed_dimensions():
    gradient_mscp = np.zeros((2, 10, 3, 3))
    gradient_mscp[..., :, :] = np.diag(np.arange(3) + 1.0)
    rchi2 = np.full((2, 10), 4.0)
    sigma = np.full(3, 0.5)
    fixed = np.full(3, False)

    a0 = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, fixed=fixed)

    fixed[0] = True
    a1 = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, fixed=fixed)
    assert np.allclose(a1[:, :, 0, 0], 2)
    assert np.allclose(a1[:, :, 1:, 0], 0)
    assert np.allclose(a1[:, :, 0, 1:], 0)
    assert not np.allclose(a1, a0)
    assert np.allclose(np.linalg.det(a1), 16)

    # Test the original matrix is returned if all dimensions are fixed
    fixed[:] = True
    a_3 = shaped_adaptive_weight_matrices(
        sigma, rchi2, gradient_mscp, fixed=fixed)
    values = a_3[a_3 != 0]
    assert np.allclose(values, 2)
