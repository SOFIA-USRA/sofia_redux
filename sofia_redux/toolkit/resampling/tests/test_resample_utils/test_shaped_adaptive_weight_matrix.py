# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    shaped_adaptive_weight_matrix)

import numpy as np


def test_bad_values():
    sigma = np.arange(3) + 1.0

    # Non-finite rchi2 values result in a NaN result.
    a = shaped_adaptive_weight_matrix(sigma, np.nan, np.eye(3))
    assert a.shape == (3, 3)
    assert np.all(np.isnan(a))

    # Negative rchi2 values return the original matrix (in different terms).
    a = shaped_adaptive_weight_matrix(sigma, -1, np.eye(3))
    assert np.allclose(np.diag(a), 0.5 / sigma ** 2)

    # Bad gradient values result in a spheroid, appropriately scaled.
    g_mscp = np.random.random((3, 3))
    g_mscp[0, 1] = np.nan
    rchi2 = 4
    a_inv = shaped_adaptive_weight_matrix(sigma, rchi2, g_mscp)
    a = np.linalg.pinv(a_inv)
    a0 = np.diag(2 * sigma ** 2)

    # Check scaling ok
    assert np.isclose(np.linalg.det(a0) / np.linalg.det(a), np.sqrt(rchi2))

    # Check spheroid
    assert np.allclose(np.diag(a_inv), a_inv[0, 0])
    assert np.allclose(np.diag(np.diag(a_inv)), a_inv)  # off-diagonal = 0


def test_rchi2():

    # Not testing scaling here
    sigma = np.ones(2)

    gradient_mscp = np.array([[1.0, -0.5], [-0.5, 4]])
    angle_0 = np.arcsin(np.linalg.svd(gradient_mscp)[0][0, 1])
    ratio_0 = gradient_mscp[0, 0] / gradient_mscp[1, 1]  # 1/4

    # For rchi2 = 1, should be spheroid
    a_inv = shaped_adaptive_weight_matrix(sigma, 1.0, gradient_mscp)

    assert np.allclose(np.diag(a_inv), 0.5)  # 1 / (2 sigma ** 2)
    assert np.allclose(np.diag(np.diag(a_inv)), a_inv)

    # Check rchi2 > 1
    rchi2 = 4.0
    a_inv = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp)
    angle_1 = np.arcsin(np.linalg.svd(a_inv)[0][0, 1])
    ratio_1 = a_inv[0, 0] / a_inv[1, 1]

    # Test orientation is the same
    assert np.isclose(angle_0, angle_1)

    # Test scale is correct
    assert np.isclose(np.linalg.det(np.linalg.pinv(a_inv)), 2)

    # Test stretch is less pronounced, but longest is still the longest
    assert ratio_1 > ratio_0

    # Now check rchi2 < 1
    rchi2 = 1 / 9
    a_inv = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp)
    angle_2 = np.arcsin(np.linalg.svd(a_inv)[0][0, 1])
    ratio_2 = a_inv[1, 1] / a_inv[0, 0]  # due to rotation

    # Check for 90 degree rotation
    assert np.isclose((angle_1 - angle_2) % (2 * np.pi), np.pi / 2)

    # Check stretch is less pronounced, but longest is still the longest
    assert ratio_2 > ratio_0

    # Check scale is correct
    # det of A_0 is 4 (2*sigma^2), rchi = 3, therefore...
    assert np.isclose(np.linalg.det(np.linalg.pinv(a_inv)), 12)


def test_check_density():

    sigma = np.ones(2)
    gradient_mscp = np.array([[1.0, -0.5], [-0.5, 4]])
    maximum_stretch = gradient_mscp[0, 0] / gradient_mscp[1, 1]
    rchi2 = 4.0

    # a_inv_uniform = shaped_adaptive_weight_matrix(sigma, rchi2,
    #                                               gradient_mscp, density=1)

    a_inv_low = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp,
                                              density=1e-6)
    # Check it's spherical in low density regions
    assert np.isclose(a_inv_low[0, 0], a_inv_low[1, 1], atol=1e-3)

    a_inv_high = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp,
                                               density=1e6)
    # Check it's spherical in low density regions
    assert np.isclose(a_inv_low[0, 0], a_inv_low[1, 1])
    high_stretch = a_inv_high[0, 0] / a_inv_high[1, 1]
    assert np.isclose(high_stretch, maximum_stretch)


def test_check_offset():
    sigma = np.ones(2)
    gradient_mscp = np.array([[1.0, -0.5], [-0.5, 4]])

    rchi2 = 4.0
    a_inv_center = shaped_adaptive_weight_matrix(
        sigma, rchi2, gradient_mscp, variance_offset=0)
    center_stretch = a_inv_center[0, 0] / a_inv_center[1, 1]

    a_inv = shaped_adaptive_weight_matrix(
        sigma, rchi2, gradient_mscp, variance_offset=9.0)

    # Check fit is more spherical away from the center of the distribution
    offset_stretch = a_inv[0, 0] / a_inv[1, 1]
    assert offset_stretch > center_stretch
    assert offset_stretch < 1


def test_fixed_dimensions():
    sigma = np.ones(3) / 2  # gives alpha = 0.5

    det_initial = np.linalg.det(np.diag(2 * sigma ** 2))

    gradient_mscp = np.array(
        [[1.0, 0.5, 0.1],
         [0.5, 2.0, 0.2],
         [0.1, 0.2, 3.0]]
    )
    rchi2 = 4.0
    a_3 = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp)
    det_3 = np.linalg.det(np.linalg.pinv(a_3))

    assert np.isclose(det_initial / det_3, np.sqrt(rchi2))

    # Fix the first dimension
    fixed = np.array([True, False, False])
    a_2 = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp,
                                        fixed=fixed)
    det_2 = np.linalg.det(np.linalg.pinv(a_2))

    # Check no scaling or rotation in fixed dimensions
    assert np.allclose(a_2[0], [2, 0, 0])
    assert np.allclose(a_2[:, 0], [2, 0, 0])
    assert np.isclose(det_3, det_2)

    # This should be easy to calculate - stretch fixed in only 1-dimension
    fixed[:2] = True
    a_1 = shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp,
                                        fixed=fixed)
    det_1 = np.linalg.det(np.linalg.pinv(a_1))
    assert np.isclose(det_1, det_2)
    assert np.allclose(a_1, [[2, 0, 0], [0, 2, 0], [0, 0, 4]])
