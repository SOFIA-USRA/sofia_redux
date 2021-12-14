# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_kernel import ResampleKernel

import numpy as np
import pytest


# block_loop and combine_blocks are tested during __call__()


@pytest.fixture
def gaussian_2d_kernel():
    y, x = np.mgrid[:23, :23]
    y2 = (y - 11.0) ** 2
    x2 = (x - 11.0) ** 2
    a = -0.1
    b = -0.3
    kernel = np.exp(a * x2 + b * y2)
    return kernel


@pytest.fixture
def data_coordinates_2d():
    data = np.zeros((51, 51))
    data[25, 25] = 1.0
    x, y = np.mgrid[:51, :51]
    coords = np.stack([x.ravel(), y.ravel()]).astype(float)
    return coords, data.ravel()


@pytest.fixture
def gaussian_2d_resampler(gaussian_2d_kernel, data_coordinates_2d):
    coords, data = data_coordinates_2d
    resampler = ResampleKernel(coords, data, gaussian_2d_kernel,
                               kernel_spacing=1.0, degrees=3)
    return resampler


def test_set_sample_tree(gaussian_2d_resampler):
    resampler = gaussian_2d_resampler
    c0 = resampler.coordinates.copy()
    ct0 = resampler.sample_tree.coordinates.copy()
    resampler.set_sample_tree(c0[::-1])
    assert np.allclose(resampler.sample_tree.coordinates, ct0[::-1])
    resampler.set_sample_tree(c0)
    assert np.allclose(resampler.sample_tree.coordinates, ct0)


def test_estimate_feature_windows(gaussian_2d_resampler):
    w = np.sqrt(2) * 11
    assert np.allclose(gaussian_2d_resampler.estimate_feature_windows(),
                       [w, w])


def test_set_kernel(gaussian_2d_kernel, data_coordinates_2d):
    kernel = gaussian_2d_kernel
    coords, data = data_coordinates_2d
    resampler = ResampleKernel(coords, data, kernel, kernel_spacing=0.5)

    with pytest.raises(ValueError) as err:
        resampler.set_kernel(kernel)
    assert "Kernel spacing or offsets must be supplied" in str(err.value)

    offsets = resampler.kernel_offsets * 2
    resampler.set_kernel(kernel.ravel(), kernel_offsets=offsets)
    assert np.allclose(offsets, resampler.kernel_offsets)
    assert resampler.kernel_spacing is None

    resampler.set_kernel(kernel, kernel_spacing=1.0, imperfect=True,
                         degrees=2, smoothing=1, eps=1.23e-6)
    assert resampler.sample_tree.smoothing == 1
    assert np.allclose(resampler.degrees, [2, 2])
    assert resampler.sample_tree.spline.eps == 1.23e-6
    assert np.allclose(resampler.kernel_spacing, [1, 1])
    y, x = np.mgrid[:kernel.shape[0], :kernel.shape[1]]
    k_coords = np.stack([x.ravel(), y.ravel()]) - 11.0
    assert np.allclose(k_coords, resampler.kernel_offsets)


def test_reduction_settings(gaussian_2d_resampler):
    resampler = gaussian_2d_resampler
    settings = resampler.reduction_settings(error_weighting=False,
                                            fit_threshold=0.5,
                                            cval=2.0,
                                            edge_threshold=0.75,
                                            edge_algorithm='distribution',
                                            is_covar=True, jobs=-1)
    assert not settings['error_weighting']
    assert settings['fit_threshold'] == 0.5
    assert settings['cval'] == 2
    assert np.allclose(settings['edge_threshold'], [0.75, 0.75])
    assert settings['edge_algorithm'] == 'distribution'
    assert settings['jobs'] == -1
    assert settings['is_covar']


def test_call(gaussian_2d_resampler):
    resampler = gaussian_2d_resampler
    # self convolution (all coordinates)
    results_flat = resampler(resampler.coordinates)
    assert isinstance(results_flat, np.ndarray)
    assert results_flat.shape == (51 * 51,)

    results = resampler(resampler.coordinates, get_error=True, get_counts=True,
                        get_weights=True, get_distance_weights=True,
                        get_rchi2=True, get_offset_variance=True)
    fit_norm, error, counts, weights, dweights, rchi2, var = results
    assert np.allclose(weights, dweights)  # no error weighting
    assert np.allclose(rchi2, 1)
    assert not resampler.fit_settings['absolute_weight']

    # Test normalization and absolute weighting
    fit = resampler(resampler.coordinates, normalize=False,
                    absolute_weight=True)
    assert np.allclose(fit, fit_norm * dweights)
    assert resampler.fit_settings['absolute_weight']


def test_process_block(gaussian_2d_resampler):
    r = gaussian_2d_resampler
    settings = r.reduction_settings()
    r.pre_fit(settings, r.sample_tree.coordinates)
    g = r.global_resampling_values()
    get_error = get_counts = get_weights = get_distance_weights = False
    get_rchi2 = get_offset_variance = False

    filename, iteration = None, 1
    args = (r.data, r.error, r.mask, r.fit_tree, r.sample_tree,
            get_error, get_counts, get_weights, get_distance_weights,
            get_rchi2, get_offset_variance, settings)

    g['args'] = args
    g['iteration'] = iteration
    g['filename'] = filename

    block_population = r.fit_tree.block_population
    hood_population = r.sample_tree.hood_population
    skip = (block_population == 0) | (hood_population == 0)
    first_block = np.nonzero(~skip)[0][0]

    result = r.process_block((filename, iteration), first_block)
    assert len(result) == 8
    fit_indices, fit, error, counts, wsum, dwsum, rchi2, var = result
    assert isinstance(fit_indices, np.ndarray) and fit_indices.size > 0
    assert isinstance(fit, np.ndarray) and fit.size > 0

    for value in [error, counts, wsum, dwsum, rchi2, var]:
        assert isinstance(value, np.ndarray) and value.size == 0
