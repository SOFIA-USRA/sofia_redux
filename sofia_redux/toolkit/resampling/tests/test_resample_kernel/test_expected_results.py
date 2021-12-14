# Licensed under a 3-clause BSD style license - see LICENSE.rs

from sofia_redux.toolkit.resampling.resample_kernel import ResampleKernel

import numpy as np
import pytest
from scipy.ndimage import convolve
from skimage.data import page


@pytest.fixture
def data_coordinates_2d():
    data = page().astype(float)
    data -= data.min()
    data /= data.max()
    y, x = np.mgrid[:data.shape[0], :data.shape[1]]
    coords = np.stack([x.ravel(), y.ravel()])
    return coords, data


def test_regular_2d(data_coordinates_2d):
    coords, data = data_coordinates_2d
    # data = np.zeros_like(data)
    # data[100, 100] = 1.0
    kernel = np.ones((3, 3))
    resampler = ResampleKernel(coords, data.ravel(), kernel, degrees=2,
                               kernel_spacing=1)
    result = resampler(coords).reshape(data.shape)
    expected = convolve(data, kernel / kernel.size, mode='constant',
                        cval=np.nan)
    # Can't check edges as scipy doesn't weight correctly
    test_mask = np.isfinite(expected)
    assert np.allclose(result[test_mask], expected[test_mask])


def test_regular_data_irregular_kernel():

    rand = np.random.RandomState(0)
    x = rand.random(1000) * 5 - 2.5
    y = rand.random(1000) * 5 - 2.5
    z = rand.random(1000) * 5 - 2.5

    kernel = np.exp(-((x ** 2) + (y ** 2) + (z ** 2)) * 0.5)
    kernel_offsets = np.stack([x, y, z])

    dz, dy, dx = np.mgrid[:25, :25, :25]
    data_coordinates = np.stack([dx.ravel(), dy.ravel(), dz.ravel()])
    data = np.zeros((25, 25, 25))
    data[7, 7, 7] = 1.0
    data[18, 18, 18] = 1.0

    resampler = ResampleKernel(data_coordinates, data.ravel(), kernel,
                               kernel_offsets=kernel_offsets, degrees=4)

    result = resampler(data_coordinates, jobs=-1).reshape(data.shape)

    zg, yg, xg = np.mgrid[:5, :5, :5]
    zg = zg - 2
    yg = yg - 2
    xg = xg - 2
    regular_kernel = np.exp(-((xg ** 2) + (yg ** 2) + (zg ** 2)) * 0.5)
    regular_kernel /= regular_kernel.sum()
    assert np.allclose(result[7, 7, 7], result[18, 18, 18])
    assert np.allclose(result[5:10, 5:10, 5:10], regular_kernel, atol=1e-2)
