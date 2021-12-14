# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_kernel import ResampleKernel

import numpy as np
import pytest


@pytest.fixture
def gaussian_2d_resampler():
    y, x = np.mgrid[:23, :23]
    y2 = (y - 11.0) ** 2
    x2 = (x - 11.0) ** 2
    a = -0.1
    b = -0.2
    kernel = np.exp(a * x2 + b * y2)
    data = np.zeros((100, 100))
    data[50, 50] = 1.0
    x, y = np.mgrid[:100, :100]
    coords = np.stack([x.ravel(), y.ravel()]).astype(float)
    resampler = ResampleKernel(coords, data.ravel(), kernel,
                               kernel_spacing=1.0, degrees=3)
    return resampler


def test_kernel(gaussian_2d_resampler):
    assert gaussian_2d_resampler.kernel.shape == (23, 23)


def test_kernel_spacing(gaussian_2d_resampler):
    assert np.allclose(gaussian_2d_resampler.kernel_spacing, [1, 1])


def test_kernel_offsets(gaussian_2d_resampler):
    y, x = np.mgrid[:23, :23]
    x, y = x.ravel() - 11.0, y.ravel() - 11.0
    assert np.allclose(gaussian_2d_resampler.kernel_offsets,
                       np.stack([x, y]))


def test_degrees(gaussian_2d_resampler):
    assert np.allclose(gaussian_2d_resampler.degrees, [3, 3])


def test_spline_exit_code(gaussian_2d_resampler):
    assert gaussian_2d_resampler.exit_code == -1


def test_spline_exit_message(gaussian_2d_resampler):
    assert "interpolating spline" in gaussian_2d_resampler.exit_message
