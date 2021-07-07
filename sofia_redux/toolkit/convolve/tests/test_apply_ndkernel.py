# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.convolve.kernel import apply_ndkernel


@pytest.fixture
def data():
    image = np.zeros((11, 11))
    image[5, 5] = 2.0
    kernel = np.zeros(11)
    kernel[4:7] = 1
    kernel = np.multiply.outer(kernel, kernel)
    return image, kernel


def test_errors(data):
    image, kernel = data
    with pytest.raises(ValueError) as err:
        apply_ndkernel(image, kernel[None])
    assert "kernel must have 1 dimension or the same" in str(
        err.value).lower()


def test_by_axis(data):
    image, kernel = data
    result = apply_ndkernel(image, kernel, normalize=True)
    assert np.allclose(result, 2 * kernel / kernel.sum())
    result = apply_ndkernel(image, np.ones(3), normalize=False)
    assert np.allclose(result, 2 * kernel)


def test_full_convolution(data):
    result = apply_ndkernel(*data, normalize=False)
    image, kernel = data
    assert np.allclose(result, 2 * kernel)
    result = apply_ndkernel(*data, normalize=True)
    assert np.allclose(result, 2 * kernel / kernel.sum())


def test_convolve_errors(data):
    result = apply_ndkernel(*data, normalize=False, is_error=True)
    assert np.allclose(result, data[1] * np.sqrt(2))
