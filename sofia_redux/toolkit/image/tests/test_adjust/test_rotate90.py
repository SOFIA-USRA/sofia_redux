# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import rotate90


@pytest.fixture
def image():
    result = np.zeros((3, 3), dtype=int)
    result[0, 0] = 1
    result[0, 2] = 2
    result[2, 0] = 3
    result[2, 2] = 4
    return result


def test_invalid_input():
    with pytest.raises(ValueError):
        rotate90(np.zeros((2, 2, 2)), 1)


def test_expected_output(image):
    img = image
    mask = img != 0
    assert np.allclose(rotate90(img, 0)[mask], [1, 2, 3, 4])
    assert np.allclose(rotate90(img, 1)[mask], [3, 1, 4, 2])
    assert np.allclose(rotate90(img, 2)[mask], [4, 3, 2, 1])
    assert np.allclose(rotate90(img, 3)[mask], [2, 4, 1, 3])
    assert np.allclose(rotate90(img, 4)[mask], [1, 3, 2, 4])
    assert np.allclose(rotate90(img, 5)[mask], [2, 1, 4, 3])
    assert np.allclose(rotate90(img, 6)[mask], [4, 2, 3, 1])
    assert np.allclose(rotate90(img, 7)[mask], [3, 4, 1, 2])


def test_1d_rotate():
    x = np.arange(10)
    assert rotate90(x, 1).ndim == 2
    assert rotate90(x, 2).ndim == 1
