# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.fill import image_naninterp


@pytest.fixture
def image():
    rand = np.random.RandomState(42)
    data = np.ones((128, 128))
    data[rand.rand(*data.shape) > 0.8] = np.nan
    return data


def test_error_or_irrelevant(image):
    assert image_naninterp(image[0]) is None
    assert image_naninterp(image * np.nan) is None
    assert np.allclose(image_naninterp(np.ones_like(image)), 1)


def test_expected(image):
    result = image_naninterp(image)
    idx = np.isfinite(result)
    assert idx.sum() > np.isfinite(image).sum()
    assert np.allclose(result[idx], 1)
