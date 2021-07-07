# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.fill import maskinterp, clough_tocher_2dfunc


@pytest.fixture
def image():
    img = np.ones((64, 64))
    rand = np.random.RandomState(42)
    corrupt = rand.rand(64, 64) > 0.7  # 70 % corrupt
    img[corrupt] = np.nan
    return img


def test_error_and_unecessary(image):
    assert maskinterp(image * np.nan) is None
    good_image = np.ones_like(image)
    assert np.allclose(good_image, maskinterp(good_image))


def test_expected(image):
    result = maskinterp(image, order=1)
    assert np.allclose(result, 1)
    r1 = maskinterp(image, func=clough_tocher_2dfunc)
    r1idx = np.isfinite(r1)
    assert np.allclose(r1[r1idx], 1)
    assert not np.isfinite(r1).all()


def test_statistical(image):
    result = maskinterp(image, func=np.nanmean, statistical=True)
    assert np.allclose(result, 1)


def test_creep(image):
    image.fill(np.nan)
    image[32, 28:35:2] = 1
    image[28:35:2, 32] = 1
    # creep doesn't actually do anything in this case
    # Should only make a slight difference when it is activated
    result = maskinterp(image, func=np.nanmean, statistical=True, creep=True)
    assert np.allclose(result[np.isfinite(result)], 1)


def test_not_all_found(image):
    result = maskinterp(image, func=np.nanmean, statistical=True,
                        maxap=2, cval=-2)
    assert np.nanmin(result) == -2
