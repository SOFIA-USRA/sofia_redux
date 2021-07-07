# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import frebin


@pytest.fixture
def image():
    y, x = np.mgrid[:64, :64]
    z = (y % 2) & (x % 2)
    return z.astype(float)


def test_no_correction(image):
    assert np.allclose(image, frebin(image, image.shape))


def test_order_determination(image):
    s2 = 128, 128  # exactly double the size - no interpolation needed
    s_frac = 100, 100  # interpolation should be required

    f2_0 = frebin(image, s2, order=0)
    f2_1 = frebin(image, s2, order=1)
    assert np.unique(f2_0).size == 2  # either 0 or 1
    assert np.unique(f2_1).size > 2  # interpolation occurred

    frac_0 = frebin(image, s_frac, order=0)
    frac_1 = frebin(image, s_frac, order=1)
    assert np.unique(frac_0).size == 2  # either 0 or 1
    assert np.unique(frac_1).size > 2  # interpolation occurred

    auto_2 = frebin(image, s2)
    auto_f = frebin(image, s_frac)
    assert np.allclose(auto_2, f2_0)  # should have chosen nearest neighbor
    assert np.allclose(auto_f, frac_1)  # should have chosen linear interp.


def test_aliasing_determination(image):
    s2 = 128, 128
    forward = frebin(image, s2)
    assert forward.shape == s2
    back = frebin(forward, image.shape)
    back_bad = frebin(forward, image.shape, anti_aliasing=True)
    assert np.allclose(back, image)
    assert not np.allclose(back_bad, image)


def test_total(image):
    forward = frebin(image, (128, 128))
    assert forward.max() == 1.0
    assert forward.min() == 0.0
    forward_t = frebin(image, (128, 128), total=True)
    assert forward_t.max() == 0.25  # spread over 4 pixels
    assert forward_t.min() == 0


def test_nan_handling(image):
    image[32, 32] = np.nan
    forward = frebin(image, (128, 128))
    nans = np.isnan(forward)
    assert nans.sum() == 4
    assert nans[64:66, 64:66].all()

    back = frebin(forward, image.shape)
    assert np.allclose(back, image, equal_nan=True)
