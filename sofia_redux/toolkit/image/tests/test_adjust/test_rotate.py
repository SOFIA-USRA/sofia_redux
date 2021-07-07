# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import rotate


@pytest.fixture
def cross():
    # squished crosshair
    image = np.zeros((64, 64))
    image[30:35, 32] = 1
    image[32, 28:37] = 1
    return image


@pytest.fixture
def single():
    # single slightly offset dot
    image = np.zeros((64, 64))
    image[33, 33] = 1
    return image


def test_errors(cross):
    assert rotate(cross[0], 90) is None
    assert rotate(cross, 90, pivot=[20]) is None


def test_expected(cross):
    rimg = rotate(cross, 90, order=0, pivot=[32, 32], missing=0)
    assert np.allclose(rimg, cross.T)

    rimg = rotate(cross, 45, order=1, pivot=[32, 32])
    ones = np.array(np.nonzero(rimg == 1))
    assert np.allclose(ones, [[30, 31, 31, 32, 33, 33, 34],
                              [34, 31, 33, 32, 31, 33, 30]])
    # the others are of mixed values
    vals = np.unique(np.around(rimg[np.isfinite(rimg)], decimals=3))
    assert np.allclose(vals, [0, 0.015, 0.172, 0.257, 0.293, 0.5, 0.757, 1])


def test_nan_handling(single):
    rimg1 = rotate(single, -45, order=1, missing_limit=0.5)
    single[34, 34] = np.nan
    rimg2 = rotate(single, -45, order=1, missing_limit=0.5)
    assert np.allclose(rimg1, rimg2, equal_nan=True)

    with pytest.raises(ValueError) as err:
        rotate(single, -45, nan_interpolation=np.nan)
    assert "nan_interpolation must be finite" in str(err.value)

    rimg = rotate(single, -45, nan_interpolation=None, order=1)
    assert np.allclose(rimg[33:35, 31:33], [[0.18933983, 0.18933983],
                                            [0.34619408, 0.34619408]])


def test_360_rotate(single):
    rimg = rotate(single, 360, order=5)  # high order should produce artifacts
    assert np.allclose(rimg, single)
