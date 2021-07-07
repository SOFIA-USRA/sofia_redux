# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import image_shift


@pytest.fixture
def image():
    image = np.zeros((64, 64))
    image[32, 32] = 1
    return image


def test_expected(image):
    result = image_shift(image, [2, 2], order=3)
    assert np.isclose(result[34, 34], 1)

    # check the rest
    assert np.isnan(result[:2]).all()
    assert np.isnan(result[:, :2]).all()
    result[34, 34] = np.nan
    assert np.allclose(result[np.isfinite(result)], 0)

    # fractional shifts
    result = image_shift(image, [1.5, 0.25], order=3)
    assert np.allclose(result[32, 30:38],
                       [-0.00806266, 0.03009027, -0.1122984, 0.52928213,
                        0.52928213, -0.1122984, 0.03009027, -0.00806266])
    assert np.allclose(result[30:38, 33],
                       [0.01981262, -0.07394171, 0.52928213, 0.16170415,
                        -0.04081446, 0.0109362, -0.00293035, 0.00078518])
