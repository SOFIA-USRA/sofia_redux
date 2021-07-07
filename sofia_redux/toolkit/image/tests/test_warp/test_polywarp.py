# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.warp import polywarp, polywarp_image


@pytest.fixture
def points():
    # This is taken from IDL polywarp document example
    xi = [24, 35, 102, 92]
    yi = [81, 24, 25, 92]
    xo = [61, 62, 143, 133]
    yo = [89, 34, 38, 105]
    kx = [[-5.3784161, -0.32094529],
          [0.75147128, 0.0022292868]]
    ky = [[-10.147952, 1.0708497],
          [-0.016875444, -0.00057621399]]
    return xi, yi, xo, yo, kx, ky


def test_expected(points):
    xi, yi, xo, yo, kx0, ky0 = points
    kx, ky = polywarp(xi, yi, xo, yo, order=1)
    assert np.allclose(kx, kx0)
    assert np.allclose(ky, ky0)


def test_error(points):
    xi, yi, xo, yo, kx0, ky0 = points
    with pytest.raises(ValueError):
        polywarp(xi, yi, xo, yo[:-1])

    with pytest.raises(ValueError):
        polywarp(xi, yi, xi, yo, order=100)


def test_polywarp_image():
    xi = [0, 0, 9, 9]
    yi = [0, 9, 0, 9]
    xo = [1, 1, 10, 10]
    yo = [1, 10, 1, 10]
    image = np.arange(100, dtype=float).reshape(10, 10)

    # output is shifted down and right, but
    # top and left edge are lost in interpolation (prior to numpy 1.19)
    expected = np.zeros((10, 10), dtype=float)
    expected[2:, 2:] = image[1:-1, 1:-1]

    # test dimension error
    with pytest.raises(ValueError) as err:
        polywarp_image(image[0], xi, yi, xo, yo)
    assert 'must be a 2-D array' in str(err)

    # test expected
    warped = polywarp_image(image, xi, yi, xo, yo, method='nearest', cval=0.0)
    assert np.allclose(warped[2:-2, 2:-2], expected[2:-2, 2:-2])
