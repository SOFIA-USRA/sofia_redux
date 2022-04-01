# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.image.warp import warp_image, PolynomialTransform


def test_warp():
    y, x = np.mgrid[:11, :11]
    y = y.astype(float)
    x = x.astype(float)
    x2 = 0.5 * x + 0.5 * y
    y2 = 0.5 * y + 2.5
    data = x + y

    warped = warp_image(data, x, y, x2, y2, order=3,
                        interpolation_order=3,
                        cval=np.nan, missing_frac=1e-3)

    masked = np.isnan(warped)
    indices = np.nonzero(~masked)
    assert np.allclose(indices[0],
                       [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                        6, 6, 6, 6, 6, 7, 7, 7, 7, 7])
    assert np.allclose(indices[1],
                       [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7,
                        4, 5, 6, 7, 8, 5, 6, 7, 8, 9])
    assert np.allclose(
        warped[indices],
        [2, 4, 6, 8, 10, 4, 6, 8, 10, 12, 6, 8, 10, 12, 14, 8, 10,
         12, 14, 16, 10, 12, 14, 16, 18])

    nan_data = data.copy()
    nan_data[5, 5] = np.nan
    warped = warp_image(nan_data, x, y, x2, y2, order=3,
                        interpolation_order=3,
                        cval=np.nan, missing_frac=1e-3)
    masked = np.isnan(warped)
    indices = np.nonzero(~masked)
    assert np.allclose(indices[0],
                       [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
                        6, 7, 7, 7, 7, 7])
    assert np.allclose(indices[1],
                       [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 6, 7, 4, 5, 6, 7,
                        8, 5, 6, 7, 8, 9])
    assert np.allclose(
        warped[indices],
        [2, 4, 6, 8, 10, 4, 6, 8, 10, 12, 6, 8, 12, 14, 8, 10, 12, 14, 16,
         10, 12, 14, 16, 18])

    warped, transform = warp_image(nan_data, x, y, x2, y2, order=3,
                                   interpolation_order=3,
                                   cval=np.nan, missing_frac=1e-3,
                                   extrapolate=False, get_transform=True)
    assert isinstance(transform, PolynomialTransform)
    masked = np.isnan(warped)
    indices = np.nonzero(~masked)
    assert np.allclose(indices[0],
                       [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6,
                        6, 6])
    assert np.allclose(indices[1],
                       [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 6, 7, 4, 5, 6, 7,
                        8])
    assert np.allclose(warped[indices],
                       [2, 4, 6, 8, 10, 4, 6, 8, 10, 12, 6, 8, 12, 14, 8, 10,
                        12, 14, 16])
