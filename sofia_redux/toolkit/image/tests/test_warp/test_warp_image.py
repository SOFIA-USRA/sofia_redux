# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from skimage.transform import PolynomialTransform

from sofia_redux.toolkit.image.warp import warp_image


def test_warping():
    y, x = np.mgrid[:11, :11]
    y = y - 5.0
    x = x - 5.0

    x2 = 0.5 * x + 0.5 * y
    y2 = 0.5 * y

    data = x + y

    polywarp = warp_image(data, x, y, x2, y2, order=2,
                          transform='polynomial', cval=np.nan,
                          missing_frac=1e-3)
    assert np.allclose(polywarp[2, 4:7], [-2, 0, 2])

    affinewarp = warp_image(data, x, y, x2, y2, order=2,
                            transform='affine', cval=np.nan,
                            missing_frac=1e-3)
    assert np.isclose(affinewarp[3, 5], 0.075193007470018)
    assert np.isclose(affinewarp[4, 6], 2.268067788409949)

    # NaN propagation
    data[4, 4] = np.nan

    polywarp, func = warp_image(data, x, y, x2, y2, order=2,
                                transform='polynomial', cval=np.nan,
                                missing_frac=1e-3, get_transform=True)
    assert np.isnan(polywarp[2, 4])
    assert np.allclose(polywarp[1:4:2, 4], -2)
    assert isinstance(func, PolynomialTransform)
