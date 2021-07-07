# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.clean_image import clean_image


def test_no_correct():
    image = np.ones((10, 10))
    assert np.allclose(clean_image(image), image)


def test_expected_results():
    y, x = np.mgrid[:9, :9]
    image = ((x - 4) ** 2) + ((y - 4) ** 2)
    original = image.copy()
    image[1, 3] = -1
    mask = np.full(image.shape, True)
    mask[1, 3] = False
    corrected = clean_image(image, mask=mask, order=2, mode='extrapolate')
    assert np.allclose(corrected, original)

    image = image.astype(float)
    error = np.ones_like(image)
    image[1, 3] = np.nan
    corrected = clean_image(image, error=error, order=2, mode='extrapolate')
    assert np.allclose(corrected, original)
