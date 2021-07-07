# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from packaging import version
import pytest
import scipy

from sofia_redux.toolkit.image.adjust import shift


@pytest.fixture
def image():
    image = np.zeros((64, 64))
    image[32, 32] = 1
    return image


def test_errors(image):
    assert shift(1, 1) is None
    assert shift(image, np.zeros(20)) is None
    assert shift(image, [0, 1], order='a') is None
    assert shift(image, [0, 1], order=-1) is None
    assert shift(image, [0, 1], order=6) is None


def test_expected(image):
    simg = shift(image, [0, 1], order=0)
    assert simg[32, 33] == 1
    assert simg[32, 32] == 0

    # test integer pixel shifts are treated accordingly
    simg = shift(image, [0, 1], order=3)
    assert simg[32, 33] == 1
    assert simg[32, 32] == 0

    # test fractional pixel shifts
    simg = shift(image, [0, 0.4], order=1)
    assert np.allclose(simg[32, 32:34], [0.6, 0.4])

    # test fractional pixel shifts over both axis
    simg = shift(image, 0.5, order=1)
    assert np.allclose(simg[32:34, 32:34], 0.25)


def test_nan_handling(image):
    image[32, 33] = np.nan
    simg = shift(image, [0, 0.4], order=1, missing_limit=0.5)
    assert np.allclose(simg[32, 32:34], [0.6, np.nan], equal_nan=True)
    simg = shift(image, [0, 0.5], order=1, missing_limit=0.5)
    assert np.allclose(simg[32, 32:34], 0.5)

    with pytest.raises(ValueError) as err:
        shift(image, [0, 0.5], nan_interpolation=np.nan)
    assert "nan_interpolation must be finite" in str(err.value)

    simg = shift(image, [0, 0.4], order=1, missing_limit=1,
                 nan_interpolation=None)

    scipy_version = version.parse(scipy.__version__)
    version_change = version.parse('1.4.0')

    if scipy_version < version_change:
        assert np.allclose(simg[32, 32:35], [[0.6, 0.49732038, 0.06488025]])
    else:
        assert np.allclose(simg[32, 32:35], [[0.6, 0.48606671, 0.05737781]])
