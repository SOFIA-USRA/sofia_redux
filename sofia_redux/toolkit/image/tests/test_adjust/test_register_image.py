# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import register_image, shift


@pytest.fixture
def data():
    dydx = [4.5, 2]
    rand = np.random.RandomState(42)
    reference = rand.rand(128, 128)
    image = shift(reference, dydx)
    return reference, image, -np.array(dydx)[::-1]


def test_error(data):
    reference, image, dxdy = data
    with pytest.raises(ValueError) as err:
        register_image(image, reference, maxshift=[1, 2, 3])
    assert "Invalid maxshift length" in str(err.value)

    with pytest.raises(ValueError) as err:
        register_image(image, reference, shift0=[1, 2, 3])
    assert "Invalid shift0 length" in str(err.value)

    with pytest.raises(ValueError) as err:
        register_image(image, reference[:-1])
    assert "Image shape does not match" in str(err.value)


def test_expected(data):
    reference, image, dxdy = data
    result = register_image(image, reference, upsample=1)
    assert np.allclose(result, [-2, -4]), 'pixel resolution'

    result = register_image(image, reference, upsample=10)
    assert np.allclose(result, dxdy), 'sub-pixel resolution'


def test_maxshift(data):
    reference, image, dxdy = data
    result = register_image(image, reference, maxshift=10)
    assert np.allclose(result, [-2, -4])
    result = register_image(image, reference, maxshift=5)
    assert np.allclose(result, [-2, -4])
    result = register_image(image, reference, maxshift=[0, 0])
    assert np.allclose(result, 0)


def test_shift0(data):
    reference, image, dxdy = data
    result = register_image(image, reference, shift0=dxdy)
    assert np.allclose(result, [-2, -4])
    result = register_image(image, reference, shift0=100, upsample=10)
    assert not np.allclose(result, [-2, -4])


def test_line():
    line1 = np.zeros((100, 1))
    line1[50] = 1
    line2 = np.zeros((100, 1))
    line2[53] = 1
    result = register_image(line1, line2)
    assert np.allclose(result, [0, 3])
