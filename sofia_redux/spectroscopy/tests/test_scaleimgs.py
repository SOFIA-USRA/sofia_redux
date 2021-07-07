# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.scaleimgs import scaleimgs
import pytest


@pytest.fixture
def data():
    images = np.zeros((5, 4, 3))
    images += np.arange(5)[:, None, None]
    variances = images ** 2
    return images, variances


def test_failure(data):
    images, variances = data
    assert scaleimgs(images, variances[0]) is None


def test_success(data):
    images, variances = data
    im = scaleimgs(images)
    assert np.isnan(im[0]).all()
    assert np.allclose(im[1:], 2)
    im, var = scaleimgs(images, variances)
    assert np.isnan(im[0]).all()
    assert np.allclose(im[1:], 2)
    assert np.isnan(var[0]).all()
    assert np.allclose(var[1:], 4)
    images = np.swapaxes(images, 0, 1)
    variances = np.swapaxes(variances, 0, 1)
    im, var = scaleimgs(images, variances, axis=1)
    im = np.swapaxes(im, 0, 1)
    var = np.swapaxes(var, 0, 1)
    assert np.isnan(im[0]).all()
    assert np.allclose(im[1:], 2)
    assert np.isnan(var[0]).all()
    assert np.allclose(var[1:], 4)
