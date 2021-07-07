# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.smooth import fiterpolate


@pytest.fixture
def data():
    y, x = np.mgrid[:256, :256]
    z = (0.1 + (0.5 * x) + y + (2.0 * x * y)
         + (0.001 * y ** 2) + (0.002 * x ** 2))
    return z


def test_success(data):
    image = data
    fit = fiterpolate(image, 16, 16)
    residual = image - fit
    relative_delta = abs(residual) / image
    # ignore boundaries (sigh)
    assert np.allclose(relative_delta[50:200, 50:250], 0, atol=0.1)
