# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.smooth import fitplane
from sofia_redux.toolkit.utilities.func import stack


@pytest.fixture
def data():
    y, x = np.mgrid[:50, :50]
    c = stack(x, y)
    return c


def test_expected(data):
    result = fitplane(data)
    assert np.allclose(result[0], [24.5, 24.5])
    assert np.allclose(result[1], [0, 1])
