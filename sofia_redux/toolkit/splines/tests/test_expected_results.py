from sofia_redux.toolkit.splines.spline import Spline

import numpy as np
import pytest


@pytest.fixture
def gaussian_2d_data():
    y, x = np.mgrid[:21, :21]
    y2 = (y - 10.0) ** 2
    x2 = (x - 10.0) ** 2
    a = -0.1
    b = -0.15
    data = np.exp(a * x2 + b * y2)
    return data


def test_gaussian_2d(gaussian_2d_data):
    data = gaussian_2d_data
    # Test symmetrical degrees
    spline = Spline(data, degrees=4, smoothing=0.0)
    assert np.allclose(spline(np.arange(21), np.arange(21)), data)
    # Test asymmetrical degrees
    spline = Spline(data, degrees=[2, 5], smoothing=0.0)
    assert np.allclose(spline(np.arange(21), np.arange(21)), data)
