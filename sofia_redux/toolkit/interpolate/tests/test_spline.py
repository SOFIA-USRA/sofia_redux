# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import spline


@pytest.fixture
def linear():
    x = np.arange(10).astype(float)
    y = np.arange(10).astype(float)
    xout = np.arange(11) - 0.5
    return x, y, xout


@pytest.fixture
def linear_sawtooth():
    x = np.arange(10).astype(float)
    y = ((x + 1) // 2) * ((-1) ** x) + 0.5  # centered around 0 for odd inds
    bendx = x[1:9] - 0.1  # "bendy" points
    ly = np.interp(bendx, x, y)  # linear fit
    return x, y, bendx, ly


def test_failures(linear):
    x, y, xout = linear
    with pytest.raises(ValueError):
        spline(np.zeros((3, 3)), y, xout)

    with pytest.raises(ValueError):
        spline(x, np.zeros((3, 3)), xout)

    with pytest.raises(ValueError):
        spline(x[:2], y[:2], xout)

    with pytest.raises(ValueError):
        spline(x[:-1], y, xout)

    with pytest.raises(ValueError):
        spline(x, y, np.zeros((3, 3)))


def test_basic_functionality(linear):
    x, y, xout = linear
    assert np.allclose(spline(x, y, xout), xout)
    scalar = spline(x, y, 100)
    assert isinstance(scalar, float) and scalar == 100
    assert spline(x, y, 100) == 100
    assert np.allclose(spline(np.flip(x), np.flip(y), xout), xout)
    assert np.allclose(spline(np.flip(x), y, xout), np.flip(xout))


def test_tension(linear_sawtooth):
    x, y, bendx, ly = linear_sawtooth
    # Examine fit closest to the "bendy" parts of the fit
    decades = 7
    sigmas = 10.0 ** (np.arange(decades) - 4)
    # higher sigma gives a tighter (more linear) fit
    fits = np.zeros((decades, bendx.size))
    for i, sigma in enumerate(sigmas):
        fits[i] = spline(x, y, bendx, sigma=sigma)

    assert np.allclose(fits[0], fits[1])  # sigma low limit test
    fits = fits[1:]

    # Check that as sigma increases, the fit becomes more linear overall
    deviation = np.abs(fits - ly[None])
    compd = deviation[1:] < deviation[:-1]
    assert np.allclose(np.median(compd, axis=1), 1)
