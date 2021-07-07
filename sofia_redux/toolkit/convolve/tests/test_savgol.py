# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.convolve.filter import savgol_windows
from sofia_redux.toolkit.convolve.filter import savgol


def test_savgol_windows_failures():
    x = np.arange(5)
    y = np.arange(5)
    with pytest.raises(ValueError):
        savgol_windows(2, 2, x, x, y)
        savgol_windows(2, 2, x[1:], y)
        savgol_windows(2, [2, 2], np.arange(5), np.arange(5),
                       np.zeros((5, 5)))


def test_savgol_windows_no_samples():
    o, w = savgol_windows(2, 3, np.arange(5))
    assert len(w) == 1 and w[0] == 3
    assert len(o) == 1 and o[0] == 2
    o, w = savgol_windows(2, 3, np.zeros((5, 5)))
    assert len(w) == 2 and np.allclose(w, 3)
    assert len(o) == 2 and np.allclose(o, 2)
    _, w = savgol_windows(2, 0, np.zeros((5, 5)))
    assert len(w) == 2 and np.allclose(w, 3)  # clipped to order, and odd
    _, w = savgol_windows(2, 10, np.zeros((5, 3)))
    assert np.allclose(w, [5, 3])  # clipped to shape


def test_savgol_windows_with_dimensions():
    x = np.arange(5)
    y = np.arange(7)
    data = np.zeros((5, 7))
    _, w = savgol_windows(2, 3, x, y, data, scale=True)
    assert len(w) == 2 and np.allclose(w, 3)
    _, w = savgol_windows(2, 10, x, y, data)
    assert np.allclose(w, [5, 7])


def test_savgol_1d():
    y = np.zeros(10)
    y[5] = 1
    result = savgol(y, 5)
    assert np.allclose(
        result[3:8], [-0.09, 0.34, 0.49, 0.34, -0.09], atol=0.01)
    assert np.allclose(result[3:8].sum(), 1)


def test_savgol_2d():
    y = np.zeros((10, 10))
    y[5, 5] = 1
    result = savgol(y, 5)
    cut = result[3:8, 3:8]
    assert np.allclose(cut,
                       [[0.007, -0.029, -0.042, -0.029, 0.007],
                        [-0.029, 0.118, 0.167, 0.118, -0.029],
                        [-0.042, 0.167, 0.236, 0.167, -0.042],
                        [-0.029, 0.118, 0.167, 0.118, -0.029],
                        [0.007, -0.029, -0.042, -0.029, 0.007]],
                       atol=0.001)
    assert np.allclose(cut.sum(), 1)


def test_savgol_check():
    y = np.zeros((10, 10))
    y[5, 5] = 1
    with pytest.raises(TypeError):
        savgol(y, 2, order=2, check=False)
    result = savgol(y, [3, 3], order=[2, 2], check=False)
    assert isinstance(result, np.ndarray)


def test_optional_arguments():
    y = np.zeros((10, 10))
    y[5, 5] = 1
    y[1, 5] = np.nan
    result = savgol(y, 5, mode='interp', delta=1.0)
    assert np.allclose(
        result[5, 3:8],
        [-0.04163265, 0.16653061, 0.23591837, 0.16653061, -0.04163265])
