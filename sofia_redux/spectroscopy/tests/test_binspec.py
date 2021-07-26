# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.binspec import binspec


def test_invalid_input():
    assert binspec('a', [1], 1) is None
    assert binspec([1], 'a', 1) is None
    assert binspec([1], [1], 1) is None
    assert binspec([1, 2], [1, 2, 3], 1) is None
    assert binspec([1, 2], [1, 2], 1, xout='a') is None
    assert binspec([1, 2], [1, 2], 'a') is None


def test_default_expected_output():
    x = np.arange(10)
    y = np.arange(10)
    result = binspec(x, y, 2)
    assert np.allclose(result,
                       [[1., 3., 5., 7., 9.], [2., 6., 10., 14., 18.]])


def test_xout_expected_output():
    x = np.arange(10)
    y = np.arange(10)
    result = binspec(x, y, 2, xout=[2, 4])
    assert np.allclose(result, [[2., 4.], [4., 8.]])
    result = binspec(x, y, 2, xout=2)
    assert np.allclose(result, [[2.], [4.]])


def test_delta_output():
    x = np.arange(10)
    y = np.arange(10)

    # delta matches xout
    result = binspec(x, y, [1, 2, 3], xout=[1, 4, 8])
    assert np.allclose(result, [[1., 4., 8], [1.375, 8, 24]])

    # delta provided, xout not
    result = binspec(x, y, [1, 2, 3])
    assert np.allclose(result, [[0.5, 2., 4.5], [0.5, 4., 13.5]])

    result = binspec(x, y, [2, 2, 2])
    assert np.allclose(result, [[1., 3., 5.], [2., 6., 10.]])

    result = binspec(x, y, [2, 2, 2], lmax=4)
    assert np.allclose(result, [[1., 3.], [2., 6.]])

    # delta larger than array
    result = binspec(x, y, [9, 11])
    assert np.allclose(result, [[4.5], [40.5]])


def test_spacing_output():
    x = np.arange(10)
    y = np.arange(10)
    result = binspec(x, y, 1, lmin=3, lmax=5)
    assert np.allclose(result, [[3.5, 4.5, 5.5], [3.5, 4.5, 5.5]])


def test_average():
    x = np.arange(10)
    y = np.arange(10)
    result = binspec(x, y, 2, average=True)
    assert np.allclose(result, [[1, 3, 5, 7, 9], [1, 3, 5, 7, 9]])
