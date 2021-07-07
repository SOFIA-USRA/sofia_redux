# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.fitting.polynomial import linear_equation


def test_standard():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([7., 8])
    alpha, beta = linear_equation(a, b, error=None, mask=None)

    assert np.allclose(alpha, [[25, 39], [39, 61]])
    assert np.allclose(beta, [53, 83])
    np.linalg.solve(alpha, beta)


def test_datavec():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([[7., 8], [7, 8]])
    alpha, beta = linear_equation(a, b, error=None, mask=None)
    assert np.allclose(alpha, [[25, 39], [39, 61]])
    assert np.allclose(beta, [53, 83])
    assert alpha.shape == (2, 2, 2)
    assert beta.shape == (2, 2)
    np.linalg.solve(alpha, beta)


def test_error():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([[7., 8], [7, 8]])
    alpha, beta = linear_equation(
        a, b, error=np.array([[0.5, 1.0], [1.0, 0.5]]), mask=None)
    assert np.allclose(alpha,
                       [[[52., 84.],
                         [84., 136.]],
                        [[73., 111.],
                         [111., 169.]]])
    assert np.allclose(beta, [[116., 188.],
                              [149., 227.]])


def test_mask():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([[7., 8], [7, 8]])
    alpha, beta = linear_equation(
        a, b, error=None, mask=np.array([[True, True], [False, False]]))
    assert np.allclose(alpha,
                       [[[25., 39.],
                         [39., 61.]],
                        [[0., 0.],
                        [0., 0.]]])
    assert np.allclose(beta,
                       [[53., 83.], [0., 0.]])

    # test no datavec
    alpha, beta = linear_equation(
        a, b[0], error=None, mask=np.array([False, False]))
    assert np.isnan(alpha).all()
    assert np.isnan(beta).all()

    alpha, beta = linear_equation(
        a, b, error=None, mask=np.array([[False, False], [False, False]]))
    assert np.isnan(alpha).all()
    assert np.isnan(beta).all()


def test_multivec():
    a = np.array([[3, 4], [5, 6.]])
    a = np.stack((a, a))
    b = np.array([7., 8])
    alpha, beta = linear_equation(a, b, error=None, mask=None)
    assert np.allclose(alpha,
                       [[[25., 39.],
                         [39., 61.]],
                        [[25., 39.],
                         [39., 61.]]])
    assert np.allclose(beta,
                       [[49., 77.],
                        [56., 88.]])
