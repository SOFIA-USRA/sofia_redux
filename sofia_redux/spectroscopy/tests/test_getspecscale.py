# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.getspecscale import getspecscale


def test_invalid_input():
    assert getspecscale(np.zeros(5)) is None


def test_expected_results():
    stack = np.repeat([np.arange(10, dtype=float)],
                      5, axis=0) + np.arange(5)[:, None]
    scales = getspecscale(stack, refidx=None)
    assert np.allclose(scales, [1.45, 1.18, 1, 0.87, 0.76], atol=0.01)
    scales = getspecscale(stack, refidx=0)
    assert np.allclose(scales, [1, 0.82, 0.69, 0.6, 0.53], atol=0.01)
    stack[...] = np.nan
    assert np.equal(getspecscale(stack), 1).all()
    stack[...] = 0
    assert np.equal(getspecscale(stack), 1).all()


def test_2d():
    stack = []
    for n in range(5):
        aps = []
        for ap in range(3):
            s = np.arange(10, dtype=float) + n
            aps.append(s)
        stack.append(aps)
    stack = np.array(stack)

    scales = getspecscale(stack, refidx=None)
    assert np.allclose(scales, [1.45, 1.18, 1, 0.87, 0.76], atol=0.01)
