# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.image.fill import polyclip


def test_invalid_input():
    """
    The point is that we want this fast for loops, so no checks are done
    """
    assert True


def test_expected_output():
    x = [0.5, 0.5, 1.5, 1.5]
    y = [0.5, 1.5, 1.5, 0.5]
    cx, cy = polyclip(0, 0, x, y)
    assert np.allclose(cx, [1, 1, 0.5, 0.5]), 'max 1 clip x'
    assert np.allclose(cy, [1, 0.5, 0.5, 1]), 'max 1 clip y'
    assert polyclip(0, 0, x, y, area=True) == 0.25

    x = [1.5, 0.75, 1.5, 2.25]
    y = [0.75, 1.5, 2.25, 1.5]
    cx, cy = polyclip(1, 1, x, y)
    assert np.allclose(cx, [2, 1.75, 1.25, 1, 1, 1.25, 1.75, 2]), 'all clip x'
    assert np.allclose(cy, [1.25, 1, 1, 1.25, 1.75, 2, 2, 1.75]), 'all clip y'

    x = [1.25, 1.25, 1.75, 1.75]
    y = [1.25, 1.75, 1.75, 1.25]
    cx, cy = polyclip(1, 1, x, y)
    assert np.allclose(cx, x), 'all inside x'
    assert np.allclose(cy, y), 'all inside y'

    cx, cy = polyclip(2, 2, x, y)
    assert cx is None, 'all outside x'
    assert cy is None, 'all outside y'
