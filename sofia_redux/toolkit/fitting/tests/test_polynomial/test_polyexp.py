# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import polyexp


def test_error():
    with pytest.raises(ValueError) as err:
        polyexp([[1, 2], [3, 4]])
    assert 'must have 1 dimension' in str(err)


def test_1d():
    assert np.allclose(polyexp(5), np.arange(6))


def test_shape():
    assert np.allclose(polyexp([2, 3], indexing='i'),
                       [[0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [3, 1],
                        [0, 2],
                        [1, 2],
                        [2, 2],
                        [3, 2]]
                       )
    assert np.allclose(polyexp([2, 3], indexing='j'),
                       [[0, 0],
                        [1, 0],
                        [2, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [0, 2],
                        [1, 2],
                        [2, 2],
                        [0, 3],
                        [1, 3],
                        [2, 3]]
                       )


def test_taylor():
    assert np.allclose(polyexp(2, ndim=3),
                       [[0, 0, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 2, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 2]]
                       )
    assert np.allclose(polyexp(3, ndim=2),
                       [[0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [0, 2],
                        [1, 2],
                        [0, 3]]
                       )
