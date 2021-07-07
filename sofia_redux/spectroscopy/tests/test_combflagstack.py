# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.combflagstack import combflagstack


def test_invalid_input():
    assert combflagstack([1, 2, 3]) is None
    assert combflagstack([[1, 2, 3], [1, 2, 3]], axis=12) is None


def test_expected_output_2d():
    bit2d = np.zeros((3, 5))
    bit2d[0] = np.arange(5) // 2
    bit2d[1] = np.arange(5) % 2
    bit2d[2, 2:] = [1, 1, 2]
    bit2d[0, 0] = 5
    assert np.equal(combflagstack(bit2d, 0), 0).all(), '2d 0-bit'
    assert np.equal(combflagstack(bit2d, 1), [1, 1, 1, 1, 0]).all(), '2d 1-bit'
    assert np.equal(combflagstack(bit2d, 8), [5, 1, 1, 1, 2]).all(), '2d 8-bit'


def test_expected_output_3d():
    bit3d = np.zeros((3, 2, 3))
    bit3d[0] = [[0, 1, 2], [0, 1, 2]]
    bit3d[1] = [[0, 1, 0], [0, 0, 1]]
    bit3d[2] = [[5, 0, 1], [1, 1, 1]]
    assert np.equal(combflagstack(bit3d, 0), 0).all(), '3d 0-bit'
    assert np.equal(combflagstack(bit3d, 2),
                    [[1, 1, 3], [1, 1, 3]]).all(), '3d 2-bit'
    assert np.equal(combflagstack(bit3d, 8),
                    [[5, 1, 3], [1, 1, 3]]).all(), '3d 8-bit'
    assert np.equal(combflagstack(bit3d, 8, axis=2),
                    [[3, 3], [1, 1], [5, 1]]).all(), '3d axis change'
