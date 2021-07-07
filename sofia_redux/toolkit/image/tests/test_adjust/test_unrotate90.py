# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import rotate90, unrotate90


@pytest.fixture
def image():
    result = np.zeros((3, 3), dtype=int)
    result[0, 0] = 1
    result[0, 2] = 2
    result[2, 0] = 3
    result[2, 2] = 4
    return result


def test_expected_results(image):
    img = image.copy()
    assert np.allclose(img, rotate90(img, 0))
    for rot in range(0, 8):
        rot1 = rotate90(img, rot)
        if rot == 0:
            assert np.allclose(rot1, image)
        else:
            assert not np.allclose(rot1, img)
        rot2 = unrotate90(rot1, rot)
        assert np.allclose(rot2, img)
