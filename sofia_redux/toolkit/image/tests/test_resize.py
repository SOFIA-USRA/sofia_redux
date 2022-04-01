# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.resize import resize


def test_resize():
    image = np.zeros((5, 5), dtype=np.float16)
    image[2, 2] = 1.0
    out = resize(image, (10, 10))
    nzi = np.nonzero(out)
    assert np.allclose(nzi[0],
                       [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6])
    assert np.allclose(nzi[1],
                       [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6])
    assert np.allclose(
        out[nzi],
        [0.0625, 0.1875, 0.1875, 0.0625, 0.1875, 0.5625, 0.5625, 0.1875,
         0.1875, 0.5625, 0.5625, 0.1875, 0.0625, 0.1875, 0.1875, 0.0625])

    out = resize(image, (4, 4))
    assert np.allclose(out,
                       [[0, 0, 0, 0],
                        [0, 0.140625, 0.140625, 0],
                        [0, 0.140625, 0.140625, 0],
                        [0, 0, 0, 0]])

    b = image.astype(bool)
    with pytest.raises(ValueError) as err:
        _ = resize(b, (10, 10), anti_aliasing=True)
    assert "anti_aliasing must be False" in str(err.value)

    with pytest.raises(ValueError) as err:
        _ = resize(image, (4, 4), anti_aliasing_sigma=-1)
    assert 'Anti-aliasing standard deviation' in str(err.value)
