# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from sofia_redux.toolkit.image.adjust import upsampled_dft


def test_standard_operation():
    data = np.ones((4, 4))
    result = upsampled_dft(data, 2)
    assert np.allclose(result,
                       [[1.60000000e+01 + 0.00000000e+00j,
                         4.89858720e-16 + 4.89858720e-16j],
                        [4.89858720e-16 + 4.89858720e-16j,
                         -2.46519033e-32 + 2.46519033e-32j]]
                       )


def test_mismatch_upsampled_region_size():
    with pytest.raises(ValueError):
        upsampled_dft(
            np.ones((4, 4)),
            upsampled_region_size=[3, 2, 1, 4])


def test_mismatch_offsets_size():
    with pytest.raises(ValueError):
        upsampled_dft(np.ones((4, 4)), 3,
                      axis_offsets=[3, 2, 1, 4])
