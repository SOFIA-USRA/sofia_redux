# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.resample_polynomial \
    import ResamplePolynomial


def test_check_call_arguments():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    r = ResamplePolynomial(coordinates, data)

    with pytest.raises(ValueError) as err:
        r._check_call_arguments(1, 2, 3, 4)

    assert "4-feature coordinates passed to 2-feature resample" in str(
        err.value).lower()

    r._check_call_arguments(1, 2)
