# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.smoothres import smoothres
import pytest


def test_invalid_input():
    with pytest.raises(ValueError) as err:
        smoothres([1, 2, 3], [1, 2], 1)
    assert "x and y array shape mismatch" in str(err.value)
    with pytest.raises(ValueError) as err:
        smoothres(*np.mgrid[:3, :3], 1)
    assert "x and y arrays must have 1 dimension" in str(err.value)
    with pytest.raises(ValueError) as err:
        smoothres([1, 2, 3], [1, 2, 3], -1)
    assert "resolution must be positive" in str(err.value)
    assert np.isnan(smoothres([np.nan] * 3, [1, 2, 3], 0.01, siglim=1)).all()


def test_expected_output():
    """
    This is compared directly with the old IDL values
    """
    x = np.arange(100) + 1
    y = np.zeros(100)
    y[50:53] = 1
    result = smoothres(x, y, 10, siglim=5)
    idl_result = [
        1.0237573e-05, 2.5295609e-05, 0.00070192904, 0.0016456661, 0.013337946,
        0.030787159, 0.083183721, 0.17168003, 0.25895768, 0.34214991,
        0.42534214, 0.36871091, 0.28929979, 0.21080966, 0.13384013,
        0.056870595, 0.028918294, 0.015090074, 0.0019396959, 0.0012582421,
        0.00057678821, 2.7674170e-05, 1.7766983e-05, 7.8597950e-06]
    assert np.allclose(result[41:65], idl_result, rtol=1e-3)


def test_zero_res():
    x = np.arange(100) + 1
    y = np.random.rand(100)
    result = smoothres(x, y, 0)
    assert np.allclose(result, y)
    result = smoothres(x, y, 1e-5)
    assert not np.allclose(result, y)
