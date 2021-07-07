# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.convolve.kernel import savitzky_golay, SavgolConvolve


@pytest.fixture
def data():
    x = np.arange(1000).astype(float)
    y = np.full_like(x, 10)
    noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
    badpts = np.arange(x.size)
    np.random.shuffle(badpts)
    badpts = badpts[:100]
    return x, y, noise, badpts


def test_expected_values(data):
    x, y, noise, badpts = data
    assert np.allclose(savitzky_golay(x, y, 5), 10)

    # Check outlier rejection
    ynoise = y + noise
    ynoise_bad = ynoise.copy()
    ynoise_bad[badpts] += 1000
    smooth1 = savitzky_golay(x, ynoise, 5)
    smooth2 = savitzky_golay(x, ynoise_bad, 5)
    assert np.allclose(smooth1, 10, 0.5)
    assert not np.allclose(smooth2, 10, 0.5)
    smooth3 = savitzky_golay(x, ynoise, 5, robust=5)
    assert np.allclose(smooth3, 10, 0.5)


def test_model(data):
    x, y, noise, badpts = data
    model = savitzky_golay(x, y, 5, model=True)
    assert isinstance(model, SavgolConvolve)
