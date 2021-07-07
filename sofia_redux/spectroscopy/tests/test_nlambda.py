# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.nlambda import nlambda
import pytest


@pytest.fixture
def data():
    wavelength = np.arange(10) / 100 + 0.01
    pressure = 800.0
    temperature = 0.0
    water = 1.0
    expected = [
        1.00007483, 1.00005905, 1.00003793, 1.00000293, 0.99994188,
        0.99982175, 0.9995057, 0.99687264, 1.00152137, 1.00077888
    ]
    return wavelength, pressure, temperature, water, expected


def test_success(data):
    wavelength, pressure, temperature, water, expected = data
    assert np.allclose(nlambda(
        wavelength, pressure, temperature, water=water), expected)
