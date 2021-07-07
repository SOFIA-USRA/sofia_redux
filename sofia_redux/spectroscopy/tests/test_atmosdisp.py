# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.atmosdisp import atmosdisp
import pytest


@pytest.fixture
def data():
    wavelength = np.arange(10) / 100 + 0.01
    refwave = 0.01
    za = 45.0
    pressure = 800.0
    temperature = 0.0
    water = 1.0
    altitude = 10  # km (same order as SOFIA altitude)
    expected = [
        0, -3.24693314, -7.5932743, -14.79513507, -27.35506815,
        -52.06782102, -117.0733797, -657.83697853, 297.86092331,
        144.91949177]
    return (wavelength, refwave, za, pressure,
            temperature, water, altitude, expected)


def test_success(data):
    (wavelength, refwave, za, pressure,
     temperature, water, altitude, expected) = data
    result = atmosdisp(wavelength, refwave, za, pressure, temperature,
                       water=water, altitude=altitude)
    assert np.allclose(result, expected)
