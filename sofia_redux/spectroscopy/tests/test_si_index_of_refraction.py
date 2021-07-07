# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.si_index_of_refraction \
    import si_index_of_refraction
import pytest


def test_failures():
    with pytest.raises(ValueError):
        si_index_of_refraction(np.arange(100), np.arange(99))


def test_success():
    wave = np.arange(1, 6).astype(float)
    temperature = 270.0
    result = si_index_of_refraction(wave, temperature)
    assert np.allclose(
        result,
        [3.56787447, 3.44932498, 3.42885567, 3.42186964, 3.41874214])

    # just testing second warning on temperature range
    result = si_index_of_refraction(wave, 10)
    assert np.allclose(
        result,
        [3.53881479, 3.42472093, 3.40472914, 3.39760479, 3.3940839])

    # test singular values
    result = si_index_of_refraction(3.0, 100)
    assert isinstance(result, float) and not hasattr(result, '__len__')
