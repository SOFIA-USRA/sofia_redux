# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.fifi_ls.channels.channel_numba_functions import (
    get_relative_channel_weights)


def test_get_relative_channel_weights():
    frames = 1
    pixels = 10
    variance = np.ones((frames, pixels))
    weights = get_relative_channel_weights(variance)
    assert np.allclose(weights, 0)
    frames = 100
    variance = np.ones((frames, pixels))
    weights = get_relative_channel_weights(variance)
    assert np.allclose(weights, 1)
