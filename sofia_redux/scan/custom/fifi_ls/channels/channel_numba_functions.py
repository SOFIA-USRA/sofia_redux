# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

from sofia_redux.scan.channels.channel_numba_functions import \
    get_typical_gain_magnitude
from sofia_redux.scan.utilities.numba_functions import smart_median_1d


nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['get_relative_channel_weights']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_relative_channel_weights(variance):  # pragma: no cover
    """
    Calculate the relative frame weights from the input variance.

    Parameters
    ----------
    variance : numpy.ndarray (float)
        The variance array of shape (n_frames, n_pixels).

    Returns
    -------
    relative_weights : numpy.ndarray (float)
        The normalized frame weights of shape (n_frames,).
    """
    n_frames, n_pixels = variance.shape
    channel_weight = np.empty(n_pixels, dtype=nb.float64)
    for pixel in range(n_pixels):
        channel_weight[pixel] = get_typical_gain_magnitude(
            variance[:, pixel])

    for pixel, value in enumerate(channel_weight):
        if value == 1:
            channel_weight[pixel] = np.nan

    typical_variance = smart_median_1d(channel_weight)[0]
    for pixel, value in enumerate(channel_weight):
        if np.isnan(value):
            channel_weight[pixel] = 0.0
            continue
        channel_weight[pixel] = typical_variance / value

    return channel_weight
