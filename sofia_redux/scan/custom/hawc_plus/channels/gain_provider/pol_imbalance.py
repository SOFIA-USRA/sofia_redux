# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.channels.gain_provider.zero_mean_gains import (
    ZeroMeanGains)

__all__ = ['HawcPlusPolImbalance']


class HawcPlusPolImbalance(ZeroMeanGains):

    def __init__(self):
        """
        A gain provider that returns 1.0 for pol 0 channels and -1.0 otherwise.

        The average value will give an indication of the relative polarization
        imbalance.
        """
        super().__init__()

    def get_relative_gain(self, channel_data):
        """
        Returns 1.0 for pol=0 channels and -1.0 for other channels.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
            Channel data or channel group data

        Returns
        -------
        gain : numpy.ndarray (float)
            The relative gain
        """
        result = np.full(channel_data.size, -1.0)
        result[channel_data.pol == 0] = 1.0
        return result

    def set_raw_gain(self, channel_data, gain):
        """
        Not implemented for polarization imbalance.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
        gain : numpy.ndarray (float)

        Returns
        -------
        None
        """
        raise NotImplementedError("Cannot set polarization imbalance gains.")
