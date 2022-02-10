# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.gain_provider.gain_provider import GainProvider
from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode
from abc import abstractmethod
import numpy as np

__all__ = ['ZeroMeanGains']


class ZeroMeanGains(GainProvider):

    def __init__(self):
        """
        An abstract class designed to subtract average gain values.
        """
        super().__init__()
        self.ave_g = 0.0

    @abstractmethod
    def get_relative_gain(self, channel_data):
        pass

    @abstractmethod
    def set_raw_gain(self, channel_data, gain):
        pass

    def get_gain(self, channel_data):
        """
        Returns gain values with the average gain removed (determined from
        `validate` of a Mode.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
            The channel data from which to extract gains.

        Returns
        -------
        gains : numpy.ndarray (float)
            The returned zero-mean gains.
        """
        gains = self.get_relative_gain(channel_data) - self.ave_g
        gains[np.isnan(gains)] = 0.0
        return gains

    def set_gain(self, channel_data, gains):
        """
        Set gains in the channel data.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
            The channel data for which to set gains.
        gains : numpy.ndarray (float)
            The gains to apply.

        Returns
        -------
        None
        """
        self.set_raw_gain(channel_data, gains + self.ave_g)

    def validate(self, mode):
        """
        Validate a given mode.

        Determine the average gain from a given mode.

        Parameters
        ----------
        mode : Mode

        Returns
        -------
        None
        """
        gains = self.get_relative_gain(mode.channel_group)
        weights = mode.channel_group.weight
        if isinstance(mode, CorrelatedMode):
            keep = mode.channel_group.is_unflagged(mode.skip_flags)
            gains = gains[keep]
            weights = weights[keep]

        sum_wg = np.sum(weights * gains)
        sum_w = np.sum(weights)
        if sum_w == 0:
            self.ave_g = 0.0
        else:
            self.ave_g = sum_wg / sum_w
