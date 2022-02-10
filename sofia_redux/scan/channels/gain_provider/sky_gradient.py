# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.channels.gain_provider.zero_mean_gains import (
    ZeroMeanGains)

__all__ = ['SkyGradient']


class SkyGradient(ZeroMeanGains):

    def __init__(self, horizontal=True):
        """
        Creates a SkyGradient gain provider.

        The sky gain provider returns the horizontal or vertical position data
        with a mean value removed.  The mean value is determined from a given
        mode during `validate`.

        Parameters
        ----------
        horizontal : bool, optional
            If `True` (default), applied to the horizontal position.
            Otherwise, operates on the vertical position data.
        """
        super().__init__()
        self.horizontal = horizontal

    def get_relative_gain(self, channel_data):
        """
        Returns the position data of the channel data.

        If the horizontal attribute is True, returns the x position coordinates
        of the channel data.  Otherwise, returns the y position coordinates.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup

        Returns
        -------
        position : numpy.ndarray (float)
            The horizontal or vertical position data.
        """
        position = getattr(channel_data, 'position', -1)
        if position == -1:
            raise TypeError(
                f"{channel_data} does not have 'position' attribute.")
        elif position is None:
            return np.full(channel_data.size, np.nan)

        if self.horizontal:
            return position.x
        else:
            return position.y

    def set_raw_gain(self, channel_data, gain):
        """
        Attempting to use this method will result in an error.

        The position data is not allowed to be set via this gain provider.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
        gain : numpy.ndarray (float)
            The gain values to set.

        Returns
        -------
        None
        """
        raise NotImplementedError("Cannot change gradient gains.")

    @classmethod
    def x(cls):
        """
        Creates and returns a horizontal sky gradient gain provider.

        Returns
        -------
        SkyGradient
            An instance of the SkyGradient class to operate on horizontal
            positions.
        """
        return SkyGradient(horizontal=True)

    @classmethod
    def y(cls):
        """
        Creates and returns a vertical sky gradient gain provider.

        Returns
        -------
        SkyGradient
            An instance of the SkyGradient class to operate on vertical
            positions.
        """
        return SkyGradient(horizontal=False)
