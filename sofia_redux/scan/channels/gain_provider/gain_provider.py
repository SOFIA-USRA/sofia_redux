# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod

__all__ = ['GainProvider']


class GainProvider(ABC):  # pragma: no cover

    def __init__(self):
        """
        The gain provider is an abstract class template, used to extract and
        set gain values for channel data.
        """
        pass

    @abstractmethod
    def get_gain(self, channel_data):
        """
        Retrieve gain values from channel data.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
            The channel data instance.

        Returns
        -------
        gains : numpy.ndarray (float)
            The gain values.
        """
        pass

    @abstractmethod
    def set_gain(self, channel_data, gain):
        """
        Set gain values in the gain provider.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
            The channel data instance.
        gain : numpy.ndarray (float)
            The gain values to set.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def validate(self, mode):
        """
        Validate a given mode.

        Parameters
        ----------
        mode : Mode
            The mode instance.

        Returns
        -------
        None
        """
        pass
