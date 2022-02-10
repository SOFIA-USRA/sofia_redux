# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.mode.mode import Mode
from abc import abstractmethod

__all__ = ['Response']


class Response(Mode):

    def __init__(self, channel_group=None, gain_provider=None, name=None):
        """
        An abstract response mode.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        The response mode contains the additional `get_signal` method to
        extract a Signal object from an integration.

        Parameters
        ----------
        channel_group : ChannelGroup, optional
            The channel group owned by the mode.
        gain_provider : str or GainProvider, optional
            If a string is provided a `FieldGainProvider` will be set to
            operate on the given field of the channel group.
        name : str, optional
            The name of the mode.  If not provided, will be determined from the
            channel group name (if available).
        """
        super().__init__(channel_group=channel_group,
                         gain_provider=gain_provider,
                         name=name)

    @abstractmethod
    def get_signal(self, integration):
        """
        Return a signal object from an integration.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        Signal
        """
        pass
