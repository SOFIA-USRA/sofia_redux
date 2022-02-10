# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
import re

from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.flags.motion_flags import MotionFlags

__all__ = ['MotionResponse']


class MotionResponse(Response):

    def __init__(self, channel_group=None, gain_provider=None, name=None):
        """
        An abstract response mode class.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        The motion response is designed to extract a signal from an integration
        based on some function of position, defined by a specific direction
        operator.  The direction operator is an instance of the `MotionFlags`
        class.

        Parameters
        ----------
        channel_group : ChannelGroup, optional
            The channel group owned by the mode.
        gain_provider : str or GainProvider, optional
            If a string is provided a `FieldGainProvider` will be set to
            operate on the given field of the channel group.
        name : str, optional
            The name of the mode.  If not provided, will be determined from the
            channel group name (if available).  To set the direction, the name
            should be of the form <name>-<direction> or <name>:<direction>.
        """
        self.direction = None
        super().__init__(channel_group=channel_group,
                         gain_provider=gain_provider,
                         name=name)

    def set_name(self, name=None):
        """
        Set the name of the mode.

        If not provided, the name will be determined from the channel group.
        If no channel group is available, the name will not be set.  If the
        direction can be determined from the name, it will be set here.

        Parameters
        ----------
        name : str, optional
            The new name of the mode.  To set the direction, the name should
            be of the form <name>-<direction> or <name>:<direction>.

        Returns
        -------
        None
        """
        super().set_name(name)
        if isinstance(self.name, str):
            direction = re.split(r'[-:]', self.name)[1]
            self.set_direction(direction)

    def set_direction(self, direction):
        """
        Set the direction attribute to a MotionFlags instance.

        Parameters
        ----------
        direction : str or MotionFlags.flags

        Returns
        -------
        None
        """
        self.direction = MotionFlags(direction)

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
        direction = MotionFlags(self.direction.direction)
        return self.get_signal_from_direction(integration, direction)

    @abstractmethod
    def get_signal_from_direction(self, integration, direction):
        """
        Return a signal object from an integration and direction.

        Parameters
        ----------
        integration : Integration
        direction : MotionFlags

        Returns
        -------
        Signal
        """
        pass
