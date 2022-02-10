# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.mode.position_response import PositionResponse
from sofia_redux.scan.flags.motion_flags import MotionFlags

__all__ = ['ChopperResponse']


class ChopperResponse(PositionResponse):

    def __init__(self, channel_group=None, gain_provider=None, name=None):
        """
        Creates a position response mode based on CHOPPER position.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        The pointing response is designed to extract a signal from an
        integration based on some function of CHOPPER position, defined by a
        specific direction operator.  The direction operator is an instance of
        the `MotionFlags` class.

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
        super().__init__(channel_group=channel_group,
                         gain_provider=gain_provider,
                         name=name,
                         position_type=MotionFlags.flags.CHOPPER)
