# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.mode.motion_response import MotionResponse
from sofia_redux.scan.flags.motion_flags import MotionFlags

__all__ = ['PositionResponse']


class PositionResponse(MotionResponse):

    def __init__(self, channel_group=None, gain_provider=None, name=None,
                 position_type=None):
        """
        Creates a position response mode class instance.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        The position response is designed to extract a signal from an
        integration based on some function of position, defined by a specific
        direction operator.  The direction operator is an instance of the
        `MotionFlags` class.

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
        position_type : str or int or MotionFlags.flags
            The position type may be one of {MotionFlags.flags.TELESCOPE,
            MotionFlags.flags.SCANNING, MotionFlags.flags.CHOPPER,
            MotionFlags.flags.PROJECT_GLS}, or {'Telescope', 'Scanning',
            'Chopper', 'Project GLS'}.  If using an integer identifier, please
            be cognizant of the mapped value.
        """
        super().__init__(channel_group=channel_group,
                         gain_provider=gain_provider,
                         name=name)
        self.type = None
        if position_type is not None:
            self.set_type(position_type)

    def set_type(self, position_type):
        """
        Sets the motion type.

        Parameters
        ----------
        position_type : str or int or MotionFlags.
            Available types are {MotionFlags.flags.TELESCOPE,
            MotionFlags.flags.SCANNING, MotionFlags.flags.CHOPPER,
            MotionFlags.flags.PROJECT_GLS}, or {'Telescope',
            'Scanning', 'Chopper', 'Project GLS'}.  If using an integer
            identifier, please be cognizant of the mapped value.

        Returns
        -------
        None
        """
        flag_type = MotionFlags.convert_flag(position_type)
        valid_types = [MotionFlags.flags.TELESCOPE,
                       MotionFlags.flags.SCANNING,
                       MotionFlags.flags.CHOPPER,
                       MotionFlags.flags.PROJECT_GLS]
        if flag_type not in valid_types:
            raise ValueError(f"Position type must be one of "
                             f"{set(valid_types)}.")
        self.type = flag_type

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
        return integration.get_position_signal(self.type, self.direction,
                                               mode=self)
