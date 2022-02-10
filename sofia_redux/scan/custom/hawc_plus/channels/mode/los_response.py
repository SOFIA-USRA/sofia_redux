# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.mode.field_response import FieldResponse

__all__ = ['LosResponse']


class LosResponse(FieldResponse):

    def __init__(self, channel_group=None, gain_provider=None, name=None):
        """
        Initialize a LOS response mode.

        The HAWC_PLUS LOS response mode extracts a signal from the "LOS" field
        of the integration frame data, returning the second order derivative
        signal which is "floating".

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
                         name=name,
                         floating=True,
                         derivative_order=2,
                         field='los')
