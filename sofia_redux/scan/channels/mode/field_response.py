# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.signal.signal import Signal

__all__ = ['FieldResponse']


class FieldResponse(Response):

    def __init__(self, channel_group=None, gain_provider=None, name=None,
                 floating=False, field=None, derivative_order=0):
        """
        Returns a field response mode.

        The field response mode is designed to return a signal based of a
        data field in the integration frame data.

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
        floating : bool, optional
            `True` if the signal is "floating".
        field : str, optional
            The data field of the frame data on which to base the signal.
        derivative_order : int, optional
            Return the derivative of the signal if greater than zero to the
            appropriate order.

        Returns
        -------
        None
        """

        self.field = str(field)
        self.is_floating = bool(floating)
        self.derivative_order = int(derivative_order)
        super().__init__(channel_group=channel_group,
                         gain_provider=gain_provider,
                         name=name)

    def get_signal(self, integration):
        """
        Get a signal from an integration.

        The retrieved signal will be based on a frame data field from the
        integration, and may be differentiated if the derivative_order
        attribute is non-zero.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        Signal
        """
        values = getattr(integration.frames, self.field)
        if values is None:
            values = np.zeros(integration.size, dtype=float)
            log.warning(f"No field named {self.field} in {integration} "
                        f"for signal.")

        values = np.asarray(values, dtype=float)
        signal = Signal(integration, mode=self, values=values,
                        is_floating=self.is_floating)
        if self.derivative_order > 0:
            for _ in range(self.derivative_order):
                signal.differentiate()
        return signal
