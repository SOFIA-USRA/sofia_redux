# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.channels.gain_provider.gain_provider import GainProvider

__all__ = ['FieldGainProvider']


class FieldGainProvider(GainProvider):

    def __init__(self, field):
        """
        Initializes a field gain provider.

        The field gain provider operates on a specified channel data field.
        Returned values (`get_gain`) retrieves that field from the channel
        data (copied, not referenced).  Also, sets the field of channel data
        using `set_gain`.

        Parameters
        ----------
        field : str
            The name of the field of the channel data on which to operate.
        """
        super().__init__()
        self.field = str(field)

    def get_gain(self, channel_data):
        """
        Retrieve gain values from channel data.

        Returned values are copied.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup
            The channel data instance.

        Returns
        -------
        gains : numpy.ndarray (float)
            The gain values.
        """
        values = getattr(channel_data, self.field)
        if values is None:
            raise ValueError(f"Channel group {channel_data} does not contain "
                             f"{self.field} field.")
        return values.astype(float)

    def set_gain(self, channel_data, gain):
        """
        Set gain values in the channel data for the field.

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
        value = np.asarray(gain, dtype=float)
        if isinstance(value, units.Quantity):
            if value.unit == units.dimensionless_unscaled:
                value = value.value

        if value.size != channel_data.size:
            raise ValueError("Gain size does not match channel size.")
        if not hasattr(channel_data, self.field):
            raise ValueError(f"{channel_data} does not have a "
                             f"{self.field} field.")
        setattr(channel_data, self.field, value)

    def validate(self, mode):
        """
        The field gain provider does not validate a mode.

        Parameters
        ----------
        mode : Mode

        Returns
        -------
        None
        """
        pass
