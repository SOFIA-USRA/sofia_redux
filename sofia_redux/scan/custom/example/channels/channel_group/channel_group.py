# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.example.channels.channel_data.channel_data \
    import ExampleChannelData
from sofia_redux.scan.channels.channel_group.channel_group import ChannelGroup

__all__ = ['ExampleChannelGroup']


class ExampleChannelGroup(ExampleChannelData, ChannelGroup):

    def __init__(self, channel_data, indices=None, name=None):
        ChannelGroup.__init__(self, channel_data, indices=indices, name=name)

    def set_channel_data(self, index, channel_info):
        """
        Set the channel info for a selected index.

        Parameters
        ----------
        index : int
            The channel index for which to set new data.
        channel_info : dict
            A dictionary of the form {field: value} where.  The attribute
            field at 'index' will be set to value.

        Returns
        -------
        None
        """
        super().set_channel_data(index, channel_info)
        if channel_info is None:
            return

        for attribute in ['coupling', 'mux_gain', 'bias_gain']:
            values = getattr(self, attribute)
            values[index] = channel_info[attribute]
            setattr(self, attribute, values)

    def validate_pixel_data(self):
        """
        Validates data read from the pixel data file.

        Returns
        -------
        None
        """
        raise NotImplementedError(
            f"Not implemented for {self.__class__} class.")

    def validate_weights(self):
        """
        Validates weight data.

        Returns
        -------
        None
        """
        raise NotImplementedError(
            f"Not implemented for {self.__class__} class.")

    def read_pixel_data(self, filename):
        """
        Read a pixel data file and apply the results.

        Parameters
        ----------
        filename : str
            File path to the pixel data file.

        Returns
        -------
        None
        """
        raise NotImplementedError(
            f"Not implemented for {self.__class__} class.")
