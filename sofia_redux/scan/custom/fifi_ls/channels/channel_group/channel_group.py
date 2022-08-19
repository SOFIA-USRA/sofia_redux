# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.fifi_ls.channels.channel_data.channel_data \
    import FifiLsChannelData
from sofia_redux.scan.channels.channel_group.channel_group import ChannelGroup

__all__ = ['FifiLsChannelGroup']


class FifiLsChannelGroup(FifiLsChannelData, ChannelGroup):

    def __init__(self, channel_data, indices=None, name=None):
        """
        Initialize a FIFI-LS channel group.

        The channel group acts on a subset of the full channel data.

        Parameters
        ----------
        channel_data : FifiLsChannelData
            The channel data on which to base the group.
        indices : numpy.ndarray (int), optional
            The indices of ChannelData that will belong to the ChannelGroup.
            If no indices are supplied, the entire ChannelData will be
            referenced.
        name : str, optional
            The name of the ChannelGroup.
        """
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

        for attribute in ['coupling', 'spexel_gain', 'spaxel_gain',
                          'row_gain', 'col_gain']:
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
