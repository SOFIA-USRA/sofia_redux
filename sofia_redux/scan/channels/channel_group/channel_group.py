# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod

from sofia_redux.scan.flags.flagged_data_group import FlaggedDataGroup
from sofia_redux.scan.channels.channel_data.channel_data import ChannelData

__all__ = ['ChannelGroup']


class ChannelGroup(FlaggedDataGroup, ChannelData):

    def __init__(self, channel_data, indices=None, name=None):
        """
        Create a channel group from channel data.

        A channel group is a subset of channels from the original channel data.
        The ChannelGroup object is a wrapper around the ChannelData class,
        accessing only certain channel indices and may be used to access or
        modify those elements of the original channel data.  However, due to
        the nature of indexing in numpy arrays, for value writing operations,
        the entire field must be modified at one.  For example
        self.gain[0]=1.234 will not result in any values being set.  To modify
        values, use self.gain=<new_array> or self.gain = <value> where <value>
        can be broadcast to the parent array.

        Parameters
        ----------
        channel_data : ChannelData or ChannelGroup or object
            The channel data to reference.  If an object is supplied, it's
            'data' attribute must contain ChannelData or ChannelGroup.
        indices : numpy.ndarray (int), optional
            The indices of ChannelData that will belong to the ChannelGroup.
            If no indices are supplied, the entire ChannelData will be
            referenced.
        name : str, optional
            The name of the ChannelGroup.
        """
        super().__init__(channel_data, indices=indices, name=name)

    def __getattr__(self, attribute):
        """
        Retrieves selected indices of the parent attribute.

        Provides additional handling for the "overlaps" field.

        Parameters
        ----------
        attribute : str
            Name of the data field.

        Returns
        -------
        numpy.ndarray
        """
        if attribute not in self.special_fields:
            return super().__getattr__(attribute)

        if attribute == 'overlaps':
            value = getattr(self.data, attribute, None)
            if value is None:
                return None
            elif self.indices is None:
                return
            else:
                return value[self.indices[:, None], self.indices[None]]
        else:
            return super().__getattr__(attribute)

    def __setattr__(self, attribute, value):
        """
        Sets selected indices of the parent data.

        Provides additional handling for the "overlaps" field.

        Parameters
        ----------
        attribute : str
            Name of the data field.
        value : numpy.ndarray
            Value to set.  value.shape[0] must be equal to the size of the
            group data.

        Returns
        -------
        None
        """
        if attribute not in self.special_fields:
            super().__setattr__(attribute, value)
            return

        parent_value = getattr(self.data, attribute, None)
        if parent_value is None:
            return

        if value is None:
            super().__setattr__(attribute, value)
            return

        if attribute == 'overlaps':
            parent_value[self.indices[:, None], self.indices[None]] = value

    def __str__(self):
        """
        Return a string representation of the channel group.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__} ({self.name}): {self.size} channels"

    def copy(self):
        """
        Return a copy of the channel group.

        Returns
        -------
        ChannelGroup
        """
        return self.__class__(self.data, indices=self.indices, name=self.name)

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
        if channel_info is None:
            return

        for attribute in ['gain', 'weight', 'flag']:
            values = getattr(self, attribute)
            values[index] = channel_info[attribute]
            setattr(self, attribute, values)

    def add_dependents(self, values):
        """
        Add dependent values for given indices.

        Parameters
        ----------
        values : numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.dependents += values

    def remove_dependents(self, values):
        """
        Subtract dependent values for given indices.

        Parameters
        ----------
        values : numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.dependents -= values

    def set_flag_defaults(self):
        """
        Sets data values based on currently set flags.

        Returns
        -------
        None
        """
        mask = self.is_flagged(self.flagspace.flags.DEAD
                               | self.flagspace.flags.DISCARD)
        for attribute in ['coupling', 'gain', 'weight', 'variance']:
            values = getattr(self, attribute)
            values[mask] = 0.0
            if attribute == 'coupling':
                values[self.is_flagged(self.flagspace.flags.BLIND)] = 0.0
            setattr(self, attribute, values)

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

    @abstractmethod
    def read_channel_data_file(self, filename):
        """
        Read a channel data file and return the information within.

        Note that this should be defined in the parent ChannelData class, not
        in one of the ChannelGroup classes.  Since the groups inherit from the
        data classes, this method is only here to avoid abstract method
        warnings.

        Parameters
        ----------
        filename : str
            The path to a channel data file.

        Returns
        -------
        channel_info : pandas.DataFrame
        """
        pass

    def create_group(self, indices=None, name=None, keep_flag=None,
                     discard_flag=None, match_flag=None):
        """
        Creates and returns a channel group.

        A channel group is a referenced subset of the channel data.  Operations
        performed on a channel group will be applied to the original channel
        data.

        Parameters
        ----------
        indices : numpy.ndarray (int), optional
            The indices to reference.  If not supplied, defaults to all
            channels.
        name : str, optional
            The name of the channel group.  If not supplied, defaults to the
            name of the channel data.
        discard_flag : int or str or ChannelFlagTypes, optional
            Flags to discard_flag from the new group.
        keep_flag : int or str or ChannelFlagTypes, optional
            Keep channels with these matching flags.
        match_flag : int or str or ChannelFlagTypes, optional
            Keep only channels with a flag exactly matching this flag.

        Returns
        -------
        ChannelGroup
            A newly created channel group.
        """
        return super().create_data_group(
            indices=indices, name=name, keep_flag=keep_flag,
            discard_flag=discard_flag, match_flag=match_flag)
