# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from copy import deepcopy
import numpy as np

from sofia_redux.scan.frames.frames import Frames
from sofia_redux.scan.channels.channel_data.channel_data import ChannelData

__all__ = ['Dependents']


class Dependents(ABC):

    referenced_attributes = ['integration']

    def __init__(self, integration, name):
        """
        Initialize a "dependents" object.

        The Dependents class contains separate float arrays of dependent values
        for both integration frames and integration channels.  Methods exist
        to add and subtract dependents from frames or channels.

        Parameters
        ----------
        integration : Integration
            The integration to which the dependents will belong.
        name : str
            The name of the dependents object.
        """
        self.name = name
        self.integration = integration
        self.for_frame = np.zeros(integration.size, dtype=float)
        self.for_channel = np.zeros(integration.channels.size)
        integration.add_dependents(self)

    def copy(self):
        """
        Return a copy of the dependents.

        Returns
        -------
        Dependents
        """
        new = self.__class__(self.integration, self.name)
        reference = self.referenced_attributes
        for attribute, value in self.__dict__.items():
            if attribute in reference:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))
        return new

    def add_async(self, channels_or_frames, value):
        """
        Add values to channel or frame dependents based on input.

        Parameters
        ----------
        channels_or_frames : Channels or ChannelData or Frames.
        value : float or numpy.ndarray (float).
            If an array, must be the same size as `channels_or_frames`.

        Returns
        -------
        None
        """
        if isinstance(channels_or_frames, ChannelData):
            indices = self.integration.channels.data.find_fixed_indices(
                channels_or_frames.fixed_index)
            self.for_channel[indices] += value
        elif isinstance(channels_or_frames, Frames):
            self.for_frame += value
        elif hasattr(channels_or_frames, 'data'):
            self.add_async(channels_or_frames.data, value)
        else:
            raise ValueError(f"Must be {ChannelData} or {Frames}.")

    def add_for_channels(self, value):
        """
        Add values to channel dependents.

        Parameters
        ----------
        value : float or numpy.ndarray (float).
            If an array, must be the same size as channels.

        Returns
        -------
        None
        """
        self.for_channel += value

    def add_for_frames(self, value):
        """
        Add values to frame dependents.

        Parameters
        ----------
        value : float or numpy.ndarray (float).
            If an array, must be the same size as frames.

        Returns
        -------
        None
        """
        self.for_frame += value

    def clear(self, channels=None, start=None, end=None):
        """
        Remove dependents from integration frames and channels.

        Removed dependents will be set to zero internally.

        Parameters
        ----------
        channels : Channels or ChannelData, optional
            If not supplied, use the integration channels.
        start : int, optional
            The starting frame to remove dependents.  The default is the
            first frame.
        end : int, optional
            The ending frame (exclusive) to remove dependents.  The default is
            the last channel.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.integration.channels

        self.integration.frames.remove_dependents(
            self.for_frame, start=start, end=end)

        indices = getattr(channels, 'indices', slice(None))
        channels.remove_dependents(self.for_channel[indices])
        self.for_channel[...] = 0.0
        self.for_frame[slice(start, end)] = 0.0

    def apply(self, channels=None, start=None, end=None):
        """
        Add dependents to integration frames and channels.

        Parameters
        ----------
        channels : Channels or ChannelData, optional
            If not supplied, use the integration channels.
        start : int, optional
            The starting frame to remove dependents.  The default is the
            first frame.
        end : int, optional
            The ending frame (exclusive) to remove dependents.  The default is
            the last channel.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.integration.channels

        self.integration.frames.add_dependents(
            self.for_frame, start=start, end=end)

        indices = getattr(channels, 'indices', None)
        if indices is None:
            channels.add_dependents(self.for_channel)
        else:
            channels.add_dependents(self.for_channel[indices])

    def get(self, channels_or_frames):
        """
        Return the frame or channel dependent values.

        Parameters
        ----------
        channels_or_frames : Channels or ChannelData or Frames

        Returns
        -------
        dependents : numpy.ndarray (float)
        """
        if isinstance(channels_or_frames, ChannelData):
            indices = getattr(channels_or_frames, 'indices', None)
            if indices is None:
                return self.for_channel.copy()
            else:
                return self.for_channel[indices]

        elif isinstance(channels_or_frames, Frames):
            return self.for_frame.copy()

        elif hasattr(channels_or_frames, 'data'):
            self.get(channels_or_frames.data)

        else:
            raise ValueError(f"Must be {ChannelData} or {Frames}.")
