# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
import numpy as np

from sofia_redux.scan.filters.filter import Filter
from sofia_redux.scan.filters import filters_numba_functions as fnf

__all__ = ['FixedFilter']


class FixedFilter(Filter):

    def __init__(self, integration=None, data=None):
        """
        Initializes a fixed filter.

        The fixed filter has a fixed point response for each channel in an
        integration.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.point_response = None
        self.rejected = None
        super().__init__(integration=integration, data=data)

    @property
    def channel_dependent_attributes(self):
        """
        Return attributes that are dependent on the parent channels.

        This is required during a slim operation (reducing number of channels).

        Returns
        -------
        set (str)
        """
        attributes = super().channel_dependent_attributes
        attributes.add('point_response')
        attributes.add('rejected')
        return attributes

    def set_integration(self, integration):
        """
        Set the filter integration.

        Sets the padding of the FFT filter, the number of frequencies, the
        frequency spacing, and retrieves the channels from the integration if
        necessary.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        super().set_integration(integration)
        self.point_response = np.ones(self.channel_size, dtype=np.float64)
        self.rejected = np.zeros(self.channel_size, dtype=np.float64)

    def get_point_response(self, channels=None):
        """
        Return the channel point responses for the filter.

        Parameters
        ----------
        channels : Channels or ChannelData, optional
            Return the point response for a given set of channels.  If not
            supplied, defaults to the filter channels.

        Returns
        -------
        response : numpy.ndarray (float)
            An array of shape (channels.size,)
        """
        if channels is None or channels is self.channels:
            return self.point_response

        channel_indices = self.channels.find_fixed_indices(
            channels.fixed_index)
        return self.point_response[channel_indices]

    def get_mean_point_response(self, channels=None):
        """
        Return the channel mean point response of the filter.

        Parameters
        ----------
        channels : Channels or ChannelData
            The set of channels for which to calculate the mean point response.
            Defaults to the filter channels.

        Returns
        -------
        response : float
        """
        return self.get_point_response(channels=channels)

    def pre_filter(self):
        """
        Perform the pre-filtering steps.

        During the pre-filtering steps, dependents are retrieved from the
        filter integration and cleared (subtracted from integration dependents,
        integration channels, and zeroed).

        Returns
        -------
        None
        """
        super().pre_filter()
        self.rejected[...] = self.count_parms()

    def reset_point_response(self, channels=None):
        """
        Calculate and store the correct point response for the given channels.

        Parameters
        ----------
        channels : ChannelGroup
            The set of channels for which to store the point response of the
            filter.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.get_channels()

        if channels is self.channels:
            channel_indices = np.arange(channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        point_response = self.calc_point_response()
        if isinstance(point_response, np.ndarray):
            self.point_response[channel_indices] = point_response[
                channel_indices]
        else:
            self.point_response[channel_indices] = point_response

    def pre_filter_channels(self, channels=None):
        """
        Performs the pre-filtering channels steps.

        The fixed filter recalculates each channel point response to update the
        channel direct and source filtering by::

            filtering /= point_response

        Parameters
        ----------
        channels : ChannelGroup
            The set of channels to filter.  Defaults to the filter channels.

        Returns
        -------
        None
        """
        super().pre_filter_channels(channels=channels)

        if channels is None:
            channels = self.get_channels()

        self.reset_point_response(channels=channels)

        if self.is_sub_filter:
            return

        if channels is self.channels:
            channel_indices = np.arange(channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        point_response = self.point_response[channel_indices]

        if np.allclose(point_response, 0):
            return

        if channels is self.channels:
            channel_indices = np.arange(channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        direct_filtering = channels.direct_filtering
        source_filtering = channels.source_filtering
        direct_filtering[channel_indices] /= point_response
        source_filtering[channel_indices] /= point_response
        channels.direct_filtering = direct_filtering
        channels.source_filtering = source_filtering

    def post_filter_channels(self, channels=None):
        """
        Performs the post-filtering channels steps.

        Parameters
        ----------
        channels : ChannelGroup
            The set of channels to filter.  Defaults to the filter channels.

        The fixed filter adds the rejected sum to the channel and frame
        dependents, sets the point response and multiplies the direct and
        source filtering by the new calculated value.  This reverses what
        occurred in the pre-filtering step.

        Returns
        -------
        None
        """
        super().post_filter_channels(channels=channels)

        if channels is None:
            channels = self.get_channels()

        if channels is self.channels:
            channel_indices = np.arange(channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        do_bugs = self.integration.configuration.get_bool('crushbugs')
        if not do_bugs:
            self.parms.add_async(channels, self.rejected)

        # This adds to self.frame_parms, which are then applied to
        # self.parms.for_frame during the apply method (Filter).
        self.add_frame_parms(channels=channels)

        self.reset_point_response(channels=channels)

        # Sub filters should not directly change the channel filtering.
        # That is the job of the parent filter based on it's accumulated
        # response.
        point_response = self.get_point_response(channels=channels)
        direct_filtering = self.channels.direct_filtering
        source_filtering = self.channels.source_filtering
        direct_filtering[channel_indices] *= point_response[channel_indices]
        source_filtering[channel_indices] *= point_response[channel_indices]
        self.channels.direct_filtering = direct_filtering
        self.channels.source_filtering = source_filtering

    def add_frame_parms(self, channels=None):
        """
        Add rejection parms to the frame parms.

        Parameters
        ----------
        channels : ChannelGroup
            The set of channels to filter.  Defaults to the filter channels.

        Returns
        -------
        None
        """
        if (self.frame_parms is None
                or self.frame_parms.size != self.integration.size):
            self.frame_parms = np.zeros(self.integration.size, dtype=float)

        if channels is None:
            channels = self.get_channels()

        frames = self.integration.frames

        fnf.add_frame_parms(
            rejected=self.rejected,
            points=self.points,
            weights=frames.relative_weight,
            frame_valid=frames.valid,
            modeling_frame=frames.is_flagged('MODELING_FLAGS'),
            frame_parms=self.frame_parms,
            sample_flags=frames.sample_flag,
            channel_indices=channels.indices)

    @abstractmethod
    def get_id(self):  # pragma: no cover
        """
        Return the filter ID.

        Returns
        -------
        filter_id : str
        """
        pass

    @abstractmethod
    def get_config_name(self):  # pragma: no cover
        """
        Return the configuration name.

        Returns
        -------
        config_name : str
        """
        pass

    @abstractmethod
    def response_at(self, fch):  # pragma: no cover
        """
        Return the response at a given frequency channel(s).

        Parameters
        ----------
        fch : int or numpy.ndarray (int or bool) or slice
            The frequency channel or channels in question.

        Returns
        -------
        response : numpy.ndarray (float)
        """
        pass
