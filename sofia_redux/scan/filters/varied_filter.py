# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy.stats import gaussian_sigma_to_fwhm
import numpy as np

from sofia_redux.scan.filters.filter import Filter
from sofia_redux.scan.filters import filters_numba_functions as fnf

__all__ = ['VariedFilter']


class VariedFilter(Filter):

    def __init__(self, integration=None, data=None):
        """
        Initialize an integration varied filter.

        The filter is designed to filter integration data using an FFT.  The
        varied filter also contains a source profile, point response, and
        `dp` or delta dependents for channels.  This is an abstract class
        used to model a varying filter response across frequencies.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.source_profile = None
        self.point_response = None
        self.dp = None
        self.source_norm = 0.0
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
        attributes.add('dp')
        attributes.add('point_response')
        return attributes

    def get_source_profile(self):
        """
        Return the source profile of the varied filter.

        Returns
        -------
        source_profile : numpy.ndarray (float)
            The source profile of shape (nf + 1) where nf = nt // 2 and
            nt = pow2ceil(integration.size).
        """
        return self.source_profile

    def set_integration(self, integration):
        """
        Set the varied filter integration.

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
        self.point_response = np.ones(self.get_channels().size,
                                      dtype=float)
        self.dp = np.zeros(self.get_channels().size, dtype=float)
        self.update_source_profile()

    def pre_filter_channels(self, channels=None):
        """
        Performs the pre-filtering channels steps.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which to perform the pre-filtering step.
            If not supplied, defaults to the filtering channels.

        Returns
        -------
        None
        """
        if not self.is_sub_filter:
            if channels is None or channels is self.channels:
                channels = self.get_channels()
                channel_indices = np.arange(self.channels.size)
            else:
                channel_indices = self.channels.find_fixed_indices(
                    channels.fixed_index)

            response = self.point_response[channel_indices]
            apply = response > 0
            if apply.any():
                factor = response[apply]
                direct_filtering = channels.direct_filtering
                source_filtering = channels.source_filtering
                direct_filtering[apply] /= factor
                source_filtering[apply] /= factor
                channels.direct_filtering = direct_filtering
                channels.source_filtering = source_filtering

        super().pre_filter_channels(channels=channels)

    def post_filter_channels(self, channels=None):
        """
        Performs the post-filtering channels steps.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which to perform the post-filtering step.
            If not supplied, defaults to the filtering channels.

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

        rejected = self.count_parms()

        if isinstance(rejected, np.ndarray) and rejected.shape != ():
            rejected_array = True
            rejected = rejected[channel_indices]
            self.parms.add_async(channels, rejected[channel_indices])
        else:
            rejected_array = False

        self.parms.add_async(channels, rejected)
        self.dp = np.zeros(self.channels.size, dtype=float)

        if isinstance(self.points, np.ndarray):
            nzi = self.points > 0
            if rejected_array:
                self.dp[nzi] = rejected[nzi] / self.points[nzi]
            else:
                self.dp[nzi] = rejected / self.points[nzi]
            self.dp[~nzi] = 0.0
        elif self.points > 0:
            self.dp[channels.indices] = rejected / self.points
        else:
            self.dp[channels.indices] = 0.0

        response = self.calc_point_response()[channel_indices]
        self.point_response[channel_indices] = response

        # Do not directly update the channel filtering if a sub-filter since
        # this is the job of the parent filter.
        if not self.is_sub_filter:
            direct_filtering = channels.direct_filtering
            source_filtering = channels.source_filtering
            direct_filtering *= response
            source_filtering *= response
            channels.direct_filtering = direct_filtering
            channels.source_filtering = source_filtering

    def remove_from_frames(self, rejected_signal, frames, channels):
        """
        Remove the rejected signal from frame data.

        Parameters
        ----------
        rejected_signal : numpy.ndarray (float)
            The rejected signal of shape (filtering_channels.size, n_frames).
        frames : Frames
            The frames for which to remove the rejected signal.
        channels : ChannelGroup
            The channel for which to subtract the signal.

        Returns
        -------
        None
        """
        super().remove_from_frames(rejected_signal, frames, channels)

        fnf.apply_rejection_to_parms(
            frame_valid=frames.valid,
            frame_weight=frames.relative_weight,
            frame_parms=self.frame_parms,
            dp=self.dp,
            channel_indices=channels.indices,
            sample_flag=frames.sample_flag)

    def dft_filter(self, channels=None):
        """
        Return the filter rejection using a discrete FFT.

        UNSUPPORTED FOR THE VARIED FILTER.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which the filtering applied.  By default,
            set to the filtering channels.

        Returns
        -------
        None
        """
        raise NotImplementedError("No DFT for varied filters.")

    def get_point_response(self, channels=None):
        """
        Return the point response for the filter.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels for which to extract the point response.  The default
            is the filtering channels.

        Returns
        -------
        point_response : numpy.ndarray (float)
            The point response for the given channels.
        """
        if channels is None or channels is self.channels:
            return self.point_response
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)
            return self.point_response[channel_indices]

    def get_mean_point_response(self):
        """
        Return the mean point response of the filter.

        Returns
        -------
        response : float
        """
        channels = self.get_channels()
        unflagged = channels.is_unflagged()
        responses = self.get_point_response()[unflagged]
        weights = channels.weight[unflagged]

        g = weights * responses
        g2 = g * responses
        return np.sum(g2) / np.sum(g)

    def update_source_profile(self):
        """
        Create the source profile based on integration crossing time.

        The source is assumed to be Gaussian.  Updates the `source_profile`
        and `source_norm` attributes.

        Returns
        -------
        None
        """
        t = self.integration.get_point_crossing_time().decompose().value
        sigma = gaussian_sigma_to_fwhm / (2 * np.pi * t * self.df)
        a = -0.5 / (sigma ** 2)
        fch = np.arange(self.nf + 1)
        self.source_profile = np.exp(a * (fch ** 2))
        self.source_norm = np.sum(self.source_profile)

    def calc_point_response(self, channels=None):
        """
        Return the point response of the source profile.

        channels : ChannelGroup, optional
            The channel group for which to calculate the point response.  The
            default is all filtering channels.

        The point response is given as:

        sum(profile <below hi-pass>) + sum(profile * response <above hi-pass>)
        divided by the `source_norm` attribute.

        Returns
        -------
        response : numpy.ndarray (float)
            The point response for each channel of shape (n_channels,).
        """
        if channels is None:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        profiles = self.response_at(np.arange(self.nf + 1))
        if profiles.ndim == 1:
            profiles = profiles[None]

        result = fnf.calculate_channel_point_responses(
            min_fch=self.get_high_pass_index(),
            source_profile=self.source_profile,
            profiles=profiles,
            channel_indices=channel_indices,
            source_norm=self.source_norm)

        return result

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
