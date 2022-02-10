# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
import numpy as np
from astropy.stats import gaussian_sigma_to_fwhm

from sofia_redux.scan.filters.varied_filter import VariedFilter
from sofia_redux.scan.filters import filters_numba_functions as fnf

__all__ = ['AdaptiveFilter']


class AdaptiveFilter(VariedFilter):

    def __init__(self, integration=None, data=None):
        """
        Initialize an integration adaptive filter.

        The adaptive filter is an abstract class where each channel has an
        individual frequency response (see :class:`VariedFilter`).

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.channel_profiles = None
        self.profile = None
        self.nF = 0
        self.dF = np.nan
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
        attributes.add('channel_profiles')
        attributes.add('profile')
        return attributes

    def get_profile(self):
        """
        Return the adaptive filter profile.

        Returns
        -------
        profile : numpy.ndarray (float)
        """
        return self.profile

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
        channels = self.get_channels()
        self.channel_profiles = np.zeros((channels.size, 0), dtype=float)

    def set_size(self, nf):
        """
        Set the number of frequencies in the adaptive filter.

        Parameters
        ----------
        nf : int
            The number of frequencies in the adaptive filter.

        Returns
        -------
        None
        """
        if self.profile is None or self.profile.shape[1] != nf:
            self.profile = np.zeros((self.channels.size, nf), dtype=float)

        channels = self.get_channels()

        dt = self.integration.info.sampling_interval.decompose().value
        self.dF = 0.5 / (nf * dt)
        self.update_source_profile()

        if self.channel_profiles.size != 0:
            old_profile = self.channel_profiles.copy()
            self.channel_profiles = np.zeros((channels.size, nf), dtype=float)
            self.resample(old_profile, self.channel_profiles)

    def resample(self, old_profile, new_profile):
        """
        ResamplePolynomial the old profile to a new profile.

        Parameters
        ----------
        old_profile : numpy.ndarray (float)
            An array of shape (n_channels, n1) containing the current profile.
        new_profile : numpy.ndarray (float)
            The new array of shape (n_channels, n2) to populate.

        Returns
        -------
        None
        """
        fnf.resample(old_profile, new_profile)
        self.channel_profiles = new_profile

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
        self.accumulate_profiles(channels=channels)
        super().post_filter_channels(channels=channels)

    def accumulate_profiles(self, channels=None):
        """
        Accumulates the channel profiles into a single profile.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which to accumulate profiles.

        The accumulated profile is the multiplication of each channel profile.

        Returns
        -------
        None
        """
        if channels is None:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        fnf.accumulate_profiles(
            profiles=self.profile,
            channel_profiles=self.channel_profiles,
            channel_indices=channel_indices)

    def response_at(self, fch):
        """
        Return the response at a given frequency channel(s).

        Parameters
        ----------
        fch : int or numpy.ndarray (int or bool) or slice
            The frequency channel or channels in question.

        Returns
        -------
        response : numpy.ndarray (float)
            The response array of shape (n_channels,) or
            (n_channels, fch.size).
        """
        n_channels = self.channels.size
        if self.profile is None:
            if not isinstance(fch, np.ndarray) or fch.shape == ():
                return np.full(n_channels, 1.0)
            return np.full((n_channels, fch.size), 1.0)

        indices = fch * self.profile.shape[1] // (self.nf + 1)
        return self.profile[:, indices]

    def get_valid_profiles(self, channels=None):
        """
        Return the valid channel profiles.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which to get profiles.  The default is
            all filtering channels.

        Returns
        -------
        channel_profiles : numpy.ndarray (float)
        """
        if channels is None:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        if self.channel_profiles is None:
            if self.profile is not None:
                n_freq = self.profile.shape[1]
                return np.full((self.channels.size, n_freq), 1.0)
            else:
                return None

        return self.channel_profiles[channel_indices]

    def count_parms(self, channels=None):
        """
        Return the rejection filter sum above the high pass frequency.

        channels : ChannelGroup, optional
            The channel group for which to determine dependents.  The
            default is all filtering channels.

        Returns
        -------
        dependents : numpy.ndarray (float)
            An array of shape (n_channels,).
        """
        if channels is None:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)
        n_channels = channel_indices.size

        if self.profile is None:
            return np.zeros(n_channels, dtype=float)

        dt = self.integration.filter_time_scale.decompose().value
        high_pass_freq = 0.5 / dt
        min_f = int(np.ceil(high_pass_freq / self.dF))
        p = self.profile[channel_indices]
        parms = np.sum(1.0 - (p[:, min_f:] ** 2), axis=1)
        return parms

    def update_source_profile(self):
        """
        Update the filter source profile.

        Returns
        -------
        None
        """
        if self.profile is None:
            return

        if self.source_profile is not None:
            if self.source_profile.size == self.profile.shape[1]:
                return

        nf = self.profile.shape[1]
        t = self.integration.get_point_crossing_time().decompose().value
        sigma = gaussian_sigma_to_fwhm / (2 * np.pi * t * self.dF)
        a = -0.5 / (sigma ** 2)
        f = np.arange(nf)
        self.source_profile = np.exp(a * (f ** 2))
        self.source_norm = float(np.sum(self.source_profile))

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
        # Start from the 1/f filter cutoff
        dt = self.integration.filter_time_scale.decompose().value
        high_pass_freq = 0.5 / dt
        min_fch = int(np.ceil(high_pass_freq / self.dF))

        if channels is None or channels is self.channels:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        return fnf.calculate_channel_point_responses(
            min_fch=min_fch,
            source_profile=self.source_profile,
            profiles=self.profile,
            channel_indices=channel_indices,
            source_norm=self.source_norm)

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
