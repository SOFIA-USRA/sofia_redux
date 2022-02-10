# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod
from astropy.stats import gaussian_sigma_to_fwhm
from copy import deepcopy
import numpy as np
import scipy

from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.channels.channels import Channels
from sofia_redux.scan.channels.channel_data.channel_data import ChannelData
from sofia_redux.scan.channels.channel_group.channel_group import ChannelGroup
from sofia_redux.scan.filters import filters_numba_functions as fnf

__all__ = ['Filter']


class Filter(ABC):

    def __init__(self, integration=None, data=None):
        """
        Initialize an integration filter.

        The filter is designed to filter integration data using an FFT.
        This is an abstract class designed to hold a few main parameters
        necessary for filtering.  Each filter requires an integration on
        which to operate, and defines a response at any given frequency.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest
            power of 2 integer above the number of integration frames. i.e.,
            if n_frames=5, nt=8, or if n_frames=13, nt=16.
        """
        self.integration = None
        self.channels = None
        self.parms = None  # The integration filter dependents
        self.frame_parms = None  # Frame dependents
        self.data = None  # Temporary or real filter data workspace
        self.points = None  # frame relative weight sums for each channel
        self.is_sub_filter = False  # Whether this is a sub-filter of main

        self.nt = 0   # pow2ceil of integration.size
        self.nf = 0   # The Nyquist frequency index
        self.df = 0.0  # The frequency spacing

        self.dft = False   # Whether to use the discrete Fourier transform
        self.pedantic = False  # If True, level filter data after application
        self.enabled = False   # Whether the filter is enabled in the config

        if data is not None:
            self.data = data

        if integration is not None:
            self.set_integration(integration)

    def copy(self):
        """
        Return a copy of the filter.

        All attributes are copied aside from the integration and channels which
        are referenced only.

        Returns
        -------
        Filter
        """
        new = self.__class__(integration=None, data=None)
        for attribute, value in self.__dict__.items():
            if attribute in self.referenced_attributes:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))
        return new

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced during the copy operation.

        Returns
        -------
        set (str)
        """
        return {'integration', 'channels'}

    @property
    def channel_dependent_attributes(self):
        """
        Return attributes that are dependent on the parent channels.

        This is required during a slim operation (reducing number of channels)
        which may happen to the filter integration.

        Returns
        -------
        set (str)
        """
        return {'points'}

    @property
    def size(self):
        """
        Return the number of integration frames.

        Returns
        -------
        n_frames : int
        """
        if self.frames is None:
            return 0
        return self.frames.size

    @property
    def channel_size(self):
        """
        Return the number of channels in the filter.

        Returns
        -------
        n_channels : int
        """
        if self.channels is None:
            return 0
        return self.channels.size

    @property
    def frames(self):
        """
        Return the integration frames.

        Returns
        -------
        Frames
        """
        if self.integration is None:
            return None
        return self.integration.frames

    @property
    def info(self):
        """
        Return the integration information.

        Returns
        -------
        Info
        """
        if self.integration is None:
            return None
        return self.integration.info

    @property
    def configuration(self):
        """
        Return the configuration (taken from the integration).

        Returns
        -------
        Configuration
        """
        if self.integration is None:
            return None
        return self.integration.configuration

    @property
    def flagspace(self):
        """
        Return the frame flagspace.

        Returns
        -------
        Flags
        """
        if self.integration is None:
            return None
        return self.integration.frames.flagspace

    @property
    def channel_flagspace(self):
        """
        Return the channel flagspace.

        Returns
        -------
        Flags
        """
        if self.channels is None:
            return None
        return self.channels.flagspace

    @property
    def valid_filtering_frames(self):
        """
        Return frames that are valid and not flagged as modeling.

        Returns
        -------
        valid_frames : numpy.ndarray (bool)
            An array of shape (n_frames,) where `True` marks a valid frame to
            be included in filtering operations.
        """
        if self.frames is None:
            return np.empty(0, dtype=bool)
        return self.frames.is_unflagged(
            self.flagspace.flags.MODELING_FLAGS) & self.frames.valid

    def reindex(self):
        """Reindex the channel groups to be consistent with parent channels.

        When the channels in the filter integration change for some reason
        (such as a slim operation in which bad channels are removed), filter
        attributes that map to integration channels will also need to be
        updated to account for such a change.

        At a base level, the filter attributes that need to be updated include
        channel dependents, and the `points` (sum of frame weights) for
        each channel.  The `data` attribute is also set to `None`, since it
        is designed to be a temporary workspace that may or may not depend on
        the integration channels depending on the type of filter in question.
        """
        if self.channels is None:
            return

        self.data = None

        keep_indices = self.channels.new_indices_in_old()
        self.channels.reindex()

        if self.parms is not None:
            self.parms = self.integration.get_dependents(
                self.get_config_name())

        channel_attributes = self.channel_dependent_attributes

        for attribute, value in self.__dict__.items():
            if attribute not in channel_attributes:
                continue
            if not isinstance(value, np.ndarray):
                continue
            setattr(self, attribute, value[keep_indices])

    def has_option(self, key):
        """
        Return whether an option is set in the configuration for this filter.
        The actual key checked for in the configuration will be:

           <filter_config_name>.<key>

        I.e., the "level" key for a whitening filter will be checked for in the
        configuration as  "filter.whiten.level".

        Parameters
        ----------
        key : str
            The name of the option for the filter.

        Returns
        -------
        is_configured : bool
        """
        if self.integration is None:
            return False
        return self.integration.has_option(f'{self.get_config_name()}.{key}')

    def option(self, key):
        """
        Return the value for a given filter option.

        The given `key` relates specifically to this filter in the
        configuration.  As such, the actual configuration will search for the
        configuration value associated with <filter_config_name>.<key>.  I.e.,
        the "level" key for a whitening filter will be checked for in the
        configuration as "filter.whiten.level".

        Parameters
        ----------
        key : str
            The name of the option for the filter.

        Returns
        -------
        str
        """
        if self.integration is None:
            return None
        return self.configuration.get(f'{self.get_config_name()}.{key}')

    def make_temp_data(self, channels=None):
        """
        Create the initial temporary data.

        Creates a temporary data array of shape (n_channels, n_frames).

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels for which to create the temporary data.  If not
            supplied, defaults to the filtering channels.

        Returns
        -------
        None
        """
        if self.data is not None:
            self.data = None
            self.points = None
        if channels is None:
            channels = self.get_channels()

        filter_size = numba_functions.pow2ceil(self.integration.size)
        n_channels = channels.size
        self.data = np.zeros((n_channels, filter_size), dtype=float)
        self.points = np.zeros(n_channels, dtype=float)

    def discard_temp_data(self):
        """
        Destroy the temporary data.

        Returns
        -------
        None
        """
        self.data = None
        self.points = None

    def is_enabled(self):
        """
        Return whether the filter is enabled.

        Returns
        -------
        bool
        """
        return self.enabled

    def get_temp_data(self):
        """
        Return the temporary data.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self.data

    def set_temp_data(self, data):
        """
        Set the temporary data.

        Parameters
        ----------
        data : numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.data = data

    def rejection_at(self, fch):
        """
        Return the filter rejection at given frequency channel(s).

        The filter rejection is equal to 1 minus the filter response:

            rejection = 1 - response

        Parameters
        ----------
        fch : int or numpy.ndarray (int or bool) or slice.

        Returns
        -------
        response : float or numpy.ndarray (float)
        """
        return 1.0 - self.response_at(fch)

    def count_parms(self):
        """
        Return the total dependent count of the filter.

        The number of filter dependents is calculated as the sum of the
        filter rejection (varies from 0 to 1) between the high-pass and
        Nyquist frequency (inclusive).

        Returns
        -------
        rejected : float or numpy.ndarray (float)
            The sum of the rejected signal above the high pass frequency.  If
            this is a filter that varies by channel, an array will be returned
            of shape (n_channels,).  Otherwise, a singular float value will be
            returned.
        """
        min_freq = self.get_high_pass_index()
        rejection = self.rejection_at(np.arange(min_freq, self.nf))
        if rejection.ndim < 2:
            return np.sum(rejection)
        else:
            return np.sum(rejection, axis=1)

    def get_channels(self):
        """
        Return the associated channel group for the integration filter.

        Returns
        -------
        ChannelGroup
        """
        return self.channels

    def set_channels(self, channels):
        """
        Set the filter channels.

        The channels attribute will be set to a ChannelGroup type.

        Parameters
        ----------
        channels : Channels or ChannelData or ChannelGroup

        Returns
        -------
        None
        """
        if channels is None:
            self.channels = None
            return
        if isinstance(channels, Channels):
            self.channels = channels.create_channel_group(name='all')
        elif isinstance(channels, ChannelData):
            if not isinstance(channels, ChannelGroup):
                group_class = self.info.get_channel_group_class()
                self.channels = group_class(
                    channels, indices=np.arange(channels.size), name='all')
            else:
                self.channels = channels.copy()
        else:
            raise ValueError(f"Channels must be {Channels} or {ChannelData}, "
                             f"not {channels}.")

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
        self.integration = integration
        self.nt = numba_functions.pow2ceil(integration.size)
        self.nf = self.nt // 2

        dt = self.info.instrument.sampling_interval.decompose().value
        self.df = 1.0 / (dt * self.nt)
        self.set_channels(integration.channels)

    def update_config(self):
        """
        Update the filter based on configuration settings.

        Returns
        -------
        None
        """
        if self.integration is None:
            return
        self.enabled = self.integration.has_option(self.get_config_name())
        self.pedantic = self.integration.configuration.get_bool(
            'filter.mrproper')

    def update_profile(self, channels=None):  # pragma: no cover
        """
        Update the profile for given channel(s).

        Parameters
        ----------
        channels : int or numpy.ndarray (int), optional
            The channel indices to update.

        Returns
        -------
        None
        """
        pass

    def apply(self, report=True):
        """
        Apply the filter to the integration.

        There are a few steps that are performed during filter application:

          1. Read the configuration to update any settings.
             - Stop if the filter is not enabled in the configuration.
          2. Pre-filter the channels.  This generally involved removing the
             filter frame and channel dependents form the integration, before
             zeroing them in the filter in order to be re-calculated.
          3. Apply the filter to the channels.
          4. Post-filter the channels.  This generally involves adding the
             filter frame and channel dependents back to the integration.

        Parameters
        ----------
        report : bool, optional
            If `True`, add messages to the integration comments on the
            filtering.

        Returns
        -------
        applied : bool
            `True` if filtering occurred, and `False` otherwise.
        """
        self.update_config()

        if not self.is_enabled():
            return False

        self.integration.comments.append(self.get_id())
        self.pre_filter()

        self.frame_parms = np.zeros(self.integration.size, dtype=float)

        self.apply_to_channels(channels=self.get_channels())

        self.parms.add_for_frames(self.frame_parms)

        self.post_filter()

        if report:
            self.report()

        return True

    def apply_to_channels(self, channels=None):
        """
        Apply the filter the integration for the channel group.

        Parameters
        ----------
        channels : ChannelGroup, optional
            If not supplied, defaults to the filter channels.

        Returns
        -------
        None
        """
        local_temp = self.data is None
        if channels is None:
            channels = self.get_channels()
        if local_temp:
            self.make_temp_data(channels=channels)
        elif channels.size != self.data.shape[0]:
            self.discard_temp_data()
            self.make_temp_data(channels=channels)

        self.load_time_streams(channels=channels)
        self.pre_filter_channels(channels=channels)

        if self.dft:
            self.dft_filter(channels=channels)
        else:
            self.fft_filter(channels=channels)

        if self.pedantic:
            self.level_data_for_channels(channels=channels)

        self.post_filter_channels(channels=channels)

        self.remove(channels=channels)

        if local_temp:
            self.discard_temp_data()

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
        if self.parms is None:
            self.parms = self.integration.get_dependents(
                self.get_config_name())

        # Sub filters should not directly change the dependents in the
        # integration since this is the job of the parent filter.
        if not self.is_sub_filter:
            self.parms.clear(self.get_channels(),
                             start=0,
                             end=self.integration.size)
        else:
            self.parms.for_frame.fill(0.0)
            self.parms.for_channel.fill(0.0)

    def post_filter(self):
        """
        Perform the post-filtering steps.

        During the post-filtering steps, dependents are added to the
        integration and integration channels.

        Returns
        -------
        None
        """
        # Sub filters should not directly change the dependents in the
        # integration since this is the job of the parent filter.
        if not self.is_sub_filter:
            self.parms.apply(self.get_channels(),
                             start=0,
                             end=self.integration.size)

    def pre_filter_channels(self, channels=None):  # pragma: no cover
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
        pass

    def post_filter_channels(self, channels=None):  # pragma: no cover
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
        # Remove the DC component...
        # level_data_for(channels)
        pass

    def remove(self, channels=None):
        """
        Subtract the filtered signal from the integration frame data.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which to remove the signal.  If not supplied,
            defaults to the filtering channels.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.get_channels()
        self.remove_from_frames(
            self.data, self.integration.frames, channels)

    @staticmethod
    def remove_from_frames(rejected_signal, frames, channels):
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
        fnf.remove_rejection_from_frames(
            frame_data=frames.data,
            frame_valid=frames.valid,
            channel_indices=channels.indices,
            rejected_signal=rejected_signal)

    def report(self):
        """
        Add messages to the integration comments regarding the filtering.

        Returns
        -------
        None
        """
        if self.integration.channels.n_mapping_channels > 0:
            msg = f'({self.get_mean_point_response()})'
        else:
            msg = '(---)'
        self.integration.comments.append(msg)

    def load_time_streams(self, channels=None):
        """
        Load time streams from integration frame data.

        Timestream data is defined as the frame data multiplied by their
        relative weights with the mean subtracted.  Output data are stored
        in the `data` temporary array.  Invalid data are set to zero.  The
        `points` array is also populated and contains the sum of frame weights
        for each channel.  Invalid points are set to zero.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels for which to load time streams.  If not supplied,
            defaults to the filtering channels.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.get_channels()

        if self.data is None:
            self.make_temp_data(channels=channels)
        elif self.data.shape[0] != channels.size:
            self.discard_temp_data()
            self.make_temp_data(channels=channels)

        frames = self.integration.frames
        fnf.load_timestream(
            frame_data=frames.data,
            frame_weights=frames.relative_weight,
            frame_valid=frames.valid,
            modeling_frames=frames.is_flagged('MODELING_FLAGS'),
            channel_indices=channels.indices,
            sample_flags=frames.sample_flag,
            timestream=self.data,
            points=self.points)

    def fft_filter(self, channels=None):
        """
        Apply the FFT filter to the temporary data.

        Converts data into a rejected (un-levelled) signal

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels for which to apply the filter.  If not supplied,
            defaults to the stored filtering channels.

        Returns
        -------
        None
        """
        freq_channels = np.arange(self.nf + 1)
        rejection = self.rejection_at(freq_channels)
        if not rejection.any():
            return

        if channels is None:
            channels = self.get_channels()

        # Pad with zeros as necessary.
        zero_fill = False
        if self.data.shape[1] > self.integration.size:
            zero_fill = True
            self.data[:, self.integration.size:] = 0.0

        self.data = scipy.fft.rfft(self.data, axis=1)
        self.update_profile(channels=channels)

        self.data[:, 0].real = 0.0
        if rejection.ndim == 1:
            self.data[:, 0].imag *= rejection[self.nf]
            self.data[:, 1:] *= rejection[1:]
        else:
            self.data[:, 0].imag *= rejection[:, self.nf]
            self.data[:, 1:] *= rejection[:, 1:]

        self.data = scipy.fft.irfft(self.data, axis=1)
        if zero_fill:
            self.data[:, self.integration.size:] = 0.0

    def dft_filter(self, channels=None):
        """
        Return the filter rejection using a discrete FFT.

        The speed of the FFT depends significantly on the number of non-zero
        rejection values.  This will be faster that the `fft_filter` method
        under certain circumstances.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which the filtering applied.  By default,
            set to the filtering channels.

        Returns
        -------
        None
        """
        freq_channels = np.arange(self.nf + 1)
        rejection = self.rejection_at(freq_channels)
        if not rejection.any():
            return

        fnf.dft_filter_channels(
            frame_data=self.data,
            rejection=rejection,
            n_frames=self.integration.size)

    def calc_point_response(self):
        """
        Returns the point response of a Gaussian source for the filter.

        It is assumed a Gaussian source profile is crossed during the
        integration crossing time such that

        sigma(t) = 1
        t/2.35 * 2pi * sigma(f) = 1
        sigma(f) = 2.35 / (2pi * t)
        df = 1 / (n dt)
        sigma(F) = sigma(f) / df = 2.35 * n * dt / (2pi * t)

        The filter response is not accounted for below the high-pass
        timescale.

        Returns
        -------
        point_response : float
        """
        t = self.integration.get_point_crossing_time()
        sigma = gaussian_sigma_to_fwhm / (2 * np.pi * t
                                          * self.df).decompose().value
        a = -0.5 / (sigma ** 2)
        f0 = self.integration.get_modulation_frequency(
            'TOTAL_POWER').decompose().value / self.df  # Usually the chopper

        # Calculate the x=0 component -- O(N)
        # Below the hi-pass time-scale, the filter has no effect, so count it
        # as such
        min_fi = self.get_high_pass_index()
        frequencies = np.arange(self.nf + 1)
        response = self.response_at(frequencies).copy()
        response[:min_fi] = 1.0

        # Calculate the true source filtering above the hi-pass timescale
        # Consider it a symmetric source profile, peaked at the origin --
        # It's peak is simply the sum of the cosine terms, which are the real
        # part of the amplitudes.  So, the filter correction is simply the
        # ratio of the sum of the filtered real amplitudes relative to the sum
        # of the original real amplitudes.
        source_response = np.exp(a * ((frequencies - f0) ** 2))
        source_response += np.exp(a * ((frequencies + f0) ** 2))
        return np.sum(source_response * response) / np.sum(source_response)

    def get_high_pass_index(self):
        """
        Return the high pass filter frequency index.

        Returns
        -------
        index : int
        """
        hi_pass_f = (0.5 / self.integration.filter_time_scale).to(
            unit='Hz').value
        if np.isnan(hi_pass_f) or hi_pass_f < 0:
            return 1
        return int(np.ceil(hi_pass_f / self.df))

    def level_data_for_channels(self, channels=None):
        """
        Level (remove average) from the filter data.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels for which to average the data, by default

        Returns
        -------
        None
        """
        self.level_for_channels(self.data, channels=channels)

    def level_for_channels(self, signal, channels=None):
        """
        Level (remove average) from a given signal.

        Parameters
        ----------
        signal : numpy.ndarray (float)
            The data to level of shape (channels.size, n_frames).
        channels : ChannelGroup, optional
            The channel group for which the data applied.  By default set
            to the filtering channels.

        Returns
        -------
        None
        """
        if channels is None or channels is self.channels:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        frames = self.integration.frames
        fnf.level_for_channels(
            signal=signal,
            valid_frame=frames.valid,
            modeling_frame=frames.is_flagged('MODELING_FLAGS'),
            sample_flag=frames.sample_flag,
            channel_indices=channel_indices)

    def level_data(self):
        """
        Level (remove average from) the filter data.

        Returns
        -------
        None
        """
        self.level(self.data)

    def level(self, signal):
        """
        Level the given signal data.

        Parameters
        ----------
        signal : numpy.ndarray (float)
            The signal data to level.  Leveling occurs for each channel over
            all frames.  The signal shape should be (n_channels, n_frames) or
            (n_frames,).

        Returns
        -------
        None
        """
        if signal.ndim == 1:
            fnf.level_1d(data=signal, n_frames=self.integration.size)
        if signal.ndim == 2:
            fnf.level(data=signal, n_frames=self.integration.size)

    def set_dft(self, value):
        """
        Set the DFT flag that determines whether to use FFT or DFT filtering.

        Parameters
        ----------
        value : bool
            If `True`, sets the filtering method to DFT.

        Returns
        -------
        None
        """
        self.dft = value

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
        response : float or numpy.ndarray (float)
        """
        pass

    @abstractmethod
    def get_mean_point_response(self):  # pragma: no cover
        """
        Return the mean point response of the filter.

        Returns
        -------
        response : float
        """
        pass
