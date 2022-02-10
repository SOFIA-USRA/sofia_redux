# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.scan.filters.adaptive_filter import AdaptiveFilter
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.filters import filters_numba_functions as fnf
from sofia_redux.scan.utilities import utils

__all__ = ['WhiteningFilter']


class WhiteningFilter(AdaptiveFilter):

    def __init__(self, integration=None, data=None):
        """
        Initialize an integration whitening filter.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.level = 1.2
        self.significance = 2.0
        self.windows = -1  # The number of stacked windows

        # The frequency channel indices of the white-level measuring range
        self.white_from = -1
        self.white_to = -1

        self.one_over_f_bin = -1
        self.white_noise_bin = -1

        self.amplitudes = None  # The amplitude at reduced resolution
        self.amplitude_weights = None
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
        attributes.add('amplitudes')
        attributes.add('amplitude_weights')
        return attributes

    def update_config(self):
        """
        Determine whether the filter is configuration and if it's pedantic.

        Loads additional settings for the whitening filter including size
        of the profiles and assigning noise bins.

        Returns
        -------
        None
        """
        super().update_config()
        self.set_size(self.integration.frames_for(
            self.integration.filter_time_scale))

        if self.has_option('level'):
            level = np.sqrt(utils.get_float(self.option('level')))
            self.level = max(1.0, level)

        window_size = numba_functions.pow2ceil(2 * self.nF)
        if self.nt < window_size:
            window_size = self.nt
        self.windows = self.nt // window_size

        if self.has_option('proberange'):
            probe_range_spec = self.option('proberange')
            if probe_range_spec.lower() == 'auto':
                # The frequency cut off (half-max) of the typical
                # point-source response
                f_point = (0.44 / self.integration.get_point_crossing_time()
                           ).to(unit='Hz').value
                probe = Range(min_val=0.2 * f_point, max_val=1.14 * f_point)
            else:
                probe = Range.from_spec(probe_range_spec, is_positive=True)
        else:
            probe = Range(min_val=0.0, max_val=self.nF * self.dF)
        probe.intersect_with(0, self.nF)

        self.white_from = int(max(1, np.floor(probe.min / self.dF)))
        self.white_to = int(min(self.nF, np.ceil(probe.max / self.dF) + 1))

        if self.has_option('1overf.freq'):
            f = utils.get_float(self.option('1overf.freq'))
            self.one_over_f_bin = int(np.clip(int(f / self.dF), 1, self.nF))
        else:
            self.one_over_f_bin = 2

        if self.has_option('1overf.ref'):
            f = utils.get_float(self.option('1overf.ref'))
            self.white_noise_bin = int(np.clip(int(f / self.dF), 1, self.nF))
        else:
            self.white_noise_bin = self.nF // 2

        # Make sure the probing range contains enough channels and that the
        # range is valid
        if self.has_option('minchannels'):
            min_probe_channels = utils.get_int(self.option('minchannels'))
        else:
            min_probe_channels = 16

        max_white = self.white_to - min_probe_channels + 1
        if self.white_from > max_white:
            self.white_from = max_white
        if self.white_from < 0:  # pragma: no cover
            # This should never happen
            self.white_from = 0
        if self.white_from > max_white:  # In case min channels is too large
            self.white_to = min(min_probe_channels + 1, self.nF)

    def set_size(self, n):
        """
        Set the size of the profiles.

        Resampling will occur if the size of the profile changes.

        Parameters
        ----------
        n : float or int
            The new size.

        Returns
        -------
        None
        """
        if int(n) == self.nF:
            return
        super().set_size(n)
        self.nF = int(n)
        shape = self.channels.size, self.nF

        self.amplitudes = np.empty(shape, dtype=float)
        self.amplitude_weights = np.empty(shape, dtype=np.float64)

    def update_profile(self, channels=None):
        """
        Update the profile for given channel(s).

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels to update.  By default, will use all filtering
            channels.

        Returns
        -------
        None
        """
        self.calc_mean_amplitudes(channels=channels)
        self.whiten_profile(channels=channels)

    def calc_mean_amplitudes(self, channels=None):
        """
        Calculates the mean amplitudes in the window of a spectrum.

        The amplitude of the FFT spectrum is calculated as the mean value
        within a given window (usually 1).  The weight of the mean operation
        will also be stored in the `amplitude_weights` attribute.  These are
        used later to calculate the channel profiles.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels to update.  By default, will use all filtering
            channels.

        Returns
        -------
        None
        """
        if channels is None or channels is self.channels:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        if self.channel_profiles is None or self.channel_profiles.size == 0:
            self.channel_profiles = np.ones((self.channels.size, self.nF),
                                            dtype=float)

        self.amplitudes.fill(0.0)
        self.amplitude_weights.fill(0.0)
        # Get coarse average spectrum (FFT is stored in the filter attribute)
        fnf.calc_mean_amplitudes(
            amplitudes=self.amplitudes,
            amplitude_weights=self.amplitude_weights,
            spectrum=self.data,
            windows=self.windows,
            channel_indices=channel_indices)

    def whiten_profile(self, channels=None):
        """
        Create the channel filtering profiles for whitening.

        Will also set channel 1/f noise statistics.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels to update.  By default, will use all filtering
            channels.

        Returns
        -------
        None
        """
        log.debug("Whitening channel profile.")
        if channels is None or channels is self.channels:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = self.channels.find_fixed_indices(
                channels.fixed_index)

        one_over_f_stat = self.channels.one_over_f_stat

        one_over_f_stat[channel_indices] = fnf.whiten_profile(
            amplitudes=self.amplitudes,
            amplitude_weights=self.amplitude_weights,
            profiles=self.profile,
            channel_profiles=self.channel_profiles,
            white_from=self.white_from,
            white_to=self.white_to,
            filter_level=self.level,
            significance=self.significance,
            one_over_f_bin=self.one_over_f_bin,
            white_noise_bin=self.white_noise_bin,
            channel_indices=channel_indices)

        self.channels.one_over_f_stat = one_over_f_stat

    def get_id(self):
        """
        Return the filter ID.

        Returns
        -------
        filter_id : str
        """
        return 'wh'

    def get_config_name(self):
        """
        Return the configuration name.

        Returns
        -------
        config_name : str
        """
        return 'filter.whiten'

    def dft_filter(self, channels=None):
        """
        Return the filter rejection using a discrete FFT.

        UNSUPPORTED FOR THE WHITENING FILTER.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which the filtering applied.  By default,
            set to the filtering channels.

        Returns
        -------
        None
        """
        super().dft_filter(channels=channels)
