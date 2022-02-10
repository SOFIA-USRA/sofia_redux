# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import numpy as np
import scipy

from sofia_redux.scan.filters.kill_filter import KillFilter
from sofia_redux.scan.flags.motion_flags import MotionFlags
from sofia_redux.scan.utilities.numba_functions import smart_median, pow2ceil
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.filters import filters_numba_functions as fnf

__all__ = ['MotionFilter']


class MotionFilter(KillFilter):

    def __init__(self, integration=None, data=None):
        """
        Initializes a motion filter.

        The motion filter kills specific frequencies based on motion
        statistics.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.critical = 10.0
        self.half_width = 0.0  # For AM noise on >5s
        self.harmonics = 1
        self.odd_harmonics_only = False
        super().__init__(integration=integration, data=data)

    def set_integration(self, integration):
        """
        Set the filter integration.

        Sets the padding of the FFT filter, the number of frequencies, the
        frequency spacing, and retrieves the channels from the integration if
        necessary.

        This is where the integration is examined to extract the relevant
        motion statistics.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        super().set_integration(integration)
        if self.integration.configuration.get_bool('lab'):
            return

        log.debug("Motion filter:")
        if self.has_option('s2n'):
            self.critical = utils.get_float(self.option('s2n'), default=10.0)

        if self.has_option('stability'):
            stability = utils.get_float(self.option('stability'))
            self.half_width = 0.5 / np.abs(stability)  # in Hz

        if self.has_option('harmonics'):
            self.harmonics = utils.get_float(self.option('harmonics'))
            self.harmonics = np.clip(self.harmonics, 1, None)

        self.odd_harmonics_only = utils.get_bool(self.option('odd'))

        positions = self.integration.get_smooth_positions('SCANNING')
        self.add_filter(positions, 'x')
        self.add_filter(positions, 'y')

        self.expand_filter()
        self.harmonize()
        self.range_check()

        n_pass = np.sum(np.logical_not(self.reject))
        log.debug(f"{100 * n_pass / self.reject.size} pass.")

    def range_check(self):
        """
        Applies the filter over a certain frequency range.

        The frequency range is supplied by the 'filter.motion.range'
        configuration option.  All previously killed frequencies outside
        of this range will be allowed to pass.

        Returns
        -------
        None
        """
        if not self.has_option('range'):
            return
        frequency_range = utils.get_range(
            self.option('range'), is_positive=True)

        reject = self.get_reject_mask()
        min_index = int(frequency_range.min / self.df)
        max_index = int(np.ceil(frequency_range.max / self.df))
        min_index = np.clip(min_index, 0, None)
        max_index = np.clip(max_index, None, reject.size)
        self.reject[:min_index] = False
        self.reject[max_index:] = False

    def add_filter(self, positions, direction):
        """
        Filters motion in a given direction.

        The FFT of the motion is calculated, and those frequencies where the
        power spectrum is above a certain threshold are added to a kill
        filter.

        The cutoff threshold is given as:

        threshold_rms = critical * RMS(power spectrum)

        where "critical` is determined from the filter.motion.s2n configuration
        setting.  If filter.motion.above is set, then the threshold is set to:

        max(threshold_rms, max(power_spectrum * above))

        Parameters
        ----------
        positions : Quantity
            An array of positions with shape (n_frames, 2)
        direction : str or int or MotionFlagTypes
            The direction of motion to examine.

        Returns
        -------
        None
        """
        reject = self.get_reject_mask()
        filter_size = pow2ceil(self.integration.size)
        data = np.zeros(filter_size, dtype=float)

        motion = MotionFlags(direction)

        position_signal = motion(positions).value

        data[:positions.size] = position_signal
        data[~np.isfinite(data)] = np.nan

        # Remove any constant scanning offset
        data -= np.nanmean(data)

        # Zero NaNs
        data[np.isnan(data)] = 0.0

        # FFT to get the scanning spectra
        data = scipy.fft.rfft(data)

        # Never
        data.real[0] = 0.0
        reject[0] = True

        critical_level = self.critical * self.get_fft_rms(data)
        signal_power = np.abs(data)
        peak_index = np.argmax(signal_power)
        peak = signal_power[peak_index]

        if self.has_option('above'):
            above = utils.get_float(self.option('above'))
            cutoff = peak * above
            critical_level = np.clip(critical_level, cutoff, None)

        reject[signal_power > critical_level] = True
        dt = self.integration.info.sampling_interval.decompose().value
        df = 1.0 / (dt * data.size)
        peak_frequency = (peak_index / 2) * df * units.Unit('Hz')

        for decade in ['mHz', 'Hz', 'kHz', 'MHz', 'GHz', 'THz']:
            print_frequency = peak_frequency.to(units.Unit(decade)).round(3)
            if 1e1 <= print_frequency.value < 1e4:
                break

        log.debug(f"{motion.direction.name} direction @ {print_frequency}")

    @staticmethod
    def get_fft_rms(fft_signal):
        """
        Return the RMS of an FFT signal.

        Parameters
        ----------
        fft_signal : numpy.ndarray (complex)
            The complex signal.

        Returns
        -------
        rms : float
        """
        variance = np.abs(fft_signal) ** 2
        median_normalized_variance = 0.454937
        rms = np.sqrt(smart_median(variance, max_dependence=1)[0])
        rms /= median_normalized_variance
        return rms

    def expand_filter(self):
        """
        Expand the kill filter based on stability.

        A number of frequency bins on either side of each rejected point will
        be added to the kill filter.  The number of additional bins on either
        side is given as half_width / frequency_spacing.

        Returns
        -------
        None
        """
        # Calculate the HWHM of the AM noise
        fnf.expand_rejection_filter(
            reject=self.get_reject_mask(),
            half_width=self.half_width,
            df=self.df)

    def harmonize(self):
        """
        Kills harmonics of the rejected frequencies.

        The number of harmonics will be determined from the
        'filter.motion.harmonics' configuration setting.

        Returns
        -------
        None
        """
        if self.harmonics < 2:
            return

        fnf.harmonize_rejection_filter(
            reject=self.get_reject_mask(),
            harmonics=self.harmonics,
            odd_harmonics_only=self.odd_harmonics_only)

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
        self.set_channels(self.integration.channels.get_observing_channels())
        super().pre_filter()

    def get_id(self):
        """
        Return the filter ID.

        Returns
        -------
        filter_id : str
        """
        return 'Mf'

    def get_config_name(self):
        """
        Return the configuration name.

        Returns
        -------
        config_name : str
        """
        return 'filter.motion'
