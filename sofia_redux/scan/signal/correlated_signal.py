# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.signal.signal import Signal
from sofia_redux.scan.integration.dependents.dependents import Dependents
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.signal import signal_numba_functions as snf

__all__ = ['CorrelatedSignal']


class CorrelatedSignal(Signal):

    def __init__(self, integration, mode):
        super().__init__(integration, mode=mode)
        self.dependents = Dependents(integration, mode.name)
        self.debug = False
        self.source_filtering = None  # per channel
        self.generation = 0

        self.resolution = mode.get_frame_resolution(integration)
        self.value = np.zeros(mode.signal_length(integration), dtype=float)
        self.weight = np.zeros(self.value.size, dtype=float)
        self.drift_n = self.value.size

    def copy(self):
        """
        Return a copy of the signal.

        Returns
        -------
        CorrelatedSignal
        """
        return super().copy()

    def weight_at(self, frame):
        """
        Return the weight at a given frame index.

        Parameters
        ----------
        frame : int

        Returns
        -------
        weight : float
        """
        return self.weight[frame // self.resolution]

    def get_variance(self):
        """
        Return the signal variance.

        The signal variance is returned as:

        v = sum(w * x^2) / sum(w)

        where x are the signal values and w are the signal weights.

        Returns
        -------
        variance : float
        """
        return snf.get_signal_variance(values=self.value, weights=self.weight)

    def level(self, start_frame=None, end_frame=None, robust=False):
        """
        Remove the mean value from the signal values.

        Parameters
        ----------
        start_frame : int, optional
            The starting frame from which to level.  The default is the first
            frame (0).
        end_frame : int, optional
            The last frame from to level (non-inclusive).  The default is the
            total number of frames.
        robust : bool, optional
            If `True`, subtract the median value rather than the mean.

        Returns
        -------
        average : float
            The average signal value that was removed.
        """
        if start_frame is None and end_frame is None:
            center = self.get_median() if robust else self.get_mean()
            self.value -= center
            return center

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.integration.size

        start_signal = start_frame // self.resolution
        end_signal = numba_functions.roundup_ratio(end_frame, self.resolution)
        select = slice(start_signal, end_signal)
        x = self.value[select]
        w = self.weight[select]
        if robust:
            center, _ = numba_functions.smart_median_1d(values=x, weights=w)
        else:
            center, _ = numba_functions.mean(values=x, weights=w)
        self.value[select] -= center
        return center

    def get_median(self):
        """
        Return the median signal value and weight.

        Returns
        -------
        median, median_weight : float, float
        """
        return numba_functions.smart_median_1d(
            values=self.value, weights=self.weight, max_dependence=1.0)

    def get_mean(self):
        """
        Return the mean signal value and weight.

        Returns
        -------
        mean, mean_weight
        """
        return numba_functions.mean(values=self.value, weights=self.weight)

    def differentiate(self):
        """
        Differentiate the signal values and weights in-place.

        Note that the spacing between sample values is assumed to be in
        seconds when calculating the gradient.

        Returns
        -------
        None
        """
        s = self.info.instrument.sampling_interval.decompose().value
        snf.differentiate_weighted_signal(
            values=self.value,
            weights=self.weight,
            dt=self.resolution * s)

    def integrate(self):
        """
        Integrate the signal values and weights in-place.

        Note that the spacing between sample values is assumed to be in
        seconds.  The `is_floating` attribute is set to `True` following this
        operation indicating that the integrated signal has an arbitrary
        DC offset.

        Returns
        -------
        None
        """
        s = self.info.instrument.sampling_interval.decompose().value
        snf.integrate_weighted_signal(
            values=self.value,
            weights=self.weight,
            dt=s * self.resolution)
        self.is_floating = True

    def get_differential(self):
        """
        Return a differentiated copy of the signal.

        Returns
        -------
        CorrelatedSignal
        """
        return super().get_differential()

    def get_integral(self):
        """
        Return an integrated copy of the signal.

        Returns
        -------
        CorrelatedSignal
        """
        return super().get_integral()

    def get_parms(self):
        """
        Return the degrees of freedom for the signal.

        The degrees of freedom are given as

        DOF = number_of(weights > 0) * (1 - 1/drift_n)

        Returns
        -------
        float
        """
        n = np.sum(self.weight > 0)
        return n * (1.0 - (1.0 / self.drift_n))

    def update(self, robust=False):
        """
        Update the frame data by the signal.

        The gain deltas are derived and subtracted from the frame data.
        Dependents are updated and new source filtering are derived.

        Parameters
        ----------
        robust : bool, optional
            If `True` use the robust method (median) to derive means.

        Returns
        -------
        None
        """
        # Get correlated for all frames, even those that are no good, but
        # only use channels that are valid, and skip over flagged samples
        channel_group = self.mode.channel_group
        good_channels = self.mode.get_valid_channels()
        frames = self.integration.frames
        resolution = self.mode.get_frame_resolution(self.integration)

        gains = self.mode.get_gains()
        self.sync_gains = delta_gains = gains - self.sync_gains
        resync_gains = (delta_gains != 0).any() & (self.value != 0).any()

        channel_group.temp = np.zeros(channel_group.size, dtype=float)
        channel_group.temp_g = gains.copy()
        channel_group.temp_wg = channel_group.weight * channel_group.temp_g
        channel_group.temp_wg2 = channel_group.temp_wg * channel_group.temp_g

        # Remove channels with zero gain/weight from good channels
        # Pre-calculate the channel dependents
        good_channels.delete_indices(good_channels.temp_wg2 == 0)

        # Correct for lowered degrees of freedom due to prior filtering
        good_channels.temp = (good_channels.temp_wg2
                              * good_channels.get_filtering(self.integration))

        # Clear the dependents in all mode channels
        self.dependents.clear(channel_group, start=0,
                              end=self.integration.size)

        # Resync gains if necessary
        if resync_gains:
            snf.resync_gains(
                frame_data=frames.data,
                signal_values=self.value,
                resolution=resolution,
                delta_gains=delta_gains,
                channel_indices=channel_group.indices,
                frame_valid=frames.valid)

        modeling_frames = frames.is_flagged('MODELING_FLAGS')

        # Calculate the gain increments and weights
        if robust:
            dc, dc_weight = self.get_robust_correlated(
                good_channels, modeling_frames=modeling_frames)
        else:
            dc, dc_weight = self.get_ml_correlated(
                good_channels, modeling_frames=modeling_frames)

        snf.apply_gain_increments(
            frame_data=frames.data,  # updated by -= channel_gain * dc
            frame_weight=frames.relative_weight,
            frame_valid=frames.valid,
            modeling_frames=modeling_frames,
            frame_dependents=self.dependents.for_frame,  # updated
            channel_g=channel_group.temp_g,
            channel_fwg2=channel_group.temp,
            channel_indices=channel_group.indices,
            channel_dependents=self.dependents.for_channel,  # updated
            sample_flags=frames.sample_flag,
            signal_values=self.value,  # updated by += dc
            signal_weights=self.weight,  # updated to dc_weight
            resolution=resolution,
            increment=dc,
            increment_weight=dc_weight)

        # Apply the mode dependents only to the channels that have contributed
        self.dependents.apply(channels=good_channels, start=0, end=frames.size)

        # Update the gain values used for signal extraction.
        self.set_sync_gains(gains)
        self.generation += 1

        # Calculate the point-source_filtering by decorrelation.
        self.calc_filtering()

    def get_ml_correlated(self, channel_group, modeling_frames=None):
        """
        Get the maximum-likelihood correlated gain increment and weight.

        The gain increments are given as:

        dC_s = sum_{f | s}{w_f * sum_{c} {w_c g_c d_{f,c}} / dW_s

        where dW_s is the gain increment weight given by

        dW_s = sum_{f | s}{w_f} sum_{c} {w_c g_c^2}

        Here {f | s} indicate the frames in each signal block, w_f is the frame
        weight, w_c is the channel_weight, g_c is the channel gain, and
        d_{f, c} is the frame data value for frame f and channel c.

        Parameters
        ----------
        channel_group : ChannelGroup
        modeling_frames : numpy.ndarray (bool), optional
            A boolean mask where `True` indicates that a frame is used for
            modeling and should be excluded from the calculations.

        Returns
        -------
        gain_increment, increment_weight : numpy.ndarray, numpy.ndarray
            The gain increments and associated weights both of shape
            (n_signal,) or (n_frames // self.resolution).
        """
        frames = self.integration.frames
        if modeling_frames is None:
            modeling_frames = frames.is_unflagged('MODELING_FLAGS')
        valid_frames = frames.valid & np.logical_not(modeling_frames)

        return snf.get_ml_correlated(
            frame_data=frames.data,
            frame_weights=frames.relative_weight,
            frame_valid=valid_frames,
            channel_indices=channel_group.indices,
            channel_wg=channel_group.temp_wg,
            channel_wg2=channel_group.temp_wg2,
            sample_flags=frames.sample_flag,
            resolution=self.resolution)

    def get_robust_correlated(self, channel_group, modeling_frames=None,
                              max_dependence=0.25):
        """

        Parameters
        ----------
        channel_group : ChannelGroup
        modeling_frames : numpy.ndarray (bool), optional
            A boolean mask where `True` indicates that a frame is used for
            modeling and should be excluded from the calculations.
        max_dependence : float, optional

        Returns
        -------
        gain_increment, increment_weight : numpy.ndarray, numpy.ndarray
            The gain increments and associated weights both of shape
            (n_signal,) or (n_frames // self.resolution).
        """
        frames = self.integration.frames
        if modeling_frames is None:
            modeling_frames = frames.is_unflagged('MODELING_FLAGS')
        valid_frames = frames.valid & np.logical_not(modeling_frames)

        return snf.get_robust_correlated(
            frame_data=frames.data,
            frame_weights=frames.relative_weight,
            frame_valid=valid_frames,
            channel_indices=channel_group.indices,
            channel_g=channel_group.temp_g,
            channel_wg2=channel_group.temp_wg2,
            sample_flags=frames.sample_flag,
            resolution=self.resolution,
            max_dependence=max_dependence)

    def calc_filtering(self):
        """
        Update the source filtering of the signal.

        Where phi is the channel dependents, they are updated by::

           phi = mean(phi + phi * channel_overlaps)

        The signal source filtering is set to 1 - phi.
        The channel source filtering has the prior correction undone, and
        then updated as::

           csf *= signal source filtering.

        Returns
        -------
        None
        """
        if self.source_filtering is None:
            self.source_filtering = np.ones(self.mode.size, dtype=float)
        channel_group = self.mode.channel_group

        cf, sf = snf.calculate_filtering(
            channel_indices=channel_group.indices,
            channel_dependents=self.dependents.for_channel,
            overlaps=channel_group.overlaps.toarray(),
            channel_valid=channel_group.is_unflagged(self.mode.skip_flags),
            n_parms=self.get_parms(),
            channel_source_filtering=channel_group.source_filtering,
            signal_source_filtering=self.source_filtering)

        channel_group.source_filtering = cf
        self.source_filtering = sf
