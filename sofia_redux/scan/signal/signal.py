# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import log, units
from copy import deepcopy
import numpy as np

from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.signal import signal_numba_functions as snf

__all__ = ['Signal']


class Signal(ABC):

    referenced_attributes = ['mode', 'integration']

    def __init__(self, integration, mode=None, values=None, is_floating=False):
        """
        Initialize a Signal object.

        The Signal object is used to extract a gain signal from an integration.

        Parameters
        ----------
        integration : Integration
            The integration to which the signal belongs.
        mode : Mode, optional
            The channel mode for which to create the signal.
        values : numpy.ndarray (float), optional
            Optional signal values to apply.
        is_floating : bool, optional
            If `True`, indicates that the signal values have some arbitrary DC
            offset.  This indicates that it would be advisable to "level"
            (subtract this offset) from the signal values before any type of
            application.
        """
        self.mode = mode
        self.integration = integration
        self.value = values
        self.weight = None
        self.drifts = None
        self.sync_gains = None
        self.resolution = None
        self.drift_n = None
        self.is_floating = is_floating

        if mode is not None:
            self.sync_gains = np.zeros(mode.size, dtype=float)
            integration.add_signal(self)

        if values is not None:
            values = np.atleast_1d(values)
            self.resolution = utils.roundup_ratio(integration.size,
                                                  values.size)
            self.value = values
            self.drift_n = values.size

    def __str__(self):
        """
        Return a string representation of the signal.

        Returns
        -------
        str
        """
        int_id = self.integration.get_full_id(separator='|')
        mode_name = self.mode.name
        return f"Signal {int_id}.{mode_name}"

    def copy(self):
        """
        Return a copy of the signal.

        Returns
        -------
        Signal
        """
        new = self.__class__(self.integration, self.mode, values=self.value,
                             is_floating=self.is_floating)
        for key, value in self.__dict__.items():
            if key in self.referenced_attributes:
                setattr(new, key, value)
            else:
                setattr(new, key, deepcopy(value))
        return new

    @property
    def size(self):
        """
        Return the length of the signal.

        Returns
        -------
        n_signal : int
        """
        if self.value is None:
            return 0
        return self.value.size

    @property
    def info(self):
        """
        Return the info object specific to the signal integration.

        Returns
        -------
        Info
        """
        return self.integration.info

    @property
    def configuration(self):
        """
        Return the configuration applicable to the signal.

        Returns
        -------
        Configuration
        """
        return self.integration.info.configuration

    def get_resolution(self):
        """
        Return the signal resolution.

        The signal resolution is basically the number of frames for which a
        single signal value may be applicable.

        Returns
        -------
        resolution : int
        """
        return self.resolution

    def value_at(self, frame):
        """
        Return the signal value for a given integration frame index.

        Parameters
        ----------
        frame : int or numpy.ndarray (int)

        Returns
        -------
        signal_value : float
        """
        return self.value[frame // self.resolution]

    def weight_at(self, frame):
        """
        Return the signal weight for a given integration frame index.

        Note that the standard signal object does not have an
        associated weight.

        Parameters
        ----------
        frame : int or numpy.ndarray (int)

        Returns
        -------
        signal_weight : float
        """
        return 1.0

    def scale(self, factor):
        """
        Scale the signal, drifts, and sync_gains by a given factor.

        Signal values and drifts are multiplied by the given factor.
        Sync gains (previous gain values) are divided by the given factor.

        Parameters
        ----------
        factor : float

        Returns
        -------
        None
        """
        self.value *= factor
        if self.drifts is not None:
            self.drifts *= factor

        self.sync_gains /= factor

    def add(self, value):
        """
        Add a DC offset to all signal and drift values.

        Parameters
        ----------
        value : float
            The value to add.

        Returns
        -------
        None
        """
        self.value += value
        if self.drifts is not None:
            self.drifts += value

    def subtract(self, value):
        """
        Subtract a DC offset from all signal and drift values.

        Parameters
        ----------
        value : float

        Returns
        -------
        None
        """
        self.add(value * -1)

    def add_drifts(self):
        """
        Add the drifts onto the signal value.

        All drifts are erased following this operation.

        Returns
        -------
        None
        """
        if self.drifts is None:
            return
        snf.add_drifts(
            signal_values=self.value, drifts=self.drifts,
            drift_length=self.drift_n)
        self.drifts = None

    def get_rms(self):
        """
        Get the signal RMS.

        Returns
        -------
        rms : float
        """
        return np.sqrt(self.get_variance())

    def get_variance(self):
        """
        Get the signal variance.

        Returns
        -------
        variance : float
        """
        return snf.get_signal_variance(self.value)

    def remove_drifts(self, n_frames=None, is_reconstructable=True,
                      robust=False):
        """
        Remove drifts from the signal.

        A drift is defined as the average signal for a given block of frames
        (the length of which is given by `n_frames`.  For each drift, the
        average signal value is calculated and subtracted from all signal
        values within that drift block.  If the drifts are marked as
        "reconstructable", they average signal values are stored in the
        `drifts` attribute.

        Parameters
        ----------
        n_frames : int, optional
            The number of frames in each drift.  The default is the number
            of frames in the signal's integration filter time scale.
        is_reconstructable : bool, optional
            If `True`, save the drift values for later use.
        robust : bool, optional
            If `True`, the robust (median) method will be used to determined
            the average signal level for each drift.

        Returns
        -------
        None
        """
        if n_frames is None:
            n_frames = self.integration.frames_for()

        n = utils.roundup_ratio(n_frames, self.resolution)

        if self.drifts is None or n != self.drift_n:
            # Add and destroy the current drifts before calculating new ones.
            self.add_drifts()
            self.drift_n = n
            drift_size = utils.roundup_ratio(self.value.size, n)
            drifts = np.zeros(drift_size, dtype=float)
            if is_reconstructable:
                self.drifts = drifts

        elif is_reconstructable:
            drifts = self.drifts

        else:
            drift_size = utils.roundup_ratio(self.value.size, n)
            drifts = np.zeros(drift_size, dtype=float)

        # Remove the mean signal and store as drifts in n_frames chunks
        snf.remove_drifts(
            signal_values=self.value,
            drifts=drifts,
            n_frames=n_frames,
            resolution=self.resolution,
            integration_size=self.integration.size,
            signal_weights=self.weight,
            robust=robust)

    def get_median(self):
        """
        Return the median signal value and weight.

        Returns
        -------
        median_value, median_weight : float, float
        """
        value, _ = numba_functions.smart_median_1d(self.value)
        weight = np.inf
        return value, weight

    def get_mean(self):
        """
        Return the mean signal value and weight.

        Returns
        -------
        mean_value, mean_weight
        """
        value, _ = numba_functions.mean(values=self.value, weights=self.weight)
        weight = np.inf
        return value, weight

    def differentiate(self):
        """
        Differentiate the signal values in-place.

        Note that the spacing between sample values is assumed to be in
        seconds when calculating the gradient.  The `is_floating` attribute
        is set to `False` following this operation indicating that the
        differentiated signal has no DC offset.

        Returns
        -------
        None
        """
        s = self.info.instrument.sampling_interval.decompose().value
        snf.differentiate_signal(values=self.value, dt=s * self.resolution)
        self.is_floating = False

    def integrate(self):
        """
        Integrate the signal values in-place using the trapezoidal rule.

        Note that the spacing between sample values is assumed to be in
        seconds.  The `is_floating` attribute is set to `True` following this
        operation indicating that the integrated signal has an arbitrary
        DC offset.

        Returns
        -------
        None
        """
        s = self.info.instrument.sampling_interval.decompose().value
        snf.integrate_signal(values=self.value, dt=s * self.resolution)
        self.is_floating = True

    def get_differential(self):
        """
        Return a differentiated copy of the signal.

        Returns
        -------
        Signal
        """
        signal = self.copy()
        signal.differentiate()
        return signal

    def get_integral(self):
        """
        Return an integrated copy of the signal.

        Returns
        -------
        Signal
        """
        signal = self.copy()
        signal.integrate()
        return signal

    def level(self, start_frame=None, end_frame=None, robust=False):
        """
        Remove the mean value from the signal values.

        The mean signal between a given start and end frame is calculated and
        subtracted from the signal.  This value is returned to the user.

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
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.integration.size

        value = self.value
        if isinstance(value, units.Quantity):
            value = value.value

        return snf.level(
            values=value,
            start_frame=start_frame,
            end_frame=end_frame,
            resolution=self.resolution,
            robust=robust)

    def smooth(self, fwhm):
        """
        Smooth the signal to a given FWHM (given in frames).

        Parameters
        ----------
        fwhm : float or int
            The FWHM in given in units of frames.

        Returns
        -------
        None
        """
        width = fwhm / self.resolution
        sigma = gaussian_fwhm_to_sigma * width
        n = 2 * (int(np.ceil(width))) + 1
        self.smooth_with_kernel(numba_functions.gaussian_kernel(n, sigma))

    def smooth_with_kernel(self, kernel):
        """
        Smooth the signal with a given kernel.

        Kernel spacing should be in units of self.resolution
        (number of frames).

        Parameters
        ----------
        kernel : numpy.ndarray (float)
            The kernel to smooth the signal values by.

        Returns
        -------
        None
        """
        numba_functions.smooth_1d(self.value, kernel)

    def set_sync_gains(self, gains):
        """
        Copy the gains to the sync_gains attribute.

        Parameters
        ----------
        gains : numpy.ndarray (float)
            An array of gains to copy to sync_gains.

        Returns
        -------
        None
        """
        self.sync_gains = gains.astype(float)

    def get_gain_increment(self, robust=False):
        """
        Return the gain increment and weight increment for each mode channel.

        For the robust method, the gain increment value is determined as:

        increment = median(frame_data / signal_value)

        while the maximum-likelihood method gives the gain increment as:

        increment = sum(frame_data * frame_weight * signal_value) / increment_w

        Both methods are weighted by frame_weight * signal_value^2 and result
        in an increment weight of:

        increment_w = sum(frame_weight * signal_value^2).

        Parameters
        ----------
        robust : bool, optional
            If `True`, use the robust method (median) to determine gain
            increments.  Otherwise, use the maximum-likelihood method (mean).

        Returns
        -------
        increment, increment_w : numpy.ndarray, numpy.ndarray
            The gain increment and increment weight, both of shape
            (n_channels,).
        """
        if self.configuration.get_bool('signal-response'):
            self.integration.comments.append(
                f'{{{self.get_covariance():.2f}}}')
            log.debug(f"covariance = {self.get_covariance():.2f}")

        # Precalculate the gain-weight products...
        # frames.temp_c = signal
        # frames.temp_wc = frame_weight * signal
        # frames.temp_wc2 = frame_weight * signal^2
        self.prepare_frame_temp_fields()

        if robust:
            return self.get_robust_gain_increment()
        else:
            return self.get_ml_gain_increment()

    def get_ml_gain_increment(self):
        """
        Get the maximum-likelihood (ML) gain increment for the signal.

        The maximum likelihood gain increment for a given channel (c) is:

        dG = sum(frame_data_{c} * frame_weight * signal_value) / dW
        dW = sum(frame_weight * signal_value^2)

        No invalid frames of modeling frames will be included in the sum.
        Additionally, no flagged samples will be included.

        Returns
        -------
        gain_increments, increment_weights : numpy.ndarray, numpy.ndarray
            The gain increments and increment weights for each channel in the
            signal mode channel group.  Both are arrays of shape (n_channels,).
        """
        frames = self.integration.frames
        valid_frames = frames.valid & frames.is_unflagged('MODELING_FLAGS')
        return snf.get_ml_gain_increment(
            frame_data=frames.data,
            signal_wc=frames.temp_wc,
            signal_wc2=frames.temp_wc2,
            sample_flags=frames.sample_flag,
            channel_indices=self.mode.channel_group.indices,
            valid_frames=valid_frames)

    def get_robust_gain_increment(self):
        """
        Get the robust (median derived) gain increment for the signal.

        The gain increment for a given channel is:

        increment = median(frame_data / signal_value)
        increment_weight = sum(frame_weight * signal_value^2)

        No invalid frames of modeling frames will be included in the sum.
        Additionally, no flagged samples will be included.

        Returns
        -------
        gain_increments, increment_weights : numpy.ndarray, numpy.ndarray
            The gain increments and increment weights for each channel in the
            signal mode channel group.  Both are arrays of shape (n_channels,).
        """
        frames = self.integration.frames
        valid_frames = frames.valid & frames.is_unflagged('MODELING_FLAGS')
        return snf.get_robust_gain_increment(
            frame_data=frames.data,
            signal_c=frames.temp_c,
            signal_wc2=frames.temp_wc2,
            sample_flags=frames.sample_flag,
            channel_indices=self.mode.channel_group.indices,
            valid_frames=valid_frames)

    def prepare_frame_temp_fields(self):
        """
        Prepare precalculated values for subsequent processing.

        The signal values are stored in the temp_* fields of the frame data.
        These contain:

        temp_c = signal_value
        temp_wc = relative_weight * signal_value
        temp_wc2 = relative_weight * signal_value^2

        Invalid frames have zero values.  Frames flagged as "MODELING" are
        set to zero weight.

        Returns
        -------
        None
        """
        frames = self.integration.frames
        signal_values = self.value_at(np.arange(frames.size))
        snf.prepare_frame_temp_fields(
            signal_values=signal_values,
            frame_weights=frames.relative_weight,
            frame_valid=frames.valid,
            frame_modeling=frames.is_flagged('MODELING_FLAGS'),
            frame_c=frames.temp_c,
            frame_wc=frames.temp_wc,
            frame_wc2=frames.temp_wc2)

    def get_covariance(self):
        """
        Return the signal covariance over all samples (frames, channels).

        The covariance only includes unflagged channels and unflagged samples.
        The signal covariance is given as:

        C = sum_{channels}(xs * xs) / sum_{channels}(ss * xx)

        where xs = sum_{frames}(w * x * s), ss = sum_{frames}(w * s * s), and
        xx = sum_{frames}(w * x * x).  Here w is the channel weight, s is the
        signal value for each frame, and x are the frame data values.

        Returns
        -------
        covariance : float
        """
        channels = self.mode.channel_group.create_data_group(match_flag=0)
        frames = self.integration.frames
        covariance = snf.get_covariance(
            signal_values=self.value_at(np.arange(frames.size)),
            frame_data=frames.data,
            frame_valid=frames.valid,
            channel_indices=channels.indices,
            channel_weights=channels.weight,
            sample_flags=frames.sample_flag)

        return covariance

    def resync_gains(self):
        """
        Resynchronize gains if any changes have occurred.

        The delta gains are calculated for each channel as:
            current_gain - last_gain

        For any non-zero delta_gain, frame data (x) for that channel will be
        updated by decrementing that value by:

            x[frame, channel] -= signal[frame] * delta_gain[channel]

        Returns
        -------
        None
        """
        gains = self.mode.get_gains()
        delta_gains = gains - self.sync_gains
        if np.all(delta_gains == 0):
            return  # nothing to do

        frames = self.integration.frames
        snf.resync_gains(
            frame_data=frames.data,
            signal_values=self.value,
            resolution=self.resolution,
            delta_gains=delta_gains,
            channel_indices=self.mode.channel_group.indices,
            frame_valid=frames.valid)

        self.set_sync_gains(gains)

    def synchronize_gains(self, sum_wc2=None, is_temp_ready=True):
        """
        Synchronize gains in the signal.

        If the gains have been updated, applies any changes to the data and
        updates dependencies.

        Parameters
        ----------
        sum_wc2 : numpy.ndarray (float), optional
            An array of gain weights of shape (n_channels,).
        is_temp_ready : bool, optional
            Indicates whether the frame temporary values have already been
            calculated.  These should contain::

                temp_c = signal_value
                temp_wc = relative_weight * signal_value
                temp_wc2 = relative_weight * signal_value^2

        Returns
        -------
        None
        """
        if self.mode.fixed_gains:
            raise ValueError("Cannot change gains for fixed gain modes.")

        channel_group = self.mode.channel_group
        parms = self.integration.get_dependents(f'gains-{self.mode.name}')

        gains = self.mode.get_gains()
        delta_gains = gains - self.sync_gains
        if (delta_gains == 0).all():
            # Do not need to resynchronize if there was no change
            return

        if not is_temp_ready:
            self.prepare_frame_temp_fields()

        if sum_wc2 is not None:

            parms.clear(channel_group)
            frames = self.integration.frames

            snf.synchronize_gains(
                frame_data=frames.data,  # Updated
                sample_flags=frames.sample_flag,
                frame_valid=frames.valid,
                modeling_frames=frames.is_flagged('MODELING_FLAGS'),
                channel_indices=channel_group.indices,
                delta_gains=delta_gains,
                frame_wc2=frames.temp_wc2,
                channel_wc2=sum_wc2,
                signal_values=frames.temp_c,
                frame_parms=parms.for_frame,  # Updated
                channel_parms=parms.for_channel)  # Updated

            # Apply the updated mode dependencies
            parms.apply(channel_group)

        # Register the gains as the ones used for the signal
        self.set_sync_gains(gains)

    def write_signal_values(self, out_file):
        """
        Write the signal information to a text file.

        Note that in crush this is the "print" method.

        Parameters
        ----------
        out_file : str
            The name of the file to write to.

        Returns
        -------
        None
        """
        n = 1 / (self.resolution * self.integration.info.integration_time)
        n = n.decompose().value
        with open(out_file, 'w') as f:
            print(f'# {n}', file=f)
            for value in self.value:
                print(f'{value:.3e}', file=f)
