# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numba as nb
from sofia_redux.scan.utilities import numba_functions

two_pi = 2 * np.pi
nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['load_timestream', 'remove_rejection_from_frames',
           'apply_rejection_to_parms', 'dft_filter_channels',
           'dft_filter_frames', 'dft_filter_frequency_channel',
           'level_for_channels', 'level', 'level_1d', 'resample',
           'accumulate_profiles', 'calculate_varied_point_response',
           'calculate_channel_point_responses', 'calc_mean_amplitudes',
           'whiten_profile', 'add_frame_parms', 'expand_rejection_filter',
           'harmonize_rejection_filter']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def load_timestream(frame_data, frame_weights, frame_valid, modeling_frames,
                    channel_indices, sample_flags, timestream, points
                    ):  # pragma: no cover
    """
    Load timestream data from the supplied frame data.

    The timestream data is defined as::

        timestream = (d * w) - mean(d * w)

    In addition, the points are also calculated at this stage as::

        points = sum(w)

    where d is the `frame_data` and w are the `frame_weights`.  This
    calculation occurs for each channel supplied in `channel_indices`.
    Only valid samples are included in the mean and sum operations, and
    equivalent timestream values for invalid samples will be set to zero
    on output.  Invalid samples are those in which the frame is marked as
    invalid or a modeling frame, the frame data is NaN or sample flag is
    nonzero.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
        The frame weights of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that a frame
        is valid and may be included in the time stream.
    modeling_frames : numpy.ndarray (bool)
        A boolean mask where `True` marks a frame as a modeling frame, and
        will therefore not be included in the output timestream.
    channel_indices : numpy.ndarray (int)
        The filter channel group indices, used to extract the correct
        channel information from the frame sample data.
    sample_flags : numpy.ndarray (int)
        The data sample flags of shape (n_frames, all_channels) where any
        non-zero value will not be included in the time-stream.
    timestream : numpy.ndarray (float)
        The data output array to populate.  Should be of shape
        (n_channels, n_frames) where n_frames is the number of integration
        frames, and n_channels is the number of channels in the filter
        channel group.
    points : numpy.ndarray (float)
        The sum of valid frame weights for each channel.  Should be of shape
        (n_channels,) where n_channels is the number of channels in the filter
        channel group.  Will be updated in-place.

    Returns
    -------
    None
    """
    n_frames, all_channels = frame_data.shape
    for i, channel in enumerate(channel_indices):
        point = 0.0
        data_sum = 0.0
        n = 0
        for frame in range(n_frames):
            if not frame_valid[frame]:
                timestream[i, frame] = np.nan
            elif modeling_frames[frame]:
                timestream[i, frame] = np.nan
            elif sample_flags[frame, channel] != 0:
                timestream[i, frame] = np.nan
            elif np.isnan(frame_data[frame, channel]):
                timestream[i, frame] = np.nan
            else:
                weight = frame_weights[frame]
                value = weight * frame_data[frame, channel]
                timestream[i, frame] = value
                data_sum += value
                point += weight
                n += 1

        points[i] = point

        # Remove the DC offset
        if n > 0:
            average = data_sum / n
            for frame in range(n_frames):
                if np.isnan(timestream[i, frame]):
                    timestream[i, frame] = 0.0
                else:
                    timestream[i, frame] -= average
        else:
            for frame in range(n_frames):
                timestream[i, frame] = 0.0


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def remove_rejection_from_frames(frame_data, frame_valid, channel_indices,
                                 rejected_signal):  # pragma: no cover
    """
    Remove the rejected signal from frame data.

    This is a simple operation that subtract the rejected signal from the
    frame data for all valid frames.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data to update in-place of shape (n_frames, all_channels).
    frame_valid : numpy.ndarray (bool)
        A boolean mask, where `False` indicates that the frame is invalid and
        should be ignored during processing.
    channel_indices : numpy.ndarray (int)
        The channel indices mapping `rejected_signal` channels onto
        `frame_data` channels of shape (n_channels,).
    rejected_signal : numpy.ndarray (float)
        The signal to remove of shape (n_channels, n_frames).

    Returns
    -------
    None
    """
    for frame in range(frame_data.shape[0]):
        if not frame_valid[frame]:
            continue
        for i, channel in enumerate(channel_indices):
            frame_data[frame, channel] -= rejected_signal[i, frame]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def apply_rejection_to_parms(frame_valid, frame_weight, frame_parms, dp,
                             channel_indices, sample_flag):  # pragma: no cover
    """
    Update the frame dependents following the remove operation.

    Apply the channel dependent updates to the frame dependents.  For a given
    frame the increment is given by::

        increment = w * sum(dp)

    where w is the frame relative weight, dp is the channel dependent delta,
    and the sum occurs over valid frames and nonzero samples.  The frame
    dependents are updated via::

       frame_dependents = frame_dependents + increment

    Parameters
    ----------
    frame_valid : numpy.ndarray (bool)
        A boolean mask, where `False` indicates that the frame is invalid and
        should be ignored during processing.  Should be of shape (n_frames,).
    frame_weight : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    frame_parms : numpy.ndarray (float)
        The frame dependents to update of shape (n_frames,).
    dp : numpy.ndarray (float)
        The channel delta dependents of shape (n_channels,).
    channel_indices : numpy.ndarray (int)
        The channel indices mapping n_channels onto all_channels of shape
        (n_channels,)
    sample_flag : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels) where any nonzero
        flag will remove that frame,channel sample from processing.

    Returns
    -------
    None
    """
    for frame in range(frame_valid.size):
        if not frame_valid[frame]:
            continue
        w = frame_weight[frame]
        if w == 0:
            continue
        for i, channel in enumerate(channel_indices):
            if sample_flag[frame, channel] != 0:
                continue
            frame_parms[frame] += w * dp[i]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def dft_filter_frequency_channel(data, fch, rejection_value, rejected, n_frames
                                 ):  # pragma: no cover
    """
    Apply DFT filtering for a given frequency channel.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The data to filter of shape (pow2ceil(n_frames),) in real-space for
        a single channel.
    fch : int
        The frequency channel.
    rejection_value : float
        The filter rejection value for the `fch` channel.
    rejected : numpy.ndarray (float)
        The current rejected overall values (in Fourier space).  This is
        updated in-place, so must be correct on entry and be the same shape
        as `data` (pow2ceil(n_frames),).
    n_frames : int
        The number of integration frames.

    Returns
    -------
    None
    """
    sum_c = 0.0
    sum_s = 0.0
    n_data = data.size
    if fch == 0:
        fch = n_data // 2

    theta = fch * two_pi / n_data
    s0 = np.sin(theta)
    c0 = np.cos(theta)
    c = 1.0
    s = 0.0

    for i in range(n_frames - 1, -1, -1):
        x = data[i]
        sum_c += c * x
        sum_s += s * x
        temp = c
        c = (temp * c0) - (s * s0)
        s = (temp * s0) + (s * c0)

    norm = rejection_value * 2.0 / n_data
    sum_c *= norm
    sum_s *= norm
    c = 1.0
    s = 0.0

    for i in range(n_frames - 1, -1, -1):
        rejected[i] += (c * sum_c) + (s * sum_s)
        temp = c
        c = (temp * c0) - (s * s0)
        s = (temp * s0) + (s * c0)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def dft_filter_frames(data, rejection, n_frames):  # pragma: no cover
    """
    Use DFT to filter single channel data.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The data to filter of shape (n,) where n = pow2ceil(n_frames).
        The data are updated in-place.
    rejection : numpy.ndarray (float)
        The rejection filter of shape (nf + 1,) where nf = n // 2.
    n_frames : int
        The number of frames in the integration (n_frames).

    Returns
    -------
    rejected_signal : numpy.ndarray (float)
        The rejected signal of shape (n_frames,)
    """
    filtered = np.zeros(n_frames, dtype=nb.float64)
    for fch in range(rejection.size - 1, -1, -1):
        rejection_value = rejection[fch]
        if rejection_value > 0:
            dft_filter_frequency_channel(
                data, fch, rejection_value, filtered, n_frames)
    data[:n_frames] = filtered[:n_frames]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def dft_filter_channels(frame_data, rejection, n_frames):  # pragma: no cover
    """
    Filter frame data using the filter rejection DFT method.

    Applies :func:`dft_filter_frames` to each channel in the frame data using
    the rejection filter.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The expanded frame data of shape (n_channels, pow2ceil(n_frames))
    rejection : numpy.ndarray (float)
        The rejection filter of shape (nf + 1,) where,
        nf = pow2ceil(n_frames) // 2.
    n_frames : int
        The number of frames in the integration (n_frames).

    Returns
    -------
    None
    """
    n_channels, n_data = frame_data.shape
    for i in range(n_channels):
        dft_filter_frames(frame_data[i], rejection, n_frames)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def level_for_channels(signal, valid_frame, modeling_frame, sample_flag,
                       channel_indices):  # pragma: no cover
    """
    Level a given signal over all frames for the given channel indices.

    The resulting operation subtracts the mean for each channel from the frame
    data.  The mean is calculated using only valid samples, but is subtracted
    from all samples.  If there are no valid samples for a given channel,  its
    data will be zeroed.  Valid samples are those that are marked as a
    `valid_frame`, is not a `modeling_frame`, and has a zero valued
    `sample_flag`.

    Parameters
    ----------
    signal : numpy.ndarray (float)
        The signal data to level of shape (n_channels, >=n_frames).  Is updated
        in-place.
    valid_frame : numpy.ndarray (bool)
        A boolean mask where `False` excludes a frame from being included in
        the average signal calculation.  Should be of shape (n_frames,).
    modeling_frame : numpy.ndarray (bool)
        A boolean mask marking modeling frames with `True`.  These frames will
        not be included in the average signal calculation.  Should be of shape
        (n_frames,).
    sample_flag : numpy.ndarray (int)
        The flag mask of shape (n_frames, all_channels) where any nonzero value
        will exclude that given channel and frame from being included in the
        averaging calculation.
    channel_indices : numpy.ndarray (int)
        The channel indices for which `signal` is applicable and maps
        n_channels onto all_channels.  An array of shape (n_channels,).

    Returns
    -------
    None
    """
    n_frames = valid_frame.size
    for i, channel in enumerate(channel_indices):
        d_sum = 0.0
        n = 0
        for frame in range(n_frames):
            if not valid_frame[frame]:
                continue
            elif modeling_frame[frame]:
                continue
            elif sample_flag[frame, channel] != 0:
                continue
            d_sum += signal[i, frame]
            n += 1

        if n > 0:
            average = d_sum / n
            for frame in range(n_frames):
                signal[i, frame] -= average
        else:
            for frame in range(n_frames):
                signal[i, frame] = 0.0


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def level_1d(data, n_frames):  # pragma: no cover
    """
    Remove the average from a given 1-D data array.

    If the average is found to be NaN, all values in the array are set to zero.

    Parameters
    ----------
    data : numpy.ndarray
        The data array to level of shape (>=n_frames,).
    n_frames : int
        The last non-inclusive index of the data that should have the average
        level removed.  Note that the average level is still calculated over
        the entire `data` provided.

    Returns
    -------
    None
    """
    mean_level = np.mean(data)
    if np.isnan(mean_level):
        for i in range(n_frames):
            data[i] = 0.0
    else:
        for i in range(n_frames):
            data[i] -= mean_level


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def level(data, n_frames):  # pragma: no cover
    """
    Remove the average level from each channel for the provided data set.

    This removes the simple average from all channels.  If any NaN values are
    present in the frame data for a given channel, that all data for that
    channel will be set to zero.

    Parameters
    ----------
    data : numpy.ndarray
        The data array to level of shape (n_channels, >=n_frames,).
    n_frames : int
        The last non-inclusive index of the data that should have the average
        level removed.  Note that the average level is still calculated over
        the entire `data` provided for each channel.

    Returns
    -------
    None
    """
    n_channels, n_data = data.shape
    for i in range(n_channels):
        level_1d(data[i], n_frames)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def resample(old, new):  # pragma: no cover
    """
    Fast implementation of the `resample` method.

    Resampling from a high number of frames in the old array to a lower number
    of frames in the new array is done by taking a simple mean of old frames
    that fall into a single bin of the new array.  Note that this is to be
    used exclusively for downsampling.

    Parameters
    ----------
    old : numpy.ndarray (float)
        The old channel profiles of shape (n_channels, n1)
    new : numpy.ndarray (float)
        The new channel profiles of shape (n_channels, n2).  Updated in-place.

    Returns
    -------
    None
    """
    n_channels, n1 = old.shape
    n2 = new.shape[1]
    n = n1 / n2

    for i in range(n2):

        start = numba_functions.round_value(i * n)
        end = numba_functions.round_value((i + 1) * n)

        for j in range(n_channels):
            if start == end:
                new[j, i] = old[j, start]
            else:
                count = 0
                data_sum = 0.0
                for k in range(start, end):
                    count += 1
                    data_sum += old[j, k]
                new[j, i] = data_sum / count


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def accumulate_profiles(profiles, channel_profiles, channel_indices
                        ):  # pragma: no cover
    """
    Accumulate the calculated profiles onto channel profiles.

    The accumulation process multiplies the existing `channel_profiles` by
    `profiles`, and once complete, sets `profiles` to the same values as
    `channel_profiles`.

    Parameters
    ----------
    profiles : numpy.ndarray (float)
        The profiles of shape (n_channels, n_freq).
    channel_profiles : numpy.ndarray (float)
        The channel profiles of shape (n_channels, n_freq).
    channel_indices : numpy.ndarray (int)
        The channel indices for which to accumulate profiles of shape
        (<=n_channels,).

    Returns
    -------
    None
    """
    n_freq = profiles.shape[1]
    for i in channel_indices:
        for f in range(n_freq):
            channel_profiles[i, f] *= profiles[i, f]
            profiles[i, f] = channel_profiles[i, f]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calculate_varied_point_response(min_fch, source_profile, response,
                                    source_norm):  # pragma: no cover
    """
    Calculate the point response for the varied filter.

    The varied point response is given as:

        response = (low_pass + high_pass) / source_norm

    where

        low_pass = sum_{0}^{min_fch-1}(source_profile)
        high_pass = sum_{min_fch}^{nf}(source_profile * response)

    and nf is the number of frequencies in the response.

    Parameters
    ----------
    min_fch : int
        The frequency channel marking the high-pass filter time-scale.
    source_profile : numpy.ndarray (float)
        The source profile of shape (n_freq,).
    response : numpy.ndarray (float)
        The filter response of shape (n_freq,).
    source_norm : float
        The source profile normalization constant.

    Returns
    -------
    point_response : float
        The point response for the varied filter.
    """
    d_sum = 0.0
    n_freq = source_profile.size
    for f in range(n_freq):
        if f < min_fch:
            d_sum += source_profile[f]
        else:
            d_sum += source_profile[f] * response[f]
    return d_sum / source_norm


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calculate_channel_point_responses(min_fch, source_profile, profiles,
                                      channel_indices, source_norm
                                      ):  # pragma: no cover
    """
    Calculate the point response for each given channel of an adaptive filter.

    This is a separate implementation of
    :func:`calculate_varied_point_response` for use over multiple channels.

    Parameters
    ----------
    min_fch : int
        The frequency channel marking the high-pass filter time-scale.
    source_profile : numpy.ndarray (float)
        The source profile of shape (n_freq,).
    profiles : numpy.ndarray (float)
        The profiles for each channel of shape (n_channels, n_freq).
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the point response of shape
        (calc_channels <= n_channels,).
    source_norm : float
        The source profile normalization constant.

    Returns
    -------
    point_response : numpy.ndarray (float)
        The point response for each given channel of shape (calc_channels,).
    """
    n_channels = channel_indices.size
    n_freq = source_profile.size
    channel_independent = profiles.shape[0] == 1
    low_sum = 0.0
    for freq in range(0, min_fch):
        low_sum += source_profile[freq]

    response = np.zeros(n_channels, dtype=nb.float64)

    for i, channel in enumerate(channel_indices):
        d_sum = low_sum
        for freq in range(min_fch, n_freq):
            profile_index = 0 if channel_independent else i
            d_sum += source_profile[freq] * profiles[profile_index, freq]
        response[i] = d_sum

    for i in range(n_channels):
        response[i] /= source_norm

    return response


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calc_mean_amplitudes(amplitudes, amplitude_weights, spectrum, windows,
                         channel_indices):  # pragma: no cover
    """
    Calculate the mean amplitudes and weights of a Fourier spectrum.

    The `amplitudes` and `amplitude_weights` are updated in-place.

    The amplitude of the spectrum at a single frequency channel (f) is
    calculated as:

        amplitude = sqrt(2 * sum_{i=0}^{window-1}(a[f+i]^2 + b[f+i]^2) / w)

    where

        w = sum_{i=0}^{window-1}((a[f+i] != 0) + (b[f+i] != 0))

    (a, b) are the (real, imaginary) components of `spectrum` and w are the
    `amplitude_weights`.  Thus, if `windows` > 1 then this effectively
    downsamples the spectrum using a box average.  Typically, `windows` is set
    to 1.  Note that the Nyquist value is purely real, and stored in the last
    element of the spectrum a[-1] (b[-1] = 0).  It's amplitude and weight will
    likewise be stored in the last elements of `amplitudes` and
    `amplitude_weights`.  The number of frequencies to calculate (nf/2) is
    determined from the input amplitude.shape[1]

    Parameters
    ----------
    amplitudes : numpy.ndarray (float)
        The empty amplitudes array to fill with the calculated amplitudes
        of shape (n_channels, >=nf/2).  Updated in-place.
    amplitude_weights : numpy.ndarray (float)
        The empty amplitude weights array to fill with the calculated
        amplitude weights of shape (n_channels, >=nf/2).  Updated in-place.
    spectrum : numpy.ndarray (complex)
        The spectrum from which to derive amplitudes.  An array of shape
        (n_channels, >=windows * nf/2).
    windows : int
        The size of the averaging window in frequency bins.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate amplitudes.

    Returns
    -------
    None
    """
    nf = amplitudes.shape[1]
    nt = spectrum.shape[1]

    # The Nyquist component is purely real and stored in the last element
    # of the spectrum.
    nyquist_bin = nt - 1

    for i, channel in enumerate(channel_indices):

        real = spectrum[channel].real
        imag = spectrum[channel].imag

        for f in range(nf):
            from_f = max(1, f * windows)
            to_f = min(from_f + windows, nyquist_bin)
            sum_p = 0.0
            pts = 0
            for f2 in range(from_f, to_f):
                dr = real[f2]
                di = imag[f2]
                if dr != 0:
                    sum_p += dr * dr
                    pts += 1
                if di != 0:
                    sum_p += di * di
                    pts += 1

            if pts == 0:
                amplitude = 0.0
            else:
                amplitude = np.sqrt(2 * sum_p / pts)
            amplitudes[channel, f] = amplitude
            amplitude_weights[channel, f] = pts

        # Add the Nyquist component to the last bin unless
        # killed by kill filter
        nyquist_value = real[-1]
        if nyquist_value != 0:
            nv = amplitudes[channel, -1]
            nw = amplitude_weights[channel, -1]
            nv = (nw * nv * nv) + (2 * nyquist_value * nyquist_value)
            nw += 1
            nv = np.sqrt(nv / nw)
            amplitudes[channel, -1] = nv
            amplitude_weights[channel, -1] = nw


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def whiten_profile(amplitudes, amplitude_weights, profiles, channel_profiles,
                   white_from, white_to, filter_level, significance,
                   one_over_f_bin, white_noise_bin, channel_indices
                   ):  # pragma: no cover
    """
    Fast implementation of the `whiten_profile` method.

    Parameters
    ----------
    amplitudes : numpy.ndarray (float)
        An array of spectrum amplitudes of shape (n_channels, nf).
    amplitude_weights : numpy.ndarray (float)
        An array of the spectrum amplitude weights derived during
        `calc_mean_amplitudes` of shape (n_channels, nf).
    profiles : numpy.ndarray (float)
        The profiles working array of shape (n_channels, nf).  Will be
        be overwritten during this process.
    channel_profiles : numpy.ndarray (float)
        The filter profile for each channel of shape (n_channels, nf).  Will
        be updated with the new relative profile changes.
    white_from : int
        The filtering start frequency channel.
    white_to : int
        The non-inclusive end filtering frequency channel.
    filter_level : float
        The filtering level active as a DC offset, above which to apply the
        whitening filter.
    significance : float
        The maximum deviation of the amplitude.  Whitening will be applied
        above this level.
    one_over_f_bin : int
        The index of the 1/f frequency channel bin.
    white_noise_bin : int
        The index of the white noise frequency bin.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to apply the whitening.

    Returns
    -------
    channel_one_over_f_stat : numpy.ndarray (float)
        The new channel 1/f statistics of shape (n_channels,).
    """
    nf = profiles.shape[1]
    n_channels = channel_indices.size
    one_over_f_stat = np.empty(n_channels, dtype=nb.float64)

    for i, channel in enumerate(channel_indices):
        profile = profiles[channel]
        last_profile = channel_profiles[channel]
        amplitude = amplitudes[channel]
        amplitude_weight = amplitude_weights[channel]

        for f in range(nf):
            profile[f] = 1.0

        median_amplitude, med_weight = numba_functions.smart_median_1d(
            values=amplitude[white_from: white_to],
            weights=amplitude_weight[white_from: white_to],
            max_dependence=1.0)

        if median_amplitude == 0:
            one_over_f_stat[i] = np.nan
            for f in range(profile.size):
                profile[f] = 0.0
                amplitude[f] = 0.0
                amplitude_weight[f] = 0.0
            one_over_f_stat[i] = np.nan
            continue

        critical = filter_level * median_amplitude
        weight_scale = 4.0 / (median_amplitude ** 2)

        for f in range(nf):
            amplitude_weight[f] *= weight_scale

        # Only whiten those frequencies which have a significant power excess
        # when compared to the specified level over the median spectral power.
        for f in range(1, nf):
            aw = amplitude_weight[f]
            last_value = last_profile[f]
            if aw > 0 and last_value != 0:
                a = amplitude[f]
                rms = 1.0 / np.sqrt(aw)
                dev = ((a / last_value) - critical) / rms

                if dev > significance:
                    new_value = median_amplitude / amplitude[f]
                else:
                    new_value = 1.0 / last_value

                profile[f] = new_value
                amplitude[f] *= new_value
                amplitude_weight[f] /= new_value ** 2
            else:
                amplitude[f] = median_amplitude

        f_value = profile[one_over_f_bin]
        if f_value != 0:
            one_over_f_stat[i] = profile[white_noise_bin] / f_value
        else:
            one_over_f_stat[i] = np.nan

    return one_over_f_stat


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def add_frame_parms(rejected, points, weights, frame_valid, modeling_frame,
                    frame_parms, sample_flags, channel_indices
                    ):  # pragma: no cover
    """
    Add to the frame dependents based on the rejected signal.

    Each frame dependent value (fp) is updated by:

        fp = fp + increment

    where:

        increment = w * dp
        dp = sum_{channels}(rejected / points)

    and w is the frame weights (`weights`).  Only valid samples (frame/channel
    combination will be updated or considered during the operation.  A valid
    sample must consist of a valid non-modeling frame with nonzero frame
    weight, and a zero valued sample flag.

    Parameters
    ----------
    rejected : numpy.ndarray (float)
        The rejected filter sum.  An array of shape (n_channels,).
    points : numpy.ndarray  (float)
        The relative weight sum over frames for each channel.  An array of
        shape (n_channels,).
    weights : numpy.ndarray (float)
        The relative frame weights as an array of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask where `False` excludes a given frame from all
        calculations or updates.
    modeling_frame : numpy.ndarray (bool)
        A boolean mask where `True` excludes a given frame from all
        calculations or updates.
    frame_parms : numpy.ndarray (float)
        The frame_parms to update.  An array of shape (nt,) where nt is
        the ceiled power of 2 number of frames.
    sample_flags : numpy.ndarray (int)
        An array of shape (n_frames, all_channels) where non-zero flagged
        samples are excluded from the calculation.
    channel_indices : numpy.ndarray (int)
        The channel indices mapping all_channels onto n_channels.

    Returns
    -------
    None
    """
    n_frames = frame_valid.size
    for i, channel in enumerate(channel_indices):

        if points[i] != 0:
            dp = rejected[i] / points[i]
        else:
            continue

        for frame in range(n_frames):
            if not frame_valid[frame]:
                continue
            elif modeling_frame[frame]:
                continue
            elif sample_flags[frame, channel] != 0:
                continue
            weight = weights[frame]
            if weight == 0:
                continue

            frame_parms[frame] += weight * dp


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def expand_rejection_filter(reject, half_width, df):  # pragma: no cover
    """
    Expand the rejection filter by a given width.

    Parameters
    ----------
    reject : numpy.ndarray (bool)
        The rejection mask of size (filter.size,).  The mask is updated
        in-place.
    half_width : float
        The half-width by which to expand the filter in frequency units.
    df : float
        The width of each frequency bin in frequency units.

    Returns
    -------
    None
    """
    d = half_width / df
    delta = numba_functions.round_value(d)

    if delta < 1:
        return

    n = reject.size
    expanded = np.full(n, False)
    last_from = n - 1
    for i in range(last_from, -1, -1):
        if reject[i]:
            from_i = max(0, i - delta)
            to_i = min(last_from, i + delta) + 1
            for j in range(from_i, to_i):
                expanded[j] = True

    # Update the mask
    for i in range(n):
        reject[i] = expanded[i]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def harmonize_rejection_filter(reject, harmonics, odd_harmonics_only
                               ):  # pragma: no cover
    """
    Add harmonics to the rejection filter.

    Parameters
    ----------
    reject : numpy.ndarray (bool)
        The rejection mask of size (filter.size,).  The mask is updated
        in-place.
    harmonics : int
        The number of harmonics to add.
    odd_harmonics_only : bool
        If `True`, only add odd integer harmonics.

    Returns
    -------
    None
    """
    n = reject.size
    spread = np.full(n, False)
    step = 2 if odd_harmonics_only else 1
    for i in range(n - 1, -1, -1):
        if not reject[i]:
            continue
        for k in range(1, harmonics + 1, step):
            j = k * i
            if j >= n:
                break
            spread[j] = True

    # update the rejection mask
    for i in range(n):
        reject[i] = spread[i]
