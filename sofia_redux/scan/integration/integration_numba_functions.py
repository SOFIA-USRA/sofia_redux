# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

from sofia_redux.scan.utilities import numba_functions

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['determine_time_weights', 'get_time_weights_by_block',
           'robust_channel_weights', 'differential_channel_weights',
           'rms_channel_weights', 'set_weights_from_var_stats',
           'despike_neighbouring', 'despike_absolute', 'despike_gradual',
           'despike_multi_resolution', 'flagged_channels_per_frame',
           'flagged_frames_per_channel', 'frame_block_expand_flag',
           'next_weight_transit', 'get_mean_frame_level',
           'weighted_mean_frame_level', 'weighted_median_frame_level',
           'remove_channel_drifts', 'level', 'apply_drifts_to_channel_data',
           'detector_stage', 'readout_stage', 'search_corners',
           'get_weighted_timestream', 'calculate_2d_velocities',
           'calculate_2d_accelerations', 'classify_scanning_speeds',
           'smooth_positions', 'get_covariance', 'get_full_covariance_matrix',
           'get_partial_covariance_matrix', 'downsample_frame_data',
           'downsample_frame_flags', 'get_downsample_start_indices',
           'get_valid_downsampling_frames']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def determine_time_weights(block_size, frame_data, frame_dof, frame_weight,
                           frame_valid, frame_dependents, frame_flags,
                           frame_weight_flag, frame_dof_flag, channel_weights,
                           channel_indices, channel_flags, time_weight_flag,
                           sample_flags):  # pragma: no cover
    """
    Determine the frame weights (noise) in time.

    The frame relative weights and degrees of freedom (DOF) will be updated in-
    place. All frame weighting flags will be removed and should be re-evaluated
    after a call to this procedure.  However, the frame DOF flag WILL be
    applied in-place.

    The frame noise (time-weights) will be determined uniformly for a given
    chunk of frames in an integration.  They will then be normalized over
    all frames in an integration such that:

        relative_weight = frame_weight / mean(frame_weight)

    Please see :func:`get_time_weights_by_block` for a description of how
    frame weights and degrees-of-freedom are determined.

    Parameters
    ----------
    block_size : int
        The number of frames to process in each chunk.
    frame_data : numpy.ndarray (float)
        The integration frame data values of shape (n_frames, all_channels).
    frame_dof : numpy.ndarray (float)
        The integration frame degrees of freedom of shape (n_frames,).  Will
        be updated in-place.
    frame_weight : numpy.ndarray (float)
        The integration frame weights of shape (n_frames,).  Will be updated
        in place.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` marks an invalid
        frame that will not be included in any calculation.
    frame_dependents : numpy.ndarray (float)
        The frame dependents of shape (n_frames,).
    frame_flags : numpy.ndarray (int)
        The frame flags of shape (n_frames,).
    frame_weight_flag : int
        The integer identifier for the FLAG_WEIGHT frame flag.  All valid
        frames will be unflagged with this identifier.
    frame_dof_flag : int
        The integer identifier for the FLAG_DOF frame flag.  All valid
        frames in the chunk will either be flagged or unflagged depending
        on whether the degrees of freedom are > 0.
    channel_weights : numpy.ndarray (float)
        The channel weights of shape (n_channels,).
    channel_indices : numpy.ndarray (int)
        The channel indices mapping n_channels -> all_channels.
    channel_flags : numpy.ndarray (int)
        The channel flags of shape (n_channels,).
    time_weight_flag : int
        The integer identifier for the TIME_WEIGHTING_FLAGS flag.  Note that
        while channel space is referred to in the original code - no such
        flag exists and I really really think this should be in frame space.
        Channels (or frames rather) flagged with this will not be included in
        the calculations.
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels).

    Returns
    -------
    None
    """
    n_points = 0.0
    weight_sum = 0.0
    n_frames, all_channels = frame_data.shape

    for start_frame in nb.prange(0, n_frames, block_size):
        block_points, block_weight = get_time_weights_by_block(
            frame_data=frame_data,
            frame_dof=frame_dof,
            frame_weight=frame_weight,
            frame_valid=frame_valid,
            frame_dependents=frame_dependents,
            frame_flags=frame_flags,
            frame_weight_flag=frame_weight_flag,
            frame_dof_flag=frame_dof_flag,
            channel_weights=channel_weights,
            channel_indices=channel_indices,
            channel_flags=channel_flags,
            time_weight_flag=time_weight_flag,
            sample_flags=sample_flags,
            frame_from=start_frame,
            frame_to=start_frame + block_size)
        n_points += block_points
        weight_sum += block_weight

    if weight_sum > 0:
        inverse_weight = n_points / weight_sum
        renorm = True
    else:
        inverse_weight = 1.0
        renorm = False

    # Normalize
    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue
        if np.isnan(frame_weight[frame]):
            frame_weight[frame] = 0.0
        elif renorm:
            frame_weight[frame] *= inverse_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_time_weights_by_block(frame_data, frame_dof, frame_weight, frame_valid,
                              frame_dependents, frame_flags, frame_weight_flag,
                              frame_dof_flag, channel_weights, channel_indices,
                              channel_flags, time_weight_flag, sample_flags,
                              frame_from=-1, frame_to=-1):  # pragma: no cover
    """Determine the frame weighting (time) for all integration frames.

    The frame weighting is determined for a given chunk of frames
    at a time.  For all frames in a chunk the time weight will be given
    as:

        chi2 = sum_{c,f} (cw * (d ** 2)) / n
        n = nc * nf
        deps = sum_{f} (frame_dependents)
        weight = (n - deps) / chi2

    where nc is the number of channels, nf is the number of frames in the
    chunk, cw is the channel weight, and d is the frame value for each
    frame and channel.  The degrees of freedom are also calculated for each
    frame in the chunk as:

        dof = 1 - (deps / n)

    At this stage all valid frames have the FLAG_WEIGHT flag removed, but
    the FLAG_DOF flag will be added or removed depending on whether
    n > deps (unflagged) or n <= deps (flagged as DOF).

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The integration frame data values of shape (n_frames, all_channels).
    frame_dof : numpy.ndarray (float)
        The integration frame degrees of freedom of shape (n_frames,).  Will
        be updated in-place.
    frame_weight : numpy.ndarray (float)
        The integration frame weights of shape (n_frames,).  Will be updated
        in place.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` marks an invalid
        frame that will not be included in any calculation.
    frame_dependents : numpy.ndarray (float)
        The frame dependents of shape (n_frames,).
    frame_flags : numpy.ndarray (int)
        The frame flags of shape (n_frames,).
    frame_weight_flag : int
        The integer identifier for the FLAG_WEIGHT frame flag.  All valid
        frames will be unflagged with this identifier.
    frame_dof_flag : int
        The integer identifier for the FLAG_DOF frame flag.  All valid
        frames in the chunk will either be flagged or unflagged depending
        on whether the degrees of freedom are > 0.
    channel_weights : numpy.ndarray (float)
        The channel weights of shape (n_channels,).
    channel_indices : numpy.ndarray (int)
        The channel indices mapping n_channels -> all_channels.
    channel_flags : numpy.ndarray (int)
        The channel flags of shape (n_channels,).
    time_weight_flag : int
        The integer identifier for the TIME_WEIGHTING_FLAGS flag.  Note that
        while channel space is referred to in the original code - no such
        flag exists and I really really think this should be in frame space.
        Channels (or frames rather) flagged with this will not be included in
        the calculations.
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels).
    frame_from : int, optional
        The starting frame of the chunk.  The default is the first (0).
    frame_to : int, optional
        The last (non-inclusive) frame of the chunk.  The default is the
        total number of frames.

    Returns
    -------
    n, weight_sum : float, float
        The total number of points (valid frames * valid channels in chunk),
        and the total weight.
    """
    points = 0
    dependents = 0.0
    sum_chi2 = 0.0
    if frame_from < 0:
        frame_from = 0
    if frame_to is None or frame_to > frame_valid.size or frame_to < 0:
        frame_to = frame_valid.size

    frame_weight_unflag = ~frame_weight_flag
    frame_dof_unflag = ~frame_dof_flag

    for frame in range(frame_from, frame_to):
        if not frame_valid[frame]:
            continue

        frame_flags[frame] &= frame_weight_unflag

        for i, channel in enumerate(channel_indices):
            # This is not in the original, but I think the original is wrong
            # and just got lucky (channel and frame flags are mixed up).
            if channel_flags[i] != 0:
                continue
            if sample_flags[frame, channel] != 0:
                continue

            sum_chi2 += channel_weights[i] * (frame_data[frame, channel] ** 2)
            points += 1

        dependents += frame_dependents[frame]

    frame_sum = 0.0
    frame_weight_sum = 0.0

    if (points - dependents) >= 1:
        if sum_chi2 > 0:
            relative_weight = (points - dependents) / sum_chi2
        else:
            relative_weight = 1.0
        dof = 1.0 - (dependents / points)
        for frame in range(frame_from, frame_to):
            if not frame_valid[frame]:
                continue

            # Remove DOF flag for all frames
            frame_flags[frame] &= frame_dof_unflag
            frame_dof[frame] = dof
            frame_weight[frame] = relative_weight
            frame_sum += 1
            frame_weight_sum += relative_weight
    else:
        for frame in range(frame_from, frame_to):
            if not frame_valid[frame]:
                continue

            # Apply DOF flag for all frames
            frame_flags[frame] |= frame_dof_flag
            frame_dof[frame] = 0.0
            frame_weight[frame] = np.nan  # Will be set to 1 when re-normalized

    return frame_sum, frame_weight_sum


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def robust_channel_weights(frame_data, relative_weights, sample_flags,
                           valid_frames, channel_indices):  # pragma: no cover
    """
    Derive robust channel variance and weights.

    The returned weight and variance for a channel `i` is given as:

        var = median(relative_weight * frame_data[:, i] ** 2)
        weight = sum(relative_weight)

    taken over all valid frames for the channel `i` and zero flagged samples
    for the channel `i`.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        An (n_frame, n_channel) array of frame data values.
    relative_weights : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    sample_flags : numpy.ndarray (int)
        The frame data sample flags of shape (n_frames, n_channels).  Only
        frames that are unflagged (flag=0) will be included in the derivation.
    valid_frames : numpy.ndarray (bool)
        An array indicating which frames are valid (`True`) and should be
        included in the derivation.
    channel_indices : numpy.ndarray (int)
        The channel indices to include in the derivation of shape (n,) where
        n <= n_channels.

    Returns
    -------
    variance_sum, variance_weight : numpy.ndarray, numpy.ndarray
        The channel variance sum (variance * weight) and channel variance
        weights, both of shape (n,) and float type.
    """
    n_channels = channel_indices.size
    channel_var_sum = np.zeros(n_channels, dtype=nb.float64)
    channel_var_weight = np.zeros(n_channels, dtype=nb.float64)
    temp = np.empty(valid_frames.sum(), dtype=nb.float64)

    for i, channel_index in enumerate(channel_indices):
        n_valid = 0
        sum_weight = 0.0
        channel_data = frame_data[:, channel_index]
        channel_flags = sample_flags[:, channel_index]
        for frame_index in range(frame_data.shape[0]):
            if (not valid_frames[frame_index]
                    or channel_flags[frame_index] != 0):
                continue
            value = channel_data[frame_index]
            weight = relative_weights[frame_index]
            temp[n_valid] = weight * (value ** 2)
            sum_weight += weight
            n_valid += 1

        if n_valid > 0:
            channel_var_weight[i] = sum_weight
            channel_var_sum[i] = np.median(temp[:n_valid])

    for i in range(n_channels):
        w = channel_var_weight[i]
        if w <= 0:
            channel_var_sum[i] = 0.0
            channel_var_weight[i] = 0.0
            continue
        channel_var_sum[i] /= 0.454937

    return channel_var_sum, channel_var_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def differential_channel_weights(frame_data, relative_weights, sample_flags,
                                 valid_frames, channel_indices,
                                 frame_delta):  # pragma: no cover
    """
    Derive channel weights using the differential method.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    relative_weights : numpy.ndarray (float)
        The frame weights of shape (n_frames,).
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels).  Any non-zero
        sample flags will be excluded from the calculations.
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from all calculations.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to include in the calculations.  Should
        be of shape (n_channels,) where n_channels < all_channels.
    frame_delta : int
        The differential frame offset.

    Returns
    -------
    variance_sum, variance_weight : numpy.ndarray, numpy.ndarray
        The channel variance sum (variance * weight) and channel variance
        weights, both of shape (n_channels,) and float type.
    """
    n_channels = channel_indices.size
    n_frames, all_channels = frame_data.shape
    channel_var_sum = np.zeros(n_channels, dtype=nb.float64)
    channel_var_weight = np.zeros(n_channels, dtype=nb.float64)

    for frame in range(n_frames):
        if not valid_frames[frame]:
            continue
        weight = relative_weights[frame]
        if weight <= 0:
            continue
        prior_frame = frame + frame_delta
        if prior_frame >= n_frames or not valid_frames[prior_frame]:
            continue
        for i, channel in enumerate(channel_indices):
            if sample_flags[frame, channel] != 0:
                continue
            if sample_flags[prior_frame, channel] != 0:
                continue
            dev = frame_data[frame, channel] - frame_data[prior_frame, channel]
            dev *= dev
            channel_var_sum[i] += weight * dev
            channel_var_weight[i] += weight

    for i in range(n_channels):
        w = channel_var_weight[i]
        if w <= 0:
            channel_var_sum[i] = 0.0
            channel_var_weight[i] = 0.0
            continue

    return channel_var_sum, channel_var_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def rms_channel_weights(frame_data, frame_weight, valid_frames, sample_flags,
                        channel_indices):  # pragma: no cover
    """
    Derive channel weights from the RMS of the frame data values.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_weight : numpy.ndarray (float)
        The frame weights of shape (n_frames,).
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels).  Any non-zero
        sample flags will be excluded from the calculations.
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from all calculations.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to include in the calculations.  Should
        be of shape (n_channels,) where n_channels < all_channels.

    Returns
    -------
    variance_sum, variance_weight : numpy.ndarray, numpy.ndarray
        The channel variance sum and channel variance weight sum, both of shape
        (n_channels,) and float type.
    """
    n_frames = frame_data.shape[0]
    n_channels = channel_indices.size
    var_sum = np.empty(n_channels, dtype=nb.float64)
    var_weight = np.empty(n_channels, dtype=nb.float64)

    for i, channel_index in enumerate(channel_indices):
        vw = 0.0
        w = 0.0
        for frame_index in range(n_frames):
            if not valid_frames[frame_index]:
                continue
            if sample_flags[frame_index, channel_index] != 0:
                continue
            weight = frame_weight[frame_index]
            if weight == 0:
                continue
            value = frame_data[frame_index, channel_index]
            if np.isnan(value) or value == 0:
                continue
            vw += weight * value * value
            w += weight
        var_sum[i] = vw
        var_weight[i] = w

    return var_sum, var_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def set_weights_from_var_stats(channel_indices, var_sum, var_weight,
                               base_dependents, base_dof, base_variance,
                               base_weight):  # pragma: no cover
    """
    Set a number of channel data parameters from variance statistics.

    Given calculated variance and variance weights, update channel
    the following channel data parameters to:

    channel_variance = variance
    channel_dof = 1 - (channel_dependents / var_weight)
    channel_weight = channel_dof / channel_variance

    Parameters
    ----------
    channel_indices : numpy.ndarray (int)
        The channel indices of shape (n_channels,) that map the `variance` and
        `var_weight` arrays onto the base channel data indices or
        n_channels -> all_channels.
    var_sum : numpy.ndarray (float)
        The calculated variance sum (weight * variance) for the channel indices
        of shape (n_channels,).  The sum is over all valid frames.
    var_weight : numpy.ndarray (float)
        The calculated variance weights for the channel indices of shape
        (n_channels,).
    base_dependents : numpy.ndarray (float)
        The channel dependents for all channels of shape (all_channels,).
    base_dof : numpy.ndarray (float)
        The channel degrees-of-freedom for all channels of shape
        (all_channels,).  Will be updated in-place.
    base_variance : numpy.ndarray (float)
        The channel variances for all channels of shape (all_channels,).  Will
        be updated in-place.
    base_weight : numpy.ndarray (float)
        The channel weights for all channels of shape (all_channels,).  Will be
        updated in-place.

    Returns
    -------
    None
    """
    for i, channel in enumerate(channel_indices):
        w = var_weight[i]
        if w <= 0:
            continue
        dof = 1.0 - (base_dependents[channel] / w)
        if dof < 0:
            dof = 0.0
        base_dof[channel] = dof
        var = var_sum[i] / w
        base_variance[channel] = var
        if var <= 0:
            base_weight[channel] = 0.0
        else:
            base_weight[channel] = dof / var


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def despike_neighbouring(frame_data, sample_flags, channel_indices,
                         frame_weight, frame_valid, channel_level,
                         delta, spike_flag, exclude_flag):  # pragma: no cover
    """
    A Numba function to despike frame samples using neighbour method.

    This is a fast implementation of the despiking neighbour method.  Please
    see `despike_neighboring` for a detailed explanation.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels).
    channel_indices : numpy.ndarray (int)
        The channel indices to despike of shape (n_channels,).
    frame_weight : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask indicating whether a frame is valid (`True`).
        Invalid frames are not included.
    channel_level : numpy.ndarray (float).
        An array of shape (n_channels,) containing a value for each channel
        indicating the maximum noise level.
    delta : int
        The number of frames of separation defining a neighbour.
    spike_flag : int
        The integer flag specifying a spike.
    exclude_flag : int
        The integer flag specifying a frame that should not be included
        in the despiking.

    Returns
    -------
    number_flagged : int
    """
    n_frames = frame_data.shape[0]
    frame_level = np.empty(n_frames, dtype=nb.float64)

    for frame in range(n_frames):
        # Clear all existing sample spike flags
        for channel in channel_indices:
            if sample_flags[frame, channel] & spike_flag != 0:
                sample_flags[frame, channel] ^= spike_flag

        w0 = frame_weight[frame]
        if not frame_valid[frame] or frame < delta or w0 <= 0:
            frame_level[frame] = np.nan
            continue

        frame_before = frame - delta
        if not frame_valid[frame_before]:
            frame_level[frame] = np.nan
            continue

        w1 = frame_weight[frame_before]
        if w1 <= 0:
            frame_level[frame] = np.nan
            continue

        frame_level[frame] = np.sqrt((1.0 / w0) + (1.0 / w1))

    # Perform the actual despiking
    n_flagged = 0
    for frame in range(n_frames - delta - 1, -1, -1):
        if not frame_valid[frame]:
            continue

        frame_after = frame + delta
        if not frame_valid[frame_after]:
            continue

        for i, channel in enumerate(channel_indices):
            f0 = sample_flags[frame, channel]
            if (f0 & exclude_flag) != 0:
                continue
            f1 = sample_flags[frame_after, channel]
            if (f1 & exclude_flag) != 0:
                continue

            d = frame_data[frame, channel] - frame_data[frame_after, channel]
            d = abs(d)
            threshold = frame_level[frame_after] * channel_level[i]
            if (f1 & spike_flag) == 0:
                if d > threshold:
                    sample_flags[frame, channel] |= spike_flag
                    sample_flags[frame_after, channel] |= spike_flag
                    n_flagged += 1
            elif d <= threshold:
                # Exonerate the prior spike if possible
                sample_flags[frame_after, channel] ^= spike_flag

    return n_flagged


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def despike_absolute(frame_data, sample_flags, channel_indices, frame_weight,
                     frame_valid, channel_level, spike_flag,
                     exclude_flag):  # pragma: no cover
    """
    A Numba function to despike frame samples using absolute method.

    This is a fast implementation of the despiking absolute method.  Please
    see `despike_absolute` for a detailed explanation.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels).
    channel_indices : numpy.ndarray (int)
        The channel indices to despike of shape (n_channels,).
    frame_weight : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask indicating whether a frame is valid (`True`).
        Invalid frames are not included.
    channel_level : numpy.ndarray (n_channels,).
        An array containing a value for each channel indicating the maximum
        noise level.
    spike_flag : int
        The integer flag specifying a spike.
    exclude_flag : int
        The integer flag specifying a frame that should not be included
        in the despiking.

    Returns
    -------
    number_flagged : int
    """
    n_flagged = 0
    n_frames = frame_data.shape[0]

    for frame in range(n_frames):

        if not frame_valid[frame]:
            # Ignore if invalid frame
            continue

        weight = frame_weight[frame]

        if weight <= 0:  # Infinite noise
            for channel in channel_indices:
                n_flagged += 1
                sample_flags[frame, channel] |= spike_flag
            continue

        chi = np.sqrt(1.0 / weight)
        for i, channel in enumerate(channel_indices):
            if (sample_flags[frame, channel] & exclude_flag) != 0:
                # Unflag if excluded
                if (sample_flags[frame, channel] & spike_flag) != 0:
                    sample_flags[frame, channel] ^= spike_flag
            elif abs(frame_data[frame, channel]) > (channel_level[i] * chi):
                # Flag
                sample_flags[frame, channel] |= spike_flag
                n_flagged += 1
            elif (sample_flags[frame, channel] & spike_flag) != 0:
                # Unflag
                sample_flags[frame, channel] ^= spike_flag

    return n_flagged


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def despike_gradual(frame_data, sample_flags, channel_indices, frame_weight,
                    frame_valid, channel_level, spike_flag, source_blank_flag,
                    exclude_flag, channel_gain, depth):  # pragma: no cover
    """
    A Numba function to despike frame samples using gradual method.

    This is a fast implementation of the despiking gradual method.  Please
    see `despike_gradual` for a detailed explanation.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, n_channels).
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, n_channels).
    channel_indices : numpy.ndarray (int)
        The channel indices to despike of shape (n_channel_indices,).
    frame_weight : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask indicating whether a frame is valid (`True`).
        Invalid frames are not included.
    channel_level : numpy.ndarray (n_channel_indices,).
        An array containing a value for each channel indicating the maximum
        noise level.
    spike_flag : int
        The integer flag specifying a spike.
    source_blank_flag : int
        The integer flag specifying a blank source.
    exclude_flag : int
        The integer flag specifying a sample that should not be included
        in the despiking.
    channel_gain : numpy.ndarray (float)
        An array of channel gains with shape (n_channel_indices,).
    depth : float
        A factor between 0 and 1 that defines the maximum allowable data
        value as a fraction of the maximum channel gain.

    Returns
    -------
    number_flagged : int
    """
    n_flagged = 0
    n_frames = frame_data.shape[0]
    for frame in range(n_frames):

        if not frame_valid[frame]:
            continue

        weight = frame_weight[frame]

        if weight <= 0:  # Infinite noise
            for channel in channel_indices:
                n_flagged += 1
                sample_flags[frame, channel] |= spike_flag
            continue

        # Find the largest unflagged spike deviation
        maximum_deviation = 0.0
        for i, channel in enumerate(channel_indices):
            # Unflag all channels
            flag = sample_flags[frame, channel]
            if (flag & spike_flag) != 0:
                flag ^= spike_flag
                sample_flags[frame, channel] = flag

            if (flag & exclude_flag) == 0:
                deviation = abs(frame_data[frame, channel] / channel_gain[i])
                if deviation > maximum_deviation:
                    maximum_deviation = deviation

        if maximum_deviation > 0:
            chi = 1.0 / np.sqrt(weight)
            minimum_signal = depth * maximum_deviation
            for i, channel in enumerate(channel_indices):
                if (sample_flags[frame, channel] & source_blank_flag) != 0:
                    continue
                critical = max(channel_gain[i] * minimum_signal,
                               channel_level[i] * chi)
                if frame_data[frame, channel] > critical:
                    sample_flags[frame, channel] |= spike_flag
                    n_flagged += 1

    return n_flagged


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def despike_multi_resolution(timestream_data, timestream_weight, sample_flags,
                             channel_indices, frame_valid, level, spike_flag,
                             max_block_size):  # pragma: no cover
    """
    A Numba function to despike frame samples using multi-resolution method.

    This is a fast implementation of the despiking resolution method.
    Please see `despike_multi_resolution` for a detailed explanation.

    Parameters
    ----------
    timestream_data : numpy.ndarray (float)
        The time-stream data of shape (n_frames, channel_indices.size)
        containing sample_data * relative_weight values.  Invalid
        samples have 0 values.
    timestream_weight : numpy.ndarray (float)
        The time-stream weights of shape (n_frames, channel_indices.size)
        containing relative_weight values.  Invalid samples have 0 values.
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, n_channels).
    channel_indices : numpy.ndarray (int)
        The channel indices for which to perform the despiking.  An array
        of shape (channel_indices.size,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` indicates that
        a frame is valid and may be included in the calculation.
    level : float
        The sigma level above which spike flagging occurs.
    spike_flag : int
        The spike flag integer identifier.
    max_block_size : int
        The maximum block size which determines the minimum resolution.
        This should be no larger than n_frames // 2 and no smaller than 1.

    Returns
    -------
    n_flagged : int
        The number of samples flagged as spikes.
    """
    n_frames = timestream_data.shape[0]
    for frame_index in range(n_frames):
        for i, channel_index in enumerate(channel_indices):
            # Clear all existing spike flags
            flag = sample_flags[frame_index, channel_index]
            if flag & spike_flag != 0:
                sample_flags[frame_index, channel_index] ^= spike_flag

    n_resolutions = int(np.log2(max_block_size)) + 1
    resolutions = 2 ** (np.arange(n_resolutions))

    resolution_max_frame = n_frames
    for resolution in resolutions:
        for i, channel_index in enumerate(channel_indices):
            v1 = timestream_data[0, i]
            w1 = timestream_weight[0, i]
            for frame_index in range(1, resolution_max_frame):
                new_frame = frame_index // 2
                if not frame_valid[frame_index] or not frame_valid[new_frame]:
                    continue
                v2 = timestream_data[frame_index, i]
                w2 = timestream_weight[frame_index, i]
                v_sum = v1 + v2
                v_diff = v1 - v2
                w_sum = w1 * w2
                v1 = v2
                w1 = w2
                if w_sum != 0:
                    w_sum /= (w1 + w2)
                else:
                    w_sum = 0.0
                timestream_data[new_frame, i] = v_sum
                timestream_weight[new_frame, i] = w_sum
                if w_sum <= 0:
                    continue

                significance = np.abs(v_diff) * np.sqrt(w_sum)
                if significance > level:
                    end_index = frame_index * resolution
                    start_index = end_index - resolution
                    for new_i in range(start_index, end_index):
                        if not frame_valid[new_i]:
                            continue
                        sample_flags[new_i, channel_index] |= spike_flag

        resolution_max_frame //= 2

    n_flagged = 0
    for frame_index in range(n_frames):
        for channel_index in channel_indices:
            if sample_flags[frame_index, channel_index] & spike_flag != 0:
                n_flagged += 1

    return n_flagged


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def flagged_channels_per_frame(sample_flags, flag, valid_frames,
                               channel_indices):  # pragma: no cover
    """
    Return the total number of flagged channels per frame.

    Parameters
    ----------
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, n_channels).
    flag : int
        The flag to check.
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames), where `True` indicates a valid
        frame that may be included in the calculations.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the number of flagged
        frames of shape (channel_indices.size,).

    Returns
    -------
    flagged_channels : numpy.ndarray (int)
        An array of shape (n_frames,) containing the number of channels
        flagged with `flag`.
    """
    n_frames = sample_flags.shape[0]

    flagged_channels = np.empty(n_frames, dtype=nb.int64)
    for frame_index in range(n_frames):
        flagged_channels[frame_index] = 0
        if not valid_frames[frame_index]:
            continue
        for channel_index in channel_indices:
            if (sample_flags[frame_index, channel_index] & flag) != 0:
                flagged_channels[frame_index] += 1

    return flagged_channels


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def flagged_frames_per_channel(sample_flags, flag, valid_frames,
                               channel_indices):  # pragma: no cover
    """
    Return the total number of flagged frames per channel.

    Parameters
    ----------
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, n_channels).
    flag : int
        The flag to check.
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames), where `True` indicates a valid
        frame that may be included in the calculations.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the number of flagged
        frames of shape (channel_indices.size,).

    Returns
    -------
    flagged_frames : numpy.ndarray (int)
        An array of shape (channel_indices.size,) containing the number
        of frames flagged with `flag`.
    """
    n_frames = sample_flags.shape[0]
    n_channels = channel_indices.size

    flagged_frames = np.empty(n_channels, dtype=nb.int64)
    for i, channel_index in enumerate(channel_indices):
        flagged_frames[i] = 0
        for frame_index in range(n_frames):
            if not valid_frames[frame_index]:
                continue

            if (sample_flags[frame_index, channel_index] & flag) != 0:
                flagged_frames[i] += 1

    return flagged_frames


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def frame_block_expand_flag(sample_flags, valid_frames, flag, block_size,
                            channel_indices):  # pragma: no cover
    """
    If a single frame is flagged in a block, flag all frames in the block.

    A block is a division of the entire frame space into chunks of size
    `block_size`.  If any frame for a given channel is flagged as `flag`,
    all frames in that block for the channel will also be flagged as `flag`.

    Parameters
    ----------
    sample_flags : numpy.ndarray (int)
        The frame sample flags of shape (n_frames, n_channels).
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` indicates that
        a frame may be included in any calculations.
    flag : int
        The flag to check and expand.
    block_size : int
        The size of the "block" or the number of frames in a single division
        of the entire frame space.
    channel_indices : numpy.ndarray (int)
        The channel indices to check.

    Returns
    -------
    None
    """
    n_frames = sample_flags.shape[0]
    max_blocks = numba_functions.roundup_ratio(n_frames, block_size)

    for block in range(max_blocks):
        start_index = block * block_size
        end_index = start_index + block_size
        if end_index > n_frames:
            end_index = n_frames
        for frame_index in range(start_index, end_index):
            if not valid_frames[frame_index]:
                continue
            for channel_index in channel_indices:
                if (sample_flags[frame_index, channel_index] & flag) != 0:
                    for index in range(start_index, end_index):
                        if not valid_frames[frame_index]:
                            continue
                        sample_flags[index, channel_index] |= flag
                    continue


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def next_weight_transit(frame_weights, level, frame_valid, frame_flags,
                        time_weighting_flags, start_frame=0,
                        above=True):  # pragma: no cover
    """
    Returns the next frame weight transit above or below a given level.

    Parameters
    ----------
    frame_weights : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    level : float
        The level for at which to find the next frame above or below this
        threshold.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where any invalid (False) frame
        will be skipped.
    frame_flags : numpy.ndarray (int)
        The frame flag array of shape (n_frames,).  Any frame flagged with the
        TIME_WEIGHTING_FLAGS frame flag will be skipped.
    time_weighting_flags : int
        The integer flag marking the TIME_WEIGHTING_FLAGS value.
    start_frame : int, optional
        The frame from which to begin looking for the next transit (inclusive).
    above : bool, optional
        If `True`, looks for the next weight transit above `level`.  If
        `False`, looks for the next weight transit below `level`.

    Returns
    -------
    frame : int
         The next weight transit.  If not found, a value of -1 is returned.
    """
    n_frames = frame_weights.size
    for frame in range(start_frame, n_frames):
        if not frame_valid[frame]:
            continue
        weight = frame_weights[frame]
        if weight <= 0 or np.isnan(weight):
            continue
        if (frame_flags[frame] & time_weighting_flags) != 0:
            continue
        if above:
            if weight > level:
                return frame
        else:
            if weight < level:
                return frame
    return -1


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_mean_frame_level(frame_data, frame_weights, frame_valid,
                         modeling_frames, sample_flags, channel_indices,
                         start_frame=None, stop_frame=None,
                         robust=False):  # pragma: no cover
    """
    Return the mean frame values for each channel.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        all processing.
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a flag as a
        modelling frame which will not be included when determining the mean.
    sample_flags : numpy.ndarray (int)
        The frame data flag mask of shape (n_frames, all_channels) where any
        non-zero value excludes a sample from inclusion in the mean
        calculation.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the mean of shape
        (n_channels,).
    start_frame : int, optional
        The start frame from which to calculate the mean.  The default is the
        first frame (0).
    stop_frame : int, optional
        The stop frame (non-inclusive) at which to terminate the mean
        calculation.  The default is the total number of frames (n_frames).
    robust : bool, optional
        If `True`, use a weighted median averaging calculation instead of a
        weighted mean.

    Returns
    -------
    mean_value, mean_weight : numpy.ndarray (float), numpy.ndarray (float)
        The mean frame values and weights for each channel.
    """
    if robust:
        return weighted_median_frame_level(
            frame_data=frame_data,
            frame_weights=frame_weights,
            frame_valid=frame_valid,
            modeling_frames=modeling_frames,
            sample_flags=sample_flags,
            channel_indices=channel_indices,
            start_frame=start_frame,
            stop_frame=stop_frame)
    else:
        return weighted_mean_frame_level(
            frame_data=frame_data,
            frame_weights=frame_weights,
            frame_valid=frame_valid,
            modeling_frames=modeling_frames,
            sample_flags=sample_flags,
            channel_indices=channel_indices,
            start_frame=start_frame,
            stop_frame=stop_frame)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def weighted_mean_frame_level(frame_data, frame_weights, frame_valid,
                              modeling_frames, sample_flags,
                              channel_indices, start_frame=None,
                              stop_frame=None):  # pragma: no cover
    """
    Return the mean frame data values for each channel.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        all processing.
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a flag as a
        modelling frame which will not be included when determining the mean.
    sample_flags : numpy.ndarray (int)
        The frame data flag mask of shape (n_frames, all_channels) where any
        non-zero value excludes a sample from inclusion in the mean
        calculation.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the mean of shape
        (n_channels,).
    start_frame : int, optional
        The start frame from which to calculate the mean.  The default is the
        first frame (0).
    stop_frame : int, optional
        The stop frame (non-inclusive) at which to terminate the mean
        calculation.  The default is the total number of frames (n_frames).

    Returns
    -------
    mean_value, mean_weight : numpy.ndarray (float), numpy.ndarray (float)
        The mean value and weight for each channel both of shape (n_channels,).
    """
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = frame_data.shape[0]

    n_channels = channel_indices.size

    mean_value = np.empty(n_channels, dtype=nb.float64)
    mean_weight = np.empty(n_channels, dtype=nb.float64)

    for i, channel in enumerate(channel_indices):
        cw, cwd = 0.0, 0.0
        same_values = True  # for floating point errors
        first_value = np.nan

        for frame in range(start_frame, stop_frame):
            if not frame_valid[frame] or modeling_frames[frame]:
                continue
            w = frame_weights[frame]
            if w == 0:
                continue
            if sample_flags[frame, channel] != 0:
                continue
            d = frame_data[frame, channel]
            wd = d if w == 1 else w * d

            if same_values and d != first_value:
                if np.isnan(first_value):
                    first_value = d
                else:
                    same_values = False

            cw += w
            cwd += wd

        if cw > 0:
            mean_weight[i] = cw
            if same_values:
                mean_value[i] = first_value
            else:
                mean_value[i] = cwd / cw
        else:
            mean_value[i] = cwd
            mean_weight[i] = 0.0

    return mean_value, mean_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def weighted_median_frame_level(frame_data, frame_weights, frame_valid,
                                modeling_frames, sample_flags,
                                channel_indices, start_frame=None,
                                stop_frame=None):  # pragma: no cover
    """
    Return the median frame data values for each channel.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        all processing.
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a flag as a
        modelling frame which will not be included when determining the mean.
    sample_flags : numpy.ndarray (int)
        The frame data flag mask of shape (n_frames, all_channels) where any
        non-zero value excludes a sample from inclusion in the mean
        calculation.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the mean of shape
        (n_channels,).
    start_frame : int, optional
        The start frame from which to calculate the mean.  The default is the
        first frame (0).
    stop_frame : int, optional
        The stop frame (non-inclusive) at which to terminate the mean
        calculation.  The default is the total number of frames (n_frames).

    Returns
    -------
    mean_value, mean_weight : numpy.ndarray (float), numpy.ndarray (float)
        The mean value and weight for each channel both of shape (n_channels,).
    """
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = frame_data.shape[0]

    n_frames = stop_frame - start_frame
    n_channels = channel_indices.size
    mean_value = np.empty(n_channels, dtype=nb.float64)
    mean_weight = np.empty(n_channels, dtype=nb.float64)
    frame_indices = np.empty(n_frames, dtype=nb.int64)
    weight_buffer = np.empty(n_frames, dtype=nb.float64)

    n_valid = 0
    for frame in range(start_frame, stop_frame):
        if not frame_valid[frame] or modeling_frames[frame]:
            continue
        w = frame_weights[frame]
        if w == 0:
            continue
        frame_indices[n_valid] = frame
        weight_buffer[n_valid] = w
        n_valid += 1

    value_buffer = np.empty(n_valid, dtype=nb.float64)
    weight_buffer = weight_buffer[:n_valid]
    frame_indices = frame_indices[:n_valid]

    for i, channel in enumerate(channel_indices):
        for j, frame in enumerate(frame_indices):
            if sample_flags[frame, channel] != 0:
                value_buffer[j] = np.nan
                continue
            value_buffer[j] = frame_data[frame, channel]

        m, w = numba_functions.smart_median_1d(
            value_buffer, weights=weight_buffer, max_dependence=0.25)
        mean_value[i] = m
        mean_weight[i] = w

    return mean_value, mean_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def remove_channel_drifts(frame_data, frame_weights, frame_valid,
                          modeling_frames, sample_flags, drift_frame_size,
                          channel_filtering, frame_dependents,
                          channel_dependents,
                          channel_indices, robust=False):  # pragma: no cover
    """
    Calculate and remove the average offset from each channel.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).  Will be updated
        in-place.
    frame_weights : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        all processing.
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a frame as
        "modeling".  Modeling frames will not be used to calculate the average
        channel offsets, but will have the average offset removed.  Modeling
        frames will not contribute to dependents.
    sample_flags : numpy.ndarray (int)
        An array of integer flags of shape (n_frames, all_channels) where
        non-zero values will not contribute to dependents or the average offset
        calculation.
    drift_frame_size : int
        The size of each block of frames for which to calculate the average
        channel values which will then be removed from that block.
    channel_filtering : numpy.ndarray (float)
        The channel filtering factor of shape (n_channels,).
    frame_dependents : numpy.ndarray (float)
        The frame dependents of shape (n_frames,).  Will be updated in-place.
    channel_dependents : numpy.ndarray (float)
        The channel dependents of shape (all_channels,).  Will be updated
        in-place.
    channel_indices : numpy.ndarray (int)
        The channels for which to calculate and subtract the average offsets.
        Should be of shape (n_channels,) and map n_channels -> all_channels.
    robust : bool, optional
        If `True`, use the robust method (median) to calculate the average
        channel offset.  Otherwise, use the weighted mean.

    Returns
    -------
    average_offset, offset_weights : numpy.ndarray, numpy.ndarray
        The average channel offset (over all drifts) and associated weight
        sums. Both are float arrays of shape (n_channels,).
    """
    n_frames = frame_data.shape[0]
    n_channels = channel_indices.size
    average_offset = np.zeros(n_channels, dtype=nb.float64)
    average_offset_weight = np.zeros(n_channels, dtype=nb.float64)

    for start_frame in range(0, n_frames, drift_frame_size):
        stop_frame = start_frame + drift_frame_size
        if stop_frame > n_frames:
            stop_frame = n_frames

        drifts, drift_weights = get_mean_frame_level(
            frame_data=frame_data,
            frame_weights=frame_weights,
            frame_valid=frame_valid,
            modeling_frames=modeling_frames,
            sample_flags=sample_flags,
            channel_indices=channel_indices,
            start_frame=start_frame,
            stop_frame=stop_frame,
            robust=robust)

        average_offset += drifts * drift_weights
        average_offset_weight += drift_weights

        level(frame_data=frame_data,  # updated here
              frame_weights=frame_weights,
              frame_valid=frame_valid,
              modeling_frames=modeling_frames,
              sample_flags=sample_flags,
              channel_indices=channel_indices,
              start_frame=start_frame,
              stop_frame=stop_frame,
              offset=drifts,
              offset_weight=drift_weights,
              frame_dependents=frame_dependents,  # updated here
              channel_filtering=channel_filtering)

        for i, channel in enumerate(channel_indices):
            if drift_weights[i] > 0:
                channel_dependents[channel] += 1

    for i in range(n_channels):
        w = average_offset_weight[i]
        if w > 0:
            average_offset[i] /= w

    return average_offset, average_offset_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def level(frame_data, frame_weights, frame_valid, modeling_frames,
          sample_flags, channel_indices, start_frame, stop_frame,
          offset, offset_weight, frame_dependents,
          channel_filtering):  # pragma: no cover
    """
    Subtract an offset per channel from the frame data between given frames.

    The frame dependents will also be updated by this operation.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).  Will be updated
        in-place.
    frame_weights : numpy.ndarray (float)
        The frame weights of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        all processing.
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a flag as a
        modelling frame which will not be included when determining the mean.
    sample_flags : numpy.ndarray (int)
        The frame data flag mask of shape (n_frames, all_channels) where any
        non-zero value excludes a sample from inclusion in the mean
        calculation.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate the mean of shape
        (n_channels,).
    start_frame : int
        The start frame from which to calculate the mean.  The default is the
        first frame (0).
    stop_frame : int
        The stop frame (non-inclusive) at which to terminate the mean
        calculation.  The default is the total number of frames (n_frames).
    offset : numpy.ndarray (float)
        The offsets of shape (n_channels,) to remove from frame data between
        the start and stop frame for the given channels.
    offset_weight : numpy.ndarray (float)
        The offset weights of shape (n_channels,) used to update the frame
        dependents.
    frame_dependents : numpy.ndarray (float)
        The frame dependents of shape (n_frames,).  Will be updated in-place.
    channel_filtering : numpy.ndarray (float)
        The channel filtering factor of shape (n_channels,).

    Returns
    -------
    None
    """
    n_channels = channel_indices.size
    p_norm = np.empty(n_channels, dtype=nb.float64)
    for i in range(n_channels):
        w = offset_weight[i]
        if w == 0:
            p_norm[i] = 0.0
        else:
            p_norm[i] = channel_filtering[i] / w

    for frame in range(start_frame, stop_frame):
        if not frame_valid[frame]:
            continue
        for i, channel in enumerate(channel_indices):
            frame_data[frame, channel] -= offset[i]

        if modeling_frames[frame]:
            continue

        fw = frame_weights[frame]
        if fw == 0:
            continue

        for i, channel in enumerate(channel_indices):
            if sample_flags[frame, channel] != 0:
                continue
            frame_dependents[frame] += fw * p_norm[i]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def apply_drifts_to_channel_data(channel_indices, offsets, average_drifts,
                                 inconsistencies, hardware_gain,
                                 filter_time_scale, source_filtering,
                                 integration_filter_time_scale,
                                 crossing_time, is_detector_stage,
                                 update_filtering):  # pragma: no cover
    """
    Apply the average drifts to channel data.

    After calculating the average channel drifts for all channels, the `offset`
    attribute of the channel data is modified by::

       offset += average_drifts * G

    where G = 1 if the detector is not staged (`is_detector_stage`=`False`) or
    set to `hardware_gain` if the detector is staged.

    If `update_filtering` is `True`, the source filtering will be updated by::

       sf *= 1 - (crossing_time / filter_time_scale)

    Note that the previous correction will be removed first.  Also, the
    filtering_time_scale will be set to the minimum of the integration
    filtering time scale and the current filtering time scale.

    Parameters
    ----------
    channel_indices : numpy.ndarray (int)
        The channel indices for which to apply the drifts to of shape
        (n_channels,).  Should map n_channels -> all_channels.
    offsets : numpy.ndarray (float)
        The channel gain offsets of shape (all_channels,).  Will be updated
        in-place.
    average_drifts : numpy.ndarray (float)
        The average drifts to apply to the channel offsets of shape
        (n_channels,).
    inconsistencies : numpy.ndarray (int)
        The channel inconsistencies of shape (all_channels,).
    hardware_gain : numpy.ndarray (float)
        The channel hardware gains of shape (all_channels,).
    filter_time_scale : numpy.ndarray (float)
        The filter time scale for all channels of shape (all_channels,).  Will
        be updated in-place.
    source_filtering : numpy.ndarray (float)
        The channel source filtering of shape (all_channels,).  Will be updated
        in-place.
    integration_filter_time_scale : float
        The integration filter time scale.
    crossing_time : float
        The point-crossing time for the integration.
    is_detector_stage : bool
        If `True` indicates that the detector is staged and hardware gains
        should be applied.
    update_filtering : bool
        If `True` updates the source filtering and filter time scale for the
        given channels.  This should be set to `False` if the drift block size
        (number of frames in a single drift) is greater than the total number
        of frames in the integration.

    Returns
    -------
    inconsistent_channels, total_inconsistencies : int, int
        The total number of channels containing one or more inconsistencies,
        and the total number of inconsistencies in the given channels.
    """
    total_inconsistencies = 0
    inconsistent_channels = 0

    for i, channel in enumerate(channel_indices):
        if is_detector_stage:
            offsets[channel] += hardware_gain[channel] * average_drifts[i]
        else:
            offsets[channel] += average_drifts[i]

        channel_inconsistency = inconsistencies[channel]
        if channel_inconsistency > 0:
            inconsistent_channels += 1
        total_inconsistencies += channel_inconsistency

        if not update_filtering:
            continue

        # set source filtering to zero if the filter time scale is <= 0
        # or the crossing time is unreasonable;
        # otherwise leave unchanged
        channel_fts = filter_time_scale[channel]
        if ((np.isfinite(channel_fts) and channel_fts <= 0)
                or not np.isfinite(crossing_time)):
            source_filtering[channel] = 0.0

        if integration_filter_time_scale < channel_fts:
            filter_time_scale[channel] = integration_filter_time_scale

    return inconsistent_channels, total_inconsistencies


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def detector_stage(frame_data, frame_valid, channel_indices,
                   channel_hardware_gain):  # pragma: no cover
    """
    Stage the detector by applying channel hardware gains to the frame data.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        processing.
    channel_indices : numpy.ndarray (int)
        The channel indices to which `channel_hardware_gain` values apply of
        shape (n_channels,).  Should map n_channels -> all_channels.
    channel_hardware_gain : numpy.ndarray (float)
        The channel hardware gains of shape (n_channels,).

    Returns
    -------
    None
    """
    for frame in range(frame_valid.size):
        if not frame_valid[frame]:
            continue
        for i, channel in enumerate(channel_indices):
            frame_data[frame, channel] /= channel_hardware_gain[i]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def readout_stage(frame_data, frame_valid, channel_indices,
                  channel_hardware_gain):  # pragma: no cover
    """
    Unstage the detector by removing channel hardware gains to the frame data.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        processing.
    channel_indices : numpy.ndarray (int)
        The channel indices to which `channel_hardware_gain` values apply of
        shape (n_channels,).  Should map n_channels -> all_channels.
    channel_hardware_gain : numpy.ndarray (float)
        The channel hardware gains of shape (n_channels,).

    Returns
    -------
    None
    """
    for frame in range(frame_valid.size):
        if not frame_valid[frame]:
            continue
        for i, channel in enumerate(channel_indices):
            frame_data[frame, channel] *= channel_hardware_gain[i]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def search_corners(sample_coordinates, valid_frames, channel_indices,
                   sample_flags, skip_flag):  # pragma: no cover
    """
    Return the x, y range of sample coordinates.

    Invalid sample coordinates (e.g. NaN) will flag a sample as SAMPLE_SKIP.

    Parameters
    ----------
    sample_coordinates : numpy.ndarray (float)
        The sample (x, y) coordinates of shape (2, n_frames, n_channels).
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from all calculations.
    channel_indices : numpy.ndarray (int)
        An array of shape (n_channels,) mapping n_channels onto
        all_channels.
    sample_flags : numpy.ndarray (int)
        An array containing the sample flags of shape
        (n_frames, all_channels).
    skip_flag : int
        The integer flag that will flag a sample if it's sample coordinate
        is invalid.

    Returns
    -------
    map_range : numpy.ndarray (float)
        An array of shape (4,) containing [min(x), min(y), max(x), max(y)].
    """
    n_dimensions, n_frames, n_channels = sample_coordinates.shape
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    result = np.empty((2, 2), dtype=nb.float64)

    for frame_index in range(n_frames):
        if not valid_frames[frame_index]:
            continue
        for i, channel_index in enumerate(channel_indices):
            x = sample_coordinates[0, frame_index, i]
            y = sample_coordinates[1, frame_index, i]
            if not np.isfinite(x) or not np.isfinite(y):
                sample_flags[frame_index, channel_index] |= skip_flag
                continue
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    result[0] = min_x, max_x
    result[1] = min_y, max_y
    return result


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_weighted_timestream(sample_data, sample_flags, frame_valid,
                            frame_weights,
                            channel_indices):  # pragma: no cover
    """
    Given sample data and flags, return weighted values, and full weights.

    The output data will only contain certain channel indices and contains
    values of `sample_data * frame_weights`.  Invalid frames or nonzero
    sample flags will result in zero weights and data in the output arrays.

    Parameters
    ----------
    sample_data : numpy.ndarray (float)
        Sample data of shape (n_frames, n_channels)
    sample_flags : numpy.ndarray (int)
        Sample flags of shape (n_frames, n_channels)
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` indicates a valid
        frame.
    frame_weights : numpy.ndarray (float)
        An array of frame weights of shape (n_frames,).
    channel_indices : numpy.ndarray (int)
        The channel indices to include in the output arrays.

    Returns
    -------
    data, weight : numpy.ndarray (float), numpy.ndarray (float)
        The data and weight arrays of shape
        (n_frames, channel_indices.size).
    """
    n_frames = sample_data.shape[0]
    n_channels = channel_indices.size
    data = np.zeros((n_frames, n_channels), dtype=nb.float64)
    weight = np.zeros((n_frames, n_channels), dtype=nb.float64)

    for frame_index in range(n_frames):
        if not frame_valid[frame_index]:
            continue

        for i, channel_index in enumerate(channel_indices):
            if sample_flags[frame_index, channel_index] != 0:
                continue
            data[frame_index, i] = (frame_weights[frame_index]
                                    * sample_data[frame_index, channel_index])
            weight[frame_index, i] = frame_weights[frame_index]
    return data, weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calculate_2d_velocities(coordinates, dt):  # pragma: no cover
    """
    Calculate the velocities for (x, y) coordinates.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The (x, y) coordinates of shape (2, N).
    dt : float
        The sampling interval.

    Returns
    -------
    velocity : numpy.ndarray (float)
        The output velocity of shape (2, N) containing the (x, y)
        velocities.
    """
    n = coordinates.shape[1]
    velocities = np.empty((2, n), dtype=nb.float64)
    i2dt = 0.5 / dt
    x = coordinates[0]
    y = coordinates[1]

    valid = np.empty(n, dtype=nb.b1)
    for i in range(n):
        if np.isnan(x[i]) or np.isnan(y[i]):
            valid[i] = False
        else:
            valid[i] = True

    for i in range(1, n - 1):
        im1 = i - 1
        ip1 = i + 1
        if not valid[im1] or not valid[ip1]:
            for dimension in range(2):
                velocities[dimension, i] = np.nan
            continue

        velocities[0, i] = (x[ip1] - x[im1]) * i2dt
        velocities[1, i] = (y[ip1] - y[im1]) * i2dt

    if n > 1:
        velocities[:, 0] = velocities[:, 1]
    if n > 2:
        velocities[:, -1] = velocities[:, -2]

    return velocities


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calculate_2d_accelerations(coordinates, dt):  # pragma: no cover
    """
    Calculate the accelerations for (x, y) coordinates.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The (x, y) coordinates of shape (2, N) in radians.
    dt : float
        The sampling interval.

    Returns
    -------
    acceleration : numpy.ndarray (float)
        The output acceleration of shape (2, N) containing the (x, y)
        accelerations.
    """
    n = coordinates.shape[1]
    valid = np.empty(n, dtype=nb.b1)
    for i in range(n):
        if np.isnan(coordinates[0, i]):
            valid[i] = False
        elif np.isnan(coordinates[1, i]):
            valid[i] = False
        else:
            valid[i] = True

    accelerations = np.empty((2, n), dtype=nb.float64)
    idt = 1.0 / dt
    two_pi = 2 * np.pi
    x = coordinates[0]
    y = coordinates[1]
    for i in range(1, n - 1):
        im1 = i - 1
        ip1 = i + 1
        if not valid[i] or not valid[im1] or not valid[ip1]:
            for dimension in range(2):
                accelerations[dimension, i] = np.nan
            continue
        ax = np.cos(y[i]) * np.fmod(x[im1] + x[ip1] - (2.0 * x[i]), two_pi)
        ay = y[im1] + y[ip1] - (2.0 * y[i])
        accelerations[0, i] = ax * idt
        accelerations[1, i] = ay * idt

    if n > 1:
        accelerations[:, 0] = accelerations[:, 1]
    if n > 2:
        accelerations[:, -1] = accelerations[:, -2]

    return accelerations


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def classify_scanning_speeds(speeds, min_speed, max_speed, valid,
                             strict):  # pragma: no cover
    """
    Classify scanning speeds.

    Speeds will be classified as keep, cut, or flag.  The valid frames will
    also be updated in-place if using the strict schema or if non-finite
    speeds are encountered.

    Parameters
    ----------
    speeds : numpy.ndarray (float)
        The scanning speeds.
    min_speed : float
        The minimum allowable scanning speed.
    max_speed : float
        The maximum allowable scanning speed.
    valid : numpy.ndarray (bool)
        An array marking a speed as invalid which will be ignored.  Will be
        updated in-place if `strict` is `True`.
    strict : bool
        If `True`, speeds outside the allowable range will be cut.  If `False`,
        they will be flagged instead.

    Returns
    -------
    keep, cut, flag : 3-tuple (numpy.ndarray (int))
        The indices to keep, cut, or flag.
    """
    n = speeds.size
    keep = np.empty(n, dtype=nb.int64)
    cut = np.empty(n, dtype=nb.int64)
    flag = np.empty(n, dtype=nb.int64)
    n_keep = n_cut = n_flag = 0

    for frame in range(speeds.size):
        if not valid[frame]:
            continue
        speed = speeds[frame]

        if not np.isfinite(speed):
            cut[n_cut] = frame
            valid[frame] = False
            n_cut += 1
            continue

        if speed < min_speed or speed > max_speed:
            if strict:
                cut[n_cut] = frame
                n_cut += 1
                valid[frame] = False
            else:
                flag[n_flag] = frame
                n_flag += 1
        else:
            keep[n_keep] = frame
            n_keep += 1

    return keep[:n_keep], cut[:n_cut], flag[:n_flag]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def smooth_positions(coordinates, bin_size,
                     fill_value=np.nan):  # pragma: no cover
    """
    Smooth (x, y) coordinates with a box kernel of given size.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The (x, y) coordinates of shape (2, n).
    bin_size : int
        The size of the smoothing kernel.
    fill_value : float, optional
        The value with which to fill in the instance that there
        are insufficient points to determine a smoothed value.

    Returns
    -------
    smoothed : numpy.ndarray (float)
        The smoothed (x, y) coordinates.
    """
    if bin_size < 2:
        return coordinates
    n = coordinates.shape[1]
    n_m = np.right_shift(bin_size, 1)
    n_p = bin_size - n_m
    tot = n - n_p + 1
    smoothed = np.full((2, n), fill_value, dtype=nb.float64)
    sum_x = 0.0
    sum_y = 0.0
    valid = np.empty(n, dtype=nb.b1)

    n_valid = bin_size
    for i in range(n):
        x, y = coordinates[0, i], coordinates[1, i]
        invalid = np.isnan(x) or np.isnan(y)
        valid[i] = ~invalid
        if i < bin_size:
            if invalid:
                n_valid -= 1
            else:
                sum_x += x
                sum_y += y

    for i in range(n_m, tot):

        if n_valid > 0:
            smoothed[0, i] = sum_x / n_valid
            smoothed[1, i] = sum_y / n_valid

        i_p = i + n_p
        if i_p >= n:
            break

        if not valid[i_p]:
            n_valid -= 1
        else:
            sum_x += coordinates[0, i_p]
            sum_y += coordinates[1, i_p]

        i_m = i - n_m
        if not valid[i_m]:
            n_valid += 1
        else:
            sum_x -= coordinates[0, i_m]
            sum_y -= coordinates[1, i_m]

    return smoothed


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_covariance(frame_data, frame_valid, frame_weight,
                   channel_flags, channel_weight,
                   sample_flags, frame_flags,
                   source_flags):  # pragma: no cover
    """
    Return the channel covariance.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_valid : numpy.ndarray (bool)
        A mask of shape (n_frames,) where `False` excludes a frame from
        processing.
    frame_weight : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,).
    channel_flags : numpy.ndarray (int)
        An integer mask of shape (all_channels,) where any non-zero value will
        exclude that channel from inclusion.
    channel_weight : numpy.ndarray (float)
        The channel weights of shape (all_channels,).
    sample_flags : numpy.ndarray (int)
        A mask of integer flags for each frame value of shape
        (n_frames, all_channels).
    frame_flags : numpy.ndarray (int)
        The frame flags of shape (n_frames,).
    source_flags : int
        The flag indicating that a frame should not be included due to having
        this associated flag.

    Returns
    -------
    covariance : numpy.ndarray (float)
        The covariance matrix of shape (all_channels, all_channels).
    """
    n_frames, all_channels = frame_data.shape
    covariance = np.zeros((all_channels, all_channels), dtype=nb.float64)
    n = np.zeros((all_channels, all_channels), dtype=nb.int64)

    for channel in range(all_channels):
        if channel_flags[channel] != 0:
            continue
        row_c = covariance[channel]
        row_n = n[channel]
        cw = channel_weight[channel]
        for frame in range(n_frames):
            if not frame_valid[frame]:
                continue
            if (frame_flags[frame] & source_flags) != 0:
                continue
            if sample_flags[frame, channel] != 0:
                continue
            fw = frame_weight[frame]
            d1 = frame_data[frame, channel]

            for c in range(channel, all_channels):
                if c == channel:
                    continue
                if channel_flags[c] != 0:
                    continue
                if sample_flags[frame, c] != 0:
                    continue
                row_c[c] += fw * d1 * frame_data[frame, c]
                row_n[c] += 1

        for c in range(channel, all_channels):
            n_c = row_n[c]
            if n_c > 0:
                row_c[c] *= np.sqrt(cw * channel_weight[c]) / n_c
            else:
                row_c[c] = 0.0
            covariance[c, channel] = row_c[c]
    return covariance


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_full_covariance_matrix(covariance, fixed_indices):  # pragma: no cover
    """
    Return the full covariance in terms of fixed channel indices.

    Parameters
    ----------
    covariance : numpy.ndarray (float)
        The covariance matrix of shape (all_channels, all_channels) where
        all_channels are those that were included in the reduction, but not
        the instrument stored channels.
    fixed_indices : numpy.ndarray (int)
        The indices of the instrument channel.

    Returns
    -------
    full_covariance : numpy.ndarray (float)
        The full covariance matrix of shape (store_channels, store_channels).
    """
    max_size = np.max(fixed_indices) + 1
    full_covariance = np.full((max_size, max_size), np.nan)
    n_channels = fixed_indices.size
    for i, ii in enumerate(fixed_indices):
        for j in range(i, n_channels):
            value = covariance[i, j]
            jj = fixed_indices[j]
            full_covariance[ii, jj] = value
            full_covariance[jj, ii] = value
    return full_covariance


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_partial_covariance_matrix(covariance, indices):  # pragma: no cover
    """
    Return the covariance matrix for certain channel indices.

    Parameters
    ----------
    covariance : numpy.ndarray (float)
        The covariance matrix of shape (all_channels, all_channels) where
        all_channels are those that were included in the reduction, but not
        the instrument stored channels.
    indices : numpy.ndarray (int)
        The indices of the instrument channel of shape (n_channels,).

    Returns
    -------
    partial_covariance : numpy.ndarray (float)
        The partial covariance matrix of shape (n_channels, n_channels).
    """
    n_channels = indices.size
    partial_covariance = np.full((n_channels, n_channels), np.nan)
    for i, ii in enumerate(indices):
        for j in range(i, n_channels):
            jj = indices[j]
            value = covariance[ii, jj]
            partial_covariance[i, j] = value
            partial_covariance[j, i] = value
    return partial_covariance


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def downsample_frame_data(data, window, valid=None):  # pragma: no cover
    """
    Downsample data using a supplied kernel.

    The size of the window kernel determines the number of frames in the
    output downsampled data, and is given as:

    (n_frames - window.size + 1) // (window.size // 2)

    In other words, convolution only occurs at points where the window is
    fully applied over the initial data values, and is downsampled by a factor
    of half the window span.  No convolution will occur if the window span
    contains a non-finite value of the data, or a data point marked as
    invalid.

    Parameters
    ----------
    data : numpy.ndarray (float)
        A data array of shape (n_frames, n_channels).
    window : numpy.ndarray (float)
        The resampling kernel of size (n,) where n is an odd integer.
    valid : numpy.ndarray (bool), optional
        If supplied, `True` indicates a valid sample available for resampling.

    Returns
    -------
    downsampled_data : numpy.ndarray (float)
        The downsampled array of shape (new_frames, n_channels).
    """
    n_frames, n_channels = data.shape
    half_window = window.size // 2
    window = window / np.sum(np.abs(window))
    new_frames = (n_frames - window.size + 1) // half_window
    new_data = np.empty((new_frames, n_channels), dtype=nb.float64)
    if valid is None:
        valid = np.full(n_frames, True)

    for channel_index in range(n_channels):
        channel_data = data[:, channel_index]
        new_channel_data = new_data[:, channel_index]

        for frame_index in range(new_frames):
            new_value = 0.0
            start_index = frame_index * half_window
            for window_index in range(window.size):
                old_frame_index = start_index + window_index
                if not valid[old_frame_index]:
                    new_value = np.nan
                    break

                old_value = channel_data[old_frame_index]
                if not np.isfinite(old_value):
                    new_value = np.nan
                    break

                new_value += window[window_index] * old_value

            new_channel_data[frame_index] = new_value

    return new_data


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def downsample_frame_flags(flags, window, valid=None):  # pragma: no cover
    """
    Downsample frame flags using a supplied kernel.

    The size of the window kernel determines the number of frames in the
    output downsampled data, and is given as:

    (n_frames - window.size + 1) // (window.size // 2)

    In other words, convolution only occurs at points where the window is
    fully applied over the initial data values, and is downsampled by a factor
    of half the window span.  No convolution will occur a data point marked as
    invalid.  In this case, the flag convolution refers to an `or` operation
    of all frame flags within the window span.

    Parameters
    ----------
    flags : numpy.ndarray (int)
        A flag array of shape (n_frames, n_channels).
    window : numpy.ndarray (float)
        The resampling kernel of size (n,) where n is an odd integer.
    valid : numpy.ndarray (bool), optional
        If supplied, `True` indicates a valid sample available for resampling.

    Returns
    -------
    downsampled_data : numpy.ndarray (float)
        The downsampled array of shape (new_frames, n_channels).
    """
    n_frames, n_channels = flags.shape
    half_window = window.size // 2
    new_frames = (n_frames - window.size + 1) // half_window
    new_flags = np.empty((new_frames, n_channels), dtype=nb.int64)
    if valid is None:
        valid = np.full(n_frames, True)

    for channel_index in range(n_channels):
        channel_flags = flags[:, channel_index]
        new_channel_flags = new_flags[:, channel_index]

        for frame_index in range(new_frames):
            new_flag = 0
            start_index = frame_index * half_window
            for window_index in range(window.size):
                old_frame_index = start_index + window_index
                if not valid[old_frame_index]:
                    new_flag = -1
                    break
                new_flag |= channel_flags[old_frame_index]
            new_channel_flags[frame_index] = new_flag

    return new_flags


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_downsample_start_indices(frame_valid, window,
                                 factor):  # pragma: no cover
    """
    Down-sample frame data using a given window kernel.

    Parameters
    ----------
    frame_valid
    window
    factor

    Returns
    -------
    downsampled_data, downsampled_flags : numpy.ndarray, numpy.ndarray
        The downsampled data and flags.  Both will be of shape
        (downsampled_frames, n_channels).
    """
    n_frames = frame_valid.size
    n_new = numba_functions.roundup_ratio(n_frames - window.size, factor)
    window_size = window.size
    downsample_start_indices = np.empty(n_new, dtype=nb.int64)
    downsample_valid = np.empty(n_new, dtype=nb.b1)

    for k in range(n_new):
        start_frame = k * factor
        valid = True
        for window_index in range(window_size):
            frame_index = window_index + start_frame
            if not frame_valid[frame_index] or frame_index >= n_frames:
                valid = False
                break

        downsample_valid[k] = valid
        downsample_start_indices[k] = start_frame

    return downsample_start_indices, downsample_valid


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_valid_downsampling_frames(valid_frames, start_indices,
                                  window_size):  # pragma: no cover
    """
    Return the valid downsampled frames that may be calculated.

    Parameters
    ----------
    valid_frames : numpy.ndarray (bool)
        The higher resolution valid frames where `True` marks a frame that may
        be included in the downsampling.  Should be of shape (n_frames,).
    start_indices : numpy.ndarray (int)
        The first frame of the higher resolution frame data to be included in
        the downsampled values.  i.e., the i(th) downsampled frame will use
        frames[start_indices[i] : start_indices[i] + window_size].  Should be
        of shape (n_downsampled,).
    window_size : int
        The size of the downsampling window function (convolution kernel).

    Returns
    -------
    valid_downsampled : numpy.ndarray (bool)
        A boolean array where `True` indicates that a downsampled value may be
        calculated for a given downsampled index.
    """
    n_downsample = start_indices.size
    n_frames = valid_frames.size
    downsampled_valid = np.empty(n_downsample, dtype=nb.b1)
    for i, start_index in enumerate(start_indices):
        if start_index >= n_frames:
            continue

        for j in range(window_size):
            frame_index = start_index + j
            if frame_index >= n_frames:
                downsampled_valid[i] = False
                break
            if not valid_frames[frame_index]:
                downsampled_valid[i] = False
                break
        else:
            downsampled_valid[i] = True

    return downsampled_valid
