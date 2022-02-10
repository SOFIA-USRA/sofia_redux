# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

from sofia_redux.toolkit.splines import spline_utils

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['calculate_coupling_increment', 'get_sample_points',
           'blank_sample_values', 'flag_out_of_range_coupling',
           'sync_map_samples', 'get_delta_sync_parms', 'flag_outside',
           'validate_pixel_indices', 'add_skydip_frames']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calculate_coupling_increment(map_indices, base_values, map_values,
                                 map_noise, sync_gains, source_gains,
                                 frame_data, frame_weight, frame_gains,
                                 frame_valid, sample_flags, channel_indices,
                                 min_s2n, max_s2n, exclude_flag
                                 ):  # pragma: no cover
    """
    Calculate and return the coupling increment factor.

    The coupling increment factor may be used to increment current coupling
    values and weights for each channel by::

        increment = sum_{frames}(fw * residual * expected) /
                    sum_{frames}(fw * expected^2)

    where fw are the frame weights, d are the frame data and::

        expected = frame_gain * channel_source_gain * map_value
        prior = frame_gain * channel_sync_gain * base_value
        residual = d + prior - expected

    The channel sync gains and base values are the channel source gains and
    map values from the previous iteration.

    Parameters
    ----------
    map_indices : numpy.ndarray (int)
        The indices of each frame/channel sample on the source map.  Should be
        of shape (n_dimensions, n_frames, all_channels) with dimensions in
        (x, y) FITS order.
    base_values : numpy.ndarray (float)
        An image of arbitrary grid shape in n_dimensions.  Contains the base
        map values (priors).
    map_values : numpy.ndarray (float)
        The current map image of arbitrary grid shape in n_dimensions.
    map_noise : numpy.ndarray (float)
        The current map noise values of arbitrary grid shape in n_dimensions.
    sync_gains : numpy.ndarray (float)
        The previous channel source gains (sync gains) of shape
        (all_channels,).
    source_gains : numpy.ndarray (float)
        The current channel source gains of shape (all_channels,).
    frame_data : numpy.ndarray (float)
        The integration frame data of shape (n_frames, all_channels,).
    frame_weight : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    frame_gains : numpy.ndarray (float)
        The frame gains of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        processing.
    sample_flags : numpy.ndarray (int)
        The integration sample flags of shape (n_frames, all_channels) where
        any sample flagged with `exclude_flag` will be excluded from
        processing.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to calculate coupling increments of shape
        (n_channels,).
    min_s2n : float
        The minimum signal-to-noise ratio on the map for which to calculate
        coupling increments.
    max_s2n : float
        The maximum signal-to-noise ratio on the map for which to calculate
        coupling increments.
    exclude_flag : int
        An integer flag used to exclude samples from processing.

    Returns
    -------
    coupling_increment : numpy.ndarray (float)
        The coupling increment factors of shape (all_channels,).
    """
    # Frame gains are integration.gain * frame.source_gain
    n_frames, n_channels = frame_data.shape
    coupling_increment = np.zeros(n_channels, dtype=nb.float64)
    coupling_weight = np.zeros(n_channels, dtype=nb.float64)

    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue

        weight = frame_weight[frame]
        if weight == 0:
            continue
        frame_gain = frame_gains[frame]
        if frame_gain == 0:
            continue

        # Remove source from all but the blind channels.
        for channel in channel_indices:

            if (sample_flags[frame, channel] & exclude_flag) != 0:
                continue

            x, y = map_indices[:, frame, channel]
            if x < 0 or y < 0:
                continue

            noise = map_noise[y, x]
            if noise <= 0:
                continue
            map_value = map_values[y, x]
            s2n = abs(map_value / noise)

            if s2n < min_s2n or s2n > max_s2n:
                continue

            base_value = base_values[y, x]

            prior = frame_gain * sync_gains[channel] * base_value
            expected = frame_gain * source_gains[channel] * map_value
            residual = frame_data[frame, channel] + prior - expected
            coupling_increment[channel] += weight * residual * expected
            coupling_weight[channel] += weight * expected * expected

    for channel in channel_indices:
        weight = coupling_weight[channel]
        if weight > 0:
            coupling_increment[channel] /= weight
        else:
            coupling_increment[channel] = 0.0

    return coupling_increment


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_sample_points(frame_data, frame_gains, frame_weights, source_gains,
                      channel_variance, valid_frames, map_indices,
                      channel_indices, sample_flags, exclude_sample_flag
                      ):  # pragma: no cover
    """
    Return the cross-product of frame and channel gains.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_gains : numpy.ndarray (float)
        The frame gains of shape (n_frames,).
    frame_weights : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,).
    source_gains : numpy.ndarray (float)
        The channel (pixel) gains of shape (all_channels,)
    channel_variance : numpy.ndarray (float)
        The channel (pixel) variances of shape (n_channels,)
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` indicates a frame
        that should be excluded from all calculations.
    map_indices : numpy.ndarray (int)
        The map indices of shape (2, n_frames, all_channels) containing the
        (ix, iy) map indices.  For pixel maps, all pixels have zero position,
        and therefore, map_indices depend solely of frame position.  In these
        cases, the shape should be (2, n_frames, 1) to indicate that only a
        single channel position is relevant.
    channel_indices : numpy.ndarray (int)
        The channel indices of shape (n_channels,) mapping n_channels onto
        all_channels.
    sample_flags : numpy.ndarray (int)
        The sample flag array of shape (n_frames, all_channels).  Any flag
        containing `exclude_sample_flag` will be ignored.
    exclude_sample_flag : int
        Indicates which sample flags should be ignored.

    Returns
    -------
    n, data, gains, weights, indices : 5-tuple (int, array+)
        The total number of mapping frames and the cross product of frame_gains
        and source_gains as an array of shape (n_frames, n_channels).  Any
        invalid frames/sample flags will result in a NaN gain value for the
        sample in question.
    """
    n_frames = frame_gains.size
    n_channels = channel_indices.size
    n_dimensions = map_indices.shape[0]
    mapping_frames = 0

    sample_data = np.empty((n_frames, n_channels), dtype=nb.float64)
    sample_gains = np.empty((n_frames, n_channels), dtype=nb.float64)
    sample_weights = np.empty((n_frames, n_channels), dtype=nb.float64)
    sample_indices = np.empty((n_dimensions, n_frames, n_channels),
                              dtype=nb.int64)

    is_pixel_map = map_indices.shape[2] == 1 and n_channels > 1

    for frame in range(n_frames):
        frame_gain = frame_gains[frame]
        frame_weight = frame_weights[frame]
        if (not valid_frames[frame] or frame_gain == 0
                or frame_weight == 0 or np.isnan(frame_gain)):
            for i in range(n_channels):
                blank_sample_values(frame, i, n_dimensions, sample_data,
                                    sample_gains, sample_weights,
                                    sample_indices)
            continue

        mapping_frames += 1
        for i, channel in enumerate(channel_indices):
            channel_gain = source_gains[channel]
            var = channel_variance[i]
            sample_flag = sample_flags[frame, channel]
            if (channel_gain == 0 or var == 0
                    or (sample_flag & exclude_sample_flag != 0)):
                blank_sample_values(frame, i, n_dimensions, sample_data,
                                    sample_gains, sample_weights,
                                    sample_indices)
                continue

            sample_gains[frame, i] = frame_gain * channel_gain
            sample_data[frame, i] = frame_data[frame, channel]
            sample_weights[frame, i] = frame_weight / var

            if is_pixel_map:
                channel_map_index = 0
            else:
                channel_map_index = channel

            for dimension in range(n_dimensions):
                sample_indices[dimension, frame, i] = (
                    map_indices[dimension, frame, channel_map_index])

    return (mapping_frames, sample_data, sample_gains, sample_weights,
            sample_indices)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def blank_sample_values(frame, channel_index, n_dimensions, sample_data,
                        sample_gains, sample_weights, sample_indices
                        ):  # pragma: no cover
    """
    Set bad frame/channel indices to blank values.

    Sets frame data and sample gains to NaN values at the frame/channel_index.
    weights are set to zero and sample indices are set to -1.

    Parameters
    ----------
    frame : int
        The bad frame index.
    channel_index : int
        The bad channel index.
    n_dimensions : int
        The number of dimensions in the data.
    sample_data : numpy.ndarray (float)
        The sample data of shape (n_frames, n_channels).
    sample_gains : numpy.ndarray (float)
        The sample gains of shape (n_frames, n_channels).
    sample_weights : numpy.ndarray (float)
        The sample weights of shape (n_frames, n_channels).
    sample_indices : numpy.ndarray (int)
        The sample map indices of shape (n_dimensions, n_frames, n_channels).

    Returns
    -------
    None
    """
    sample_gains[frame, channel_index] = np.nan
    sample_data[frame, channel_index] = np.nan
    sample_weights[frame, channel_index] = 0.0
    for dimension in range(n_dimensions):
        sample_indices[dimension, frame, channel_index] = -1


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def flag_out_of_range_coupling(channel_indices, coupling_values, min_coupling,
                               max_coupling, flags, blind_flag
                               ):  # pragma: no cover
    """
    Flags channels with an out-of-range coupling value as "BLIND"

    Note that only previously unflagged channels will be flagged.

    Parameters
    ----------
    channel_indices : numpy.ndarray (int)
        The channel indices to check of shape (n_channels,).
    coupling_values : numpy.ndarray (float)
        The coupling values of shape (all_channels,).
    min_coupling : float
        The minimum allowable coupling value.
    max_coupling : float
        The maximum allowable coupling value.
    flags : numpy.ndarray (int)
        The channel flags of shape (all_channels,).  Will be updated in-place.
    blind_flag : int
        The integer flag marking a "BLIND" channel.

    Returns
    -------
    None
    """
    for channel in channel_indices:
        if flags[channel] != 0:
            continue
        coupling_value = coupling_values[channel]
        if (coupling_value < min_coupling) or (coupling_value > max_coupling):
            flags[channel] |= blind_flag


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def sync_map_samples(frame_data, frame_valid, frame_gains, channel_indices,
                     map_values, map_valid, map_masked, map_indices,
                     base_values, base_valid, source_gains, sync_gains,
                     sample_flags, sample_blank_flag):  # pragma: no cover
    """
    Remove map source gains from frame data and flag samples.

    For a given sample at frame i and channel j, frame data d_{i,j} will be
    decremented by dg where::

        dg = fg * ( (gain(source) * map[index]) - (gain(sync) * base[index]) )

    Here, fg is the frame gain and index is the index on the map of sample
    (i,j).

    Any masked map value will result in matching samples being flagged.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).  Data will be updated
        in-place.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        all processing.
    frame_gains : numpy.ndarray (float)
        An array of frame gains of shape (n_frames,).
    channel_indices : numpy.ndarray (int)
        The channel indices of shape (n_channels,) for which to sync map gains.
    map_values : numpy.ndarray (float)
        The current map supplied as a regular grid of arbitrary shape and
        dimensions (image_shape,).
    map_valid : numpy.ndarray (bool)
        A boolean mask of shape (image_shape,) where `False` excludes a map
        element from synchronization.
    map_masked : numpy.ndarray (bool)
        A boolean mask of shape (image_shape,) where `True` will result in
        any frame/channel sample being flagged with the `sample_blank_flag`
        flag.  Similarly, `False` will unflag samples.
    map_indices : numpy.ndarray (int)
        The sample map indices of shape (n_dimensions, n_frames, all_channels)
        where dimensions are ordered in (x,y) FITS format.  These contain map
        (pixel) indices for each sample.  For pixel maps, all pixels will
        have zero position values, so the array should be of shape
        (n_dimensions, n_frames, 1) in this case.
    base_values : numpy.ndarray (float)
        The base map values (priors) of shape (image_shape,).
    base_valid : numpy.ndarray (bool)
        A boolean mask of shape (image_shape,) where `False` indicates that the
        corresponding base map value is invalid and will be set to zero during
        processing.
    source_gains : numpy.ndarray (float)
        The channel source gains of shape (all_channels,).
    sync_gains : numpy.ndarray (float)
        The prior channel source gains of shape (all_channels,).
    sample_flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, all_channels,).  Will be updated
        in-place.
    sample_blank_flag : int
        The integer flag with which to flag samples that contain a masked
        map value.

    Returns
    -------
    None
    """
    map_shape = np.asarray(map_values.shape)
    _, _, row_col_steps = spline_utils.flat_index_mapping(map_shape)

    fits_shape = map_shape[::-1]

    # Convert from (row, col) to (col, row) for FITS type data indexing
    steps = row_col_steps[::-1]
    n_dimensions = steps.size

    flat_map_values = map_values.flat
    flat_map_valid = map_valid.flat
    flat_map_masked = map_masked.flat
    flat_base_values = base_values.flat
    flat_base_valid = base_valid.flat
    unflag_blank = ~sample_blank_flag

    is_pixel_map = map_indices.shape[2] == 1 and channel_indices.size > 1

    n_frames = frame_data.shape[0]
    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue

        frame_gain = frame_gains[frame]

        for channel in channel_indices:
            if is_pixel_map:
                map_index = map_indices[:, frame, 0]
            else:
                map_index = map_indices[:, frame, channel]
            flat_index = 0
            for dimension in range(n_dimensions):
                index = map_index[dimension]
                if index < 0 or index >= fits_shape[dimension]:
                    flat_index = -1
                    break
                flat_index += steps[dimension] * index
            if flat_index < 0:
                continue  # Invalid index

            if not flat_map_valid[flat_index]:
                continue

            # Remove from frame data
            # Do not check for flags to get a true difference image.
            if frame_gain > 0:
                if not flat_base_valid[flat_index]:
                    base_value = 0.0
                else:
                    base_value = flat_base_values[flat_index]
                map_value = flat_map_values[flat_index]

                decrement = source_gains[channel] * map_value
                decrement -= sync_gains[channel] * base_value
                decrement *= frame_gain
                frame_data[frame, channel] -= decrement

            # Blank samples here.
            if flat_map_masked[flat_index]:
                sample_flags[frame, channel] |= sample_blank_flag
            else:
                sample_flags[frame, channel] &= unflag_blank


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_delta_sync_parms(channel_source_gains, channel_indices, channel_flags,
                         channel_variance, frame_weight, frame_source_gains,
                         frame_valid, frame_flags, source_flag, n_points
                         ):  # pragma: no cover
    """
    Calculate frame/channel dependent deltas following source synchronization.

    Source synchronization is the process of removing the source model from the
    integration frame data.  Following this, dependents (scaled degrees-of-
    freedom) need to be updated to account for the operation.  The dependent
    increments are calculated as follows::

        channel_dp = n_p * channel_gain^2 / variance
        frame_dp = n_f * frame_weight * |frame_gain|

        n_p = n_points / sum_{channels}(channel_gain^2/variance)
        n_f = n_points / sum_{frames}(frame_weight * frame_gain)

    where n_points effective number of pixels in the source model (possibly
    accounting for smoothing).

    Parameters
    ----------
    channel_source_gains : numpy.ndarray (float)
        The channel source gains of shape (all_channels,).
    channel_indices : numpy.ndarray (int)
        The indices of the channels to process of shape (n_channels,).
    channel_flags : numpy.ndarray (int)
        The integer flags marking various status types of shape (n_channels,).
        Only zero-values flags will be included in the calculation.
    channel_variance : numpy.ndarray (float)
        The variance associated with each channel of shape (n_channels,).
    frame_weight : numpy.ndarray (float)
        The relative weight of each frame of shape (n_frames,).
    frame_source_gains : numpy.ndarray (float)
        The frame source gains of shape (n_frames,).
    frame_valid : numpy.ndarray (bool)
        A boolean mask where `False` excludes a given frame from inclusion in
        the calculation and results in that same frame receiving a zero-valued
        dependent increment.  Should be of shape (n_frames,).
    frame_flags : numpy.ndarray (int)
        The integer flags marking various status types for each frame of shape
        (n_frames,).  Any frames marked with `source_flag` will be excluded
        from all calculations and receive a zero-valued dependent increment.
    source_flag : int
        The integer used to mark a frame with the SOURCE status.
    n_points : int or float
        The effective number of points in the source model.

    Returns
    -------
    frame_dp, channel_dp : numpy.ndarray, numpy.ndarray
        The frame and channel dependent deltas of shape (n_frames,) and
        (n_channels,), both of float type.
    """
    n_channels = channel_indices.size
    n_frames = frame_valid.size
    sum_pw = 0.0
    for i, channel in enumerate(channel_indices):
        if channel_flags[i] != 0:
            continue
        gain = channel_source_gains[channel]
        sum_pw += gain * gain / channel_variance[i]

    sum_fw = 0.0
    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue
        if (frame_flags[frame] & source_flag) != 0:
            continue
        sum_fw += frame_weight[frame] * frame_source_gains[frame]

    n_p = n_points / sum_pw if sum_pw > 0 else 0.0
    n_f = n_points / sum_fw if sum_fw > 0 else 0.0

    frame_dp = np.empty(n_frames, dtype=nb.float64)
    channel_dp = np.empty(n_channels, dtype=nb.float64)
    for i, channel in enumerate(channel_indices):
        if channel_flags[i] != 0:
            channel_dp[i] = 0.0
            continue
        gain = channel_source_gains[channel]
        channel_dp[i] = n_p * gain * gain / channel_variance[i]

    for frame in range(n_frames):
        if not frame_valid[frame] or (frame_flags[frame] & source_flag) != 0:
            frame_dp[frame] = 0.0
            continue
        frame_dp[frame] = (n_f * frame_weight[frame]
                           * np.abs(frame_source_gains[frame]))

    return frame_dp, channel_dp


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def flag_outside(sample_coordinates, valid_frames, channel_indices,
                 sample_flags, skip_flag, map_range):  # pragma: no cover
    """
    Flag samples outside of the allowed mapping range.

    Frames with no channels inside the allowable range will be flagged
    as invalid.

    Parameters
    ----------
    sample_coordinates : numpy.ndarray (float)
        An array of shape (2, n_frames, n_channels) containing the (x, y)
        coordinates of each sample measurement.
    valid_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,).  Frames flagged as `False` will
        not be included in any calculations.
    channel_indices : numpy.ndarray (int)
        The indices mapping n_channels onto all_channels.
    sample_flags : numpy.ndarray (int)
        The sample flags to update.  An array of shape
        (n_frames, all_channels).
    skip_flag : int
        The integer flag to mark a sample as SKIP (outside mapping range).
    map_range : numpy.ndarray (float)
        An array of shape (2, 2) containing the
        [[min(x), max(x)], [min(y), max(y)]] allowable map range.

    Returns
    -------
    None
    """
    min_x, max_x = map_range[0]
    min_y, max_y = map_range[1]

    for frame_index in range(sample_coordinates.shape[1]):
        valid = False
        if not valid_frames[frame_index]:
            continue
        for i, channel_index in enumerate(channel_indices):
            x = sample_coordinates[0, frame_index, i]
            y = sample_coordinates[1, frame_index, i]
            if min_x <= x <= max_x and min_y <= y <= max_y:
                valid = True
                continue
            sample_flags[frame_index, channel_index] |= skip_flag
        valid_frames[frame_index] = valid


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def validate_pixel_indices(indices, x_size, y_size, valid_frame=None
                           ):  # pragma: no cover
    """
    Set pixel indices outside of the map range to (0, 0).

    Parameters
    ----------
    indices : numpy.ndarray (int)
        Pixel indices of shape (2, n_frames, n_channels) containing the
        (x, y) pixel indices.  Any invalid sample (frame, channel) will be
        updated to (-1, -1) in-place.
    x_size : int
        The size of the map in x.
    y_size : int
        The size of the map in y.
    valid_frame : numpy.ndarray (bool), optional
        An optional flag mask where `False` excludes the given frame from the
        bad sample count (return value).

    Returns
    -------
    bad_samples : int
        The number of pixels that fall outside the range of the map extent.
    """
    bad_samples = 0
    n_dimensions, n_frames, n_channels = indices.shape

    if valid_frame is None:
        check_valid = False
        valid = np.empty(0, dtype=nb.b1)
    else:
        check_valid = True
        valid = valid_frame

    for frame_index in range(n_frames):
        if check_valid:
            frame_is_valid = valid[frame_index]
        else:
            frame_is_valid = True

        for i in range(n_channels):  # the channel indices
            px = indices[0, frame_index, i]
            py = indices[1, frame_index, i]
            if not (0 <= px < x_size):
                indices[:, frame_index, i] = -1
                if frame_is_valid:
                    bad_samples += 1
                continue
            if not (0 <= py < y_size):
                indices[:, frame_index, i] = -1
                if frame_is_valid:
                    bad_samples += 1
                continue

    return bad_samples


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def add_skydip_frames(data, weight, signal_values, signal_weights,
                      frame_weights, frame_valid, data_bins
                      ):  # pragma: no cover
    """
    Add signal values to the sky dip model data.

    The skydip model data (`data`) and weights (`weight`) are added to using
    increments of::

        data += frame_weights * signal_weights * signal_values
        weight += frame_weights * signal_weights

    Parameters
    ----------
    data : numpy.ndarray (float)
        The skydip model data of shape (n_data,).
    weight : numpy.ndarray (float)
        The skydip model data weight of shape (n_data,).
    signal_values : numpy.ndarray (float)
        The signal values of shape (n_frames,)
    signal_weights : numpy.ndarray (float)
        The signal value weights of shape (n_frames,)
    frame_weights : numpy.ndarray (float)
        The frame relative weights of shape (n_frames,)
    frame_valid : numpy.ndarray (bool)
        A mask of shape (n_frames,) where `False` exclude a given frame from
        processing.
    data_bins : numpy.ndarray (int)
        A mapping array of shape (n_frames,) in which data_bins[i] maps frame
        i onto the `data` and `weight` array indices.

    Returns
    -------
    None
    """
    n_frames = frame_valid.size
    data_size = data.size
    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue
        data_bin = data_bins[frame]
        if data_bin < 0 or data_bin >= data_size:
            continue
        w = frame_weights[frame] * signal_weights[frame]
        if w == 0:
            continue
        data[data_bin] += w * signal_values[frame]
        weight[data_bin] += w
