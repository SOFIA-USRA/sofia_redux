# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['find_inconsistencies', 'fix_jumps', 'fix_block',
           'flag_block', 'level_block', 'correct_jumps',
           'flag_zeroed_channels', 'check_jumps']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def find_inconsistencies(frame_valid, frame_data, frame_weights,
                         modeling_frames, frame_parms, sample_flags,
                         exclude_sample_flag, channel_indices, channel_parms,
                         min_jump_level_frames, jump_flag, fix_each,
                         fix_subarray, has_jumps, subarray, jump_counter,
                         drift_size, flag_before=0, flag_after=0
                         ):  # pragma: no cover
    """
    Find, fix (or flag) jumps and return the number corrected per channel.

    This function is a wrapper around :func:`fix_jumps` the processes chunks
    of frames (the size of which is determined by `drift_size`) sequentially,
    and returns the number of chunk blocks found with jumps for each channel.
    Jumps may be ignored, levelled, or flagged depending on
    `min_jump_level_frames`, `fix_each`, and `fix_subarray`.

    Parameters
    ----------
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        any processing.
    frame_data : numpy.ndarray (float)
        The frame data values of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a frame as a
        modeling frame.  Modeling frames will still be levelled, but will not
        be included when updating the frame or channel dependents.
    frame_parms : numpy.ndarray (float)
        The frame dependents.  Will be updated in-place if levelling occurs.
    sample_flags : numpy.ndarray (int)
        The frame data sample flags.  Typically non-zero samples will be
        excluded from processing.  However, those samples not flagged with
        only `exclude_sample_flag` will be included.
    exclude_sample_flag : int
        The sample flag to explicitly exclude from processing.
    channel_indices : numpy.ndarray (int)
        The channel indices of shape (n_channels,) indicating which channels
        to process and mapping n_channels onto all_channels.
    channel_parms : numpy.ndarray (float)
        The channel dependents of shape (all_channels,).  Will be updated
        in-place when levelling occurs.
    min_jump_level_frames : int
        The minimum number of frames in a jump block required for levelling.
        If this value is not reached, all samples in the block are flagged
        with `jump_flag` instead.
    jump_flag : int
        The integer flag identifier with which to flag samples if the
        jump block length to which they belong is less than
        `min_jump_level_frames` and cannot be levelled.
    fix_each : bool
        If `False`, do not fix any channel with jumps.
    fix_subarray : numpy.ndarray (bool)
        An array of shape (n_subarrays,) where `True` indicates that any
        channel belonging to that subarray should  have jumps corrected.
        Ignored if fix_each is False.
    has_jumps : numpy.ndarray (bool)
        A boolean mask of shape (n_channels,) where `True` indicates that a
        channel has jumps that may be corrected.
    subarray : numpy.ndarray (int)
        An array of shape (n_channels,) containing the subarray number for
        each channel.
    jump_counter : numpy.ndarray (int)
        The channel jumps of shape (n_frames, n_channels).
    drift_size : int
        The number of frames that constitute a single chunk that will have
        jumps either levelled or flagged (see `min_jump_level_frames`).  Each
        chunk will be processed separately.
    flag_before : int, optional
        The number of frames to flag in the sample flags prior to a jump
        detection with `jump_flag`.
    flag_after : int, optional
        The number of frames to flag in the sample flags following a jump
        detection with `jump_flag`.

    Returns
    -------
    inconsistencies : numpy.ndarray (int)
        An array of shape (n_channels,) containing the number of frame blocks
        found in the data that contain jumps which have either been levelled
        or flagged for each channel.
    """

    n_frames = frame_data.shape[0]
    n_channels = channel_indices.size
    inconsistencies = np.zeros(n_channels, dtype=nb.int64)

    for start_frame in range(0, n_frames, drift_size):
        end_frame = start_frame + drift_size
        if end_frame > n_frames:
            end_frame = n_frames
        no_jumps = fix_jumps(
            frame_valid=frame_valid,
            frame_data=frame_data,
            frame_weights=frame_weights,
            modeling_frames=modeling_frames,
            frame_parms=frame_parms,
            sample_flags=sample_flags,
            exclude_sample_flag=exclude_sample_flag,
            channel_indices=channel_indices,
            channel_parms=channel_parms,
            min_jump_level_frames=min_jump_level_frames,
            jump_flag=jump_flag,
            fix_each=fix_each,
            fix_subarray=fix_subarray,
            has_jumps=has_jumps,
            subarray=subarray,
            jump_counter=jump_counter,
            start_frame=start_frame,
            end_frame=end_frame,
            flag_before=flag_before,
            flag_after=flag_after)

        for i in range(n_channels):
            if not no_jumps[i]:
                inconsistencies[i] += 1

    return inconsistencies


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def fix_jumps(frame_valid, frame_data, frame_weights,
              modeling_frames, frame_parms, sample_flags, exclude_sample_flag,
              channel_indices, channel_parms, min_jump_level_frames, jump_flag,
              fix_each, fix_subarray, has_jumps, subarray, jump_counter,
              start_frame=None, end_frame=None, flag_before=0, flag_after=0
              ):  # pragma: no cover
    """
    Detect and fix jumps in the frame data for each channel.

    If a given channel has jumps and settings imply that jumps may be fixed,
    the average frame data level between the jump is subtracted (frame and
    channel dependents are also updated).  A change in the jump level occurs
    when the `jump_counter` value for a given channel changes to a new value.
    a block is defined as all frames having the same jump value before a jump
    change.

    If the number of frames in a block is less than `min_jump_level_frames`,
    those frames will be flagged with the `jump_flag` sample flag.  Otherwise,
    the frame data will be levelled as described above.

    Parameters
    ----------
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from any processing.
    frame_data : numpy.ndarray (float)
        The frame data values of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a frame as a
        modeling frame.  Modeling frames will still be levelled, but will not
        be included when updating the frame or channel dependents.
    frame_parms : numpy.ndarray (float)
        The frame dependents.  Will be updated in-place if levelling occurs.
    sample_flags : numpy.ndarray (int)
        The frame data sample flags.  Typically non-zero samples will be
        excluded from processing.  However, those samples not flagged with
        only `exclude_sample_flag` will be included.
    exclude_sample_flag : int
        The sample flag to explicitly exclude from processing.
    channel_indices : numpy.ndarray (int)
        The channel indices of shape (n_channels,) indicating which channels
        to process and mapping n_channels onto all_channels.
    channel_parms : numpy.ndarray (float)
        The channel dependents of shape (all_channels,).  Will be updated
        in-place when levelling occurs.
    min_jump_level_frames : int
        The minimum number of frames in a jump block required for levelling.
        If this value is not reached, all samples in the block are flagged
        with `jump_flag` instead.
    jump_flag : int
        The integer flag identifier with which to flag samples if the jump
        block length to which they belong is less than `min_jump_level_frames`
        and cannot be levelled.
    fix_each : bool
        If `False`, do not fix any channel with jumps.
    fix_subarray : numpy.ndarray (bool)
        An array of shape (n_subarrays,) where `True` indicates that any
        channel belonging to that subarray should  have jumps corrected.
        Ignored if fix_each is False.
    has_jumps : numpy.ndarray (bool)
        A boolean mask of shape (n_channels,) where `True` indicates that a
        channel has jumps that may be corrected.
    subarray : numpy.ndarray (int)
        An array of shape (n_channels,) containing the subarray number for
        each channel.
    jump_counter : numpy.ndarray (int)
        The channel jumps of shape (n_frames, n_channels).
    start_frame : int, optional
        The start frame from which to begin correction.  The default is the
        first frame (0).
    end_frame : int, optional
        The last frame at which to conclude correction (non-inclusive).  The
        default is the total number of frames (n_frames).
    flag_before : int, optional
        The number of frames to flag in the sample flags prior to a jump
        detection with `jump_flag`.
    flag_after : int, optional
        The number of frames to flag in the sample flags following a jump
        detection with `jump_flag`.

    Returns
    -------
    no_jumps : numpy.ndarray (bool)
        A boolean mask of shape (n_channels,) where `True` indicates that a
        channel has no jumps.
    """
    no_jumps = np.full(channel_indices.size, True)

    if not fix_each or not fix_subarray.any():
        return no_jumps

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = frame_data.shape[0]

    for i, channel in enumerate(channel_indices):
        if not has_jumps[i]:
            continue
        if not fix_subarray[subarray[i]]:
            continue

        blocks_fixed = 0
        started = False
        jump_start = -1
        from_frame = -1
        first_frame = -1
        for frame in range(start_frame, end_frame):
            if not frame_valid[frame]:
                continue
            if not started:
                jump_start = jump_counter[frame, channel]
                first_frame = from_frame = frame
                started = True
            jump = jump_counter[frame, channel]
            if jump == jump_start:
                continue

            fix_block(from_frame=from_frame,
                      to_frame=frame,
                      frame_valid=frame_valid,
                      frame_data=frame_data,
                      frame_weights=frame_weights,
                      modeling_frames=modeling_frames,
                      frame_parms=frame_parms,
                      sample_flags=sample_flags,
                      exclude_sample_flag=exclude_sample_flag,
                      channel=channel,
                      channel_parms=channel_parms,
                      min_jump_level_frames=min_jump_level_frames,
                      jump_flag=jump_flag)

            if flag_before != 0:
                flag_start = max(0, frame - flag_before)
                for flag_frame in range(flag_start, frame):
                    if frame_valid[flag_frame]:
                        sample_flags[flag_frame, channel] |= jump_flag

            if flag_after != 0:
                flag_stop = min(frame + flag_after, end_frame)
                for flag_frame in range(frame, flag_stop):
                    if frame_valid[flag_frame]:
                        sample_flags[flag_frame, channel] |= jump_flag

            blocks_fixed += 1
            from_frame = frame
            jump_start = jump

        if first_frame != from_frame:  # Fix the end
            fix_block(from_frame=from_frame,
                      to_frame=end_frame,
                      frame_valid=frame_valid,
                      frame_data=frame_data,
                      frame_weights=frame_weights,
                      modeling_frames=modeling_frames,
                      frame_parms=frame_parms,
                      sample_flags=sample_flags,
                      exclude_sample_flag=exclude_sample_flag,
                      channel=channel,
                      channel_parms=channel_parms,
                      min_jump_level_frames=min_jump_level_frames,
                      jump_flag=jump_flag)
            blocks_fixed += 1

        no_jumps[i] = blocks_fixed == 0
    return no_jumps


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def fix_block(from_frame, to_frame, frame_valid, frame_data, frame_weights,
              modeling_frames, frame_parms, sample_flags, exclude_sample_flag,
              channel, channel_parms, min_jump_level_frames, jump_flag
              ):  # pragma: no cover
    """
    Jump correct a block of frames for a given channel.

    See :func:`fix_jumps` for further details.  This function essentially
    compares the jump block length with `min_jump_level_frames` and determines
    whether the block should be flagged or levelled.  If the number of frames
    in the block (`to_frame` - `from_frame`) is >= `min_jump_level_frames`,
    then leveling will occur.  Otherwise, all samples in the block will be
    flagged with the `jump_flag`.

    Parameters
    ----------
    from_frame : int
        The starting frame from which to begin jump correction.
    to_frame : int
        The end frame (non-inclusive) at which to conclude jump correction.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from any processing.
    frame_data : numpy.ndarray (float)
        The frame data values of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a frame as a
        modeling frame.  Modeling frames will still be levelled, but will not
        be included when updating the frame or channel dependents.
    frame_parms : numpy.ndarray (float)
        The frame dependents.  Will be updated in-place if levelling occurs.
    sample_flags : numpy.ndarray (int)
        The frame data sample flags.  Typically non-zero samples will be
        excluded from processing.  However, those samples not flagged with
        only `exclude_sample_flag` will be included.
    exclude_sample_flag : int
        The sample flag to explicitly exclude from processing.
    channel : int
        The channel to process.
    channel_parms : numpy.ndarray (float)
        The channel dependents of shape (all_channels,).  Will be updated
        in-place when levelling occurs.
    min_jump_level_frames : int
        The minimum number of frames in a jump block required for levelling.
        If this value is not reached, all samples in the block are flagged
        with `jump_flag` instead.
    jump_flag : int
        The integer flag identifier with which to flag samples if the jump
        block length to which they belong is less than `min_jump_level_frames`
        and cannot be levelled.

    Returns
    -------
    None
    """
    if (to_frame - from_frame) < min_jump_level_frames:
        flag_block(from_frame=from_frame,
                   to_frame=to_frame,
                   frame_valid=frame_valid,
                   sample_flags=sample_flags,
                   jump_flag=jump_flag,
                   channel=channel)
    else:
        level_block(from_frame=from_frame,
                    to_frame=to_frame,
                    frame_valid=frame_valid,
                    frame_data=frame_data,
                    frame_weights=frame_weights,
                    modeling_frames=modeling_frames,
                    frame_parms=frame_parms,
                    sample_flags=sample_flags,
                    exclude_sample_flag=exclude_sample_flag,
                    channel=channel,
                    channel_parms=channel_parms)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def flag_block(from_frame, to_frame, frame_valid, sample_flags, jump_flag,
               channel):  # pragma: no cover
    """
    Flag all samples within a block for a given channel with a jump flag.

    Parameters
    ----------
    from_frame : int
        The starting frame from which to begin jump correction.
    to_frame : int
        The end frame (non-inclusive) at which to conclude jump correction.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from any processing.
    sample_flags : numpy.ndarray (int)
        The frame data sample flags.  Typically non-zero samples will be
        excluded from processing.  However, those samples not flagged with
        only `exclude_sample_flag` will be included.
    jump_flag : int
        The integer flag identifier with which to flag samples if the jump
        block length to which they belong is less than `min_jump_level_frames`
        and cannot be levelled.
    channel : int
        The channel to process.

    Returns
    -------
    None
    """
    for frame in range(from_frame, to_frame):
        if not frame_valid[frame]:
            continue
        sample_flags[frame, channel] |= jump_flag


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def level_block(from_frame, to_frame, frame_valid, frame_data, frame_weights,
                modeling_frames, frame_parms, sample_flags,
                exclude_sample_flag, channel,
                channel_parms):  # pragma: no cover
    """
    Level a frame block for a given channel.

    For a given channel i, all frames that are valid (but may be modeling
    frames) will be corrected by:

    d[f:t, i] -= sum_{f|nm}(w[f] * d[f]) / w_sum

    w_sum = sum_{f|nm}(w[f])

    Where f is the `from_frame`, t is the `to_frame`, d are the `frame_data`,
    w are the `frame_weights`, and f|nm are the set of frames in the range t:f
    that are valid and non-modeling frames.

    If w_sum > 0 then the channel dependents are incremented by 1 and the frame
    dependents for frame j are updated by:

    dp[j] += w[j] / w_sum

    if j is valid, non-modeling and the associated sample flag is not excluded
    by `exclude_sample_flag`.

    Parameters
    ----------
    from_frame : int
        The starting frame from which to begin jump correction.
    to_frame : int
        The end frame (non-inclusive) at which to conclude jump correction.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        any processing.
    frame_data : numpy.ndarray (float)
        The frame data values of shape (n_frames, all_channels).
    frame_weights : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).
    modeling_frames : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `True` marks a frame as a
        modeling frame.  Modeling frames will still be levelled, but will not
        be included when updating the frame or channel dependents.
    frame_parms : numpy.ndarray (float)
        The frame dependents of shape (n_frames,).  Will be updated in-place.
    sample_flags : numpy.ndarray (int)
        The frame data sample flags.  Any samples flagged with
        `exclude_sample_flag` will not be included in the mean value
        calculation or contribute to the frame dependents.  However, the
        mean value will still be subtracted from them.
    exclude_sample_flag : int
        The sample flag to explicitly exclude from processing.
    channel : int
        The channel to process.
    channel_parms : numpy.ndarray (float)
        The channel dependents of shape (all_channels,).  Will be updated
        in-place.

    Returns
    -------
    None
    """
    d_sum = 0.0
    w_sum = 0.0
    for frame in range(from_frame, to_frame):
        if not frame_valid[frame] or modeling_frames[frame]:
            continue
        if (sample_flags[frame, channel] & exclude_sample_flag) != 0:
            continue
        w = frame_weights[frame]
        if w == 0:
            continue
        d_sum += w * frame_data[frame, channel]
        w_sum += w

    if w_sum <= 0:
        return

    channel_parms[channel] += 1
    average = d_sum / w_sum

    for frame in range(from_frame, to_frame):
        if not frame_valid[frame]:
            continue
        frame_data[frame, channel] -= average
        if modeling_frames[frame]:
            continue
        if (sample_flags[frame, channel] & exclude_sample_flag) != 0:
            continue
        frame_parms[frame] += frame_weights[frame] / w_sum


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def correct_jumps(frame_data, frame_valid, jump_counter, channel_indices,
                  channel_jumps, jump_range):  # pragma: no cover
    """
    Correct DAC jumps in the frame data.

    Frame data are corrected in-place by:

    d[f, c] -= channel_jumps[c] * n_jumps_{f,c}

    where d are the frame data, f is the frame, c is the channel and n_jumps
    are the number of jumps detected since the first valid frame or:

    n_jumps_{f,c} = jump_counter[f, c] - jump_counter[f0, c]

    where f0 is the first valid frame.  Since the jump counter have byte
    values, wrap around values are considered.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).  Values are updated
        in-place.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        any processing.
    jump_counter : numpy.ndarray (int)
        The jump counter of shape (n_frames, all_channels).
    channel_indices : numpy.ndarray (int)
        The channel indices for which to correct jumps of shape (n_channels,).
        Should map n_channels -> all_channels.
    channel_jumps : numpy.ndarray (float)
        The jump values of shape (n_channels,).
    jump_range : int
        The maximum jump range (bytes) to check for wrap around values.

    Returns
    -------
    None
    """
    n_frames = frame_data.shape[0]
    max_jump = jump_range // 2
    min_jump = -max_jump
    start_frame = -1
    for frame in range(n_frames):
        if frame_valid[frame]:
            start_frame = frame
            break
    if start_frame < 0:  # no valid frames
        return

    start_jumps = np.empty(channel_indices.size, dtype=nb.int64)

    for i, channel in enumerate(channel_indices):
        start_jumps[i] = jump_counter[start_frame, channel]

    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue
        for i, channel in enumerate(channel_indices):
            if channel_jumps[i] == 0:
                continue

            n_jumps = jump_counter[frame, channel] - start_jumps[i]
            if n_jumps == 0:
                continue  # no change

            elif n_jumps > max_jump:
                n_jumps -= jump_range  # Wrap around in positive direction

            elif n_jumps < min_jump:
                n_jumps += jump_range  # Wrap around in negative direction

            jump_correction = n_jumps * channel_jumps[i]
            frame_data[frame, channel] -= jump_correction


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def flag_zeroed_channels(frame_data, frame_valid, channel_indices,
                         channel_flags, discard_flag):  # pragma: no cover
    """
    Flag channels with the DISCARD flag if all frame data are zero valued.

    Parameters
    ----------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from checks.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to check zero-levels of shape
        (n_channels,).  Should map n_channels -> all_channels.
    channel_flags : numpy.ndarray (int)
        The channel flags of shape (n_channels,).
    discard_flag : int
        The flag to apply to channel flags if all frames are zero valued.

    Returns
    -------
    None
    """
    n_frames = frame_data.shape[0]

    for i, channel in enumerate(channel_indices):
        channel_is_flagged = (channel_flags[i] & discard_flag) != 0
        for frame in range(n_frames):
            if frame_valid[frame] and frame_data[frame, channel] != 0:
                # Unflag if necessary
                if channel_is_flagged:
                    channel_flags[i] ^= discard_flag
                break
        else:
            # Flag if all frame data are invalid or zero-valued.
            if not channel_is_flagged:
                channel_flags[i] |= discard_flag


@nb.jit(cache=True, nogil=False, parallel=False, fastmath=True)
def check_jumps(start_counter, jump_counter, frame_valid, has_jumps,
                channel_indices):  # pragma: no cover
    """
    Checks for jumps in each channel.

    A jump will be recorded for a channel if the jump counter value is not
    constant over all frames.

    Parameters
    ----------
    start_counter : numpy.ndarray (int)
        The first valid frame jump counter value for each channel of shape
        (all_channels,).
    jump_counter : numpy.ndarray (int)
        The jump counter of shape (n_frames, all_channels).
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame from
        standard processing, although it is not used here except to determine
        the number of frames.
    has_jumps : numpy.ndarray (bool)
        A boolean mask indicating whether a channel has jumps of shape
        (n_channels,).  Will be updated in-place.
    channel_indices : numpy.ndarray (int)
        An array containing the channel indices for all channels to be checked
        of shape (n_channels,).  Should map n_channels -> all_channels.

    Returns
    -------
    jumps_found : int
        The number of channels found with jumps.
    """
    n_frames = frame_valid.size

    jumps_found = 0
    for i, channel in enumerate(channel_indices):
        jump_start = start_counter[channel]
        for frame in range(n_frames):
            if jump_counter[frame, channel] != jump_start:
                has_jumps[i] = True
                jumps_found += 1
                break
        else:
            has_jumps[i] = False

    return jumps_found
