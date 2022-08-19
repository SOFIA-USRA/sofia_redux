# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['flag_zeroed_channels']


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
