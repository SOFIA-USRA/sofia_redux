# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['add_dependents', 'validate_frames', 'downsample_data']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def add_dependents(dependents, dp, frame_valid, start_frame=None,
                   end_frame=None, subtract=False):
    """
    Add dependents from frame data.

    Parameters
    ----------
    dependents : numpy.ndarray (float)
        Must be the same size as self.size.
    dp : numpy.ndarray (float) or float
        The dependent values to add.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes that frame
        from any processing.
    start_frame : int, optional
        The starting frame.
    end_frame : int, optional
        The non-inclusive ending frame.
    subtract : bool, optional
        If `True`, remove the dependents rather than adding.

    Returns
    -------
    None
    """
    n_frames = frame_valid.size
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames

    add_values = np.asarray(dp).flat
    singular = len(add_values) == 1
    if singular:
        singular_value = add_values[0]
    else:
        singular_value = 0.0

    if subtract:
        for frame in range(start_frame, end_frame):
            if not frame_valid[frame]:
                continue
            if singular:
                dependents[frame] -= singular_value
            else:
                dependents[frame] -= add_values[frame]
    else:
        for frame in range(start_frame, end_frame):
            if not frame_valid[frame]:
                continue
            if singular:
                dependents[frame] += singular_value
            else:
                dependents[frame] += add_values[frame]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def validate_frames(valid, cos_a, sin_a, native_sin_lat, native_cos_lat,
                    validated, has_telescope_info, mount,
                    cassegrain, gregorian,
                    nasmyth_corotating, prime_focus,
                    left_nasmyth, right_nasmyth):
    """
    Sets the cos(angle) and sin(angle) values is NaN.

    Parameters
    ----------
    valid : numpy.ndarray (bool)
        Set to `True` if a frame is valid and `False` otherwise.
    cos_a : numpy.ndarray (float)
        The cos(angle) values of shape (n_frames,).  Updated in-place.
    sin_a : numpy.ndarray (float)
        The sin(angle) values of shape (n_frames,).  Updated in-place.
    native_sin_lat : numpy.ndarray (float)
        The sin(latitude) values of the native coordinates with shape
        (n_frames,).
    native_cos_lat : numpy.ndarray (float)
        The cos(latitude) values of the native coordinates with shape
        (n_frames,).
    validated : numpy.ndarray (bool)
        Indicates if a frame has been checked.  If `True`, no action is taken.
        Otherwise, the validated flag will be set to `True` and cos_a/sin_a
        values will be fixed if NaN.
    has_telescope_info : numpy.ndarray (bool)
        If `True`, indicates that telescope information is available.
    mount : int
        The mount flag.
    cassegrain : int
        The Cassegrain mount flag.
    gregorian : int
        The Gregorian mount flag.
    nasmyth_corotating : int
        The Nasmyth corotating mount flag.
    prime_focus : int
        The prime focus mount flag.
    left_nasmyth : int
        The left Nasmyth mount flag.
    right_nasmyth : int
        The right Nasmyth mount flag.

    Returns
    -------
    None
    """
    n_frames = cos_a.size
    for frame in range(n_frames):
        if validated[frame]:
            continue

        if not valid[frame]:
            validated[frame] = True
            continue

        validated[frame] = True
        if not has_telescope_info[frame]:
            continue

        # Fix bad angles
        if np.isfinite(cos_a[frame]) and np.isfinite(sin_a[frame]):
            continue

        if mount == prime_focus:
            sin_a[frame] = 0.0
            cos_a[frame] = 1.0
            continue
        elif mount == left_nasmyth:
            sin_a[frame] = -native_sin_lat[frame]
            cos_a[frame] = native_cos_lat[frame]
            continue
        elif mount == right_nasmyth:
            sin_a[frame] = native_sin_lat[frame]
            cos_a[frame] = native_cos_lat[frame]
            continue
        else:
            sin_a[frame] = 0.0
            cos_a[frame] = 1.0


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def downsample_data(data, sample_flag, valid, window, start_indices):

    n_window = window.size
    high_frames, n_channels = data.shape
    low_frames = valid.size
    new_data = np.empty((low_frames, n_channels), dtype=nb.float64)
    new_flag = np.empty((low_frames, n_channels), dtype=nb.int64)

    for channel in range(n_channels):

        for new_frame in range(low_frames):
            if not valid[new_frame]:
                new_data[new_frame, channel] = np.nan
                new_flag[new_frame, channel] = 0
                continue
            start_index = start_indices[new_frame]
            d_value = 0.0
            d_flag = 0

            for window_index in range(n_window):
                scaling = window[window_index]
                if scaling == 0:
                    continue
                frame_index = start_index + window_index
                value = scaling * data[frame_index, channel]
                flag = sample_flag[frame_index, channel]
                d_value += value
                d_flag |= flag

            new_data[new_frame, channel] = d_value
            new_flag[new_frame, channel] = d_flag

    return new_data, new_flag
