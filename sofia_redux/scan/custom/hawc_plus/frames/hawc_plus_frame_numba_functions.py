# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['validate', 'dark_correct', 'downsample_hwp_angle']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def validate(valid, validated, status, chop_length, chopping,
             use_between_scans, normal_observing_flag, between_scan_flag,
             transit_tolerance, chopper_amplitude, check_coordinates,
             non_sidereal, equatorial_null, equatorial_nan, object_null,
             horizontal_nan, chopper_nan, lst, site_lon, site_lat,
             telescope_vpa, instrument_vpa):  # pragma: no cover
    r"""
    Utility function to validate HAWC+ frames following a data read.

    Checks the frame attributes specific to HAWC+ observations for validity.

    Parameters
    ----------
    valid : numpy.ndarray (bool)
        The validity array where `False` marks an invalid frame that should
        not be included in any subsequent operations of shape (n_frames,).
        Will be updated in-place.  A `False` value on entry overrides any
        further analysis and will remain so.
    validated : numpy.ndarray (bool)
        A boolean array of shape (n_frames,) where `True` indicates that
        a frame has already gone through the validation process and should
        not be re-validated.
    status : numpy.ndarray (int)
        A status array in which values correspond to an observation type.
        Those that do not correspond to FITS_FLAG_NORMAL_OBSERVING will be
        flagged as invalid.  If `use_between_scans` is `True` and the status
        is equal to FITS_FLAG_BETWEEN_SCANS, the frame will also be marked
        as invalid. Should be an array of shape (n_frames,).
    chop_length : numpy.ndarray (float)
        The chopper offset distance from the nominal (x=0, y=0) position in
        arcseconds.  If
        \|`chop_length` - `chopper_amplitude`\| > `transit_tolerance`, then
        the frame will be marked as invalid.  Should be an array of shape
        (n_frames,).
    chopping : bool
        `True` if the chopper was used during the observation, and `False`
        otherwise.
    use_between_scans : bool
        `True` if data was continually taken between scans.
    normal_observing_flag : int
        The integer flag equivalent to the FITS_FLAG_NORMAL_OBSERVING flag.
        Any `status` value that is not equal to this is flagged as invalid.
    between_scan_flag : int
        The integer flag equivalent to the FITS_FLAG_BETWEEN_SCANS flag.  If
        `use_between_scans` is `True`, data will be flagged as invalid if it's
        `status` matches this value.
    transit_tolerance : float
        The chopper maximum deviation above the chopper amplitude above which
        a frame will be marked as invalid in arcseconds.  Please see
        `chop_length` for further details.
    chopper_amplitude : float
        The expected chopper amplitude in arcseconds.  Please see `chop_length`
        for further details.
    check_coordinates : numpy.ndarray (bool)
        If `True`, check the frame coordinates for validity.  Should be set to
        `False` if this was a lab observation without any real coordinates.
        If `False`, the equatorial, object, horizontal, chopper, lst, site,
        and vpa type values will not impact the validity of the frames.
        Should be an array of shape (n_frames,)
    non_sidereal : bool
        If `True`, the frames should contain valid object coordinates.  If not,
        they will be marked as invalid.
    equatorial_null : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that the
        equatorial coordinates are zero valued.  Any such frames will be
        marked as invalid.
    equatorial_nan : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that one or
        more of the equatorial coordinates for a given frame is NaN.  Any
        such frames will be marked as invalid.
    object_null : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that the object
        coordinates for a given frame are zeroed.  Any such frames will be
        marked as invalid.
    horizontal_nan : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that the
        horizontal coordinates for a given frame are NaN.  Any such frames
        will be marked as invalid.
    chopper_nan : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that one or
        more of the chopper coordinates for a given frame is set to NaN.
        If the chopper is used, then any such frame will be marked as invalid.
    lst : numpy.ndarray (float)
        The local-sidereal-time values in unix seconds of shape (n_frames,).
        If any LST value is set to NaN, the associated frame will be marked
        as invalid.
    site_lon : numpy.ndarray (float)
        The site longitude coordinates of shape (n_frames,) in arbitrary units.
        If any longitude value is NaN, the associated frame will be marked as
        invalid.
    site_lat : numpy.ndarray (float)
        The site latitude coordinates of shape (n_frames,) in arbitrary units.
        If any latitude value is NaN, the associated frame will be marked as
        invalid.
    telescope_vpa : numpy.ndarray (float)
        The telescope VPA in arbitrary angle units of shape (n_frames,).
        If any value is set to NaN, the associated frame will be marked as
        invalid.
    instrument_vpa : numpy.ndarray (float)
        The instrument VPA in arbitrary angle units of shape (n_frames,).
        If any value is set to NaN, the associated frame will be marked
        as invalid.

    Returns
    -------
    None
    """
    n_frames = valid.size
    check_chopper = (chopping and np.isfinite(transit_tolerance)
                     and np.isfinite(chopper_amplitude))

    for i in range(n_frames):
        if validated[i]:
            continue
        if not valid[i]:
            validated[i] = True
            continue

        valid[i] = False  # Unflag at the end

        # Skip data that is not normal observing
        if status[i] != normal_observing_flag:
            continue
        # In practice this would never be hit, but oh well.
        if use_between_scans and (status[i] == between_scan_flag):
            continue

        if check_chopper:
            dev = np.abs(chop_length[i] - chopper_amplitude)
            if dev > transit_tolerance:
                continue

        if not check_coordinates[i]:
            valid[i] = True
            continue

        if equatorial_null[i] or equatorial_nan[i] or horizontal_nan[i]:
            continue
        if non_sidereal and object_null[i]:
            continue
        if np.isnan(lst[i]) or np.isnan(site_lon[i]) or np.isnan(site_lat[i]):
            continue
        if np.isnan(telescope_vpa[i]) or np.isnan(instrument_vpa[i]):
            continue
        if chopper_nan[i]:
            continue

        valid[i] = True


@nb.njit(cache=True, nogil=False, parallel=False)
def dark_correct(data, valid_frame, channel_indices, squid_indices
                 ):  # pragma: no cover
    """
    Perform the dark correction.

    The dark squid correction simply subtracts the squid channel frame data
    from all given channel frame data.  I.e.::

       data_out[:, channel[i]] = data_in[:, channel[i]] - data_in[:, squid[i]]

    Parameters
    ----------
    data : numpy.ndarray (float)
        The frame data of shape (n_frames, all_channels,).  The data will be
        updated in-place.
    valid_frame : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` excludes a frame
        from processing.
    channel_indices : numpy.ndarray (int)
        The channel indices for which to apply the dark correction of shape
        (n_channels,).  A value of -1 is ignored.
    squid_indices : numpy.ndarray (int)
        The reference indices used to subtract the correction of shape
        (n_channels,).  A value of -1 is ignored and not applied.

    Returns
    -------
    None
    """
    for frame in range(data.shape[0]):
        if not valid_frame[frame]:
            continue
        for channel_index, squid_index in zip(channel_indices, squid_indices):
            if squid_index != -1 and channel_index != -1:
                data[frame, channel_index] -= data[frame, squid_index]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def downsample_hwp_angle(hwp_angle, start_indices, valid, window
                         ):  # pragma: no cover
    """
    Return a downsampled HWP angle array.

    Downsamples the half-wave-plate angle via convolution with a window kernel.

    Parameters
    ----------
    hwp_angle : numpy.ndarray (float)
        The HWP angles to downsample of shape (high_res_frames,)
    start_indices : numpy.ndarray (int)
        The indices of marking the start index of the high resolution HWP
        angle array for each low resolution index.  Should be of shape
        (low_res_frames,).
    valid : numpy.ndarray (bool)
        A boolean array indicating whether any downsampled value will
        be valid. Should be of shape (low_res_frames,).  Any invalid
        low-resolution frames will be set to NaN.
    window : numpy.ndarray (float)
        The convolution kernel of shape (n_window,).  Since no re-weighting
        is performed, the window should be normalized if applicable.

    Returns
    -------
    low_res_hwp : numpy.ndarray (float)
    """
    n_window = window.size
    low_frames = valid.size
    low_res_hwp = np.zeros(low_frames, dtype=nb.float64)
    max_frames = hwp_angle.size
    for frame in range(low_frames):
        if not valid[frame]:
            low_res_hwp[frame] = np.nan
            continue
        start_index = start_indices[frame]
        for window_index in range(n_window):
            scaling = window[window_index]
            if scaling == 0:
                continue
            frame_index = start_index + window_index
            # The user should make sure this doesn't happen,
            # but just in case...
            if frame_index >= max_frames:
                low_res_hwp[frame] = np.nan
                break
            else:
                low_res_hwp[frame] += scaling * hwp_angle[frame_index]
    return low_res_hwp
