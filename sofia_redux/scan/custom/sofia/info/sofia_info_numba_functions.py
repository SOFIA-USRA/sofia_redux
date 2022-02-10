# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['get_drift_corrections']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_drift_corrections(frame_utc, frame_valid, drift_utc_ranges,
                          drift_deltas):  # pragma: no cover
    """
    Get the drift corrections by linear interpolation between drifts.

    The drift correction will be given as:

    delta_i * (frame_time - start_i) / (end_i - start_i)

    where the start and end are the UTC start and end times for a given drift
    where start < frame_time <= end.  If the frame_time is greater than the
    maximum available drift end time, then extrapolation will be used.

    Parameters
    ----------
    frame_utc : numpy.ndarray (float)
        An array of shape (n_frames,) containing the frame UTC times.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` skips the correction
        calculation for that frame.
    drift_utc_ranges : numpy.ndarray (float)
        The start and end times for the drifts of shape (n_drifts, 2) where
        index [0, 0] gives the start time of the first drift, and index [0, 1]
        gives the end time of the first drift.
    drift_deltas : numpy.ndarray (float)
        The drift offset deltas of shape (n_drifts, 2) where index [0, 0] gives
        the delta of the first drift in the x-direction, and index [0, 1] gives
        the delta of the first drift in the y-direction.

    Returns
    -------
    drift_correction, extrapolation_frame : numpy.ndarray (float), int
        The drift correction is an array of shape (n_frames, 2) where index
        [0, 0] gives the correction for frame 0 in the x-direction and
        index [0, 1] gives the correction for frame 0 in the y-direction.
        The extrapolation frame will be set to a positive number indicating the
        frame from which extrapolation was required.  If no extrapolation
        occurred, this value is -1.
    """
    extrapolate_index = -1
    drift_index = 0
    n_frames = frame_utc.size
    n_drifts = drift_utc_ranges.shape[0]
    drift_correction = np.empty((n_frames, 2), dtype=nb.float64)
    if n_drifts == 0:
        for frame in range(n_frames):
            drift_correction[frame, 0] = 0.0
            drift_correction[frame, 1] = 0.0
        return drift_correction, extrapolate_index

    min_utc = drift_utc_ranges[0, 0]
    max_utc = drift_utc_ranges[0, 1]
    utc_span = max_utc - min_utc
    dx = drift_deltas[0, 0]
    dy = drift_deltas[0, 1]

    for frame in range(frame_utc.size):
        if not frame_valid[frame]:
            drift_correction[frame, 0] = 0.0
            drift_correction[frame, 1] = 0.0
            continue
        utc = frame_utc[frame]

        if extrapolate_index < 0 and (utc < min_utc or utc > max_utc):
            drift_index += 1
            if drift_index >= n_drifts:
                extrapolate_index = frame
                drift_index = -1

            min_utc = drift_utc_ranges[drift_index, 0]
            max_utc = drift_utc_ranges[drift_index, 1]
            utc_span = max_utc - min_utc
            dx = drift_deltas[drift_index, 0]
            dy = drift_deltas[drift_index, 1]

        dt = (utc - min_utc) / utc_span
        drift_correction[frame, 0] = dx * dt
        drift_correction[frame, 1] = dy * dt

    return drift_correction, extrapolate_index
