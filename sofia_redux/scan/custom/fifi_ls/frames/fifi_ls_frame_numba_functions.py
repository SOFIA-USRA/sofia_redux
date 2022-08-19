# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

from sofia_redux.scan.channels.channel_numba_functions import \
    get_typical_gain_magnitude
from sofia_redux.scan.utilities.numba_functions import smart_median_1d


nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['validate', 'get_relative_frame_weights']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def validate(valid, validated, weight, check_coordinates, equatorial_null,
             equatorial_nan, horizontal_nan, chopper_nan, lst,
             site_lon, site_lat, telescope_vpa, instrument_vpa
             ):  # pragma: no cover
    r"""
    Utility function to validate HAWC+ frames following a data read.

    Checks the frame attributes specific to FIFI-LS observations for validity.

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
    weight : numpy.ndarray (float)
        The relative frame weights of shape (n_frames,).  Zero weighted frames
        will be marked as invalid.
    check_coordinates : numpy.ndarray (bool)
        If `True`, check the frame coordinates for validity.  Should be set to
        `False` if this was a lab observation without any real coordinates.
        If `False`, the equatorial, object, horizontal, chopper, lst, site,
        and vpa type values will not impact the validity of the frames.
        Should be an array of shape (n_frames,)
    equatorial_null : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that the
        equatorial coordinates are zero valued.  Any such frames will be
        marked as invalid.
    equatorial_nan : numpy.ndarray (bool)
        An array of shape (n_frames,) where `True` indicates that one or
        more of the equatorial coordinates for a given frame is NaN.  Any
        such frames will be marked as invalid.
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
    for i in range(n_frames):
        if validated[i]:
            continue
        if not valid[i]:
            validated[i] = True
            continue

        valid[i] = False  # Unflag at the end

        if weight[i] <= 0 or not np.isfinite(weight[i]):
            continue

        if not check_coordinates[i]:
            valid[i] = True
            continue

        if equatorial_null[i] or equatorial_nan[i] or horizontal_nan[i]:
            continue
        if np.isnan(lst[i]) or np.isnan(site_lon[i]) or np.isnan(site_lat[i]):
            continue
        if np.isnan(telescope_vpa[i]) or np.isnan(instrument_vpa[i]):
            continue
        if chopper_nan[i]:
            continue

        valid[i] = True


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_relative_frame_weights(variance):  # pragma: no cover
    """
    Calculate the relative frame weights from the input variance.

    Parameters
    ----------
    variance : numpy.ndarray (float)
        The variance array of shape (n_frames, n_pixels).

    Returns
    -------
    relative_weights : numpy.ndarray (float)
        The normalized frame weights of shape (n_frames,).
    """
    n_frames, n_pixels = variance.shape
    frame_weight = np.empty(n_frames, dtype=nb.float64)
    for frame in range(n_frames):
        frame_weight[frame] = get_typical_gain_magnitude(variance[frame])

    for frame, value in enumerate(frame_weight):
        if value == 1:
            frame_weight[frame] = np.nan

    typical_frame_variance = smart_median_1d(frame_weight)[0]
    for frame, value in enumerate(frame_weight):
        if np.isnan(value):
            frame_weight[frame] = 0.0
            continue
        frame_weight[frame] = typical_frame_variance / value

    return frame_weight
