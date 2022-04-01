# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.lib import NumpyVersion
import scipy
from scipy import ndimage


__all__ = ['to_ndimage_mode', 'fix_ndimage_mode', 'clip_output',
           'map_coordinates']


def to_ndimage_mode(mode):
    """
    Convert from `numpy.pad` mode name to the corresponding ndimage mode.

    Parameters
    ----------
    mode : str

    Returns
    -------
    ndimage_mode : str
    """
    mode_translation_dict = dict(constant='constant', edge='nearest',
                                 symmetric='reflect', reflect='mirror',
                                 wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(
            (f"Unknown mode: '{mode}', or cannot translate mode. The "
             f"mode should be one of 'constant', 'edge', 'symmetric', "
             f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
             f"more info."))
    return fix_ndimage_mode(mode_translation_dict[mode])


def fix_ndimage_mode(mode):  # pragma: no cover
    """
    Allow translation of modes for scipy versions >= 1.6.0

    SciPy 1.6.0 introduced grid variants of constant and wrap which
    have less surprising behavior for images. Use these when available

    Parameters
    ----------
    mode : str
        The ndimage mode that may need to be fixed depending on the SciPy
        version.

    Returns
    -------
    ndimage_mode : str
    """
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    if NumpyVersion(scipy.__version__) >= '1.6.0':
        mode = grid_modes.get(mode, mode)
    return mode


def clip_output(original, warped, mode, cval, clip):
    """
    Clip the array to the range of original values.

    This operation is performed in-place, and only if `clip` is `True`.

    Parameters
    ----------
    original : numpy.ndarray
        The original data array values.
    warped : numpy.ndarray
        The warped data array values.
    mode : str
        Can take values of {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        for which points outside the boundaries of the input are filled
        according to the given mode.  Modes match the behaviour of
        :func:`np.pad`.
    cval : float
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.

    Returns
    -------
    None
    """
    if not clip:
        return

    min_val = np.nanmin(original)
    max_val = np.nanmax(original)
    nan_cval = np.isnan(cval)
    if mode == 'constant':
        if nan_cval:
            preserve_cval = True
        else:
            preserve_cval = min_val <= cval <= max_val
    else:
        preserve_cval = False

    if preserve_cval:
        if nan_cval:
            cval_mask = np.isnan(warped)
        else:
            cval_mask = warped == cval
    else:
        cval_mask = None

    np.clip(warped, min_val, max_val, out=warped)
    if cval_mask is not None:
        warped[cval_mask] = cval


def map_coordinates(data, coordinates, order=3, mode='constant', cval=np.nan,
                    output=None, clip=True, threshold=0.5):
    """
    A drop in replacement for :func:`ndimage.map_coordinates`.

    This function has been modified to handle NaN `cval` values with the
    'constant' `mode`.  The original method results in inconsistent results
    and occasionally fills the output with all NaN values.

    The array of coordinates is used to find, for each point in the output,
    the corresponding coordinates in the input. The value of the input at
    those coordinates is determined by spline interpolation of the
    requested order.

    The shape of the output is derived from that of the coordinate
    array by dropping the first axis. The values of the array along
    the first axis are the coordinates in the input array at which the
    output value is found.

    Parameters
    ----------
    data : numpy.ndarray
        The data array to map with n_dimensions dimensions with a size of N.
    coordinates : numpy.ndarray
        The coordinates at which `data` is evaluated.  Must be of shape
        (n_dimensions, N) or (n_dimensions, shape,) where product(shape) = N.
        Dimensions are ordered using the Numpy (y, x) convention.
    order : int, optional
        The order of the spline interpolation.  Must be in the range 0-5.
    mode : str, optional
        Can take values of {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        for which points outside the boundaries of the input are filled
        according to the given mode.  Modes match the behaviour of
        :func:`np.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    output : numpy.ndarray, optional
        The output array to fill with the results.  Should generally be the
        same shape as `data`.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    threshold : float, optional
        Used in conjunction with `cval`=NaN and `mode`='constant'.  Should
        generally take values in the range -1 to 1 with a default of 0.5.
        This is used to better apply NaN `cval` boundaries as expected.  Points
        inside the boundaries are mapped to 1, and values outside are mapped to
        -1.  Points which map to values >= `threshold` are considered valid,
        while others will be set to NaN in the output.

    Returns
    -------
    mapped_coordinates : numpy.ndarray
        The result of transforming the data. The shape of the output is
        derived from that of `coordinates` by dropping the first axis, or
        via `output`.
    """
    # Pre-filtering not necessary for order 0 or 1 interpolation
    prefilter = order > 1
    ndi_mode = to_ndimage_mode(mode)
    if np.isnan(cval) and ndi_mode == 'grid-constant':
        # Need to do the masking manually
        if output is not None:
            mask_output = output.copy()
        else:
            mask_output = None
        warped = ndimage.map_coordinates(
            data, coordinates, prefilter=prefilter, output=output,
            mode='nearest', order=order, cval=0.0)
        mask = np.ones_like(data, dtype=float)
        warped_mask = ndimage.map_coordinates(
            mask, coordinates, prefilter=prefilter, output=mask_output,
            mode='grid-constant', order=order, cval=-1.0)
        replace = warped_mask < threshold
        warped[replace] = cval
    else:
        warped = ndimage.map_coordinates(
            data, coordinates, prefilter=prefilter, output=output,
            mode=ndi_mode, order=order, cval=cval)

    clip_output(data, warped, mode, cval, clip)
    return warped
