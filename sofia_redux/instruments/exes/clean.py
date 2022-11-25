# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
import bottleneck as bn
import numpy as np

from sofia_redux.instruments.exes.get_badpix import get_badpix
from sofia_redux.toolkit.utilities.fits import set_log_level

__all__ = ['clean']


def clean(data, header, std, mask=None, radius=10, threshold=20.0,
          propagate_nan=False):
    """
    Correct bad pixels.

    Bad pixels to correct may be indicated in an input mask, with good
    values indicated with 1 or True and bad values indicated with 0 or False.

    Bad pixels may also be identified from a reference bad pixel mask on disk.
    The filename should be passed in as header['BPM'].  In this FITS image,
    pixels that are known to be bad are marked with a value of 0; good pixels
    are marked with a value of 1.

    Alternatively, bad pixels may be identified from their noise
    characteristics: if the standard deviation associated with a pixel is
    greater than a threshold value times the mean standard deviation
    for the frame, then it is marked as a bad pixel.  Bad pixels are
    corrected by using neighboring good values to linearly interpolate over
    the bad ones.  The search for good pixels checks first in the
    y-direction, then in the x-direction.  If good pixels cannot be
    identified within a 10-pixel radius, then the bad pixel will not be
    corrected.  If there is a different uncertainty frame for each input
    data frame, this algorithm should be run in a loop on each frame
    individually.

    Parameters
    ----------
    data : numpy.ndarray
        Data cube of shape (nframe, nspec, nspat) or image (nspec, nspat).
    header : fits.Header
        Header associated with the input data.  Will be updated in-place.
    std : numpy.ndarray
        2D or 3D uncertainty array (i.e. sqrt(variance)) of shape
        (nspec, nspat) or (nframe, nspec, nspat).  If a 2D array is
        passed in, it will be applied to all frames.
    mask : numpy.ndarray of int or bool, optional
        Bad pixel array of shape (nspec, nspat) or (nframe, nspec, nspat)
        indicating pixels to correct (good=1, bad=0).  If not None, will be
        updated with additional bad pixels found.  It is permitted for a
        2D mask to be applied over all frames.
    radius : int or array-like of int, optional
        The maximum distance over which to perform interpolation.  If an
        array is supplied, this is the maximum distance for each dimension
        in numpy (row, col) order (y, x).
    threshold : float, optional
        Threshold for bad pixel identification, as a factor to multiply
        by the standard deviation.
    propagate_nan : bool, optional
        If True, bad pixels in the data will be replaced by NaN rather
        than interpolated over.

    Returns
    -------
    data, std : numpy.ndarray, numpy.ndarray
        The cleaned data and error.
    """
    data, std, mask = _check_inputs(data, std, mask=mask)
    mask = _apply_badpix_mask(mask, header)

    if data.ndim == 3:
        cleaned = np.empty_like(data)
        cleaned_error = np.empty_like(data)
        separate_mask = mask.ndim == 3
        separate_std = std.ndim == 3

        for frame in range(data.shape[0]):
            m = mask[frame] if separate_mask else mask
            e = std[frame] if separate_std else std
            e, m = _check_noise(e, m, header, threshold)
            cleaned[frame], cleaned_error[frame] = _clean_frame(
                data[frame], e, m, radius=radius, propagate_nan=propagate_nan)

        if not separate_std:
            cleaned_error = cleaned_error[0]
        result = cleaned, cleaned_error
    else:
        std, mask = _check_noise(std, mask, header, threshold)
        result = _clean_frame(data, std, mask, radius=radius,
                              propagate_nan=propagate_nan)

    return result


def _apply_badpix_mask(mask, header):
    """
    Update mask with bad pixels from a reference file.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Bad pixel mask to be updated (True = good, False = bad).
    header : fits.Header
        Header containing bad pixel mask filename, in the BPM keyword.

    Returns
    -------
    mask : numpy.ndarray of bool
        The updated mask.
    """

    with set_log_level('WARNING'):
        bpm = get_badpix(header, clip_reference=True, apply_detsec=True)
    if bpm is None:
        return mask

    # update mask in place to track bad pixels
    if mask.ndim == 3:
        mask &= bpm[None]
    else:
        mask &= bpm
    return mask


def _check_noise(std, mask, header, stdfac):
    """
    Look for noisy or too quiet pixels.

    Parameters
    ----------
    std : numpy.ndarray
        Error values to check.
    mask : numpy.ndarray
        Bad pixel mask values.  Good pixel=True, bad pixel=False
    header : fits.Header
        FITS header associated with the data.
    stdfac : float
        Noisy pixel threshold, as a factor times the error.

    Returns
    -------
    std, mask : numpy.ndarray, numpy.ndarray
        The std and mask are updated in-place. Pixels with high error
        values are added to the mask as a False value. Pixels with low
        values are set to a new minimum.
    """

    if header.get('SCAN') in ['MAP', 'FILEMAP']:
        return std, mask

    if stdfac <= 0:
        return std, mask

    stdmean = bn.nanmean(std[mask])
    if stdmean == 0 or np.isnan(stdmean):
        return std, mask

    stdmax = stdmean * stdfac
    stdmin = stdmean / 16

    # todo: revisit statistics
    #  There may be better options for finding noisy pixels.

    # Mark pixels as bad if std > stdmax
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        high_std = std > stdmax
        nhigh = np.sum(high_std)
        if nhigh != 0:
            log.info(f"Found {nhigh} noisy pixels.")
            mask[high_std] = False

        # Set pixels with std < stdmin to stdmin
        np.clip(std, stdmin, None, out=std)

    return std, mask


def _clean_frame(data, std, mask, radius=10, propagate_nan=False):
    """
    Clean a single frame.

    Bad pixels are linearly interpolated over, using neighboring
    data.  Neighbors in the y-direction are preferentially used.

    Parameters
    ----------
    data : numpy.ndarray
        (M, N) of float
    std : numpy.ndarray
        (M, N) of float
    mask : numpy.ndarray
        (M, N) of bool
    radius : int or array_like of int, optional
        The maximum distance over which to perform interpolation.  If an
        array is supplied, this is the maximum distance for each dimension
        in numpy (row, col) order (y, x).
    propagate_nan : bool, optional
        If True, bad pixels in the data will be replaced by NaN rather
        than interpolated over.

    Returns
    -------
    cleaned_data, cleaned_std : numpy.ndarray, numpy.ndarray
        The cleaned data and updated error both of shape (M, N)
    """
    # mark any NaNs in the data if not already marked
    mask &= ~np.isnan(data)

    if mask.all():
        return data, std

    if propagate_nan:
        data[~mask] = np.nan
        std[~mask] = np.nan
        return data, std

    # todo: There may be better ways to do this interpolation
    ny, nx = data.shape
    nxny = nx * ny
    result = data.copy().ravel()
    error = std.copy().ravel()
    missing = ~mask
    y, x = np.mgrid[:ny, :nx]
    w = np.asarray(radius, dtype=int)
    if w.size == 1:
        w = np.full(2, int(w))

    missing_inds = (y * nx + x)[missing]
    offsety_inds = (np.arange(w[0] - 1) + 1)
    offsetx_inds = (np.arange(w[1] - 1) + 1)

    flat_mask = mask.ravel()
    x_missing = x[missing]
    y_missing = y[missing]
    y0inds = (y_missing * nx)[..., None]
    x0inds = x_missing[..., None]

    # search left of missing points
    search = np.subtract.outer(x_missing, offsetx_inds)
    left = search >= 0
    inds = search + y0inds
    inds[~left] = 0
    left &= flat_mask[inds]
    search[~left] = -1
    left = np.max(search, axis=1)

    # search right of missing points
    search = np.add.outer(x_missing, offsetx_inds)
    right = search < nx
    inds = search + y0inds
    inds[~right] = 0
    right &= flat_mask[inds]
    search[~right] = nxny
    right = np.min(search, axis=1)

    # search below missing points
    search = np.subtract.outer(y_missing, offsety_inds)
    bottom = search >= 0
    inds = search * nx + x0inds
    inds[~bottom] = 0
    bottom &= flat_mask[inds]
    search[~bottom] = -1
    bottom = np.max(search, axis=1)

    # search above missing points
    search = np.add.outer(y_missing, offsety_inds)
    top = search < ny
    inds = search * nx + x0inds
    inds[~top] = 0
    top &= flat_mask[inds]
    search[~top] = nxny
    top = np.min(search, axis=1)

    # Check if interpolation is possible and get necessary deltas
    x_ok = (left >= 0) & (right < nx)
    y_ok = (bottom >= 0) & (top < ny)
    dx0 = x_missing - left
    dy0 = y_missing - bottom
    xdist = dx0 + right - x_missing
    ydist = dy0 + top - y_missing

    # Interpolate across 1 pixel in x if nothing else is closer
    select = (x_ok & xdist == 2) & (~y_ok | (ydist > xdist))
    y0 = y_missing[select]
    d0 = data[y0, left[select]]
    d1 = data[y0, right[select]]
    e0 = std[y0, left[select]]
    e1 = std[y0, right[select]]
    replace = missing_inds[select]
    result[replace] = d0 + (d1 - d0) * dx0[select] / xdist[select]
    error[replace] = np.sqrt((e0 ** 2) + (e1 ** 2))

    # Otherwise use y-interpolation
    found = select.copy()
    select = ~select & y_ok
    x0 = x_missing[select]
    d0 = data[bottom[select], x0]
    d1 = data[top[select], x0]
    e0 = std[bottom[select], x0]
    e1 = std[top[select], x0]
    replace = missing_inds[select]
    result[replace] = d0 + (d1 - d0) * dy0[select] / ydist[select]
    error[replace] = np.sqrt((e0 ** 2) + (e1 ** 2))

    # Finally use x-interpolation if not possible any other way
    found |= select
    select = ~found & x_ok
    y0 = y_missing[select]
    d0 = data[y0, left[select]]
    d1 = data[y0, right[select]]
    e0 = std[y0, left[select]]
    e1 = std[y0, right[select]]
    replace = missing_inds[select]
    result[replace] = d0 + (d1 - d0) * dx0[select] / xdist[select]
    error[replace] = np.sqrt((e0 ** 2) + (e1 ** 2))

    found |= select
    if not found.all():
        log.warning("%i pixels could not be cleaned" % np.sum(~found))
    log.debug(f'Cleaned {(~mask).sum() - (~found).sum()} bad pixels.')

    return result.reshape((ny, nx)), error.reshape((ny, nx))


def _check_inputs(data, std, mask=None):
    """
    Check input parameters and standardize them to the required format.

    Parameters
    ----------
    data : numpy.ndarray
        Data cube of shape (nframe, nspec, nspat) or image (nspec, nspat).
    std : numpy.ndarray
        2D or 3D uncertainty array (i.e. sqrt(variance)) of shape
        (nspec, nspat) or (nframe, nspec, nspat).
    mask : numpy.ndarray of int or bool, optional
        Bad pixel array of shape (nspec, nspat) or (nframe, nspec, nspat)
        indicating pixels to correct (good=1, bad=0).

    Returns
    -------
    data, std, mask : 3-tuple of numpy.ndarray
        Standardized `data`, `std`, and `mask` arrays.
    """
    if mask is not None:
        log.debug(f'Mask: {mask.shape}, {mask.ndim}')
    data = np.asarray(data, dtype=float)
    if data.ndim not in [2, 3]:
        raise ValueError("Data must be a 2 or 3 dimensional array")
    shape = data.shape

    std = np.asarray(std, dtype=float)
    if std.ndim not in [2, 3]:
        raise ValueError("Std must be a 2 or 3 dimensional array")
    elif std.shape[-2:] != shape[-2:]:
        raise ValueError("Std and data shape mismatch")

    if mask is None:
        mask = np.ones(shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim not in [2, 3]:
            raise ValueError("Mask must be a 2 or 3 dimensional array")
        elif mask.shape[-2:] != shape[-2:]:
            raise ValueError("Mask and data shape mismatch")

    return data, std, mask
