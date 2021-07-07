# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import bottleneck as bn
import numpy as np
from scipy.stats import describe

from sofia_redux.toolkit.utilities.func import slicer

__all__ = ['find_outliers', 'meancomb', 'medcomb', 'moments', 'robust_mask']


def get_sigma_ratio(data):
    med = bn.nanmedian(data)
    d0 = data - med
    dabs = np.abs(d0)
    mad = 1.482 * bn.nanmedian(dabs)
    if mad == 0:
        result = np.ones(data.shape)
        result[dabs != 0] = np.inf
        return result
    else:
        return dabs / mad


def find_outliers(data, threshold=5, keepnans=False, axis=None):
    """
    Determines the outliers in a distribution of data

    Computes the median and the median absolute deviation (MAD)
    1.482 * median(abs(x_i-x_med)) of the data and identifies data
    values as outliers if abs(x_i - x_med) / MAD > threshold where
    x_med is the median.

    Taken from mc_findoutliers in spextool

    Parameters
    ----------
    data : array_like of (int or float)
    threshold : int or float, optional
        The sigma threshold
    keepnans : bool, optional
        If True, do not flag NaNs as outliers
    axis : int, optional
        If axis is set, outliers are determined for each slice
        along the set axis.  i.e. if axis=0 for a 2-d array,
        then outliers will be determined by using statistics
        derived along each row rather than over the entirety
        of the data set.  This could be is useful when each
        "row" contains a unique dataset such as a spectrum.

    Returns
    -------
    numpy.ndarray
        boolean array of the same shape as `data`.  False
        indicates an outlier
    """
    data = np.asarray(data, dtype=float)
    threshold = abs(threshold)
    if not np.isfinite(data).any():
        return np.full(data.shape, keepnans)

    if axis is None:
        sr = get_sigma_ratio(data)
    else:
        sr = np.empty_like(data)
        for axi in range(data.shape[axis]):
            islice = slicer(data, axis, axi, ind=True)
            sr[islice] = get_sigma_ratio(data[islice])

    sr[np.isnan(sr)] = 0 if keepnans else np.inf
    return sr <= threshold


def meancomb(data, variance=None, mask=None, rms=False, axis=None,
             ignorenans=True, robust=0, returned=True, info=None):
    """
    (Robustly) averages arrays along arbitrary axes.

    This routine will combine data using either a straight mean
    or weighted mean.  If datavar are not given, then a straight
    mean, <x>, and square of the standard error of the mean,
    sigma_mu^2 = sigma^2/N are computed.  If datavar are given,
    then a weighted mean and corresponding variance on the mean
    are computed.

    Parameters
    ----------
    data : array_like of (int or float)
        (shape1) input data array
    mask : array_like of bool, optional
        (shape1) An optional mask of the same shape as data identifying
        pixels to use in the combination (True=good, False=bad).
    variance : array_like of float, optional
        (shape1) The variances of the data points with the same shape as
        data.  If given a weighted median is performed.
    rms : bool, optional
        Set to return the RMS error instead of the error on the
        mean.
    axis : int, optional
        Specifies on which axis the mean operation should be
        performed.
    ignorenans : bool, optional
       If True, NaNs will be ignored in all calculations
    robust : float, optional
        Set to the sigma threshold to throw bad data out.  A data
        point is identified as an outlier if abs(x_i - x_med)/MAD > thresh
        where x_med is the median and MAD is the median absolute
        deviation defined as 1.482*median(abs(x_i - x_med)).  Set to
        a non-integer/non-float or 0 to skip outlier rejection.
    returned : bool, optional
        Flag indicating whether a tuple ``(mean, meanvar)`` should be
        returned as output (True), or just the mean (False).  Default
        is True.
    info : dict, optional
        If supplied will be updated with:
            'mask' -> output mask as modified by mask, NaNs, and robust

    Returns
    -------
    mean, [variance_of_mean] : (tuple of) float or numpy.ndarray
        The mean along the specified axis.  When returned is `True`,
        return a tuple with the average as the first element and the
        variance of the mean as the second element.  The return type
        `numpy.float64`.  If returned, `variance_of_mean` is
        `numpy.float64`.  If keepdims then
    """
    data, variance, valid, outshape = prepare_combination(
        data, variance, mask, robust=robust, ignorenans=ignorenans,
        axis=axis, info=info)

    if not valid.any() or not np.isfinite(data[valid]).any():
        if outshape is None:
            return np.nan, 0.0 if returned else np.nan
        else:
            if returned:
                return np.full(outshape, np.nan), np.zeros(outshape)
            else:
                return np.full(outshape, np.nan)

    dovar = variance is not None
    weights = np.zeros(data.shape)
    if not dovar:
        weights[valid] = 1.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            weights[valid] = 1 / variance[valid]

    mean = np.ma.average(data, weights=weights, axis=axis)
    masked, badmask = False, None
    if isinstance(mean, np.ma.MaskedArray):
        badmask = mean.mask
        mean = mean.data
        mean[badmask] = np.nan
        masked = True

    if not returned:
        return mean

    # Calculate variance of mean
    if dovar:
        mvar = np.sum(weights, axis=axis)
        if axis is None:
            if mvar != 0:
                mvar = 1 / mvar
        else:
            if masked:
                mvar[badmask] = 0
            nzi = mvar != 0
            mvar[nzi] = 1 / mvar[nzi]
    else:
        ndat = np.sum(weights != 0, axis=axis)
        mvar = np.sum(data ** 2, axis=axis) - (ndat * (mean ** 2))
        nzi = ndat > 1
        if axis is None:
            if nzi:
                mvar /= ndat - 1
                if not rms:
                    mvar /= ndat
            else:
                mvar = 0.0  # pragma: no cover
        else:
            if masked:
                nzi[badmask] = False
            mvar[nzi] /= ndat[nzi] - 1
            mvar[~nzi] = 0
            if not rms:
                mvar[nzi] /= ndat[nzi]

    if axis is not None:
        mvar[~np.isfinite(mvar)] = 0

    return mean, mvar


def medcomb(data, variance=None, mask=None, mad=False, axis=None,
            ignorenans=True, robust=0, returned=True, info=None):
    """
    Combines a data set using median

    Combines the data together using a median.  An estimate of the
    error is given by computing the Median Absolute Deviation (MAD),
    where MAD = 1.482 * median(abs(x_i - x_med)), and then
    med_var = MAD^2 / N.

    Parameters
    ----------
    data : array_like (shape)
    mask : array_like (shape), optional
        An optional mask of the same shape as data identifying
        pixels to use in the combination (True=good, False=bad).
    variance : array_like (shape), optional
        The variances of the data points with the same shape as
        data.  If given a weighted median is performed.
    mad : bool, optional
        If True, the Median Absolute Deviation (MAD^2) is returned
        instead of MAD^2 / N.
    axis : int, optional
        Specifies on which axis the mean operation should be
        performed.
    ignorenans : bool, optional
       If True, NaNs will be ignored in all calculations.
    robust : float, optional
        Set to the sigma threshold to throw bad data out.  A data
        point is identified as an outlier if abs(x_i - x_med)/MAD > thresh
        where x_med is the median and MAD is the median absolute
        deviation defined as 1.482*median(abs(x_i - x_med)).  Set to
        a non-integer/non-float or 0 to skip outlier rejection.
    returned : bool, optional
        Flag indicating whether a tuple ``(mean, meanvar)`` should be
        returned as output (True), or just the mean (False).  Default
        is True.
    info : dict, optional
        If supplied will be updated with:
            'mask' -> output mask as modified by mask, NaNs, and robust

    Returns
    -------
    median, [variance] : float or numpy.ndarray
        The median array and the optional variance array (MAD^2 / N) if
        `returned=True`.
    """
    data, variance, valid, outshape = prepare_combination(
        data, variance, mask, robust=robust, ignorenans=ignorenans,
        axis=axis, info=info)

    if not ignorenans:
        nan_loc = np.all(np.isfinite(data), axis=axis, keepdims=True)
        valid &= nan_loc

    invalid = ~valid
    data[invalid] = np.nan
    if variance is not None:
        variance[invalid] = np.nan

    if not valid.any() or not np.isfinite(data[valid]).any():
        if outshape is None:
            return (np.nan, 0.0) if returned else np.nan
        else:
            if returned:
                return np.full(outshape, np.nan), np.zeros(outshape)
            else:
                return np.full(outshape, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', (RuntimeWarning, UserWarning))
        median = np.nanmedian(data, axis=axis, keepdims=True)
        inv_n = np.sum(valid, axis=axis, keepdims=True).astype(float)
        nzi = inv_n != 0
        inv_n[nzi] = 1 / inv_n[nzi]
        median = median.ravel()[0] if axis is None else slicer(median, axis, 0)

        if returned:
            if isinstance(variance, np.ndarray):
                var = (np.pi / 2) * np.nansum(
                    variance, axis=axis, keepdims=True) * (inv_n ** 2)
            else:
                var = (1.4826 * np.nanmedian(
                    np.abs(data - median), axis=axis, keepdims=True)) ** 2
                if not mad:
                    var *= inv_n
            var[~np.isfinite(var)] = 0
            var = var.ravel()[0] if axis is None else slicer(var, axis, 0)
            return median, var
        else:
            return median


def prepare_combination(data, variance, mask, robust=0.0, ignorenans=True,
                        axis=None, info=None):

    data = np.asarray(data).astype(float)
    shape = data.shape
    dovar = variance is not None
    if dovar:
        variance = np.atleast_1d(np.asarray(variance).astype(float))
        if variance.size == 1:
            variance = np.full(data.shape, variance[0])
        if variance.shape != shape:
            raise ValueError("data and variance shape mismatch")
    else:
        variance = None

    if mask is not None:
        invalid = np.logical_not(np.asarray(mask, dtype=bool))
        if invalid.shape != shape:
            raise ValueError("data and mask shape mismatch")
        data[invalid] = 0
        if dovar:
            variance[invalid] = 0
    else:
        invalid = np.full(data.shape, False)

    if robust > 0:
        invalid |= ~find_outliers(
            data, threshold=robust, keepnans=not ignorenans, axis=axis)
        data[invalid] = 0
        if dovar:
            variance[invalid] = 0

    if ignorenans:
        invalid |= ~np.isfinite(data)
        if dovar:
            invalid |= ~np.isfinite(variance)
        data[invalid] = 0
        if dovar:
            variance[invalid] = 0

    if dovar:
        invalid |= variance == 0

    valid = np.logical_not(invalid, out=invalid)
    if axis is None:
        outshape = None
    else:
        outshape = list(data.shape)
        del outshape[axis]
        outshape = tuple(outshape)

    if isinstance(info, dict):
        info['mask'] = valid

    return data, variance, valid, outshape


def robust_mask(data, threshold, mask=None, axis=None, mask_data=False,
                cval=np.nan):
    """
    Computes a mask derived from data Median Absolute Deviation (MAD).

    Calculates a robust mask based on the input data and optional input mask.
    If :math:`threshold > 0`, the dataset is searched for outliers.  Outliers
    are identified for point :math:`i` if

    .. math::

        \\frac{|y_i - median[y]|}{MAD} > threshold

    where :math:`MAD` is the Median Absolute Deviation defined as

    .. math::

        MAD = 1.482 * median[|y_i - median[y]|]

    Parameters
    ----------
    data : array_like of float
        The data on which to derive a robust mask.
    threshold : float
        Threshold as described above.
    mask : array_like of bool, optional
        If supplied, must be the same shape as `data`.  Any masked (`False`)
        `data` values will not be included in the :math:`MAD` calculation.
        Additionally, masked elements will also be masked (`False`) in the
        output mask.
    axis : int, optional
        Axis over which to calculate the :math:`MAD`.  The default (`None`)
        derives the :math:`MAD` from the entire set of `data`.
    mask_data : bool, optional
        If `True`, return a copy of `data` with masked values replaced by
        `cval` in addition to the output mask.  The default is `False`.
        Note that the output type will
    cval : int or float, optional
        if `mask_data` is set to `True`, masked values will be replaced by
        `cval`.  The default is `numpy.nan`.

    Returns
    -------
    numpy.ndarray of bool, [numpy.ndarray of numpy.float64]
        The output mask where `False` indicates a masked value, while `True`
        indicates that associated data deviation is below the `threshold`
        limit.  If `mask_data` was `True`, also returns a copy of `data`
        with masked values replaced by `cval`.
    """
    d = np.asarray(data, dtype=float).copy()
    valid = np.isfinite(d)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != d.shape:
            raise ValueError("data and mask shape mismatch")
        valid &= mask

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        if threshold is not None and threshold > 0:
            d[~valid] = np.nan
            if axis is None:
                med = bn.nanmedian(d)
                mad = 1.482 * bn.nanmedian(np.abs(d - med))
            else:
                med = np.expand_dims(bn.nanmedian(d, axis=axis), axis)
                mad = np.expand_dims(
                    1.482 * bn.nanmedian(
                        np.abs(d - med), axis=axis), axis)

            ratio = np.abs(d - med) / mad
            valid &= ratio <= threshold

    if mask_data:
        d[~valid] = cval
        return valid, d
    else:
        return valid


def moments(data, threshold=None, mask=None, axis=None, get_mask=False):
    """
    Computes statistics on a data set avoiding deviant points if requested

    Moments are calculated for a given set of data.  If a value is passed
    to threshold, then the dataset is searched for outliers.  A data point
    is identified as an outlier if abs(x_i - x_med)/MAD > threshold, where
    x_med is the median, MAD is the median absolute deviation defined as
    1.482 * median(abs(x_i - x_med)).

    Parameters
    ----------
    data : array_like of float
        (shape1) Data on which to calculate moments
    mask : array_like of bool
        (shape1) Mask to apply to data
    threshold : float, optional
        Sigma threshold over which values are identified as outliers
    axis : int, optional
        Axis over which to calculate statistics
    get_mask : bool, optional
        If True, only return the output mask

    Returns
    -------
    dict or numpy.ndarray
        If `get_mask` is False, returns a dictionary containing the
        following statistics: mean, var, stddev, skew, kurt, stderr,
        mask.  Otherwise, returns the output mask.
    """

    valid = robust_mask(data, threshold, mask=mask, axis=axis,
                        mask_data=np.logical_not(get_mask), cval=np.nan)

    if get_mask:
        return valid
    else:
        valid, d = valid

    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        stats = describe(d, nan_policy='omit', axis=axis)

    result['mask'] = valid
    result['mean'] = getattr(stats, 'mean')
    result['var'] = getattr(stats, 'variance')
    result['stddev'] = np.sqrt(result['var'])
    result['skewness'] = getattr(stats, 'skewness')
    result['kurtosis'] = getattr(stats, 'kurtosis')
    for k, v in result.items():
        if isinstance(v, np.ma.MaskedArray):
            result[k] = v.data

    return result
