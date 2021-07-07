# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import bottleneck as bn
import numpy as np
from scipy.ndimage import convolve1d, correlate1d
from scipy.signal import savgol_filter, savgol_coeffs

from sofia_redux.toolkit.utilities.func import to_array_shape


__all__ = ['savgol', 'savgol_windows', 'sobel']


def savgol(data, window, order=2, axes=None, check=True,
           is_error=False, scale=False, **kwargs):
    """
    Apply Savitzky-Golay filter to an array of arbitrary features

    Parameters
    ----------
    data : array_like (shape)
        Data to be filtered.  shape must have ndim elements.
    window : float or array_like of float (ndim,)
        The width of the filtering window in units of data spacing
        in each dimension.
    order : int or array_like of int (ndim,), optional
        The order of polynomial used to fit in each dimension.
    axes : array_like of int, optional
        The order in which to apply the filtering.  i.e. filter along
        dimension 0, then dimension 1 etc., or alternatively, just
        filter along select features.
    check : bool, optional
        If True, skip all checks on data validity and just solve.
        Note that if this is the case, both window and order should
        be supplied as (ndim,) arrays.
    is_error : bool, optional
        If True, assumes input data are error values and propagates
        accordingly
    scale : bool, optional
        If True, scale window to the average spacing between samples
        over each dimension.  Note that this replaces "width" in the
        old IDL version.  This option should not be used if working in
        multiple non-orthogonal features, as average spacing per
        dimension is taken as the average separation between ordered
        dimensional coordinates.
    kwargs : dict, optional
        Optional keywords to pass into scipy.signal.savgol_filter

    Returns
    -------
    filtered_data : numpy.ndarray
        The output type is of the same type and shape as "data", so be
        careful if using unsigned integers with kernels containing negative
        values.
    """
    if check:
        order, window = savgol_windows(
            order, window, data, scale=scale)

    have_nans = np.isnan(data).any()
    c_kwargs, s_kwargs = {}, {}
    for key in ['deriv', 'delta']:
        if key in kwargs:
            s_kwargs[key] = kwargs[key]
    for key in ['mode', 'cval']:
        if key in kwargs:
            c_kwargs[key] = kwargs[key]
    if have_nans:
        mode = c_kwargs.get('mode')
        if mode is None or mode == 'interp':
            c_kwargs['mode'] = 'nearest'

    if axes is None:
        axes = np.arange(data.ndim)
    result = data ** 2 if is_error else data

    for axis in axes:
        if not have_nans and not is_error:
            result = savgol_filter(
                result, window[axis], order[axis], axis=axis, **kwargs)
        else:
            coeffs = savgol_coeffs(window[axis], order[axis], **s_kwargs)
            if is_error:
                coeffs **= 2
            result = convolve1d(result, coeffs, axis=axis, **c_kwargs)

    return np.sqrt(result) if is_error else result


def savgol_windows(order, window, *samples, scale=False):
    """
    Creates the correct windows for given order and samples

    Also, performs error checks on samples.  Note that the order of
    samples is the same as the order of the features in the
    data values (samples[-1]).  For example, if working in two
    features samples[0] should be the y-coordinates and
    samples[1] should be the x-coordinates while samples[2]
    should have shape (len(samples[0]), len(samples[1])

    Parameters
    ----------
    order : int or array_like of int
        The order of the polynomial used to fit the windows. order
        must be less than "window".
    window : float or array_like of float
        The length of the Savitzky-Golay filter window.  Will be converted
        to a positive odd integer between order and less than the
        number of samples in that dimension.
    samples : (ndim+1)-tuple of array_like
        samples[-1] is a data array of ndim features.  samples[:ndim]
        give the coordinates of each dimension (1d-arrays).
    scale : bool, optional
        If True, scale window to the average spacing between samples
        over each dimension.  This option should be used with
        caution if working in multiple features that are not
        orthogonal to one another as the average spacing for each
        dimension is calculated by first ordering all unique
        coordinates.

    Returns
    -------
    orders, windows : numpy.ndarray, numpy.ndarray
        Both orders and windows are expanded to (ndim).  These values
        are suitable for passing into "savgol".
    """
    values = np.asarray(samples[-1])
    shape = values.shape
    ndim = len(samples) - 1
    if ndim == 0:
        scale = False
        ndim = values.ndim
        samples = tuple(np.arange(n, dtype=float) for n in values.shape)
    elif values.ndim != ndim:
        raise ValueError("samples[-1] must have len(samples)-1 features")

    windows = to_array_shape(window, ndim, dtype=float)
    orders = to_array_shape(order, ndim, dtype=int)

    if scale:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for dimi, win in enumerate(windows):
                ordered = np.unique(samples[dimi])
                deltas = ordered[1:] - ordered[:-1]
                windows[dimi] /= bn.nanmean(deltas)

    # must be a positive odd integer >= order < nsamples
    for dimi, (w, o) in enumerate(zip(windows, orders)):
        w = np.clip(abs(w), o, shape[dimi] - 1)
        windows[dimi] = w

    windows = np.asarray(windows, dtype=int)
    windows[windows % 2 == 0] += 1

    return orders, windows


def sobel(input_array, kderiv=(-1, 0, 1), kperp=(1, 2, 1), pnorm=1, doabs=True,
          axis=None, mode='reflect', cval=0.0, origin=0):
    """
    Edge enhancement Sobel filter for n-dimensional images.

    Applies a Sobel filter over each dimension and returns the p-norm
    (default=1).  When calculating G_i for dimension i, the
    convolution kernel will be formed from a convolution of `kaxis` along
    dimension i, and `kother` along all remaining dimensions.

    Parameters
    ----------
    input_array : array_like of (int or float)
        (shape) in n-features
    kderiv : array, optional
        The kernel operator in the derivate direction.
    kperp : array, optional
        The kernel operator perpendicular to the derivative direction.
    pnorm : int or float, optional
        If axis is `None`, 'p' value of the p-norm applied to the addition
        of convolution results over each axis.
    doabs : bool, optional
        If True, and `axis=None`, the absolute value of the result for each
        axis will be taken when calculating the p-norm for the final result.
    axis : int, optional
        If provided, the sobel filter will only be applied over this axis.
        Otherwise, the results for each axis will be combined according to
        `pnorm` and `doabs`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the input array is extended
        beyond its boundaries. Default is 'reflect'. Behavior for each valid
        value is as follows:

        'reflect' (`d c b a | a b c d | d c b a`)
            The input is extended by reflecting about the edge of the last
            pixel.

        'constant' (`k k k k | a b c d | k k k k`)
            The input is extended by filling all values beyond the edge with
            the same constant value, defined by the `cval` parameter.

        'nearest' (`a a a a | a b c d | d d d d`)
            The input is extended by replicating the last pixel.

        'mirror' (`d c b | a b c d | c b a`)
            The input is extended by reflecting about the center of the last
            pixel.

        'wrap' (`a b c d | a b c d | a b c d`)
            The input is extended by wrapping around to the opposite edge.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : int, optional
        Controls the placement of the filter on the input array's pixels.
        A value of 0 (the default) centers the filter over the pixel, with
        positive values shifting the filter to the left, and negative ones
        to the right.

    Returns
    -------
    edges : numpy.ndarray
        (shape) of the same type as `image`.
    """
    input_array = np.asarray(input_array)
    if pnorm == 0:
        raise ValueError("pnorm must not equal zero")

    if axis is not None:
        result = input_array.copy()
        for dimi in range(input_array.ndim):
            kernel = kderiv if dimi == axis else kperp
            correlate1d(result, kernel, axis=dimi, output=result,
                        mode=mode, cval=cval, origin=origin)
    else:
        result = np.zeros_like(input_array)
        for one_axis in range(input_array.ndim):
            output = np.copy(input_array)
            for dimi in range(input_array.ndim):
                kernel = kderiv if dimi == one_axis else kperp
                correlate1d(output, kernel, axis=dimi, output=output,
                            mode=mode, cval=cval, origin=origin)
            if doabs:
                output = abs(output)

            if pnorm == 1:
                result += output
            else:
                result += output ** pnorm

        if pnorm != 1:
            result **= (1.0 / pnorm)

    return result
