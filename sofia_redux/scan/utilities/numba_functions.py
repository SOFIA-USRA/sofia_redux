# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math
import numba as nb
import numpy as np

from sofia_redux.toolkit.splines import spline_utils

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['smart_median_1d', 'smart_median_2d', 'smart_median',
           'roundup_ratio', 'level', 'smooth_1d', 'gaussian_kernel',
           'mean', 'box_smooth_along_zero_axis', 'log2round', 'log2ceil',
           'pow2round', 'pow2floor', 'pow2ceil', 'regular_kernel_convolve',
           'regular_coarse_kernel_convolve', 'smooth_values_at',
           'smooth_value_at', 'point_aligned_smooth', 'point_smooth',
           'sequential_array_add', 'index_of_max', 'robust_mean']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def smart_median_1d(values, weights=None, max_dependence=1.0
                    ):  # pragma: no cover
    """
    Returns the weighted median of values in one dimension.

    The weighted median is defined as the value at which the cumulative weight
    of the sorted values is equal to half of the weight sum.  If this occurs
    at a midpoint between two values, the weighted mean of the two neighboring
    values is returned.  NaN values and weights are not included in any
    computation.

    This is a numba compiled function.

    Parameters
    ----------
    values : numpy.ndarray (float)
        An array of values (size,).
    weights : numpy.ndarray (float), optional
        An array of weights (size,).
    max_dependence : float
        A value between 0 and 1.  If the maximum weight value is greater than
        max_dependence * sum(weights), the median calculation is aborted and
        the weighted mean is calculated instead.

    Returns
    -------
    median, weight : float, float
        The median value and weight.
    """
    do_weights = weights is not None

    if do_weights:
        wts = weights
    else:
        wts = np.empty(0, dtype=nb.float64)

    if values.size == 1:
        if do_weights:
            return values[0], wts[0]
        else:
            return values[0], 1.0

    idx = np.argsort(values)
    values = values[idx]

    if do_weights:
        wts = wts[idx]

    n = values.size
    if do_weights:
        wt = 0.0
        w_max = 0.0
        for i in range(n):
            w = wts[i]
            if np.isnan(w):
                continue
            wt += w
            if w > w_max:
                w_max = w

        # If a single datum dominates, return the weighted mean
        if w_max >= (wt * max_dependence):
            v_sum = 0.0
            w_sum = 0.0
            for i in range(n):
                v = values[i]
                if np.isnan(v):
                    continue
                w = wts[i]
                if np.isnan(w):
                    continue
                v_sum += w * v
                w_sum += w
            if w_sum == 0:
                return 0.0, 0.0
            else:
                return v_sum / w_sum, w_sum
    else:
        wt = np.nan

    # Trim and NaNs from the end
    new_n = 0
    for i in range(n):
        if np.isnan(values[i]):
            break
        new_n += 1

    if new_n != n:
        values = values[:n]
        if do_weights:
            wts = wts[:n]
        n = new_n

    # If all weights are zero, return the arithmetic median...
    if wt == 0 or np.isnan(wt):
        if n % 2 == 0:
            nd2 = n // 2
            result = 0.5 * (values[nd2 - 1] + values[nd2])
        else:
            result = values[(n - 1) // 2]
        if wt == 0:
            return result, 0.0  # All weights were zero (should not happen)
        else:
            return result, float(n)  # no weights were provided

    mid_w = 0.5 * wt
    i = 0
    last_weight, last_value = 0.0, 0.0
    w, v = wts[i], values[i]
    wi = w

    while wi < mid_w and i < n:
        i += 1
        w_test = wts[i]
        if w_test <= 0 or np.isnan(w_test):
            continue
        last_value = v
        last_weight = w
        v = values[i]
        w = w_test
        wi += 0.5 * (last_weight + w)

    w_plus = wi
    w_minus = wi - (0.5 * (last_weight + w))
    w1 = (w_plus - mid_w) / (w_plus + w_minus)
    value = (w1 * last_value) + ((1.0 - w1) * v)
    weight = wt
    return value, weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def smart_median_2d(values, weights=None, max_dependence=0.0
                    ):  # pragma: no cover
    """
    Returns the weighted median of values in two dimensions.

    For a 2-dimensional array of shape (m, n), the median calculation is
    performed over the 2nd dimension such that the resulting output arrays
    are of shape (m,).

    The weighted median is defined as the value at which the cumulative weight
    of the sorted values is equal to half of the weight sum.  If this occurs
    at a midpoint between two values, the weighted mean of the two neighboring
    values is returned.  NaN values and weights are not included in any
    computation.

    This is a numba compiled function.

    Parameters
    ----------
    values : numpy.ndarray (float)
        An array of values (m, n).
    weights : numpy.ndarray (float)
        An array of weights (m, n).
    max_dependence : float
        A value between 0 and 1.  If the maximum weight value is greater than
        max_dependence * sum(weights) the median calculation is aborted, and
        the weighted mean is calculated instead.

    Returns
    -------
    median, weight : numpy.ndarray (float), numpy.ndarray (float)
        The median value and weights evaluated over the 2nd dimension (n).  The
        returned arrays will be of shape (m,).
    """
    n = values.shape[0]
    result_v = np.empty(n, dtype=nb.float64)
    result_w = np.empty(n, dtype=nb.float64)
    do_weights = weights is not None

    for i in range(n):
        if do_weights:
            v, w = smart_median_1d(values=values[i], weights=weights[i],
                                   max_dependence=max_dependence)
        else:
            v, w = smart_median_1d(values=values[i],
                                   max_dependence=max_dependence)
        result_v[i] = v
        result_w[i] = w

    return result_v, result_w


def smart_median(values, weights=None, axis=None, max_dependence=1.0):
    """
    Returns the weighted median of values in K-dimensions.

    The weighted median is defined as the value at which the cumulative weight
    of the sorted values is equal to half of the weight sum.  If this occurs
    at a midpoint between two values, the weighted mean of the two neighboring
    values is returned.  NaN values and weights are not included in any
    computation.

    This is a numba compiled function.

    Parameters
    ----------
    values : numpy.ndarray (float)
        An array values of shape (shape)
    weights : numpy.ndarray (float)
        An array of weights of shape (shape)
    axis : int, optional
        The axis over which to perform the median calculation.  The default of
        `None` applies the median calculation over all values.
    max_dependence : float
        A value between 0 and 1.  If the maximum weight value is greater than
        max_dependence * sum(weights) the median calculation is aborted, and
        the weighted mean is calculated instead.

    Returns
    -------
    median, weight : numpy.ndarray (float), numpy.ndarray (float)
        The median value(s) and weight(s).
    """

    if weights is None:
        weights = np.ones(values.shape, dtype=np.float64)

    if axis is None or (values.ndim == 1):
        return smart_median_1d(values.ravel(), weights.ravel(),
                               max_dependence=max_dependence)

    v = np.moveaxis(values, axis, -1)
    w = np.moveaxis(weights, axis, -1)
    out_shape = v.shape[:-1]
    flat_shape = np.prod(out_shape), v.shape[-1]
    v = v.reshape(flat_shape)
    w = w.reshape(flat_shape)

    v_out, wout = smart_median_2d(v, w, max_dependence=max_dependence)
    v_out = np.reshape(v_out, out_shape)
    wout = np.reshape(wout, out_shape)
    return v_out, wout


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def roundup_ratio(a, b):  # pragma: no cover
    """
    Returns int((a + b - 1) / b).

    Parameters
    ----------
    a : int or float
    b : int or float

    Returns
    -------
    ratio : int
        The roundup ratio.
    """
    return int((a + b - 1) / b)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def level(values, start=0, end=-1, resolution=1.0):  # pragma: no cover
    """
    Subtract the average value from all values and return the average.

    The average is calculated between the start and end indices of the supplied
    values.  Leveling is applied inplace on the values (between start and end
    index), and the average value is returned to the caller.

    Parameters
    ----------
    values : numpy.ndarray (float)
        The array to level
    start : int, optional
        The start index from which to start the leveling.  The default is the
        first index (0).
    end : int, optional
        The index at which to stop leveling.  The default is the last index
        (-1).
    resolution : int or float, optional
        The resolution of the indices.  If not 1, the start and end index are
        modified to start // resolution, and ceil(end / resolution).

    Returns
    -------
    average : float
        The average value subtracted from all values.
    """
    if end < 0:
        end = end + values.size + 1

    start = int(start // resolution)
    end = min(roundup_ratio(end, resolution), values.size)
    v_sum = 0.0
    n = 0

    for i in range(start, end):
        value = values[i]
        if not np.isnan(value):
            v_sum += value
            n += 1

    if n == 0:
        return 0.0

    ave = v_sum / n
    for i in range(start, end):
        values[i] -= ave

    return ave


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def smooth_1d(values, kernel):  # pragma: no cover
    """
    Apply 1-D kernel convolution in place.

    Smooths an array of values using kernel convolution of the form:

        s_i = sum_{j=1}^{n}(k_j * x_{i - n//2 + j}) / sum(k)

    I.e. the smoothed value for point i is the convolution of the kernel with
    the values where the kernel is centered over point i, where the center of
    the kernel is defined as n // 2 (integer) and n is the size of the kernel.
    Therefore, smoothing with an odd sized kernel is advisable to avoid any
    offset in the output convolution.

    Any NaN values will be ignored, resulting in zero-contribution to the
    smoothed values.  As such, no NaN values will be present following the
    smooth operation.

    Parameters
    ----------
    values : numpy.ndarray of float
        The array to convolve.  Will be updated in-place.
    kernel : numpy.ndarray of float
        The kernel by which to convolve.

    Returns
    -------
    None
    """
    ic = kernel.size // 2
    smoothed = np.empty(values.size, dtype=nb.float64)
    width = kernel.size

    for i in range(values.size):

        sum_wv = 0.0
        sum_w = 0.0

        for j in range(width):  # kernel index

            vi = i - ic + j  # value index for the kernel index centered on i
            if vi < 0 or vi >= values.size:
                continue

            v = values[vi]
            w = kernel[j]
            if np.isnan(v):
                continue

            sum_wv += v * w
            sum_w += np.abs(w)

        if sum_w > 0:
            smoothed[i] = sum_wv / sum_w
        else:
            smoothed[i] = 0.0

    for i in range(values.size):
        values[i] = smoothed[i]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def gaussian_kernel(n, sigma):  # pragma: no cover
    """
    Create a Gaussian convolution kernel.

    The Gaussian will be centered on the `n // 2` element with a maximum value
    of 1 when n is odd.  The resultant kernel will be of size n, and be created
    according to:

        g_i = exp(-(i - n//2)^2 / 2sigma^2)

    where i ranges from 0 to n - 1.

    Parameters
    ----------
    n : int
        Size of the kernel.
    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    kernel : numpy.ndarray (float)
        The Gaussian kernel
    """
    kernel = np.empty(n, dtype=nb.float64)
    ic = n // 2
    a = -0.5 / (sigma * sigma)
    for i in range(n):
        x2 = i - ic
        x2 *= x2
        kernel[i] = math.exp(a * x2)
    return kernel


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def mean(values, weights=None, start=0, end=-1):  # pragma: no cover
    """
    Return the (optionally weighted) mean value of an array.

    NaN and zero weighted values are not included in the calculation.  All
    arrays must be one dimensional.

    Parameters
    ----------
    values : numpy.ndarray (float)
        An array of values with shape (n,).
    weights : numpy.ndarray (float), optional
        An array of weights with shape (n,).
    start : int, optional
        The inclusive start index of the values included in the mean
        calculation.  The default is the first value (0).
    end : int, optional
        The non-inclusive end index for the values included in the mean
        calculation.  Negative values are wrapped to values.size + 1 + end.
        The default is the last value (-1).

    Returns
    -------
    mean, weight : float, float
        The mean value and weight sum.
    """
    if end < 0:
        end = end + values.size + 1

    sum_v = 0.0
    sum_w = 0.0

    weighted = weights is not None

    for i in range(start, end):
        v = values[i]
        if np.isnan(v):
            continue
        if weighted:
            w = weights[i]
            if w == 0:
                continue
        else:
            w = 1.0

        sum_v += w * v
        sum_w += w

    if sum_w == 0:
        return 0.0, 0.0

    return sum_v / sum_w, sum_w


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def box_smooth_along_zero_axis(values, bin_size, valid=None, fill_value=np.nan
                               ):  # pragma: no cover
    """
    Smooth a 2-D array of values along the zeroth axis.

    Results will only be generated for those indices where the bin range is
    fully inside the array range.  All other values will be set to the
    `fill_value` (NaN by default).

    Parameters
    ----------
    values : numpy.ndarray (float)
        The values to smooth as an array of shape (dimensions, N).  Smoothing
        occurs along each dimension (axis=1).
    bin_size : int
        The smoothing bin size.
    valid : numpy.ndarray (int)
        A boolean masked array where `True` indicates a value that may be used.
    fill_value : float, optional
        The value with which to fill smoothed values that do not have enough
        data to calculate.

    Returns
    -------
    smoothed : numpy.ndarray (float)
        The smoothed values of shape (dimensions, N).
    """
    if bin_size <= 1:
        return values.astype(nb.float64)

    dimensions, size = values.shape
    result = np.full(values.shape, fill_value, dtype=nb.float64)
    sums = np.zeros(dimensions, dtype=nb.float64)

    bin_left = bin_size // 2
    bin_right = bin_size - bin_left
    if (bin_size % 2) == 0:
        bin_left -= 1
    else:
        bin_right -= 1

    max_i = size - bin_right

    if valid is None:
        valid = np.full(size, True)

    # Looping over values at start of the array (not included in result)
    n_valid = 0
    for i in range(bin_size - 1):
        if valid[i]:
            n_valid += 1
            for dimension in range(dimensions):
                sums[dimension] += values[dimension, i]

    # Start populating results
    for i in range(bin_left, max_i):

        right_i = i + bin_right
        if valid[right_i]:
            n_valid += 1
            for dimension in range(dimensions):
                sums[dimension] += values[dimension, right_i]

        if n_valid > 0:
            for dimension in range(dimensions):
                result[dimension, i] = sums[dimension] / n_valid

        left_i = i - bin_left
        if valid[left_i]:
            n_valid -= 1
            for dimension in range(dimensions):
                sums[dimension] -= values[dimension, left_i]

    return result


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def log2round(x):  # pragma: no cover
    """
    Return the rounded value of log_2(x).

    Parameters
    ----------
    x : int or float

    Returns
    -------
    int
    """
    return int(round_value(np.log2(x)))


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def log2ceil(x):  # pragma: no cover
    """
    Return the ceiled value of log_2(x).

    Parameters
    ----------
    x : int or float

    Returns
    -------
    int
    """
    return int(np.ceil(np.log2(x)))


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def pow2round(x):  # pragma: no cover
    """
    Return 2 to the power of round(log_2(x)).

    Rounds a number to the nearest (log_2 scale) power of 2.

    Parameters
    ----------
    x : int or float

    Returns
    -------
    int
    """
    return 1 << log2round(x)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def pow2floor(x):  # pragma: no cover
    """
    Return 2 to the power of floor(log_2(x)).

    Finds where 2^n <= x < 2^(n+1) and returns 2^n.

    Parameters
    ----------
    x : int or float

    Returns
    -------
    int
    """
    return 2 ** int(np.log2(x))


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def pow2ceil(x):  # pragma: no cover
    """
    Return 2 to the power of ceil(log_2(x)).

    Finds where 2^(n - 1) <= x < 2^n and returns 2^n.

    Parameters
    ----------
    x : int or float

    Returns
    -------
    int
    """
    value = pow2floor(x)
    return value if value == x else value * 2


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def regular_kernel_convolve(data, kernel, kernel_reference_index=None,
                            weight=None, valid=None):  # pragma: no cover
    """
    Direct convolution of ND data with a given kernel.

    Fast Numba implementation of direct kernel convolution for arbitrary
    dimensions.  The actual convolution is performed on a point-by-point basis
    with :func:`point_aligned_smooth`, while the majority of this function
    relates to process reducing N-dimensions to a single dimension.  Note that
    this is not suitable for 1-dimensional convolution.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The data to convolve.
    kernel : numpy.ndarray (float)
        The kernel to convolve the data with.  Should have the same
        dimensionality as `data`.
    kernel_reference_index : numpy.ndarray (int), optional
        The index marking the center of the kernel.  If not provided, defaults
        to ceil((kernel.shape - 1) / 2).
    weight : numpy.ndarray (float), optional
        An optional weighting array.
    valid : numpy.ndarray (bool), optional
        An optional validating array where `False` excludes a datum from
        inclusion in processing.  Effectively the same as setting weight[i] = 0
        for index i.

    Returns
    -------
    convolved, smoothed_weights : numpy.ndarray (float), numpy.ndarray (float)
    """
    weighted = weight is not None
    validated = valid is not None

    if weighted:
        flat_weight = weight.ravel()
    else:  # For numba compilation
        flat_weight = np.empty(0, dtype=nb.float64)

    if validated:
        flat_valid = valid.ravel()
    else:  # For numba compilation
        flat_valid = np.empty(0, dtype=nb.b1)

    kernel_shape = np.atleast_1d(np.asarray(kernel.shape))

    if kernel_reference_index is None:
        reference_index = (kernel_shape - 1) / 2.0
        offset = (reference_index % 1) != 0
        reference_index = reference_index.astype(nb.int64) + offset
    else:
        reference_index = kernel_reference_index.astype(nb.int64)

    result = np.empty(data.shape, dtype=nb.float64)
    result_weight = np.empty(data.shape, dtype=nb.float64)
    flat_result = result.ravel()
    flat_result_weight = result_weight.ravel()

    data_shape = np.asarray(data.shape)
    kernel_indices, _, kernel_steps = spline_utils.flat_index_mapping(
        kernel_shape)
    data_indices, _, data_steps = spline_utils.flat_index_mapping(data_shape)

    n_data = data.size
    flat_kernel = kernel.ravel()
    flat_data = data.ravel()

    for i in range(n_data):
        flat_result[i], flat_result_weight[i] = point_aligned_smooth(
            flat_data=flat_data,
            flat_kernel=flat_kernel,
            flat_weight=flat_weight,
            flat_valid=flat_valid,
            data_index=data_indices[:, i],
            kernel_indices=kernel_indices,
            kernel_reference_index=reference_index,
            data_shape=data_shape,
            data_steps=data_steps,
            validated=validated,
            weighted=weighted)

    return result, result_weight


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def regular_coarse_kernel_convolve(data, kernel, steps,
                                   kernel_reference_index=None,
                                   weight=None,
                                   valid=None):  # pragma: no cover
    """
    Direct convolution of ND data with a given kernel onto a coarse grid.

    This function is similar to :func:`regular_kernel_convolve` in that
    it results in an output that is the convolution of regular data with
    a regular grid, except that the output dimensions are altered.  The size
    of the output grid is::

       shape_out = ceil(data.shape / steps)

    and is generally intended to speed up processing time by reducing the
    number of points at which a solution is required.  Later, if necessary and
    mathematically sound, the solution can be interpolated back onto a fine
    grid.  Only downsampling is supported using integer step values > 1.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The data to convolve.
    kernel : numpy.ndarray (float)
        The kernel to convolve the data with.  Should have the same
        dimensionality as `data`.
    steps : numpy.ndarray (int)
        The steps for the coarse convolution.
    kernel_reference_index : numpy.ndarray (int), optional
        The index marking the center of the kernel.  If not provided, defaults
        to ceil((kernel.shape - 1) / 2).
    weight : numpy.ndarray (float), optional
        An optional weighting array.
    valid : numpy.ndarray (bool), optional
        An optional validating array where `False` excludes a datum from
        inclusion in processing.  Effectively the same as setting weight[i] = 0
        for index i.

    Returns
    -------
    convolved, weights, shape : array (float), array (float), array (int)
        The convolved result and weights on a coarse grid as 1-D arrays.
        The arrays may be reshaped using `shape` which contains the shape of
        the coarse grids (numba can't do arbitrary array shapes in N-D with
        Numba).
    """
    weighted = weight is not None
    validated = valid is not None

    data_shape = np.asarray(data.shape)
    n_dimensions = kernel.ndim
    ratio = np.empty(n_dimensions, dtype=nb.int64)
    for dimension in range(n_dimensions):
        ratio[dimension] = roundup_ratio(data_shape[dimension],
                                         steps[dimension])

    n_course = int(np.prod(ratio))
    course_signal = np.empty(n_course, dtype=nb.float64)
    course_weight = np.empty(n_course, dtype=nb.float64)

    if kernel_reference_index is None:
        reference_index = (np.asarray(kernel.shape) - 1) / 2.0
        offset = (reference_index % 1) != 0
        reference_index = reference_index.astype(nb.int64) + offset
    else:
        reference_index = kernel_reference_index.astype(nb.int64)

    if weighted:
        flat_weight = weight.ravel()
    else:
        flat_weight = np.empty(0, dtype=nb.float64)

    if validated:
        flat_valid = valid.ravel()
    else:
        flat_valid = np.empty(0, dtype=nb.b1)

    kernel_indices, _, kernel_steps = spline_utils.flat_index_mapping(
        np.asarray(kernel.shape))
    data_indices, _, data_steps = spline_utils.flat_index_mapping(data_shape)
    course_indices, _, course_steps = spline_utils.flat_index_mapping(ratio)

    n_dimensions, n_kernel = kernel_indices.shape
    flat_kernel = kernel.ravel()
    flat_data = data.ravel()
    flat_result = course_signal.ravel()
    flat_result_weight = course_weight.ravel()
    offset_indices = kernel_indices.copy()
    for dimension in range(n_dimensions):
        offset_indices[dimension] -= reference_index[dimension]

    for i in range(n_course):
        flat_data_index = 0
        for dimension in range(n_dimensions):
            index = course_indices[dimension, i] * steps[dimension]
            flat_data_index += data_steps[dimension] * index

        data_index = data_indices[:, flat_data_index]
        w_sum = 0.0
        wd_sum = 0.0
        for j in range(n_kernel):
            k = flat_kernel[j]
            if k == 0:
                continue

            ref_data_index = 0
            offset_index = offset_indices[:, j]
            for dimension in range(n_dimensions):
                oi = offset_index[dimension] + data_index[dimension]
                if oi < 0:
                    break
                elif oi >= data_shape[dimension]:
                    break
                ref_data_index += oi * data_steps[dimension]
            else:
                if validated and not flat_valid[ref_data_index]:
                    continue
                if weighted:
                    w = flat_weight[ref_data_index]
                    if w == 0:
                        continue
                else:
                    w = 1.0

                kw = k * w
                wd_sum += kw * flat_data[ref_data_index]
                w_sum += kw

        if w_sum > 0:
            wd_sum /= w_sum
        else:
            w_sum = 0.0
            wd_sum = 0.0

        flat_result_weight[i] = w_sum
        flat_result[i] = wd_sum

    return course_signal, course_weight, ratio


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def smooth_values_at(data, kernel, indices, kernel_reference_index, knots,
                     coefficients, degrees, panel_mapping, panel_steps,
                     knot_steps, nk1, spline_mapping, weight=None, valid=None
                     ):  # pragma: no cover
    """
    Return the values of data, smoothed by a kernel at the given indices.

    A "smooth value" is one in which the convolution of the data with a kernel
    centered over a specific point is calculated and returned.  This function
    is essentially a wrapper around :func:`smooth_value_at` to process multiple
    points in a single pass.  This is a moderately low-level function, and
    requires a spline representation of the kernel to have been previously
    defined (see :class:`sofia_redux.toolkit.splines.spline.Spline` for further
    details).

    If the indices align perfectly with existing data indices (i.e., the
    difference between the `indices` and `kernel_reference_index` can be
    represented exactly as an integer), no spline representation is required as
    direct convolution with the kernel is possible.

    Notes
    -----
    For the purposes of debugging, please remember that any spline parameters
    have their dimensionality expressed in (x, y, z, ...) order, but all other
    parameters (such as `indices`) use native numpy ordering (..., z, y, x).
    If the parameters were calculated (as expected) via the spline class and
    other native `sofia_scan` functions, then no conversion is necessary.
    However, if any of these parameters were manually created, please be aware
    of this difference.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The data from which to calculate smoothed values.  Must be an array
        of arbitrary shape, but have n_dimensions.
    kernel : numpy.ndarray (float)
        The kernel to smooth with.  Must be an array of arbitrary shape, but
        match the same number of dimensions as `data`.
    indices : numpy.ndarray (int)
        The indices in relation to `data` for which to calculate smoothed
        values.  Must be of shape (n_dimensions, n) and use numpy dimensional
        ordering (y, x).
    kernel_reference_index : numpy.ndarray (int)
        The kernel reference index specifying the center of the kernel.
        Must be of shape (n_dimensions,) and use numpy dimensional ordering
        (y, x).
    knots : numpy.ndarray (float)
        The knots as calculated by the :class:`Spline` object on `kernel`.
        These should be of shape (n_dimensions, max_knots) where max_knots is
        the maximum possible number of knots for a spline representation of
        `kernel` over all dimensions.  Dimensions should be ordered as (x, y).
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The spline degrees for each dimension of shape (n_dimensions,).
         Dimensions should be ordered as (x, y).
    panel_mapping : numpy.ndarray (int)
        The panel mapping translation for the spline fit.  Should be of shape
        (n_dimensions, n_panels).   Dimensions should be ordered as (x, y).
    panel_steps : numpy.ndarray (int)
        The panel steps translation for the spline fit.  Should be of shape
        (n_dimensions,).   Dimensions should be ordered as (x, y).
    knot_steps : numpy.ndarray (int)
        The spline knot steps translation for the spline fit.  Should be of
        shape (n_dimensions,).   Dimensions should be ordered as (x, y).
    nk1 : numpy.ndarray (int)
        Another spline mapping parameter which is equal to
        n_knots - degrees - 1 for the spline.  Should be of shape
        (n_dimensions,).   Dimensions should be ordered as (x, y).
    spline_mapping : numpy.ndarray (int)
        A spline mapping translation for the spline knots.  Should be of shape
        (n_dimensions, max_knots).   Dimensions should be ordered as (x, y).
    weight : numpy.ndarray (float), optional
        The optional `data` weights.  Should have the same shape as `data`.
    valid : numpy.ndarray (bool), optional
        An optional array that marks good `data` values as True, and all others
        that should not be used in the fit as `False`.  Should be the same
        shape as `data`.

    Returns
    -------
    smooth_values, smooth_weights : numpy.ndarray, numpy.ndarray
        The smoothed values and weights as determined by convolution of `data`
        with `kernel`, and possibly adjusted for position by a spline fit.
        Both will be of shape (n,) matching the number of indices passed in as
        an argument.
    """
    data_shape = np.asarray(data.shape)
    kernel_shape = np.asarray(kernel.shape)
    kernel_indices, _, kernel_steps = spline_utils.flat_index_mapping(
        kernel_shape)
    data_indices, _, data_steps = spline_utils.flat_index_mapping(data_shape)

    if weight is None:
        flat_weight = np.empty(0, dtype=nb.float64)
        weighted = False
    else:
        flat_weight = weight.ravel()
        weighted = True

    if valid is None:
        flat_valid = np.empty(0, dtype=nb.b1)
        validated = False
    else:
        flat_valid = valid.ravel()
        validated = True

    n = indices.shape[1]
    smooth_values = np.empty(n, dtype=nb.float64)
    smooth_weights = np.empty(n, dtype=nb.float64)

    for i in range(n):
        smooth_values[i], smooth_weights[i] = smooth_value_at(
            data=data,
            kernel=kernel,
            index=indices[:, i],
            kernel_indices=kernel_indices,
            kernel_reference_index=kernel_reference_index,
            knots=knots,
            coefficients=coefficients,
            degrees=degrees,
            panel_mapping=panel_mapping,
            panel_steps=panel_steps,
            knot_steps=knot_steps,
            nk1=nk1,
            spline_mapping=spline_mapping,
            data_shape=data_shape,
            kernel_shape=kernel_shape,
            data_steps=data_steps,
            flat_weight=flat_weight,
            flat_valid=flat_valid,
            weighted=weighted,
            validated=validated)

    return smooth_values, smooth_weights


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def smooth_value_at(data, kernel, index, kernel_indices,
                    kernel_reference_index, knots, coefficients,
                    degrees, panel_mapping, panel_steps,
                    knot_steps, nk1, spline_mapping, data_shape,
                    kernel_shape, data_steps,
                    flat_weight, flat_valid, weighted, validated
                    ):  # pragma: no cover
    """
    Return the convolution of data with a kernel at a single point.

    This low-level function basically sorts a convolution of `data` with
    `kernel` at a single point (`index`) into processing by one of two
    algorithms.  If the index aligns perfectly with a point on the kernel,
    the direct convolution is calculated via :func:`point_aligned_smooth`.
    Otherwise, a slightly shifted version of the kernel is generated using
    spline interpolation so that the shifted kernel indices align with those
    of the data array, and direct convolution is possible.  In this case,
    convolution is performed using :func:`point_smooth`.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The data from which to calculate smoothed values.  Must be an array
        of arbitrary shape, but have n_dimensions.
    kernel : numpy.ndarray (float)
        The kernel to smooth with.  Must be an array of arbitrary shape, but
        match the same number of dimensions as `data`.
    index : numpy.ndarray (int or float)
        The index on `data` at which to calculate the smooth value.  This
        should be an array of shape (n_dimensions,) using numpy (y, x)
        ordering.
    kernel_indices : numpy.ndarray (int)
        An array of shape (n_dimensions, kernel.size) as returned by
        :func:`spline_utils.flat_index_mapping`.  This gives the N-dimensional
        kernel index for a flat kernel.  I.e., kernel.ravel()[i] is the same
        as kernel[kernel_indices[i]].  Note that dimensions are ordered using
        the Numpy (y, x) convention.
    kernel_reference_index : numpy.ndarray (int)
        The kernel reference index specifying the center of the kernel.
        Must be of shape (n_dimensions,) and use numpy dimensional ordering
        (y, x).
    knots : numpy.ndarray (float)
        The knots as calculated by the :class:`Spline` object on `kernel`.
        These should be of shape (n_dimensions, max_knots) where max_knots is
        the maximum possible number of knots for a spline representation of
        `kernel` over all dimensions.  Dimensions should be ordered as (x, y).
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The spline degrees for each dimension of shape (n_dimensions,).
         Dimensions should be ordered as (x, y).
    panel_mapping : numpy.ndarray (int)
        The panel mapping translation for the spline fit.  Should be of shape
        (n_dimensions, n_panels).   Dimensions should be ordered as (x, y).
    panel_steps : numpy.ndarray (int)
        The panel steps translation for the spline fit.  Should be of shape
        (n_dimensions,).   Dimensions should be ordered as (x, y).
    knot_steps : numpy.ndarray (int)
        The spline knot steps translation for the spline fit.  Should be of
        shape (n_dimensions,).   Dimensions should be ordered as (x, y).
    nk1 : numpy.ndarray (int)
        Another spline mapping parameter which is equal to
        n_knots - degrees - 1 for the spline.  Should be of shape
        (n_dimensions,).   Dimensions should be ordered as (x, y).
    spline_mapping : numpy.ndarray (int)
        A spline mapping translation for the spline knots.  Should be of shape
        (n_dimensions, max_knots).   Dimensions should be ordered as (x, y).
    data_shape : numpy.ndarray (int)
        The shape of `data` as an array of shape (n_dimensions,) in
        Numpy (y, x) order.
    kernel_shape : numpy.ndarray (int)
        The shape of `kernel` as an array of shape (n_dimensions,) in Numpy
        (y, x) order.
    data_steps : numpy.ndarray (int)
        An array of shape (n_dimensions,) in Numpy (y, x) order giving the
        number of elements one would need to jump on a flattened `data` for
        a single increment along a given dimension on the ND-array.  Please
        see :func:`spline_utils.flat_index_mapping` for further details.
    flat_weight : numpy.ndarray (float)
        The associated weight values for data, flattened to a single dimension
        array of shape (data.size,).  I.e., `flat_weight` = weight.ravel().
        If no weighting is required, this should be an empty array of
        shape (0,).
    flat_valid : numpy.ndarray (bool)
        An array marking data values as valid (`True`) or invalid (`False`).
        Any invalid data will not be included in the calculation.  This should
        be a flattened singular dimension array taken from one that was
        originally the same shape as `data` and be of shape (data.size,).
        I.e., `flat_valid` = valid.ravel().  If no validity checking is
        required (assumes all data points are valid), set this to an empty
        array of shape (0,).
    weighted : bool
        If `True`, indicates that weighting is required and `flat_weight`
        should be of shape (data.size,).  Otherwise, no weighting is required,
        and `flat_weight` may be of shape (0,).
    validated : bool
        If `True`, indicates that validity checking is required and
        `flat_valid` should be of shape (data.size,).  Otherwise, `flat_valid`
        may be of shape (0,).

    Returns
    -------
    smooth_value, smooth_weight : float, float
        The derived smooth value and associated weight by convolving `data`
        with `kernel` at `index`.
    """

    n_dimensions = index.size
    index_difference = index - kernel_reference_index

    for dimension in range(n_dimensions):
        if (index_difference[dimension]
                != np.floor(index_difference[dimension])):
            # Must shift kernel using spline interpolation
            return point_smooth(
                flat_data=data.ravel(),
                index=index,
                kernel_indices=kernel_indices,
                kernel_reference_index=kernel_reference_index,
                knots=knots,
                coefficients=coefficients,
                degrees=degrees,
                panel_mapping=panel_mapping,
                panel_steps=panel_steps,
                knot_steps=knot_steps,
                nk1=nk1,
                spline_mapping=spline_mapping,
                data_shape=data_shape,
                kernel_shape=kernel_shape,
                data_steps=data_steps,
                weighted=weighted,
                validated=validated,
                flat_weight=flat_weight,
                flat_valid=flat_valid)
    else:
        # Perform direct convolution
        return point_aligned_smooth(
            flat_data=data.ravel(),
            flat_kernel=kernel.ravel(),
            flat_weight=flat_weight,
            flat_valid=flat_valid,
            data_index=np.asarray(index, dtype=nb.int64),
            kernel_indices=kernel_indices,
            kernel_reference_index=kernel_reference_index,
            data_shape=data_shape,
            data_steps=data_steps,
            validated=validated,
            weighted=weighted)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def point_aligned_smooth(flat_data, flat_kernel, flat_weight, flat_valid,
                         data_index, kernel_indices, kernel_reference_index,
                         data_shape, data_steps, validated, weighted
                         ):  # pragma: no cover
    """
    Perform direct kernel convolution at a single point.

    This low-level function is designed to return the convolved result and
    associated weight of data with a kernel at a given point, where all kernel
    points perfectly align with points in the data array.

    Parameters
    ----------
    flat_data : numpy.ndarray (float)
        A flattened version of N-dimensional data (data.ravel()) of shape (n,).
    flat_kernel : numpy.ndarray (float)
        A flattened version of an N-dimensions kernel (kernel.ravel()) of shape
        (m,).
    flat_weight : numpy.ndarray (float)
        A flattened version of N-dimensional weights (weight.ravel()).  If
        `weighted` is `True`, must be of shape (n,).
    flat_valid : numpy.ndarray (bool)
        A flattened version of an N-dimensional validity array (valid.ravel()).
        If `validated` is `True`, must be of shape (n,).
    data_index : numpy.ndarray (int)
        The N-D index on the original data array at which to determine the
        smoothed value.  Must be of shape (n_dimensions,) and by in Numpy
        (y, x) order.
    kernel_indices : numpy.ndarray (int)
        An array of shape (n_dimensions, kernel.size) as returned by
        :func:`spline_utils.flat_index_mapping`.  This gives the N-dimensional
        kernel index for a flat kernel.  I.e., kernel.ravel()[i] is the same
        as kernel[kernel_indices[i]].  Note that dimensions are ordered using
        the Numpy (y, x) convention.
    kernel_reference_index : numpy.ndarray (int)
        The kernel reference index specifying the center of the kernel.
        Must be of shape (n_dimensions,) and use numpy dimensional ordering
        (y, x).
    data_shape : numpy.ndarray (int)
        The shape of `data` as an array of shape (n_dimensions,) in
        Numpy (y, x) order.
    data_steps : numpy.ndarray (int)
        An array of shape (n_dimensions,) in Numpy (y, x) order giving the
        number of elements one would need to jump on a flattened `data` for
        a single increment along a given dimension on the ND-array.  Please
        see :func:`spline_utils.flat_index_mapping` for further details.
    weighted : bool
        If `True`, indicates that weighting is required and `flat_weight`
        should be of shape (data.size,).  Otherwise, no weighting is required,
        and `flat_weight` may be of shape (0,).
    validated : bool
        If `True`, indicates that validity checking is required and
        `flat_valid` should be of shape (data.size,).  Otherwise, `flat_valid`
        may be of shape (0,).

    Returns
    -------
    smooth_value, smooth_weight : float, float
        The derived smooth value and associated weight by convolving data with
        kernel at `data_index`.
    """
    w_sum = 0.0
    wd_sum = 0.0
    n_dimensions = data_steps.size

    for j in range(flat_kernel.size):
        k = flat_kernel[j]
        if k == 0:
            continue

        ref_data_index = 0
        for dimension in range(n_dimensions):
            oi = (kernel_indices[dimension, j]
                  - kernel_reference_index[dimension]
                  + data_index[dimension])
            if oi < 0:
                break
            elif oi >= data_shape[dimension]:
                break
            ref_data_index += int(oi) * data_steps[dimension]
        else:
            if validated and not flat_valid[ref_data_index]:
                continue
            if weighted:
                w = flat_weight[ref_data_index]
                if w == 0:
                    continue
            else:
                w = 1.0

            kw = k * w
            w_sum += kw
            wd_sum += kw * flat_data[ref_data_index]

    if w_sum > 0:
        wd_sum /= w_sum
    else:
        w_sum = 0.0
        wd_sum = 0.0
    return wd_sum, w_sum


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def point_smooth(flat_data, index, kernel_indices, kernel_reference_index,
                 knots, coefficients, degrees, panel_mapping, panel_steps,
                 knot_steps, nk1, spline_mapping, data_shape, kernel_shape,
                 data_steps, weighted, validated, flat_weight, flat_valid
                 ):  # pragma: no cover
    """
    Convolution with a spline representation of a kernel at a single point.

    This is a more complex interpretation of :func:`point_aligned_smooth`
    intended for use when convolution occurs at a point on the kernel that does
    not have an exact value.  I.e. we need to calculate the intermediate kernel
    value between two definite kernel values.  For these purposes, spline
    interpolation is used to derive offset kernel values to be used during
    convolution.  There are many different spline parameters which will need
    to have been previously defined.  However, these can all be calculated
    easily by creating a :class:`sofia_redux.toolkit.splines.spline.Spline`
    representation of the kernel and parsing the various attributes into this
    function.

    Parameters
    ----------
    flat_data : numpy.ndarray (float)
        A flattened version of N-dimensional data (data.ravel()) of
        shape (n,).
    index : numpy.ndarray (int)
        The N-D index on the original data array at which to determine the
        smoothed value.  Must be of shape (n_dimensions,) and by in Numpy
        (y, x) order.
    kernel_indices : numpy.ndarray (int)
        An array of shape (n_dimensions, kernel.size) as returned by
        :func:`spline_utils.flat_index_mapping`.  This gives the N-dimensional
        kernel index for a flat kernel.  I.e., kernel.ravel()[i] is the same
        as kernel[kernel_indices[i]].  Note that dimensions are ordered using
        the Numpy (y, x) convention.
    kernel_reference_index : numpy.ndarray (int)
        The kernel reference index specifying the center of the kernel.
        Must be of shape (n_dimensions,) and use numpy dimensional
        ordering (y, x).
    knots : numpy.ndarray (float)
        The knots as calculated by the :class:`Spline` object on `kernel`.
        These should be of shape (n_dimensions, max_knots) where max_knots is
        the maximum possible number of knots for a spline representation of
        `kernel` over all dimensions.  Dimensions should be ordered as (x, y).
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The spline degrees for each dimension of shape (n_dimensions,).
         Dimensions should be ordered as (x, y).
    panel_mapping : numpy.ndarray (int)
        The panel mapping translation for the spline fit.  Should be of shape
        (n_dimensions, n_panels).   Dimensions should be ordered as (x, y).
    panel_steps : numpy.ndarray (int)
        The panel steps translation for the spline fit.  Should be of shape
        (n_dimensions,).   Dimensions should be ordered as (x, y).
    knot_steps : numpy.ndarray (int)
        The spline knot steps translation for the spline fit.  Should be of
        shape (n_dimensions,).   Dimensions should be ordered as (x, y).
    nk1 : numpy.ndarray (int)
        Another spline mapping parameter which is equal to
        n_knots - degrees - 1 for the spline.  Should be of shape
        (n_dimensions,).   Dimensions should be ordered as (x, y).
    spline_mapping : numpy.ndarray (int)
        A spline mapping translation for the spline knots.  Should be of shape
        (n_dimensions, max_knots).   Dimensions should be ordered as (x, y).
    data_shape : numpy.ndarray (int)
        The shape of `data` as an array of shape (n_dimensions,) in
        Numpy (y, x) order.
    kernel_shape : numpy.ndarray (int)
        The shape of `kernel` as an array of shape (n_dimensions,) in Numpy
        (y, x) order.
    data_steps : numpy.ndarray (int)
        An array of shape (n_dimensions,) in Numpy (y, x) order giving the
        number of elements one would need to jump on a flattened `data` for
        a single increment along a given dimension on the ND-array.  Please
        see :func:`spline_utils.flat_index_mapping` for further details.
    weighted : bool
        If `True`, indicates that weighting is required and `flat_weight`
        should be of shape (data.size,).  Otherwise, no weighting is required,
        and `flat_weight` may be of shape (0,).
    validated : bool
        If `True`, indicates that validity checking is required and
        `flat_valid` should be of shape (data.size,).  Otherwise, `flat_valid`
        may be of shape (0,).
    flat_weight : numpy.ndarray (float)
        A flattened version of N-dimensional weights (weight.ravel()).  If
        `weighted` is `True`, must be of shape (n,).
    flat_valid : numpy.ndarray (bool)
        A flattened version of an N-dimensional validity array (valid.ravel()).
        If `validated` is `True`, must be of shape (n,).

    Returns
    -------
    smooth_value, smooth_weight : float, float
        The derived smooth value and associated weight by convolving data with
        spline representation of the kernel at `index`.
    """
    n_dimensions, n_kernel = kernel_indices.shape
    kernel_coordinate = np.empty(n_dimensions, dtype=nb.float64)

    k1 = degrees + 1
    n_spline = int(np.prod(k1))
    work_spline = np.empty((n_dimensions, np.max(k1)), dtype=nb.float64)
    lower_bounds = np.zeros(n_dimensions, dtype=nb.float64)
    upper_bounds = kernel_shape - 1

    i0 = index - kernel_reference_index

    # Continue here, since splines are in (x, y) order...
    wd_sum = 0.0
    w_sum = 0.0

    nd1 = n_dimensions - 1

    for j in range(n_kernel):
        flat_data_index = 0
        for dimension in range(n_dimensions):
            coordinate = i0[dimension] + kernel_indices[dimension, j]
            data_index = int(np.ceil(coordinate))
            if data_index < 0:
                break
            elif data_index >= data_shape[dimension]:
                break

            # Need to convert numpy (y, x) ordering to
            # spline (x, y) ordering...
            kernel_coordinate[nd1 - dimension] = data_index - i0[dimension]
            flat_data_index += data_steps[dimension] * data_index
        else:
            if validated and not flat_valid[flat_data_index]:
                continue
            if weighted:
                w = flat_weight[flat_data_index]
                if w == 0:
                    continue
            else:
                w = 1.0

            kernel_value = spline_utils.single_fit(
                coordinate=kernel_coordinate,
                knots=knots,
                coefficients=coefficients,
                degrees=degrees,
                panel_mapping=panel_mapping,
                panel_steps=panel_steps,
                knot_steps=knot_steps,
                nk1=nk1,
                spline_mapping=spline_mapping,
                k1=k1,
                n_spline=n_spline,
                work_spline=work_spline,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds)
            if np.isnan(kernel_value):
                continue

            kw = w * kernel_value
            w_sum += kw
            wd_sum += kw * flat_data[flat_data_index]

    if w_sum == 0:
        return 0.0, 0.0
    return wd_sum / w_sum, w_sum


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def sequential_array_add(array, add_values, at_indices, valid_array=None,
                         valid_indices=None):  # pragma: no cover
    """
    Add one array of values to another.

    Unlike standard adding, values can be added from `add_values` to a single
    index of `array` multiple times.

    Examples
    --------
    >>> x = np.zeros(5)
    >>> y = np.ones(5)
    >>> indices = np.array([0, 0, 3, 3, 3])
    >>> added = sequential_array_add(x, y, indices)
    >>> print(added, x)
    [ True False False  True False] [2. 0. 0. 3. 0.]

    Parameters
    ----------
    array : numpy.ndarray (float or int)
        The array to add to of shape (shape1).  Addition will be performed
        in-place.
    add_values : numpy.ndarray (float or int)
        The values to add to the array of shape (shape2)
    at_indices : numpy.ndarray (int)
        The indices on `array` for which to add `add_values`.  Must be of
        shape (n_dimensions, values.size or add_values.shape).
        One-dimensional arrays may be of shape (values.size,).
    valid_array : numpy.ndarray (bool), optional
        An array of shape (shape1) where `True` indicates that `array` may be
        added to at that index.
    valid_indices : numpy.ndarray (bool), optional
        An array of shape (shape2) where `False` excludes the same element of
        `add_values` from being added to `array`.

    Returns
    -------
    added : numpy.ndarray (bool)
        An array the same shape as `array` where `True` indicates that an
        element has been added to.
    """
    array_shape = np.asarray(array.shape)
    (array_indices, _,
     array_steps) = spline_utils.flat_index_mapping(array_shape)

    flat_array = array.ravel()
    added_to = np.full(array.shape, False)
    flat_add = add_values.ravel()
    at_indices = np.atleast_2d(at_indices)
    n_dimensions = array.ndim
    flat_added_to = added_to.ravel()

    n_add = int(np.prod(np.array(at_indices.shape[1:])))
    valid = np.full(n_add, True)
    flat_indices = np.zeros(n_add, dtype=nb.int64)

    if valid_array is None:
        do_valid_array = False
        flat_valid_array = np.empty(0, dtype=nb.b1)
    else:
        do_valid_array = True
        flat_valid_array = valid_array.ravel()

    if valid_indices is None:
        do_valid_indices = False
        flat_valid_indices = np.empty(0, dtype=nb.b1)
    else:
        do_valid_indices = True
        flat_valid_indices = valid_indices.ravel()

    for dimension in range(n_dimensions):
        step = array_steps[dimension]
        index_line = at_indices[dimension].ravel()
        max_index = array_shape[dimension]
        for i in range(n_add):
            if not valid[i]:
                continue
            if do_valid_indices and not flat_valid_indices[i]:
                valid[i] = False
                continue
            index = index_line[i]
            if index < 0 or index >= max_index:
                valid[i] = False
                continue
            flat_indices[i] += step * index

    for i in range(n_add):
        if not valid[i]:
            continue
        index = flat_indices[i]
        if do_valid_array and not flat_valid_array[index]:
            continue
        flat_array[index] += flat_add[i]
        flat_added_to[index] = True

    return added_to


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def index_of_max(array, valid=None, sign=1):  # pragma: no cover
    """
    Return the maximum value and index of that value.

    Parameters
    ----------
    array : numpy.ndarray (float)
        The array of data values to check.
    valid : numpy.ndarray (bool), optional
        Optionally exclude elements by setting the corresponding element in
        valid to `False`.
    sign : int or float, optional
        If sign < 0, return the minimum value/index.  If sign > 0 (default),
        return the maximum value/index.  If sign == 0, return the maximum
        absolute value/index.

    Returns
    -------
    value, index : 2-tuple (float, numpy.ndarray (int))
        The max/min/max-absolute value and corresponding index.
    """
    index = -1
    if valid is None:
        check_valid = False
        flat_valid = np.empty(0, dtype=nb.b1)
    else:
        check_valid = True
        flat_valid = valid.ravel()

    flat_indices, _, flat_steps = spline_utils.flat_index_mapping(
        np.asarray(array.shape))
    flat_array = array.ravel()

    positive = False
    absolute = False
    negative = False
    if sign > 0:
        check_value = -np.inf
        positive = True
    elif sign < 0:
        check_value = np.inf
        negative = True
    else:
        check_value = 0.0
        absolute = True

    for i in range(flat_array.size):
        value = flat_array[i]
        if np.isnan(value):
            continue
        if check_valid and not flat_valid[i]:
            continue
        if positive and value > check_value:
            check_value = value
            index = i
        elif negative and value < check_value:
            check_value = value
            index = i
        elif absolute:
            value = np.abs(value)
            if value > check_value:
                check_value = value
                index = i

    if index == -1:
        return np.nan, np.full(array.ndim, -1)
    else:
        return flat_array[index], flat_indices[:, index]


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def robust_mean(values, tails=None):  # pragma: no cover
    """
    Calculate the robust mean value excluding tails.

    Parameters
    ----------
    values : numpy.ndarray (float)
    tails : float, optional
        Calculate the mean between the sorted values within the tails of the
        distribution.

    Returns
    -------
    mean : float
    """
    if tails is None:
        return np.nanmean(values)

    sorted_values = np.sort(values)
    n = sorted_values.size
    for last in range(n):
        if np.isnan(sorted_values[last]):
            break
    else:
        last = n

    dn = round_value(tails * last)
    start = dn
    end = last - dn
    if start >= end:
        return np.nan

    value_sum = 0.0
    for i in range(start, end):
        value_sum += sorted_values[i]
    return value_sum / (end - start)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def round_value(x):  # pragma: no cover
    """
    A fix to :func:`np.round` so that values are always rounded as expected.

    In special cases, when `int(x)` is even, and it's decimal value is exactly
    #.5, :func:`np.round` floors the value instead of performing a ceil.
    This function fixes that.

    Parameters
    ----------
    x : float or int
        The value to round.

    Returns
    -------
    rounded_x : int
    """
    if x % 1 != 0.5:
        return int(np.round(x))
    int_x = int(x)
    if int_x % 2 != 0:
        return int(np.round(x))
    if x < 0:
        return int_x - 1
    else:
        return int_x + 1


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def round_array(x):  # pragma: no cover
    """
    Correctly round a given array.

    Please see :func:`round_value` for a description of the problem with
    :func:`np.round`.

    Parameters
    ----------
    x : numpy.ndarray
        The array to round.

    Returns
    -------
    rounded_x : numpy.ndarray (int)
    """
    result = np.empty(x.shape, dtype=nb.int64)
    flat_result = result.flat
    flat_x = x.flat
    for i, value in enumerate(flat_x):
        rounded_value = round_value(value)
        flat_result[i] = rounded_value
    return result
