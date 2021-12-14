# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math
import numpy as np
import numba as nb
from numba import njit
from sofia_redux.toolkit.splines.spline_utils import perform_fit
from sofia_redux.toolkit.resampling.resample_utils import (
    update_mask, check_edges, calculate_fitting_weights, array_sum,
    weighted_mean_variance, solve_mean_fit, offset_variance
)

_condition_limit = 1 / np.finfo(float).eps
_fast_flags = {'nsz', 'nnan', 'ninf'}
_fast_flags_all = {'nnan',  # no NaNs
                   'ninf',  # no infinities
                   'ninf',  # no signed zeros
                   'nsz',  # no signed zeros
                   'arcp',  # allow reciprocal
                   'contract',  # allow floating-point contraction
                   'afn',  # approximate functions
                   'reassoc'  # allow re association transformations
                   }

__all__ = ['solve_kernel_fits', 'solve_kernel_fit', 'calculate_kernel_weights',
           'apply_mask_to_kernel_set_arrays']


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=True, cache=True, parallel=False)
def solve_kernel_fits(sample_indices, sample_coordinates, sample_data,
                      sample_error, sample_mask, fit_coordinates,
                      knots, coefficients, degrees, panel_mapping, panel_steps,
                      knot_steps, nk1, spline_mapping, n_knots,
                      is_covar=False, cval=np.nan,
                      error_weighting=True,
                      absolute_weight=False,
                      edge_algorithm_idx=1,
                      edge_threshold=None,
                      get_error=True, get_counts=True, get_weights=True,
                      get_distance_weights=True, get_rchi2=True,
                      get_offset_variance=True):  # pragma: no cover
    r"""
    Solve all fits within one intersection block.

    This function is a wrapper for :func:`solve_kernel_fit` over all data sets
    and fit points.  The main computations here involve:

        1. Creating and populating the output arrays.
        2. Selecting the correct samples within the region of each fitting
           window.
        3. Calculating the full weighting factors for the fits.

    For further details on the actual fitting, please see
    :func:`solve_kernel_fit`.

    Parameters
    ----------
    sample_indices : numba.typed.List
        A list of 1-dimensional numpy.ndarray (dtype=int) of length n_fits.
        Each list element `sample_indices[i]`, contains the indices of samples
        within the "window" region of `fit_indices[i]`.
    sample_coordinates : numpy.ndarray (n_dimensions, n_samples)
        The independent coordinates for each sample in n_dimensions
    sample_data : numpy.ndarray (n_sets, n_samples)
        The dependent values of the samples for n_sets, each containing
        n_samples.
    sample_error : numpy.ndarray (n_sets, n_samples)
        The associated 1-sigma error values for each sample in each set.  The
        user may also supply an array of shape (n_sets, 1) in which case all
        samples in a set will share the same associated error value.  If
        the shape is set to (n_sets, 0), this indicates that no error values
        are available for the samples.
    sample_mask : numpy.ndarray (n_sets, n_samples)
        A mask where `False` indicates that the associated sample should be
        excluded from all fits.
    fit_coordinates : numpy.ndarray (n_dimensions, n_fits)
        The independent variables at each fit coordinate in d_dimensions.
    knots : list or tuple or numpy.ndarray, optional
        A set of starting knot coordinates for each dimension.  If a list or
        tuple is supplied it should be of length n_dimensions where element
        i is an array of shape (n_knots[i]) for dimension i.  If an array
        is supplied, it should be of shape (n_dimension, max(n_knots)).
        Note that there must be at least 2 * (degree + 1) knots for each
        dimension.  Unused or invalid knots may be set to NaN, at the end
        of each array.  Knot coordinates must also be monotonically
        increasing in each dimension.
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The degrees of the spline in each dimension (n_dimensions,).
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    panel_steps : numpy.ndarray (int)
        The flat index mapping steps in panel-space of shape (n_dimensions,).
        These are returned by passing the shape `Spline.panel_shape` into
        :func:`flat_index_mapping`.
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    nk1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) containing the values n_knots - k1
        where n_knots are the number of knots in each dimension, and k1 are the
        spline degrees + 1 in each dimension.
    spline_mapping : numpy.ndarray (int)
        An array containing the spline mapping (flat to n-D) indices.  This is
        created by passing the spline shape (degrees + 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_spline_coefficients).
    n_knots : numpy.ndarray (int)
        The number of knots in each dimension (n_dimensions,).
    is_covar : bool, optional
        If `True`, indicates that `sample_data` contains covariance values
        that should be propagated through algorithm.
    cval : float, optional
        In a case that a fit is unable to be calculated at certain location,
        `cval` determines the fill value for the output `fit` array at those
        locations.
    error_weighting : bool, optional
        If `True`, weight the samples in the fit by the inverse variance
        (1 / sample_error^2) in addition to distance weighting.
    absolute_weight : bool, optional
        If the kernel weights are negative, can lead to almost zero-like
        divisions in many of the algorithms.  If set to `True`, the sum of the
        absolute weights are used for normalization.
    edge_algorithm_idx : int, optional
        Integer specifying the algorithm used to determine whether a fit should
        be attempted with respect to the sample distribution.  Please see
        :func:`check_edges` for further information.  The default (1), is
        always the most robust of the available algorithms.
    edge_threshold : numpy.ndarray (n_dimensions,)
        A threshold parameter determining how close an edge should be to the
        center of the distribution during :func:`check_edges`.  Higher values
        result in an edge closer to the sample mean.  A value should be
        provided for each dimension.  A zero value in any dimension will result
        in an infinite edge for that dimension.
    get_error : bool, optional
        If `True`, return the error on the fit.
    get_counts : bool, optional
        If `True`, return the number of samples used when determining the fit
        at each fitting point.
    get_weights : bool, optional
        If `True`, return the sum of all sample weights used in determining the
        fit at each point.
    get_distance_weights : bool, optional
        If `True`, return the sum of only the distance weights used in
        determining the fit at each point.
    get_rchi2 : bool, optional
        If `True`, return the reduced chi-squared statistic for each of the
        fitted points.
    get_offset_variance : bool optional
        If `True`, return the offset of the fitting point from the sample
        distribution.  See :func:`offset_variance` for further information.

    Returns
    -------
    fit_results : 7-tuple of numpy.ndarray
        fit_results[0]: Fitted values.
        fit_results[1]: Error on the fit.
        fit_results[2]: Number of samples in each fit.
        fit_results[3]: Weight sums.
        fit_results[4]: Distance weight sums.
        fit_results[5]: Reduced chi-squared statistic.
        fit_results[6]: Offset variances from the sample distribution center.

        All arrays have the shape (n_sets, n_fits) or (0, 0) depending on
        whether `get_<name>` is `True` or `False` respectively.
    """
    n_sets = sample_data.shape[0]
    n_dimensions, n_fits = fit_coordinates.shape

    fit_out = np.empty((n_sets, n_fits), dtype=nb.float64)

    if get_error:
        error_out = np.empty((n_sets, n_fits), dtype=nb.float64)
    else:
        error_out = np.empty((0, 0), dtype=nb.float64)

    if get_counts:
        counts_out = np.empty((n_sets, n_fits), dtype=nb.i8)
    else:
        counts_out = np.empty((0, 0), dtype=nb.i8)

    if get_weights:
        weights_out = np.empty((n_sets, n_fits), dtype=nb.float64)
    else:
        weights_out = np.empty((0, 0), dtype=nb.float64)

    if get_distance_weights:
        distance_weights_out = np.empty((n_sets, n_fits), dtype=nb.float64)
    else:
        distance_weights_out = np.empty((0, 0), dtype=nb.float64)

    if get_rchi2:
        rchi2_out = np.empty((n_sets, n_fits), dtype=nb.float64)
    else:
        rchi2_out = np.empty((0, 0), dtype=nb.float64)

    if get_offset_variance:
        offset_variance_out = np.empty((n_sets, n_fits), dtype=nb.float64)
    else:
        offset_variance_out = np.empty((0, 0), dtype=nb.float64)

    if n_sets == 0 or n_fits == 0:  # pragma: no cover
        return (fit_out, error_out,
                counts_out, weights_out, distance_weights_out,
                rchi2_out, offset_variance_out)

    for fit_index in range(len(sample_indices)):

        fit_coordinate = fit_coordinates[:, fit_index]

        # Arrays for all datasets within window region
        window_indices = sample_indices[fit_index]
        window_coordinates = sample_coordinates[:, window_indices]
        window_values = sample_data[:, window_indices]
        window_mask = sample_mask[:, window_indices]

        if sample_error.shape[1] > 1:
            window_error = sample_error[:, window_indices]
        else:
            # Single values are expanded in apply_mask_to_set_arrays
            window_error = sample_error

        # Determine kernel weighting
        # The kernel weighting should be calculated here...
        kernel_weights = calculate_kernel_weights(
            coordinates=window_coordinates,
            reference=fit_coordinate,
            knots=knots,
            n_knots=n_knots,
            coefficients=coefficients,
            degrees=degrees,
            panel_mapping=panel_mapping,
            panel_steps=panel_steps,
            knot_steps=knot_steps,
            nk1=nk1,
            spline_mapping=spline_mapping)

        for data_set in range(n_sets):

            (fitted_value,
             fitted_error,
             counts,
             weightsum,
             distance_weight,
             rchi2,
             variance_offset
             ) = solve_kernel_fit(window_coordinates,
                                  window_values[data_set],
                                  window_error[data_set],
                                  window_mask[data_set],
                                  kernel_weights, fit_coordinate,
                                  is_covar=is_covar,
                                  cval=cval,
                                  error_weighting=error_weighting,
                                  absolute_weight=absolute_weight,
                                  edge_algorithm_idx=edge_algorithm_idx,
                                  edge_threshold=edge_threshold,
                                  get_error=get_error,
                                  get_weights=get_weights,
                                  get_distance_weights=get_distance_weights,
                                  get_rchi2=get_rchi2,
                                  get_offset_variance=get_offset_variance)

            fit_out[data_set, fit_index] = fitted_value
            if get_error:
                error_out[data_set, fit_index] = fitted_error

            if get_counts:
                counts_out[data_set, fit_index] = counts

            if get_weights:
                weights_out[data_set, fit_index] = weightsum

            if get_distance_weights:
                distance_weights_out[data_set, fit_index] = distance_weight

            if get_rchi2:
                rchi2_out[data_set, fit_index] = rchi2

            if get_offset_variance:
                offset_variance_out[data_set, fit_index] = variance_offset

    return (fit_out, error_out, counts_out, weights_out, distance_weights_out,
            rchi2_out, offset_variance_out)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_kernel_fit(window_coordinates, window_values, window_error,
                     window_mask, kernel_weights, fit_coordinate,
                     is_covar=False, cval=np.nan, error_weighting=True,
                     absolute_weight=False, edge_algorithm_idx=1,
                     edge_threshold=None,
                     get_error=True, get_weights=True,
                     get_distance_weights=True, get_rchi2=True,
                     get_offset_variance=True):  # pragma: no cover
    r"""
    Solve for a kernel convolution at a single coordinate.

    Generally, the kernel convolution value is of the form:

    .. math::

        f(x) = \sum_{j}{d_j k_j} / \sum_{j}{k_j}

    where :math:`x` is the coordinate at the fit point, :math:`j` are the
    indices of all samples within the fitting window, :math:`d` are the sample
    values, and :math:`k` are the kernel weights.

    EDGE CHECKING

    The first of the on-the-fly calculation is the "edge check".  Generally,
    polynomial fits are not well-defined away from the sample distribution
    from which they were derived.  This is especially true for higher order
    fits that may fit the sample distribution well, but start to deviate wildly
    when evaluated outside of the distribution.  The edge check step defines a
    border around the distribution, outside of which the fit will be aborted.
    There are a number of algorithms available which vary in robustness and
    speed.  Please see :func:`check_edges` for details on available algorithms.

    FITTING

    If the above checks pass, a fit can be attempted.  There are two types of
    fit that may be performed.  The first is the standard kernel fit described
    above.  However, if `is_covar` was set to `True`, the `window_values` are
    considered covariances to propagate, and a fit will derived by propagating
    a weighted variance (this was created for the SOFIA HAWC+ pipeline).

    Parameters
    ----------
    window_coordinates : numpy.ndarray (n_dimensions, n_samples)
        The independent coordinates within the window region of the fitting
        coordinate.
    window_values : numpy.ndarray (n_samples,)
        The dependent values of the samples.
    window_error : numpy.ndarray (n_samples,)
        The associated 1-sigma error values for each sample in each set.  The
        user may also supply an array of shape (1,) in which case all
        samples in a set will share the same associated error value.  If the
        shape is set to (0,) this indicates that no error values are available
        for the samples.
    window_mask : numpy.ndarray (n_samples,)
        A mask where `False` indicates that the associated sample should be
        excluded from the fit.
    kernel_weights : numpy.ndarray (n_samples,)
        The kernel weighting factors applied to each sample in the fit.
    fit_coordinate : numpy.ndarray (n_dimensions,)
        The coordinate of the fitting point.
    is_covar : bool, optional
        If `True`, indicates that `window_values` contains covariance values
        that should be propagated through algorithm.  If this is the case,
        polynomial fitting is disabled, and a weighted variance is calculated
        instead.
    cval : float, optional
        In a case that a fit is unable to be calculated at certain location,
        `cval` determines the returned fit value.
    error_weighting : bool, optional
        If `True`, weight the samples in the fit by the inverse variance
        (1 / window_error^2) in addition to distance weighting.
    absolute_weight : bool, optional
        If the kernel weights are negative, can lead to almost zero-like
        divisions in many of the algorithms.  If set to `True`, the sum of the
        absolute weights are used for normalization.
    edge_algorithm_idx : int, optional
        Integer specifying the algorithm used to determine whether a fit should
        be attempted with respect to the sample distribution.  Please see
        :func:`check_edges` for further information.  The default (1), is
        always the most robust of the available algorithms.
    edge_threshold : numpy.ndarray (n_dimensions,)
        A threshold parameter determining how close an edge should be to the
        center of the distribution during :func:`check_edges`.  Higher values
        result in an edge closer to the sample mean.  A value should be
        provided for each dimension.  A zero value in any dimension will result
        in an infinite edge for that dimension.
    get_error : bool, optional
        If `True`, return the error on the fit.
    get_weights : bool, optional
        If `True`, return the sum of all sample weights used in determining the
        fit.
    get_distance_weights : bool, optional
        If `True`, return the sum of only the distance weights used in
        determining the fit.
    get_rchi2 : bool, optional
        If `True`, return the reduced chi-squared statistic for each of
        the fits.
    get_offset_variance : bool optional
        If `True`, return the offset of the fitting point from the sample
        distribution.  See :func:`offset_variance` for further information.

    Returns
    -------
    fit_result : 7-tuple
        fit_result[0]: Fitted value (float).
                       Set to `cval` on fit failure.
        fit_result[1]: Error on the fit (float).
                       Set to NaN on fit failure.
        fit_result[2]: Number of samples included in the fit (int).
                       Set to 0 on fit failure.
        fit_result[3]: Weight sum (float).
                       Set to 0.0 on fit failure.
        fit_result[4]: Distance weight sum (float).
                       Set to 0.0 on fit failure.
        fit_result[5]: Reduced chi-squared statistic (float).
                       Set to NaN on fit failure.
        fit_result[6]: Offset variance from the distribution center (float).
                       Set to NaN on fit failure.
    """
    # need to update mask and counts for zero/bad weights
    counts = update_mask(kernel_weights, window_mask)

    # The order is: fitted_value, fitted_error, counts,
    #               weight, distance_weight, rchi2, offset_variance
    failure_values = (cval, np.nan, counts, 0.0, 0.0, np.nan, np.nan)

    if counts == 0:
        return failure_values

    # Check edges
    if edge_threshold is None:
        edge_thresh = np.empty(0, dtype=nb.float64)
    else:
        edge_thresh = np.asarray(edge_threshold, dtype=nb.float64)

    if not check_edges(window_coordinates, fit_coordinate, window_mask,
                       edge_thresh, algorithm=edge_algorithm_idx):
        return failure_values

    # Remove masked values
    window_values, window_error, kernel_weights = \
        apply_mask_to_kernel_set_arrays(window_mask, window_values,
                                        window_error, kernel_weights,
                                        counts=counts)

    # Calculate fitting weights
    window_full_weights = calculate_fitting_weights(
        window_error, kernel_weights, error_weighting=error_weighting)

    if not absolute_weight:
        weightsum = array_sum(window_full_weights)
    else:
        weightsum = array_sum(np.abs(window_full_weights))

    if weightsum == 0:
        return failure_values

    if is_covar:
        # Propagate variance
        fitted_value = weighted_mean_variance(
            window_values, window_full_weights, weightsum=weightsum)

        if get_distance_weights:
            if not absolute_weight:
                total_distance_weights = array_sum(kernel_weights)
            else:
                total_distance_weights = array_sum(np.abs(kernel_weights))
        else:
            total_distance_weights = 0.0

        if get_weights:
            if not absolute_weight:
                total_weights = array_sum(window_full_weights)
            else:
                total_weights = array_sum(np.abs(window_full_weights))
        else:
            total_weights = 0.0

        return (fitted_value,
                np.nan,  # error
                counts,
                total_weights,
                total_distance_weights,
                np.nan,  # rchi2
                np.nan  # offset_variance
                )

    fitted_value, fitted_variance, rchi2 = solve_mean_fit(
        window_values, window_error, window_full_weights,
        weightsum=weightsum,
        calculate_variance=get_error,
        calculate_rchi2=get_rchi2)

    fitted_error = math.sqrt(fitted_variance) if get_error else np.nan

    if get_distance_weights:
        if absolute_weight:
            distance_weight = array_sum(np.abs(kernel_weights))
        else:
            distance_weight = array_sum(kernel_weights)
    else:
        distance_weight = 0.0

    if get_offset_variance:
        variance_offset = offset_variance(
            window_coordinates, fit_coordinate, mask=window_mask)
    else:
        variance_offset = np.nan

    return (fitted_value,
            fitted_error,
            counts,
            weightsum,
            distance_weight,
            rchi2,
            variance_offset)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_kernel_weights(coordinates, reference, knots, n_knots,
                             coefficients, degrees, panel_mapping, panel_steps,
                             knot_steps, nk1, spline_mapping, eps=1e-8,
                             ):  # pragma: no cover
    r"""
    Calculate values of a kernel centered on a reference for given coordinates.

    The kernel must have been transformed into a spline representation.  A
    value of zero on the spline knots represents the center of the kernel.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The coordinates for which to calculate weights of shape
        (n_dimensions, n).
    reference : numpy.ndarray (float)
        The location of the kernel center of shape (n_dimensions,).
    knots : numpy.ndarray (float)
        The kernel spline knots of shape (n_dimensions, >max(n_knots)).
    n_knots : numpy.ndarray (int)
        The number of spline knots in each dimension of shape (n_dimensions,).
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The spline degrees in each dimension of shape (n_dimensions,).
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    panel_steps : numpy.ndarray (int)
        The flat index mapping steps in panel-space of shape (n_dimensions,).
        These are returned by passing the shape `Spline.panel_shape` into
        :func:`flat_index_mapping`.
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    nk1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) containing the values n_knots - k1
        where n_knots are the number of knots in each dimension, and k1 are the
        spline degrees + 1 in each dimension.
    spline_mapping : numpy.ndarray (int)
        An array containing the spline mapping (flat to n-D) indices.  This is
        created by passing the spline shape (degrees + 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_spline_coefficients).
    eps : float, optional
        Due to rounding errors, sometimes a value is flagged as invalid if it
        is exactly on the edge of the kernel.  This value allows a certain
        tolerance to those incorrect results.


    Returns
    -------
    weights : numpy.ndarray (float)
        The interpolated weights of shape (n,).
    """
    ndim, n = coordinates.shape
    kernel_extent = np.empty((ndim, 2), dtype=nb.float64)
    for dim in range(ndim):
        kernel_extent[dim, 0] = knots[dim, 0] - eps
        kernel_extent[dim, 1] = knots[dim, n_knots[dim] - 1] + eps

    relative_position = np.empty((ndim, n), dtype=nb.float64)
    valid = np.full(n, True)
    weights = np.empty(n, dtype=nb.float64)

    n_valid = 0
    for i in range(n):
        for dim in range(ndim):
            diff = coordinates[dim, i] - reference[dim]
            if kernel_extent[dim, 0] <= diff <= kernel_extent[dim, 1]:
                relative_position[dim, n_valid] = diff
            else:
                valid[i] = False
                break
        else:
            valid[i] = True
            n_valid += 1

    if n_valid == 0:
        for i in range(n):
            weights[i] = 0.0
    else:
        fit = perform_fit(
            coordinates=relative_position[:, :n_valid],
            knots=knots,
            coefficients=coefficients,
            degrees=degrees,
            panel_mapping=panel_mapping,
            panel_steps=panel_steps,
            knot_steps=knot_steps,
            nk1=nk1,
            spline_mapping=spline_mapping,
            n_knots=n_knots)
        n_valid = 0
        for i in range(n):
            if valid[i]:
                weights[i] = fit[n_valid]
                n_valid += 1
            else:
                weights[i] = 0.0

    return weights


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def apply_mask_to_kernel_set_arrays(mask, data, error, weights, counts=None
                                    ):  # pragma: no cover
    """
    Set certain arrays to a fixed size based on a mask array.

    Parameters
    ----------
    mask : numpy.ndarray of bool (N,)
        Mask where `True` values indicate the associated element should be
        kept, and `False` will result in exclusion from the output arrays.
    data : numpy.ndarray (N,)
        The data array.
    error : numpy.ndarray
        An array of shape (1,) or (N,).  If an array of size 1 is supplied,
        it will be expanded to an array of `counts` size.
    weights : numpy.ndarray
        An array of shape (1,) or (N,).  If an array of size 1 is supplied,
        it will be expanded to an array of `counts` size.
    counts : int, optional
        The number of `True` values in the mask, optionally passed in for
        speed.  Determines the output size of all arrays.

    Returns
    -------
    data_out, error_out, weight_out : 3-tuple of numpy.ndarray.
       Resized arrays in which the last axis is of size `counts`.
    """

    if counts is None:
        counts = 0
        for i in range(mask.size):
            counts += mask[i]

    n_data = data.size
    data_out = np.empty(counts, dtype=nb.float64)
    weight_out = np.empty(counts, dtype=nb.float64)
    single_weight = weights.size == 1

    valid_error = error.size > 0
    if valid_error:
        single_error = error.size == 1
        error_out = np.empty(counts, dtype=nb.float64)
    else:
        error_out = np.empty(0, dtype=nb.float64)
        single_error = False

    counter = 0
    for i in range(n_data):
        if mask[i]:
            data_out[counter] = data[i]
            if valid_error:
                if single_error:
                    error_out[counter] = error[0]
                else:
                    error_out[counter] = error[i]
            if single_weight:
                weight_out[counter] = weights[0]
            else:
                weight_out[counter] = weights[i]
            counter += 1
            if counter == counts:
                break

    return data_out, error_out, weight_out
