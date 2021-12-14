# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math
import sys
import warnings
import numpy as np
import numba as nb
from numba import njit
from numba.typed import List, Dict
from numba.core import boxing
from scipy.integrate import nquad
from scipy.special import gamma
from types import ModuleType, FunctionType
from gc import get_referents

from sofia_redux.toolkit.utilities.func import taylor

nb.config.THREADING_LAYER = 'threadsafe'
assert List
assert Dict
assert boxing

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

__all__ = ['polynomial_exponents', 'polynomial_derivative_map',
           'evaluate_derivative', 'evaluate_derivatives',
           'scale_coordinates', 'scale_forward_scalar', 'scale_forward_vector',
           'scale_reverse_scalar', 'scale_reverse_vector',
           'polynomial_terms', 'single_polynomial_terms',
           'multiple_polynomial_terms',
           'sscp', 'solve_coefficients', 'solve_amat_beta', 'fit_residual',
           'weighted_mean', 'weighted_variance', 'weighted_mean_variance',
           'weighted_fit_variance', 'fit_phi_value', 'fit_phi_variance',
           'solve_inverse_covariance_matrices',
           'covariance_matrix_inverse', 'estimated_covariance_matrix_inverse',
           'solve_rchi2_from_error', 'solve_rchi2_from_variance',
           'solve_mean_fit', 'calculate_fitting_weights',
           'fasttrapz',
           'relative_density',
           'array_sum', 'update_mask',
           'multivariate_gaussian',
           'shaped_adaptive_weight_matrices',
           'shaped_adaptive_weight_matrix',
           'scaled_adaptive_weight_matrices',
           'scaled_adaptive_weight_matrix',
           'calculate_adaptive_distance_weights_scaled',
           'calculate_adaptive_distance_weights_shaped',
           'calculate_distance_weights_from_matrix',
           'calculate_distance_weights',
           'coordinate_mean', 'coordinate_covariance', 'offset_variance',
           'variance_from_offsets', 'distribution_variances',
           'derivative_mscp',
           'check_edges', 'check_edge_with_ellipsoid',
           'check_edge_with_distribution', 'check_edge_with_box',
           'check_edge_with_range',
           'check_orders', 'check_orders_with_bounds',
           'check_orders_without_bounds', 'check_orders_with_counts',
           'apply_mask_to_set_arrays', 'no_fit_solution',
           'solve_polynomial_fit', 'sigmoid', 'logistic_curve',
           'half_max_sigmoid', 'stretch_correction', 'solve_fits', 'solve_fit',
           'convert_to_numba_list']


def polynomial_exponents(order, ndim=None, use_max_order=False):
    r"""
    Define a set of polynomial exponents.

    The resampling algorithm uses defines a set of polynomial exponents as an
    array of shape (dimensions, terms) for an equation of the form:

    .. math::

        f( \Phi ) = \sum_{m=1}^{M}{c_m \Phi_m}

    for :math:`M` terms.  Here, :math:`\Phi_m` represents the product of
    independent variables, each raised to an appropriate power as defined by
    `exponents`. For example, consider the equation for 2-dimensional data
    with independent variables :math:`x` and :math:`y`:

    .. math::

        f(x, y) = c_1 + c_2 x + c_3 x^2 + c_4 y + c_5 x y + c_6 y^2

    In this case::

        exponents = [[0, 0],  # represents a constant or x^0 y^0
                     [1, 0],  # represents x
                     [2, 0],  # represents x^2
                     [0, 1],  # represents y
                     [1, 1],  # represents xy
                     [0, 2]]  # represents y^2

    The resampling algorithm solves for the coefficients (:math:`c`) by
    converting :math:`f(X) \rightarrow f(\Phi)` for
    :math:`K-\text{dimensional}` independent variables (:math:`X`)
    and `exponents` (:math:`p`) by setting:

    .. math::

        \Phi_m = \prod_{k=1}^{K}{X_{k}^{p_{m, k}}}

    In most of the code, the :math:`\Phi` terms are interchangable with
    "polynomial terms", and in the above example :math:`\Phi_5 = xy` since
    exponents[4] = [1, 1] representing :math:`x^1 y^1`.

    Note that for all terms (:math:`m`) in each dimension :math:`k`,
    :math:`\sum_{k=1}^{K}{p_{m, k}} \leq max(\text{order})`.  In addition,
    if `use_max_order` is `False` (default),
    :math:`p_{m,k} \leq \text{order}[k]`.

    Parameters
    ----------
    order : int or array_like of int
        Polynomial order for which to generate exponents.  If an array
        will create full polynomial exponents over all len(order)
        dimensions.

    ndim : int, optional
        If set, return Taylor expansion for `ndim` dimensions for
        the given `order` if `order` is not an array.

    use_max_order : bool, optional
        This keyword is only applicable for multi-dimensional data when orders
        are unequal across dimensions.  When `True`, the maximum exponent for
        each dimension is equal to max(order).  If `False`, the maximum
        available exponent for dimension k is equal to order[k].

    Returns
    -------
    exponents : numpy.ndarray
        (n_terms, n_dimensions) array of polynomial exponents.

    Examples
    --------
    >>> polynomial_exponents(3)
    array([[0],
           [1],
           [2],
           [3]])

    >>> polynomial_exponents([1, 2])
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1],
           [0, 2]])

    >>> polynomial_exponents(3, ndim=2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [3, 0],
           [0, 1],
           [1, 1],
           [2, 1],
           [0, 2],
           [1, 2],
           [0, 3]])
    """
    order = np.atleast_1d(np.asarray(order, dtype=int))
    if order.ndim > 1:
        raise ValueError("Order must have 0 or 1 dimensions")
    if order.size > 1:
        ndim = order.size
        check_maximum_order = not use_max_order
        max_order = np.max(order)
    else:
        if ndim is None:
            ndim = 1
        check_maximum_order = False
        max_order = order[0]

    exponents = np.asarray([list(e) for e in taylor(max_order, int(ndim))])
    exponents = np.flip(exponents, axis=-1)

    if check_maximum_order:
        keep = np.logical_not(np.any(exponents > order[None], axis=1))
        exponents = exponents[keep]

    return exponents


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def polynomial_derivative_map(exponents):  # pragma: no cover
    r"""
    Creates a mapping from polynomial exponents to derivatives.

    Please see :func:`polynomial_exponents` for details on how a polynomial
    equation is defined within the resampling algorithm, and the use of the
    `exponents` array in defining the polynomial terms (:math:`\Phi`).

    Within the confines of the resampling algorithm, the polynomial exponents
    should have always been defined in a way that will always allow the
    derivative of a polynomial fit to be calculated from existing,
    pre-calculated :math:`\Phi` terms.

    For example, consider the 2-dimensional 2nd order polynomial equation and
    its derivatives in each dimension:

    .. math::

        f(x, y) = c_1 + c_2 x + c_3 x^2 + c_4 y + c_5 x y + c_6 y^2

    .. math::

        \frac{\partial f}{\partial x} = c_2 + 2 c_3 x + c_5 y

    .. math::

        \frac{\partial f}{\partial y} = c_4 + c_5 x + 2 c_6 y

    Converting :math:`f(x, y) \rightarrow f(\Phi)` we get:

    .. math::

        f(\Phi) = c_1 \Phi_1 + c_2 \Phi_2 + c_3 \Phi_3 + c_4 \Phi_4 +
                  c_5 \Phi_5 + c_6 \Phi_6

    It can then be seen that

    .. math::

        \frac{\partial f}{\partial x} = c_2 \Phi_1 + 2 c_3 \Phi_2 + c_5 \Phi_4

    .. math::

        \frac{\partial f}{\partial y} = c_4 \Phi_1 + c_5 \Phi_2 + 2 c_6 \Phi_4

    Generalizing for a polynomial equation consisting of :math:`M` terms of the
    independent variable :math:`X` in :math:`K-\text{dimensions}`, a mapping
    (:math:`h`) can be devised enabling calculation of the derivatives from
    pre-existing terms.  For dimension :math:`k`:

    .. math::

        \frac{\partial f}{\partial X_k} = \sum_{m=1}^{M}
            {h_{k, 0, m} \cdot c_{h_{k, 1, m}} \cdot \Phi_{h_{k, 2, m}}}

    This allows the derivative to be calculated from the existing polynomial
    terms (:math:`\Phi`) and coefficients (:math:`c`).  In addition, the
    mapping can be calculated prior to reduction and will therefore only need
    to be calculated once along with :math:`\Phi`.  Once the coefficients are
    known, the derivatives can then be calculated using :math:`h`.

    Derivatives may be evaluated using :func:`evaluate_derivative` and
    :func:`evaluate_derivatives`.

    Parameters
    ----------
    exponents : numpy.ndarray (n_terms, n_dimensions)
        The exponents defining a polynomial equation.

    Returns
    -------
    derivative_map : numpy.ndarray of int
        An array of shape (n_dimensions, 3, n_valid_terms).  The second
        dimension (of size 3) gives a constant multiplier in the first element,
        the coefficient index in the second element, and the phi index in the
        second element.  The third dimension will generally be of a smaller
        size than the number of terms in the polynomial equation as not all
        are required to calculate the derivative.  Due to the fact that some
        dimensions may contain more valid terms than others, `n_valid_terms`
        is set to the maximum number of valid terms over all dimensions.  Any
        invalid terms still remaining in the mapping array will have
        multipliers set to zero, and index pointers set to -1.
    """

    n_terms, n_dimensions = exponents.shape
    derivative_map = np.empty((n_dimensions, 3, n_terms), dtype=nb.i8)
    terms_found_per_dimension = np.zeros(n_dimensions, dtype=nb.i8)

    for term_index in range(n_terms):
        term_exponents = exponents[term_index]

        for dimension in range(n_dimensions):
            terms_found = terms_found_per_dimension[dimension]
            derivative_exponent = term_exponents[dimension] - 1

            if derivative_exponent < 0:  # the term vanished
                continue

            # Now search for matching exponents in the original exponent set
            for j in range(n_terms):
                check_match = exponents[j]
                for i in range(n_dimensions):
                    if i == dimension:
                        if check_match[i] != derivative_exponent:
                            break
                    else:
                        if check_match[i] != term_exponents[i]:
                            break
                else:
                    # match found
                    # The lowered power as a constant in front of the new term
                    derivative_map[dimension, 0, terms_found] = term_exponents[
                        dimension]
                    # The coefficient index
                    derivative_map[dimension, 1, terms_found] = term_index
                    # The index of the phi term that can be used
                    derivative_map[dimension, 2, terms_found] = j
                    terms_found_per_dimension[dimension] += 1
                    break

    # Clean up
    max_found = 0
    for i in range(n_dimensions):
        n_found = terms_found_per_dimension[i]
        if n_found > max_found:
            max_found = n_found
        derivative_map[i, 0, n_found:] = 0
        derivative_map[i, 1, n_found:] = -1
        derivative_map[i, 2, n_found:] = -1

    return derivative_map[:, :, :max_found]


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def evaluate_derivative(coefficients, phi_point, derivative_map
                        ):  # pragma: no cover
    r"""
    Calculates the derivative of a polynomial at a single point.

    Please see :func:`polynomial_derivative_map` for a full description of
    how the derivatives are calculated from a polynomial equation defined by
    :func:`polynomial_exponents`.  These also explain how one should transform
    the independent variables to the "phi" (:math:`\Phi`) terms (which may be
    done using :func:`polynomial_terms`).

    The derivative at a point is calculated by:

    .. math::

        \frac{\partial f}{\partial X_k} = \sum_{m=1}^{M}
            {h_{k, 0, m} \cdot c_{h_{k, 1, m}} \cdot \Phi_{h_{k, 2, m}}}

    for a polynomial equation consisting of :math:`M` terms at the coordinate
    :math:`X` in dimension :math:`k`, where :math:`h` is the derivative map.

    Parameters
    ----------
    coefficients : numpy.ndarray (n_terms,)
        The coefficients of the polynomial equation for each term.
    phi_point : numpy.ndarray (n_terms,)
        The polynomial terms of the fittin equation at a single coordinate.
    derivative_map : numpy.ndarray
        An array of shape (n_dimensions, 3, n_valid_terms).  The second
        dimension (of size 3) gives a constant multiplier in the first element,
        the coefficient index in the second element, and the phi index in the
        second element.  The third dimension will generally be of a smaller
        size than the number of terms in the polynomial equation as not all
        are required to calculate the derivative.  Due to the fact that some
        dimensions may contain more valid terms than others, `n_valid_terms`
        is set to the maximum number of valid terms over all dimensions.  Any
        invalid terms still remaining in the mapping array will have
        multipliers set to zero, and index pointers set to -1.

    Returns
    -------
    derivative : numpy.ndarray (n_dimensions,)
        The partial derivative of the polynomial equation with respect to each
        dimension.
    """

    ndim = derivative_map.shape[0]
    nterms = derivative_map.shape[2]
    gradients = np.empty(ndim, dtype=nb.float64)

    for i in range(ndim):
        dimension_map = derivative_map[i]
        if dimension_map[0, 0] == -1:
            gradients[i] = 0.0
            continue

        value = 0.0
        for term in range(nterms):
            lowered_power_constant = dimension_map[0, term]
            if lowered_power_constant == 0:
                continue

            coefficient = coefficients[dimension_map[1, term]]
            term_phi = phi_point[dimension_map[2, term]]
            value += lowered_power_constant * coefficient * term_phi

        gradients[i] = value

    return gradients


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def evaluate_derivatives(coefficients, phi_points, derivative_map
                         ):  # pragma: no cover
    r"""
    Calculates the derivative of a polynomial at multiple points.

    Please see :func:`polynomial_derivative_map` for a full description of
    how the derivatives are calculated from a polynomial equation defined by
    :func:`polynomial_exponents`.  These also explain how one should transform
    the independent variables to the "phi" (:math:`\Phi`) terms (which may be
    done using :func:`polynomial_terms`).

    The derivative at a point is calculated by:

    .. math::

        \frac{\partial f}{\partial X_k} = \sum_{m=1}^{M}
            {h_{k, 0, m} \cdot c_{h_{k, 1, m}} \cdot \Phi_{h_{k, 2, m}}}

    for a polynomial equation consisting of :math:`M` terms at the coordinate
    :math:`X` in dimension :math:`k`, where :math:`h` is the derivative map.

    Parameters
    ----------
    coefficients : numpy.ndarray (n_terms,)
        The coefficients of the polynomial equation for each term.
    phi_points : numpy.ndarray (n_terms, n_points)
        The polynomial terms of the fitting equation at a multiple points.
    derivative_map : numpy.ndarray
        An array of shape (n_dimensions, 3, n_valid_terms).  The second
        dimension (of size 3) gives a constant multiplier in the first element,
        the coefficient index in the second element, and the phi index in the
        second element.  The third dimension will generally be of a smaller
        size than the number of terms in the polynomial equation as not all
        are required to calculate the derivative.  Due to the fact that some
        dimensions may contain more valid terms than others, `n_valid_terms`
        is set to the maximum number of valid terms over all dimensions.  Any
        invalid terms still remaining in the mapping array will have
        multipliers set to zero, and index pointers set to -1.

    Returns
    -------
    derivatives : numpy.ndarray (n_dimensions, n_points)
        The partial derivative of the polynomial equation with respect to each
        dimension at each point.
    """

    ndim = derivative_map.shape[0]
    ndata = phi_points.shape[1]
    nterms = derivative_map.shape[2]
    gradients = np.zeros((ndim, ndata), dtype=nb.float64)

    for i in range(ndim):
        dimension_map = derivative_map[i]
        if dimension_map[0, 0] == -1:
            continue

        for term in range(nterms):
            lowered_power_constant = dimension_map[0, term]
            if lowered_power_constant == 0:
                continue

            coefficient = coefficients[dimension_map[1, term]]
            if coefficient == 0:
                continue

            term_phi = phi_points[dimension_map[2, term]]
            multiplier = lowered_power_constant * coefficient

            for k in range(ndata):
                gradients[i, k] += multiplier * term_phi[k]

    return gradients


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def derivative_mscp(coefficients, phi_samples, derivative_map,
                    sample_weights):  # pragma: no cover
    r"""
    Return the weighted mean-square-cross-product (mscp) of sample derivatives.

    Given a polynomial equation of the form:

    .. math::

        f(\Phi) = c \cdot \Phi

    The derivative is calculated as:

    .. math::

        \frac{\partial f}{\partial X_k} = \sum_{m=1}^{M}
        {h_{k, 0, m} \cdot c_{h_{k, 1, m}} \cdot \Phi_{h_{k, 2, m}}}

    for an equation of :math:`M` terms at the coordinate :math:`X` in
    dimension :math:`k`, where :math:`h` is the `derivative_map` and :math:`c`
    are the `coefficients`.  Please see :func:`polynomial_derivative_map` for
    a more complete description of the derivative calculation.

    One the derivatives (:math:`g = \frac{df}{dX}`) are calculated for all
    samples, they are averaged, and the cross-product is returned as:

    .. math::

        \bar{g}^2 = \frac{1}{tr(W W^T)} g^T W W^T g

    where :math:`W = diag(\text{weights})`.

    For example, for polynomial fit of 2-dimensional data :math:`f(x, y)`, the
    returned matrix will be:

    .. math::

        \bar{g}^2 =
            \begin{bmatrix}
                \frac{\partial f}{\partial x} \frac{\partial f}{\partial x} &
                \frac{\partial f}{\partial x} \frac{\partial f}{\partial y} \\
                \frac{\partial f}{\partial y} \frac{\partial f}{\partial x} &
                \frac{\partial f}{\partial y} \frac{\partial f}{\partial y}
            \end{bmatrix}

    Parameters
    ----------
    coefficients : numpy.ndarray (n_terms,)
        The coefficients of a polynomial fit for each term.
    phi_samples : numpy.ndarray (n_terms, n_samples)
        The polynomial terms of the sample coordinates.  Please see
        :func:`polynomial_exponents` for a description of this variable.
    derivative_map : numpy.ndarray
        An array of shape (n_dimensions, 3, n_valid_terms).  Please see
        :func:`polynomial_derivative_map` for an explanation of this variable.
    sample_weights : numpy.ndarray (n_samples,)
        The weighting to apply to each sample when determining the weighted
        mean (as a multiplier).

    Returns
    -------
    mscp : numpy.ndarray (n_dimensions, n_dimensions)
        An array where sscp[i, j] = derivative[i] * derivative[j].
        where derivative is the weighted mean derivatives over all samples.
    """
    gradients = evaluate_derivatives(coefficients, phi_samples, derivative_map)
    gradient_mscp = sscp(gradients, weight=sample_weights, normalize=True)

    return gradient_mscp


def scale_coordinates(coordinates, scale, offset, reverse=False):
    r"""
    Apply scaling factors and offsets to N-dimensional data.

    The two available transforms are controlled by the `reverse`.  The
    transform functions apply the following functions:

    +-----------------+----------------------+
    |    Reverse      | f(x)                 |
    +=================+======================+
    | False (default) | (x - offset) / scale |
    +-----------------+----------------------+
    | True            | (x * scale) + offset |
    +-----------------+----------------------+

    Parameters
    ----------
    coordinates : numpy.ndarray (N, M) or (N,)
        Either a 1 or 2-dimensional array may be supplied.  If a 1-dimensional
        array is supplied, it is assumed that it represents a single
        coordinates in N-dimensions.  If a 2-dimensional array is supplied,
        it should be of shape (N, M) where N is the number of dimensions, and
        M is the number of coordinates.
    scale : numpy.ndarray (N,)
        The scaling factor to apply to each dimension.
    offset : numpy.ndarray (N,)
        The offset to apply to each dimension.
    reverse : bool, optional
        If `True`, apply the reverse transform.  The default is `False`.

    Returns
    -------
    numpy.ndarray of numpy.float64 (N, M) or (N,)
        The scaled `coordinates` array.
    """
    scalar = coordinates.ndim == 1
    if reverse:
        if scalar:
            return scale_reverse_scalar(coordinates, scale, offset)
        else:
            return scale_reverse_vector(coordinates, scale, offset)
    else:
        if scalar:
            return scale_forward_scalar(coordinates, scale, offset)
        else:
            return scale_forward_vector(coordinates, scale, offset)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def scale_forward_scalar(coordinate, scale, offset):  # pragma: no cover
    r"""
    Applies the function `f(x) = (x - offset) / scale` to a single coordinate.

    This is a :mod:`numba` jit compiled function.

    Parameters
    ----------
    coordinate : numpy.ndarray (N,)
        An array where N is the number of dimensions.
    scale : numpy.ndarray (N,)
        The scaling factor to apply to each dimension.
    offset : numpy.ndarray (N,)
        The offset to apply to each dimension.

    Returns
    -------
    numpy.ndarray of numpy.float64 (N,)
        The scaled `coordinates` array.
    """
    features = coordinate.size
    result = np.empty(features, dtype=nb.float64)
    for k in range(features):
        result[k] = (coordinate[k] - offset[k]) / scale[k]
    return result


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def scale_forward_vector(coordinates, scale, offset):  # pragma: no cover
    r"""
    Applies the function `f(x) = (x - offset) / scale` to a coordinate array.

    This is a :mod:`numba` jit compiled function.

    Parameters
    ----------
    coordinates : numpy.ndarray (N, M)
        An array where N is the number of dimensions, and M is the number of
        coordinates.
    scale : numpy.ndarray (N,)
        The scaling factor to apply to each dimension.
    offset : numpy.ndarray (N,)
        The offset to apply to each dimension.

    Returns
    -------
    numpy.ndarray of numpy.float64 (N, M)
        The scaled `coordinates` array.
    """
    features, ndata = coordinates.shape
    result = np.empty((features, ndata), dtype=nb.float64)
    for k in range(features):
        for i in range(ndata):
            result[k, i] = (coordinates[k, i] - offset[k]) / scale[k]
    return result


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def scale_reverse_vector(coordinates, scale, offset):  # pragma: no cover
    r"""
    Applies the function `f(x) = (x * scale) + offset` to a coordinate array.

    This is a :mod:`numba` jit compiled function.

    Parameters
    ----------
    coordinates : numpy.ndarray (N, M)
        An array where N is the number of dimensions, and M is the number of
        coordinates.
    scale : numpy.ndarray (N,)
        The scaling factor to apply to each dimension.
    offset : numpy.ndarray (N,)
        The offset to apply to each dimension.

    Returns
    -------
    numpy.ndarray of numpy.float64 (N, M)
        The scaled `coordinates` array.
    """
    features, ndata = coordinates.shape
    result = np.empty((features, ndata), dtype=nb.float64)
    for k in range(features):
        for i in range(ndata):
            result[k, i] = coordinates[k, i] * scale[k] + offset[k]
    return result


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def scale_reverse_scalar(coordinate, scale, offset):  # pragma: no cover
    r"""
    Applies the function `f(x) = (x * scale) + offset` to a single coordinate.

    This is a :mod:`numba` jit compiled function.

    Parameters
    ----------
    coordinate : numpy.ndarray (N,)
        An array where N is the number of dimensions.
    scale : numpy.ndarray (N,)
        The scaling factor to apply to each dimension.
    offset : numpy.ndarray (N,)
        The offset to apply to each dimension.

    Returns
    -------
    numpy.ndarray of numpy.float64 (N,)
        The scaled `coordinates` array.
    """
    features = coordinate.size
    result = np.empty(features, dtype=nb.float64)
    for k in range(features):
        result[k] = coordinate[k] * scale[k] + offset[k]
    return result


def polynomial_terms(coordinates, exponents):
    r"""
    Derive polynomial terms given coordinates and polynomial exponents.

    Raises a single coordinate or multiple coordinates by a power and then
    calculates the product over all dimensions.  For example, the output of
    an (x, y) vector with `exponent=[[2, 3]]` would be :math:`x^2y^3`.

    Note that multiple sets of exponents are expected to be provided during
    this operation, so the `exponents` parameter should be a 2-dimensional
    array.  If a single N-dimensional vector is provided, the output will be
    a 1-dimensional array with a single value for each exponent set.  If
    multiple vectors are provided, the output will be of shape (number of
    exponent sets, number of vectors).

    Parameters
    ----------
    coordinates : numpy.ndarray (N, n_vectors) or (N,)
        Sets of coordinates in N-dimensions or a single coordinate of
        N-dimensions.
    exponents : numpy.ndarray (n_exponents, N)
        Sets of polynomial exponents to apply to coordinates.

    Returns
    -------
    numpy.ndarray of numpy.float64 (n_exponents, n_vectors) or (n_exponents,)
        The polynomial terms.
    """
    if coordinates.ndim == 2:
        return multiple_polynomial_terms(coordinates, exponents)
    else:
        return single_polynomial_terms(coordinates, exponents)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def single_polynomial_terms(coordinate, exponents):  # pragma: no cover
    r"""
    Derive polynomial terms for a single coordinate given polynomial exponents.

    Raises a single coordinate by a power and then calculates the product over
    all dimensions.  For example, the output of an (x, y) vector with
    `exponent=[[2, 3]]` would be :math:`x^2y^3`.

    Note that multiple sets of exponents are expected to be provided during
    this operation, so the `exponents` parameter should be a 2-dimensional
    array.  The return value will be a 1-dimensional array with size equal
    to the number of exponent sets provided.

    Parameters
    ----------
    coordinate : numpy.ndarray (N,)
        The coordinate in each dimension.
    exponents : numpy.ndarray (n_exponents, N)
        Sets of exponents to apply to the coordinate.

    Returns
    -------
    numpy.ndarray of numpy.float64 (n_exponents,)
        The polynomial terms for the coordinate.
    """
    n_coefficients, n_dimensions = exponents.shape
    pp = np.empty(n_coefficients, dtype=nb.float64)
    for i in range(n_coefficients):
        x = 1.0
        for j in range(n_dimensions):
            val = coordinate[j]
            exponent = exponents[i, j]
            val_e = 1.0
            for _ in range(exponent):
                val_e *= val
            x *= val_e
        pp[i] = x
    return pp


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def multiple_polynomial_terms(coordinates, exponents):  # pragma: no cover
    r"""
    Derive polynomial terms for a coordinate set given polynomial exponents.

    Raises multiple coordinates by a power and then calculates the product over
    all dimensions.  For example, the output of an (x, y) vector with
    `exponent=[[2, 3]]` would be :math:`x^2y^3`.

    Note that multiple sets of exponents are expected to be provided during
    this operation, so the `exponents` parameter should be a 2-dimensional
    array.  The return value will be a 2-dimensional array with the size of
    the first dimension equal to the number of exponent sets, and the size of
    the second dimension equal to the number of vector sets.

    Parameters
    ----------
    coordinates : numpy.ndarray (N, n_vectors)
        Sets of vectors in N-dimensions.
    exponents : numpy.ndarray (n_exponents, N)
        Sets of exponents by which to raise the vector.

    Returns
    -------
    numpy.ndarray of numpy.float64 (n_exponents, n_vectors)
        The product of the exponentiation of the vectors.
    """
    n_coefficients, n_dimensions = exponents.shape
    n_vectors = coordinates.shape[1]
    pp = np.empty((n_coefficients, n_vectors))

    for k in range(n_vectors):
        for i in range(n_coefficients):
            x = 1.0
            for j in range(n_dimensions):
                val = coordinates[j, k]
                exponent = exponents[i, j]
                val_e = 1.0
                for _ in range(exponent):
                    val_e *= val
                x *= val_e
            pp[i, k] = x
    return pp


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def sscp(matrix, weight=None, normalize=False):  # pragma: no cover
    r"""
    Calculate the sum-of-squares-and-cross-products of a matrix.

    For the matrix :math:`A`, calculates :math:`A^TA`.  If weights (:math:`W`)
    are provided.

    .. math::

        sscp = WA^TAW^T

    Note that the `weight` should only contain the diagonal elements of
    :math:`W`, and as such should be a 1-dimensional array.

    If `normalize=True`:

    .. math::

        sscp = \frac{WA^TAW^T}{trace(W^TW)}

    where :math:`W = I`, the identity matrix if `weight` is not supplied.

    Parameters
    ----------
    matrix : numpy.ndarray (M, N)
        Input matrix.
    weight : numpy.ndarray (N,), optional
        Weights to be applied during sscp.
    normalize : bool, optional
        If True, scales result as described above.  Default is False.

    Returns
    -------
    numpy.ndarray of numpy.float64 (M, M)
        Square output array containing the sum-of-squares-and-cross-products
        matrix.
    """
    m, n = matrix.shape
    ata = np.empty((m, m), dtype=nb.float64)

    if weight is None:
        for i in range(m):
            for j in range(i, m):
                a_sum = 0.0
                for k in range(n):
                    a_sum += matrix[i, k] * matrix[j, k]
                ata[i, j] = a_sum
                if i != j:
                    ata[j, i] = a_sum
        w2sum = n

    else:
        w2 = np.empty(n, dtype=nb.float64)
        w2sum = 0.0
        for k in range(n):
            w2[k] = weight[k] * weight[k]
            if normalize:
                w2sum += w2[k]

        for i in range(m):
            for j in range(i, m):
                a_sum = 0.0
                for k in range(n):
                    a_sum += w2[k] * matrix[i, k] * matrix[j, k]
                ata[i, j] = a_sum
                if i != j:
                    ata[j, i] = a_sum

    if normalize:
        ata /= w2sum

    return ata


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def scaled_matrix_inverse(matrix, n=None, rank=None):  # pragma: no cover
    """
    Returns the inverse of a matrix scaled by N / (N - rank(matrix)).

    The return value given `matrix` :math:`A` is

    .. \frac{N}{N - rank(A)} A^{-1}

    Note that if `n` is not provided or `rank` >= `n`, the return value will
    be :math:`A^{-1}` and :math:`A^{-1}A = I`.  Otherwise, if scaling is
    applied, the diagonal elements of :math:`A^{-1}A` will be equal to
    :math:`N / (N - rank(A))`, with offset elements equal to zero.

    Parameters
    ----------
    matrix : (N, M)
        Matrix to invert
    n : int or float, optional
        Refers to N in the above description.  If not passed in, no scaling
        will occur.
    rank : int or float, optional
        The rank of the matrix, optionally passed in for speed.

    Returns
    -------
    inverse_matrix (M, N)
        The inverse of matrix A where the diagonal elements of A^(-1)A are
        equal to N / (N - rank(A)) if scaling options were provided, else
        A^(-1)A = I.
    """
    if n is not None:
        if rank is None:
            rank = np.linalg.matrix_rank(matrix)
        if rank < n:
            scale = n / (n - rank)
        else:
            scale = 1.0
    else:
        scale = 1.0

    return scale * np.linalg.pinv(matrix)


@njit(fastmath=True, nogil=False, cache=True, parallel=False)
def solve_coefficients(amat, beta):  # pragma: no cover
    r"""
    Find least squares solution of Ax=B and rank of A.

    Parameters
    ----------
    amat : numpy.ndarray (ncoeffs, ndata)
        The coefficient array.
    beta : numpy.ndarray (ncoeffs,) or (ncoeffs, N)
        Dependent values.  If 2-dimensional, the least-squares solution is
        calculated for each column.

    Returns
    -------
    rank, x : int, numpy.ndarray (min(M, N),)
        The rank of `amat` and least-squares solution of x.

    """
    coefficients, residuals, rank, s = np.linalg.lstsq(amat, beta)
    return rank, coefficients


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_amat_beta(phi, data, weights):  # pragma: no cover
    r"""
    Convenience function returning matrices suitable for linear algebra.

    Given independent variables :math:`\Phi`, data :math:`y`, and weights
    :math:`W`, returns matrices :math:`A` and :math:`B` where:

    .. math::

        A = \Phi W \Phi^T

        B = \Phi W y

    Parameters
    ----------
    phi : numpy.ndarray (n_terms, n_samples)
        Polynomial terms of independent variables for each sample in the fit.
    data : numpy.ndarray (n_samples,)
        Sample data values.
    weights : numpy.ndarray (n_samples,)
        Squared weights.

    Returns
    -------
    A, B : numpy.ndarray (n_terms, n_terms), numpy.ndarray (n_terms,)
        The :math:`A` and :math:`B` terms described above.
    """
    ncoeffs, ndata = phi.shape
    alpha = np.empty((ncoeffs, ndata), dtype=nb.float64)
    beta = np.empty(ncoeffs, dtype=nb.float64)

    sqrt_weight = np.empty(ndata, dtype=nb.float64)
    for k in range(ndata):
        sqrt_weight[k] = math.sqrt(weights[k])

    for i in range(ncoeffs):
        b = 0.0
        for k in range(ndata):
            w = sqrt_weight[k]
            wa = w * phi[i, k]
            b += wa * w * data[k]
            alpha[i, k] = wa
        beta[i] = b

    amat = sscp(alpha)

    return amat, beta


def relative_density(sigma, counts, weight_sum, tolerance=None,
                     max_dim=4):
    r"""
    Returns the relative density of samples compared to a uniform distribution.

    The relative local density is defined as 1 for a uniform distribution,
    < 1 for a distribution that is sparse near the center, and > 1 when
    clustered around the center.

    The sum of the `distance_weights` returned from a Gaussian weighting
    function on the samples is required for this calculation.  The
    weighting function should be of the form:

    .. math::

        w(\Delta x) = exp \left(
            -\sum_{k=1}^{K}{\frac{-\Delta x_k^2}{2 \sigma_k^2}}
            \right)

    over :math:`K` dimensions where :math:`\Delta x_k` is the offset of a
    sample in dimension :math:`k` from the point of interest, and
    :math:`\sigma` must be supplied to `relative_density` as `sigma`, where
    `distance_weights` = :math:`\sum_{i=1}^{N}{w(x_i)}` and
    `counts` = :math:`N`.  Note that :math:`\sigma` and :math:`x` must be
    scaled such that the principle axis of an ellipsoid window
    containing all samples are equal to unity (principle axis in dimension
    :math:`k` is :math:`\Omega_k = 1` such that
    :math:`\prod_{k=1}^{K}{\Omega_k} = 1` below).

    The local relative density is then given as:

    .. math::

        \rho = \frac{\rho(\text{measured})}{\rho(\text{uniform})}

    where,

    .. math::

        \rho(\text{uniform}) = N \frac{\Gamma \left( 1 + \frac{K}{2} \right)}
                                      {\pi^{K/2} \prod_{k=1}^{K}{\Omega_k}}

    .. math::

        \rho(\text{measured}) = \frac
            {\sum_{i=1}^{N}{w_i}}
            {\int \cdots \int_{R} w(\mathbf{\Delta x}) \, {dx}_1 \cdots {dx}_K}

    and region :math:`R` satisfies the requirement
    :math:`\| \mathbf{\Delta x} \|_2 \leq 1`

    Parameters
    ----------
    sigma : np.ndarray (n_dimensions,)
        The standard deviation of the Gaussian weighting function used to
        calculate the `distance_weights` for each dimension.
    counts : int or float or numpy.ndarray (N,)
        The number of data samples included in the sum of distance weights.
    weight_sum : int or float or numpy.ndarray (N,)
        The sum of weights as returned from a Gaussian weighting function.
    tolerance : float, optional
        Relative error tolerance passed to `scipy.integrate.quad` when
        determining the integral of the weighting function.  The default of
        10^(2*dim - 7) determined by testing, balancing precision with
        speed and convergence.
    max_dim : int, optional
        If the number of dimensions is greater than max_dim, do not attempt
        to calculate the relative density since the integral calculation
        is unlikely to converge and will take a vast amount of time.  The
        return output will be 1.0 or an array of ones (N,).  The maximum
        recommended number of dimensions is 4 (default).

    Returns
    -------
    float or numpy.ndarray of float64 (N,)
        The relative density.
    """
    m = sigma.size  # number of dimensions

    if m > max_dim:
        return (weight_sum * 0.0) + 1

    if tolerance is None and m > 1:
        tolerance = 10 ** ((2 * m) - 7)

    # Assumes the window is 1 in all dimensions.  Coordinates passed into the
    # distance weighting function and sigma should be scaled accordingly.
    prod_omega = 1.0

    window_volume = np.pi ** (m / 2) * prod_omega
    window_volume /= gamma(1 + (m / 2))
    uniform_density = counts / window_volume

    alpha = 2 * sigma ** 2
    mean = np.zeros(m)
    limits = [[-1.0, 1.0]] * m

    def nd_weighting(*args):
        x = np.array([x for x in args])
        return calculate_windowed_distance_weight(x, mean, alpha)

    def opts(*args):
        options = {}
        if tolerance is not None:
            options['epsrel'] = float(tolerance)

        # Assign break points in the function
        if len(args) == 1:
            options['points'] = [-1.0, 1.0]
        else:
            points = 1.0 - np.linalg.norm(args[1:], ord=2) ** 2
            if points >= 0:
                points = np.sqrt(points)
                options['points'] = [-points, points]

        return options

    integration = nquad(nd_weighting, limits, opts=[opts] * m)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = weight_sum / uniform_density / integration

    return result


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_windowed_distance_weight(
        coordinate, center, alpha):  # pragma: no cover
    r"""
    Calculates exp(-(dx^2) / alpha) for a single coordinate.

    If the L2 norm of dx > 1, then the returned weight is zero.

    Parameters
    ----------
    coordinate : numpy.ndarray (ndim,)
        The coordinate in each dimension.
    center : numpy.ndarray (ndim,)
        The center in each dimension where dx = coordinate - center.
    alpha : numpy.ndarray (ndim,)
        The alpha values for each dimension.

    Returns
    -------
    weight : float
        The returned weight.
    """
    features = coordinate.size

    weight = 0.0
    r = np.sum((coordinate ** 2))
    if r > 1:
        return 0.0
    for i in range(features):
        d = coordinate[i] - center[i]
        d *= d / alpha[i]
        weight += d

    return math.exp(-weight)


@njit(fastmath=True, nogil=False, cache=True, parallel=False)
def fit_residual(data, phi, coefficients):  # pragma: no cover
    r"""
    Calculates the residual of a polynomial fit to data.

    The residual is calculated using the matrix operation Y - CX where
    Y is the `dataset`, C are the `coefficients` and X is the `phi` polynomial
    terms.

    Parameters
    ----------
    data : numpy.ndarray (n_samples,)
        Data from which coefficients were derived.
    phi : numpy.ndarray (n_terms, n_samples)
        Polynomial terms of independent values of each sample.
    coefficients : numpy.ndarray (n_terms,)
        Coefficient values.

    Returns
    -------
    residual : numpy.ndarray (n_samples,)
        The residual
    """
    residual = -np.dot(coefficients, phi)
    for i in range(residual.size):
        residual[i] += data[i]
    return residual


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def weighted_mean(data, weights, weightsum=None):  # pragma: no cover
    r"""
    Calculate the weighted mean of a data set.

    The weighted mean of data :math:`y` with weights :math:`w` is given as:

    .. math::

        \bar{y} = \frac{\sum_{i=1}^{N}{w_i y_i}}
                       {\sum_{i=1}^{N}{w_i}}

    This is a jit compiled :mod:`numba` function for use within other
    functions in `sofia_redux.toolkit.resampling`.

    Parameters
    ----------
    data : numpy.ndarray (ndata,)
        Data.
    weights : numpy.ndarray (ndata,)
        Weights.
    weightsum : int or float, optional
        Sum of `weights`, optionally passed in for speed if pre-calculated.

    Returns
    -------
    weighted_mean : float
        The weighted mean.
    """
    n = data.size
    data_sum = 0.0

    if weightsum is None:
        weightsum = 0.0
        for i in range(n):
            weightsum += weights[i]

    for i in range(n):
        data_sum += data[i] * weights[i]
    return data_sum / weightsum


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def weighted_variance(
        error, weights, weightsum=None):   # pragma: no cover
    r"""
    Utility function to calculate the biased weighted variance.

    Calculates the biased weighted variance from data errors as:

    .. math::

        V = \frac{\sum{(w\sigma)^2}}{(\sum{w})^2}

    Parameters
    ----------
    error : numpy.ndarray (ndata,)
        1-sigma error values.
    weights : numpy.ndarray (ndata,)
        Data weights.
    weightsum : int or float, optional
        Sum of weights.  Optionally passed in for speed if pre-calculated.

    Returns
    -------
    weighted_variance : float
        The weighted variance.
    """
    n = error.size
    v_sum = 0.0

    if weightsum is None:
        weightsum = 0.0
        for i in range(n):
            weightsum += weights[i]

    for i in range(n):
        w = weights[i]
        ew = error[i] * w
        v_sum += ew * ew

    return v_sum / weightsum / weightsum


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def weighted_mean_variance(
        variance, weights, weightsum=None):  # pragma: no cover
    r"""
    Calculated mean weighted variance.

    Propagate variance as:

    .. math::

        \bar{V} = \frac{\sum_{i=1}^{N}{w_i^2 V_i}}
                       {(\sum_{i=1}^{N}{w_i})^2}

    Parameters
    ----------
    variance : numpy.ndarray (ndata,)
        Variance array.
    weights : numpy.ndarray (ndata,)
        Weights.
    weightsum : int or float, optional
        Sum of weights.  Passed in for speed if pre-calculated.

    Returns
    -------
    mean_variance : float
        The propagated variance.
    """
    n = variance.size
    v_sum = 0.0

    if weightsum is None:
        weightsum = 0.0
        for i in range(n):
            weightsum += weights[i]

    for i in range(n):
        w = weights[i]
        v_sum += variance[i] * w * w

    return v_sum / weightsum / weightsum


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def weighted_fit_variance(
        residuals, weights, weightsum=None, rank=1):  # pragma: no cover
    r"""
    Calculate variance of a fit from the residuals of the fit to data.

    For data :math:`y`, weights :math:`w`, and fitted function
    :math:`f(x) = fit(x, y, w)`, the residual is given as
    :math:`r = y - f(x)`.  The variance is then given as:

    .. math::

        V = \frac{1}{N - M}
            \frac{\sum_{i=1}^{N}{w_i r_i^2}}
                 {\sum_{i=1}^{N}{w_i}}

    where :math:`M` = `dof` if :math:`M < N` and :math:`M = N - 1` otherwise.

    Parameters
    ----------
    residuals : numpy.ndarray (ndata,)
        The residuals given as data - fit.
    weights : numpy.ndarray (ndata,)
        The weights.
    weightsum : int or float, optional
        The sum of weights optionally passed in for speed if pre-calculated.
    rank : int, optional
        The degrees of freedom used in the variance calculation is taken as
        ndata - rank.  The default is 1 and applies the Bessel correction.
        If ndata < rank, rank is automatically set to ndata - 1.

    Returns
    -------
    variance : float
        Variance calculated from residuals.
    """
    n = residuals.size

    if weightsum is None:
        weightsum = 0.0
        for i in range(n):
            weightsum += weights[i]

    r2sum = 0.0
    for i in range(n):
        r = residuals[i]
        r2sum += weights[i] * r * r

    if n > rank:
        variance = r2sum / weightsum / (n - rank)
    else:
        variance = r2sum / weightsum

    return variance


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def fit_phi_value(phi, coefficients):  # pragma: no cover
    r"""
    Returns the dot product of phi and coefficients.

    A utility function for use in calculating the polynomial fit based on the
    polynomial terms of the independent values (`phi`), and a set of calculated
    `coefficients`.

    The return value for `phi` (:math:`\Phi`) terms and coefficients
    (:math:`c`) each consisting of :math:`L` terms is:

    .. math::

        y = \sum_{l=1}^{L}{c_l \Phi_l}

    The polynomial terms :math:`\Phi` are pre-calculated and used in place of
    regular independent values :math:`x` in the resampling algorithm to avoid
    the unnecessary recalculation of terms in a polynomial equation.  For
    example, if fitting

    .. math::

        y = 5 x_0 x_1 + 6 x_0 x_3^2 + 7 x_1^3 x_4^2 + 8 x_0 x_1 x_2 x_3

    we set

    .. math::

       c = [5,\, 6,\, 7,\, 8]

       \Phi = [x_0 x_1,\, x_0 x_3^2,\, x_1^3 x_4^2,\, x_0 x_1 x_2 x_3]

    and then only need to perform the simple fast calculation

    .. math::

        y = c \cdot \Phi

    Parameters
    ----------
    phi : numpy.ndarray (n_coefficients,)
        Polynomial terms of independent values.
    coefficients : numpy.ndarray (n_coefficients,)
        Coefficients used to determine fit.

    Returns
    -------
    fit : float
        The fitted value.
    """
    fit = 0.0
    for i in range(coefficients.size):
        fit += coefficients[i] * phi[i]
    return fit


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def fit_phi_variance(phi, inv_covariance):  # pragma: no cover
    r"""
    Calculates variance given the polynomial terms of a coordinate.

    The output variance for given polynomial terms `phi` (:math:`\Phi`) is
    given as:

    .. math::

        V = \Phi^T Var(\hat{c}) \Phi

    where :math:`Var(\hat{c})` is the covariance matrix inverse of the
    fit coefficients (`inv_covariance`) such that :math:`Var(\hat{c})_{i, j}`
    gives the covariance between the coefficients for terms :math:`i` and
    :math:`j`, and the coefficients :math:`\hat{c}` define the fit:

    .. math::

        y = \hat{c} \cdot \Phi

    Parameters
    ----------
    phi : numpy.ndarray (ncoeffs,)
        The polynomial terms.
    inv_covariance : numpy.ndarray (ncoeffs, ncoeffs)
        The covariance matrix inverse of the fit coefficients.

    Returns
    -------
    variance : float
        The calculated variance.
    """

    ncoeffs = phi.size
    var = 0.0
    # Note that for this specific usage, covariance matrix C may
    # not always = C^T

    for i in range(ncoeffs):
        phi_i = phi[i]
        for j in range(i, ncoeffs):
            if i == j:
                phi_ij = phi_i * phi_i
                inv_cov_ij = inv_covariance[i, j]
            else:
                phi_ij = phi_i * phi[j]
                inv_cov_ij = inv_covariance[i, j] + inv_covariance[j, i]

            var += inv_cov_ij * phi_ij

    return var


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_inverse_covariance_matrices(phi, error, residuals, weights,
                                      error_weighted_amat=None,
                                      rank=None,
                                      calculate_error=True,
                                      calculate_residual=True,
                                      estimate_covariance=False
                                      ):  # pragma: no cover
    r"""
    Inverse covariance matrices on fit coefficients from errors and residuals.

    A utility function to calculate the inverse covariance matrices of the fit
    coefficients (:math:`c`) based on the :math:`1\sigma` error values of the
    sample measurements and/or the residuals of the fit
    :math:`y - c \cdot \Phi`.

    The function used to calculate the error covariance may be either
    :func:`estimated_covariance_matrix_inverse` or
    :func:`covariance_matrix_inverse`.  However,
    :func:`estimated_covariance_matrix_inverse` will always be used to
    calculate the covariance matrix derived from residuals.

    Parameters
    ----------
    phi : numpy.ndarray (nterms, N)
        The polynomial terms for each of the N samples.
    error : numpy.ndarray (N,)
        The 1-sigma error values for each sample.
    residuals : numpy.ndarray (N,)
        The residuals of the fit y - c.phi.
    weights : numpy.ndarray (N,)
        The weighting of each sample in the fit.
    error_weighted_amat : numpy.ndarray (nterms, nterms), optional
        The matrix :math:`A = \Phi^T W Var(y) W \Phi`, optionally passed in for
        speed if pre-calculated.
    rank : int or float, optional
        The rank of `error_weighted_amat`, if provided, and it's rank was
        pre-calculated.  Otherwise, it will be solved for.
    calculate_error : bool, optional
        If True, calculate the covariance of the fit coefficients based upon
        the `error` values.
    calculate_residual : bool, optional
        If True, calculate the covariance of the fit coefficients based upon
        `residuals` of the fit.
    estimate_covariance : bool, optional
        If True, calculate the covariance of the fit coefficients from the
        `error` values using :func:`estimated_covariance_matrix_inverse`.
        Otherwise, use :func:`covariance_matrix_inverse`.

    Returns
    -------
    e_inv, r_inv : numpy.ndarray, numpy.ndarray
        The inverse covariance calculated from `error`, and the inverse
        covariance calculated from `residuals`.  If `calculate_error` is True,
        the shape of e_cov will be (nterms, nterms) or (0, 0) otherwise.  The
        same is true for `calculate_residual` and r_cov.
    """

    if not calculate_error and not calculate_residual:
        cov = np.empty((0, 0), dtype=nb.float64)
        return cov, cov

    # e_cov is the covariance determined from error values
    if calculate_error:
        if estimate_covariance:
            e_inv = estimated_covariance_matrix_inverse(
                phi, error, weights, rank=rank)
        else:
            if error_weighted_amat is None:
                amat = np.empty((0, 0), dtype=nb.float64)
            else:
                amat = np.asarray(error_weighted_amat)
            e_inv = covariance_matrix_inverse(
                amat, phi, error, weights, rank=rank)
    else:
        e_inv = np.empty((0, 0), dtype=nb.float64)

    # r_cov is the covariance determined from residuals
    if calculate_residual:
        # Always use the estimated covariance for residuals since they
        # would otherwise define a poor solution.
        r_inv = estimated_covariance_matrix_inverse(
            phi, residuals, weights, rank=rank)
        if not calculate_error:
            e_inv = r_inv
    else:
        r_inv = e_inv

    return e_inv, r_inv


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def covariance_matrix_inverse(amat, phi, error, weights, rank=None
                              ):  # pragma: no cover
    r"""
    Calculates the inverse covariance matrix inverse of the fit coefficients.

    If the least-squares solution to a fit is given as
    :math:`y = \hat{c} \cdot \Phi` when :math:`y = c \cdot \Phi + \epsilon`,
    the inverse covariance on the estimated fit coefficients :math:`\hat{c}` is
    given as:

    .. math::

        \Sigma^{-1} = \frac{N}{N - M} (\Phi^T W Var(y) \Phi)^{-1}

    where :math:`N` are the number of samples fit, :math:`M` are the lost
    degrees of freedom, `W` are the fit `weights`, and :math:`Var(y)` is
    related to the `error` (:math:`\sigma`) by:

    .. math::

        Var(y) = diag(1 / \sigma^2)

    Note that during the fitting process, it is common for the inverse of the
    covariance matrix :math:`\Sigma` to have already been calculated (not
    factoring in lost degrees of freedom) if inverse variance was included as a
    weighting factor for the fit.  If so, it should be passed in as `amat`
    (:math:`A`), and the final covariance is simply given by:

    .. math::

        \Sigma^{-1} = \frac{N}{N - M}A^{-1}

    If `amat` was not calculated, it should be supplied as an array of shape
    (0, 0) (because :mod:`numba`).

    Parameters
    ----------
    amat : numpy.ndarray (n_terms, n_terms)
        The matrix A as described above.  If the shape of amat is set
        to (0, 0), it will be calculated using the error, weights, and
        phi terms.  This should only be done if A was weighted by both
        `error` and `weights`.
    phi : numpy.ndarray (n_terms, N)
        The polynomial terms for each of the N data samples.
    error : numpy.ndarray (N,)
        The 1-sigma errors.
    weights : numpy.ndarray (N,)
        The weighting factor applied to each sample when determining the
        least squares solution of the fit.  Note that this *must not*
        factor in any error based weighting.  Therefore, in the case of
        the resampling algorithm, it should refer to the distance
        weighting factor of each sample from the resampling point, or
        an array of ones (np.ones(N)) if distance weighting was not
        applied.
    rank : int or float, optional
        The matrix rank of `amat`, optionally passed in for speed if
        pre-calculated.

    Returns
    -------
    inverse_covariance : numpy.ndarray (nterms, nterms)
        The covariance matrix inverse of fit coefficients.
    """
    n = error.size

    if amat.size != 0:
        return scaled_matrix_inverse(amat, n=n, rank=rank)

    weighting = np.empty(n, dtype=nb.float64)
    for i in range(n):
        if error[i] != 0:
            weighting[i] = np.sqrt(weights[i]) / error[i]
        else:
            weighting[i] = 0.0

    # Here, amat = phi.T @ diag(weights / error^2) @ phi
    amat = sscp(phi, weight=weighting)
    return scaled_matrix_inverse(amat, n=n, rank=rank)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def estimated_covariance_matrix_inverse(phi, error, weights, rank=None
                                        ):  # pragma: no cover
    r"""
    Calculates covariance matrix inverse of fit coefficients from mean error.

    An estimate to the covariance of fit parameters is given by:

    .. math::

        \Sigma = \frac{N}{N - M} \bar{\sigma}^2 (\Phi^T W \Phi)^{-1}

    where `N` are the number of samples used in the fit, :math:`M` are the
    number of lost degrees of freedom, :math:`W` are the `weights` applied to
    the samples during the fit, and the estimated coefficients :math:`\hat{c}`
    are used to fit :math:`y = \hat{c} \cdot \Phi + \sigma` to the sample
    population :math:`y = c \cdot \Phi + \epsilon`.  The weighted mean of the
    squared `error` (:math:`\bar{\sigma}^2`) is given by:

    .. math::

        \bar{\sigma}^2 = \frac{\sum_{i=1}^{N}{w_i e_i^2}}
                              {\sum_{i=1}^{N}{w_i}}

    The final returned matrix is :math:`\Sigma^{-1}`.

    Parameters
    ----------
    phi : numpy.ndarray (nterms, N)
        The polynomial terms for each of the N data samples.
    error : numpy.ndarray (N,)
        The 1-sigma errors or residuals to the fit.
    weights : numpy.ndarray (N,)
        The weighting of each sample in the fit, not including any error
        weighting.
    rank : int or float, optional
        The matrix rank of :math:`\Phi^T W \Phi`, optionally passed in for
        speed if pre-calculated.

    Returns
    -------
    covariance_inverse : numpy.ndarray (nterms, nterms)
        The inverse covariance matrix.
    """
    n = error.size
    v_sum = 0.0
    w_sum = 0.0
    weighting = np.empty(n, dtype=nb.float64)

    for i in range(n):
        weighting[i] = np.sqrt(weights[i])
        w_sum += weights[i]
        v_sum += weights[i] * error[i] * error[i]

    v_mean = v_sum / w_sum
    amat = sscp(phi, weight=weighting)

    return scaled_matrix_inverse(amat, n=n, rank=rank) * v_mean


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_rchi2_from_error(residuals, weights, errors,
                           weightsum=None, rank=1):  # pragma: no cover
    r"""
    Return the reduced chi-squared given residuals and sample errors.

    For `weights` :math:`w`, `errors` :math:`\sigma`, and residuals :math:`r`
    where :math:`r = y - f(x)`, the reduced chi-squared is given as:

    .. math::

       \chi_r^2 = \frac{N}{N - M}
                  \frac{\sum_{i=1}^{N}{w_i r_i^2 / \sigma_i^2}}
                       {\sum_{i=1}^{N}{w_i}}

    where :math:`M` is given by `rank`.

    Parameters
    ----------
    residuals : numpy.ndarray (N,)
        The residuals to the fit, or y - f(x).
    weights : numpy.ndarray (N,)
        The weights to each sample in the fit.
    errors : numpy.ndarray (N,)
        The 1-sigma measurement errors for each sample in the fit.
    weightsum : int or float, optional
        The sum of the sample weights, optionally passed in for speed.
    rank : int or float, optional
        The degrees of freedom used in the reduced chi-squared value is taken
        as N - rank.  The default is 1 and applies the Bessel correction.
        If N < rank, rank is automatically set to N - 1.

    Returns
    -------
    rchi2 : float
        The reduced chi-squared value for the fit.
    """
    r2sum = 0.0

    n = residuals.size
    calculate_weightsum = weightsum is None
    if calculate_weightsum:
        weightsum = 0.0

    for i in range(residuals.size):
        sig = residuals[i] / errors[i]
        r2sum += weights[i] * sig * sig
        if calculate_weightsum:
            weightsum += weights[i]

    if rank < n:
        rchi2 = r2sum / weightsum * n / (n - rank)
    else:
        rchi2 = r2sum / weightsum

    return rchi2


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_rchi2_from_variance(residuals, weights, variance,
                              weightsum=None, rank=1):  # pragma: no cover
    r"""
    Return the reduced chi-squared given residuals and constant variance.

    For `weights` :math:`w`, `variance` :math:`V`, and residuals :math:`r`
    where :math:`r = y - f(x)`, the reduced chi-squared is given as:

    .. math::

       \chi_r^2 = \frac{1}{N - M}
                  \frac{\sum_{i=1}^{N}{w_i r_i^2}}
                       {V \sum_{i=1}^{N}{w_i}}

    where :math:`M` is given by `rank`.

    Parameters
    ----------
    residuals : numpy.ndarray (N,)
        The residuals to the fit, or y - f(x).
    weights : numpy.ndarray (N,)
        The weights to each sample in the fit.
    variance : int or float
        The constant variance of the fit.
    weightsum : int or float, optional
        The sum of the sample weights, optionally passed in for speed.
    rank : int or float, optional
        The degrees of freedom used in the reduced chi-squared value is taken
        as N - rank.  The default is 1 and applies the Bessel correction.
        If N < rank, rank is automatically set to N - 1.

    Returns
    -------
    rchi2 : float
        The reduced chi-squared value for the fit.

    """
    if variance == 0:
        return 0.0
    r2sum = 0.0
    n = residuals.size

    calculate_weightsum = weightsum is None
    if calculate_weightsum:
        weightsum = 0.0

    for i in range(residuals.size):
        r = residuals[i]
        r2sum += weights[i] * r * r
        if calculate_weightsum:
            weightsum += weights[i]

    rchi2 = r2sum / weightsum / variance
    if rank < n:
        rchi2 /= n - rank

    return rchi2


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def solve_mean_fit(data, error, weight, weightsum=None,
                   calculate_variance=True,
                   calculate_rchi2=True):  # pragma: no cover
    r"""
    Return the weighted mean of data, variance, and reduced chi-squared.

    For `data` (:math:`y`), `error` (:math:`\sigma`), and weights (:math:`w`),
    the weighted mean is given as:

    .. math::

        \bar{y} = \frac{\sum_{i=1}^{N}{w_i y_i}}
                       {\sum_{i=1}^{N}{w_i}}

    The returned variance (:math:`V`) will depend on `use_error`.  If
    `use_error` is `True`:

    .. math::

        V = \frac{\sum_{i=1}^{N}{(w_i\sigma_i)^2}}{(\sum_{i=1}^{N}{w_i})^2}

    If `use_error` is `False`:

    .. math::

        V = \frac{1}{N - 1}
            \frac{\sum_{i=1}^{N}{w_i (y_i - \bar{y})^2}}
                 {\sum_{i=1}^{N}{w_i}}

    Finally, the reduced chi-squared statistic is given as:

    .. math::

       \chi_r^2 = \frac{N}{N - 1}
          \frac{\sum_{i=1}^{N}{w_i (y_i - \bar{y})^2 / \sigma_i^2}}
               {\sum_{i=1}^{N}{w_i}}

    Note that :math:`\chi_r^2 = 1` is `use_error` is `False`.

    Parameters
    ----------
    data : numpy.ndarray (N,)
        The data array consisting of N samples.
    error : numpy.ndarray (N,)
        The associated 1-sigma error values for each of the N data samples.
    weight : numpy.ndarray (N,)
        The weighting applied to each of the N data samples.
    weightsum : int or float, optional
        The sum of the weights, optionally passed in for speed if
        pre-calculated.
    calculate_variance : bool, optional
        If `True`, calculate the variance.  Otherwise, variance will be
        returned as a float value of zero.
    calculate_rchi2 : bool, optional
        If `True`, calculate the reduced chi-squared statistic.  Otherwise, it
        will be returned as a float value of zero.

    Returns
    -------
    mean, variance, rchi2 : float, float, float
        The weighted mean, variance and reduced chi-squared statistic.
    """

    fitted = weighted_mean(data, weight, weightsum=weightsum)
    if calculate_rchi2:
        residuals = data - fitted
    else:
        residuals = data  # dummy for Numba compilation success

    use_error = error.size != 0
    if calculate_variance:
        if use_error:
            variance = weighted_variance(error, weight, weightsum=weightsum)
        else:
            variance = weighted_fit_variance(
                residuals, weight, weightsum=weightsum, rank=1)
    else:
        variance = 0.0

    if calculate_rchi2:
        if use_error:
            rchi2 = solve_rchi2_from_error(
                residuals, weight, error, weightsum=weightsum, rank=1)
        else:
            rchi2 = 1.0
    else:
        rchi2 = 0.0

    return fitted, variance, rchi2


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_fitting_weights(errors, weights, error_weighting=True
                              ):  # pragma: no cover
    r"""
    Calculate the final weighting factor based on errors and other weights.

    If `error_weighting` is applied, the return value is `weights` / `error`^2.
    Otherwise, simply returns `weights`.

    Notes
    -----
    The square root of the weight is used in the polynomial system of
    equations rather than the actual weight due to how the least-squares
    solution is derived.

    For the linear system of equations A.x = B, we are solving
    (A^T.W.A).C = A^T.W.B, where C are the coefficients we wish to find.
    If we set X = sqrt(W).A, and Y = sqrt(W).B, this is the same as
    (X^T.X).C = X^T.Y, which can be solved easily.  Another way of thinking
    about it is that we are minimizing the squared residuals (y - f(x))^2.

    Parameters
    ----------
    errors : numpy.ndarray (N,)
        1-sigma error values to apply to weighting.
    weights : numpy.ndarray (N,)
        Other weighting factors, not including any type of error weighting.
    error_weighting : bool
        If False, returns `weights`, otherwise returns weights / errors^2.

    Returns
    -------
    fit_weighting : numpy.ndarray (N,)
        The final fitting weights.
    """

    if not error_weighting:
        return weights
    else:
        n = errors.size
        fit_weighting = np.empty(n)
        for i in range(n):
            e = 1.0 / errors[i]
            w2 = e * e * weights[i]
            fit_weighting[i] = w2

    return fit_weighting


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def array_sum(mask):  # pragma: no cover
    r"""
    Return the sum of an array.

    Utility function for fast :mod:`numba` calculation of the sum of a 1-D
    numpy array.

    Parameters
    ----------
    mask : numpy.ndarray (N,)
        The input array.

    Returns
    -------
    count : int or float
        The sum of values.
    """
    count = 0
    for i in range(mask.size):
        count += mask[i]
    return count


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_distance_weights(coordinates, reference, alpha
                               ):  # pragma: no cover
    r"""
    Returns a distance weighting based on coordinate offsets.

    Given a set of :math:`K` dimensional `coordinates` (:math:`x`), a single
    `reference` position (:math:`x_{ref}`), and the scaling factor `alpha`
    (:math:`\alpha`), returns the weighting factor:

    .. math::

        w(x) = exp \left(
                   \sum_{k=1}^{K}{\frac{-(x_{ref, k} - x_k)^2}{\alpha_k}}
                   \right)

    Notes
    -----
    `alpha` relates to the standard deviation (:math:`\sigma`) in a normal
    distribution as :math:`\alpha = 2\sigma^2`.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, N)
        An array of N coordinates in n_dimensions.
    reference : numpy.ndarray (n_dimensions,)
        The reference position from which to determine distance offsets for the
        weighting function.
    alpha : numpy.ndarray (1 or n_dimensions,)
        The distance scaling factor.  If an array of size 1 is supplied, it
        will be applied over all dimensions.  Otherwise, a value must be
        provided for each dimension.

    Returns
    -------
    weights : numpy.ndarray (N,)
        The distance weighting factors.
    """
    ndim, n = coordinates.shape
    weights = np.empty(n, dtype=nb.float64)
    symmetric = alpha.size == 1

    if symmetric and alpha[0] == 0:
        for i in range(n):
            weights[i] = 1.0
        return weights
    else:
        for i in range(n):
            weights[i] = 0.0

    for k in range(ndim):
        a = alpha[0] if symmetric else alpha[k]
        if a == 0:
            continue
        for i in range(n):
            d = reference[k] - coordinates[k, i]
            d *= d
            d /= a
            weights[i] += d

    for i in range(n):
        weights[i] = math.exp(-weights[i])

    return weights


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_distance_weights_from_matrix(
        coordinates, reference, alpha_matrix):  # pragma: no cover
    r"""
    Returns distance weights based on coordinate offsets and matrix operation.

    Given a set of :math:`K` dimensional `coordinates` (:math:`x`), a single
    `reference` position (:math:`o`), and the symmetric matrix `alpha_matrix`
    (:math:`A`), returns the weighting factor:

    .. math::

        w(x) = exp(-\Delta x^T A \Delta x)

    where :math:`{\Delta x}_k = o_k - x_k` for dimension :math:`k`.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        An array of N coordinates in n_dimensions.
    reference : numpy.ndarray (n_dimensions,)
        The reference position from which to determine distance offsets for the
        weighting function.
    alpha_matrix : numpy.ndarray (n_dimensions, n_dimensions)
        Defines the matrix operation to perform on the coordinate offsets.
        Note that in this implementation, it should be a symmetric matrix
        such that :math:`A = A^T`.  As such, only the upper triangle is
        iterated over and any off-diagonal elements in the lower triangle are
        ignored.

    Returns
    -------
    weights : numpy.ndarray (N,)
        The distance weighting factors.
    """
    n_dimensions, n = coordinates.shape
    weights = np.empty(n, dtype=nb.float64)

    for k in range(n):
        weight = 0.0
        for i in range(n_dimensions):
            xi = reference[i] - coordinates[i, k]
            for j in range(i, n_dimensions):
                if i == j:
                    xj = xi
                else:
                    xj = reference[j] - coordinates[j, k]
                if i == j:
                    weight += xi * xj * alpha_matrix[i, j]
                else:
                    weight += 2 * xi * xj * alpha_matrix[i, j]
        weights[k] = np.exp(-weight)

    return weights


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_adaptive_distance_weights_scaled(
        coordinates, reference, adaptive_alpha):  # pragma: no cover
    r"""
    Returns distance weights based on offsets and scaled adaptive weighting.

    Given a set of :math:`K` dimensional `coordinates` (:math:`x`), a single
    `reference` position (:math:`x_{ref}`), and scaling factors
    `adaptive_alpha` (:math:`A`), returns the weighting factor:

    .. math::

        w(x) = exp
            \left(
            -\sum_{i=1}^{K} \sum_{j=1}^{K} {{\Delta x}_i A_{i,j} {\Delta x}_j}
            \right)

    where :math:`{\Delta x}_k = x_{ref, k} - x_k` for dimension :math:`k`.

    Unlike :func:`calculate_distance_weights_from_matrix`, this function
    applies the function over multiple sets.  In this context, a single set
    will contain the same independent values (coordinates) as all other sets
    in a reduction, but the dependent data values may vary between sets.
    Therefore, it is necessary for the adaptive weighting values to also vary
    between sets.

    The algorithm is equivalent to
    :func:`calculate_adaptive_distance_weights_shaped` except when no rotation
    is applied, and therefore only a 1-dimensional array, rather than a matrix,
    is required to perform the transform.  The third axis of `adaptive_alpha`
    is set to one to allow :mod:`numba` to compile the function successfully,
    also indicating that there is no need to store the off-diagonal elements
    since they are all zero.  All stretching will therefore occur along the
    dimensional axes.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
    reference : numpy.ndarray (n_dimensions,)
    adaptive_alpha : numpy.ndarray
       (n_coordinates, n_sets, 1, n_dimensions) array containing the scaled
       adaptive weighting factors.

    Returns
    -------
    weights : numpy.ndarray (n_sets, n_coordinates)
        The distance weighting factors.
    """
    n_coordinates, n_sets, _, n_dimensions = adaptive_alpha.shape
    weights = np.zeros((n_sets, n_coordinates), dtype=nb.float64)

    for i in range(n_dimensions):
        r_ij = reference[i]
        x_ij = coordinates[i]
        a_ij = adaptive_alpha[:, :, 0, i]
        for k in range(n_coordinates):
            dx2 = r_ij - x_ij[k]
            dx2 *= dx2
            for set_number in range(n_sets):
                weights[set_number, k] += a_ij[k, set_number] * dx2

    for set_number in range(n_sets):
        for k in range(n_coordinates):
            weights[set_number, k] = math.exp(-weights[set_number, k])

    return weights


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def calculate_adaptive_distance_weights_shaped(
        coordinates, reference, shape_matrices):  # pragma: no cover
    r"""
    Returns distance weights based on offsets and shaped adaptive weighting.

    Given a set of :math:`K` dimensional `coordinates` (:math:`x`), a single
    `reference` position (:math:`o`), and the symmetric matrix `shape_matrices`
    (:math:`A`), returns the weighting factor:

    .. math::

        w(x) = exp(-\Delta x^T A^{-1} \Delta x)

    where :math:`{\Delta x}_k = o_k - x_k` for dimension :math:`k`.

    This function is applied to multiple coordinates over multiple sets.  In
    this context, a single set will contain the same independent values
    (coordinates) as all other sets in a reduction, but the dependent data
    values may vary between sets.  Therefore, it is necessary for the adaptive
    weighting values to also vary between sets.

    Unlike :func:`calculate_adaptive_distance_weights_scaled`, the matrix
    :math:`A` allows for the kernel weighting function to be stretched along
    arbitrarily rotated orthogonal axes.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
    reference : numpy.ndarray (n_dimensions,)
    shape_matrices : numpy.ndarray
        (n_coordinates, n_sets, n_dimensions, n_dimensions) array containing
        the shaped adaptive weighting matrix for all coordinates and sets.

    Returns
    -------
    weights : numpy.ndarray (n_sets, n_coordinates)
        The distance weighting factors.
    """
    n_coordinates, n_sets, _, n_dimensions = shape_matrices.shape
    weights = np.zeros((n_sets, n_coordinates), dtype=nb.float64)

    for i in range(n_dimensions):
        x_i = coordinates[i]
        r_i = reference[i]
        for j in range(i, n_dimensions):
            a_ij = shape_matrices[:, :, i, j]
            if i == j:
                x_j = x_i
                r_j = r_i
            else:
                x_j = coordinates[j]
                r_j = reference[j]

            for k in range(n_coordinates):
                dx2 = (r_i - x_i[k]) * (r_j - x_j[k])
                for set_number in range(n_sets):
                    value = a_ij[k, set_number] * dx2
                    if i != j:
                        # times 2 because we're only looping over
                        # upper triangle and it's certain a = a^T
                        value *= 2
                    weights[set_number, k] += value

    for set_number in range(n_sets):
        for k in range(n_coordinates):
            weights[set_number, k] = math.exp(-weights[set_number, k])

    return weights


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def update_mask(weights, mask):  # pragma: no cover
    r"""
    Updates a mask, setting False values where weights are zero or non-finite.

    Utility function update a boolean `mask` in place given `weights`.  mask
    values where weights are zero or non-finite will be set to `False`, and
    the total number of `True` values is returned as output.

    Parameters
    ----------
    weights : numpy.ndarray (N,)
        The weight values.
    mask : numpy.ndarray of bool (N,)
        The mask array to update in place.

    Returns
    -------
    counts : int
        The number of `True` mask values following the update.
    """
    counts = 0
    for i in range(weights.size):
        w = weights[i]
        if mask[i] and np.isfinite(w) and w != 0:
            counts += 1
        else:
            mask[i] = False

    return counts


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def coordinate_mean(coordinates, mask=None):  # pragma: no cover
    r"""
    Returns the mean coordinate of a distribution.

    Given a distribution of :math:`N` coordinates in :math:`K` dimensions, the
    mean coordinate in dimension :math:`k` is given as:

    .. math::

        \bar{x_k} = \frac{\sum_{i=1}^{N}{w_i x_{k,i}}}{\sum_{i=1}^{N}{w_i}}

    where :math:`w_i = 0` if mask[i] is `False` and :math:`w_i = 1` if
    mask[i] is `True`.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        The coordinate distribution.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An array of bool values where `True` indicates a coordinate should
        be included in the calculation, and `False` indicates that a coordinate
        should be ignored.  By default, all coordinates are included.

    Returns
    -------
    mean_coordinate : numpy.ndarray (n_dimensions,)
        The mean coordinate of the distribution.
    """
    n_dimensions, n_coordinates = coordinates.shape
    means = np.empty(n_dimensions, dtype=nb.float64)
    have_mask = mask is not None
    if n_coordinates == 0:
        for i in range(n_dimensions):
            means[i] = 0.0
        return means

    for i in range(n_dimensions):
        x_sum = 0.0
        w_sum = 0
        for k in range(n_coordinates):
            if have_mask and not mask[k]:
                continue
            x_sum += coordinates[i, k]
            w_sum += 1
        means[i] = x_sum / w_sum

    return means


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def coordinate_covariance(coordinates, mean=None, mask=None, dof=1
                          ):  # pragma: no cover
    r"""
    Calculate the covariance of a distribution.

    Given the sample distribution of :math:`N` `coordinates` (:math:`X`) in
    :math:`K` dimensions, the sample covariance is given as:

    .. math::

        \Sigma = E[(X - E[X])(X - E[X])^T]

    where :math:`\Sigma` is a :math:`K \times K` matrix and :math:`E` denotes
    the expected value.  In the general case where the expected value of
    :math:`X` is unknown and derived from the distribution itself, the
    covariance of the samples between dimension :math:`i` and :math:`j` is:

    .. math::

        \Sigma_{ij} = \frac{1}{N - M} \sum_{k=1}^{N}
                      {(X_{ki} - \bar{X}_i)(X_{kj} - \bar{X}_j)}

    where :math:`M` is the number of degrees of freedom lost (`dof`) in
    determining the `mean` (:math:`\bar{X}`).  If the `mean` is not provided,
    it will be calculated using :func:`coordinate_mean` in which case the
    default `dof` of 1 is appropriate.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        The coordinates of the distribution.
    mean : numpy.ndarray (n_dimensions,), optional
        The mean of the coordinate distribution in each dimension.  If not
        provided, the expected value in each dimension will be calculated
        using :func:`coordinate_mean`.
    mask : numpy.ndarray (n_coordinates,), optional
        An array of bool values where `True` indicates a coordinate should
        be included in the calculation, and `False` indicates that a coordinate
        should be ignored.  By default, all coordinates are included.
    dof : int or float, optional
        The lost degrees of freedom, typically 1 to indicate that the
        population mean is not known and is replaced by the sample mean.

    Returns
    -------
    covariance : numpy.ndarray of numpy.float64 (n_dimensions, n_dimensions)
        The covariance of the sample distribution.
    """
    n_dimensions, n_coordinates = coordinates.shape
    covariance = np.empty((n_dimensions, n_dimensions), dtype=nb.float64)
    have_mask = mask is not None

    if have_mask:
        n = 0
        for k in range(n_coordinates):
            if mask[k]:
                n += 1
    else:
        n = n_coordinates

    n -= dof
    if n <= 0:
        for i in range(n_dimensions):
            for j in range(n_dimensions):
                covariance[i, j] = 0.0
        return covariance

    if mean is None:
        mean = coordinate_mean(coordinates, mask=mask)

    delta = np.empty((n_dimensions, n_coordinates), dtype=nb.float64)
    for i in range(n_dimensions):
        for k in range(n_coordinates):
            delta[i, k] = coordinates[i, k] - mean[i]

    for i in range(n_dimensions):
        for j in range(i, n_dimensions):
            d_ij = 0.0
            for k in range(n_coordinates):
                if have_mask and not mask[k]:
                    continue
                d_ij += delta[i, k] * delta[j, k]

            cov_ij = d_ij / n
            covariance[i, j] = cov_ij
            if i != j:
                covariance[j, i] = cov_ij

    return covariance


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def offset_variance(coordinates, reference, mask=None, mean=None,
                    sigma_inv=None, scale=1.0, dof=1):  # pragma: no cover
    r"""
    Variance at reference coordinate derived from distribution uncertainty.

    Given a distribution of `coordinates` (:math:`X`), calculate the variance
    at a `reference` coordinate (:math:`X_{ref}`) based upon the uncertainty in
    the coordinate distribution.  Firstly, the distribution covariance matrix
    (:math:`\Sigma`) is calculated by :func:`coordinate_covariance` enabling
    the variance at the reference position to be given as:

    .. math::

        V(X_{ref}) = (X_{ref} - E[X])^{T} \Sigma^{-1} (X_{ref} - E[X])

    If the expected value of the distribution is known (:math:`E[X]`) or
    pre-calculated, it can be passed in to the function using the `mean`
    optional parameter along with the lost degrees of freedom (`dof`) spent in
    determining :math:`E[X]`.  If not, the default is to use :math:`\bar{X}`
    and `dof` = 1.

    The user may optionally specify a `scale` factor (:math:`\beta`) such that:

    .. math::

        V(X_{ref}) = (\beta(X_{ref} - E[X]))^{T} \Sigma^{-1}
                     (\beta(X_{ref} - E[X]))

    or:

    .. math::

        V(X_{ref}) = \beta^2 (X_{ref} - E[X])^{T} \Sigma^{-1} (X_{ref} - E[X])

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        The coordinates of the distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An array of bool values where `True` indicates a coordinate should
        be included in the calculation, and `False` indicates that a coordinate
        should be ignored.  By default, all coordinates are included.
    mean : numpy.ndarray (n_dimensions,), optional
        The mean of the coordinate distribution in each dimension.  If not
        provided, the expected value in each dimension will be calculated
        using :func:`coordinate_mean`.
    scale : int or float, optional
        The scaling factor described above.
    dof : int or float, optional
        The lost degrees of freedom, typically 1 to indicate that the
        population mean is not known and is replaced by the sample mean.
    sigma_inv : numpy.ndarray (n_dimensions, n_dimensions), optional
        If the covariance matrix of the coordinate distribution has already
        been calculated, the matrix inverse may be passed in as sigma_inv for
        speed.

    Returns
    -------
    variance : float
        The variance at the reference coordinate determined from the coordinate
        distribution.
    """
    if mean is None:
        mean = coordinate_mean(coordinates, mask=mask)

    offset = (mean - reference) * scale
    if sigma_inv is None:
        covariance = coordinate_covariance(
            coordinates, mean=mean, mask=mask, dof=dof)
    else:
        n_dimensions = coordinates.shape[0]
        covariance = np.empty((n_dimensions, n_dimensions), dtype=nb.float64)

    return variance_from_offsets(offset, covariance, sigma_inv=sigma_inv)


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def variance_from_offsets(offsets, covariance, sigma_inv=None
                          ):  # pragma: no cover
    r"""
    Determine the variance given offsets from the expected value.

    The output variance is:

    .. math::

        V = M^T \Sigma^{-1} M

    where `offsets` (:math:`M`) are the deviations :math:`X - E[X]` and
    :math:`\Sigma` is the `covariance`.

    Parameters
    ----------
    offsets : numpy.ndarray (n_dimensions,)
        The observational offsets from the expected value.
    covariance : numpy.ndarray (n_dimensions, n_dimensions)
        The covariance matrix of observations.
    sigma_inv : numpy.ndarray (n_dimensions, n_dimensions), optional
        The matrix inverse of the covariance matrix, optionally passed in for
        speed.

    Returns
    -------
    variance : float
        The variance as described above.
    """
    if sigma_inv is None:
        sigma_inv = np.linalg.pinv(covariance)

    features = offsets.size
    relative_variance = 0.0
    for i in range(features):
        offset_i = offsets[i]
        for j in range(i, features):
            cij = sigma_inv[i, j]
            if cij == 0:
                continue
            var_ij = offset_i * offsets[j] * cij

            if i == j:
                relative_variance += var_ij
            else:
                relative_variance += 2 * var_ij

    return relative_variance


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def distribution_variances(coordinates, mean=None, covariance=None, mask=None,
                           sigma_inv=None, dof=1):  # pragma: no cover
    r"""
    Return variance at each coordinate based on coordinate distribution.

    Given a normal sample distribution :math:`X`, returns the variance at each
    sample coordinate.  For example, consider a population of zero mean
    (:math:`\bar{X} = 0`) and a standard deviation of one (:math:`\sigma = 1`).
    Samples located at :math:`\bar{X} \pm \sigma` will return a variance of
    1, while samples located at :math:`\bar{X} \pm 2\sigma` will return a
    variance of 4.

    By default, the distribution variance is derived using
    :func:`coordinate_covariance`, and the sample mean is derived using
    :func:`coordinate_mean` assuming the loss of 1 degree of freedom. However,
    the expected value (:math:`E[X]`) may be supplied with the `mean` optional
    parameter along with the `covariance`, and the number of lost degrees of
    freedom (`dof`).

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.
    mean : numpy.ndarray (n_dimensions,), optional
        The expected mean value of the distribution.
    covariance : numpy.ndarray (n_dimensions, n_dimensions), optional
        The covariance matrix (if known) for the sample distribution.
    mask : numpy.ndarray of bool (n_samples,), optional
        An array of bool values where `True` indicates a sample should
        be included when calculating the mean and covariance, and `False`
        indicates that a sample should be ignored.  By default, all samples
        are included.  The output variance will still be calculated for all
        samples.
    sigma_inv : numpy.ndarray (n_dimensions, n_dimensions), optional
        The inverse of the covariance matrix, optionally passed in for speed
        if the covariance is known, and it's inverse has been pre-calculated.
    dof : int or float, optional
        The lost degrees of freedom, typically 1 to indicate that the
        population mean is not known and is replaced by the sample mean.

    Returns
    -------
    variance : numpy.ndarray (n_samples,)
        The variance at each sample coordinate based on the sample
        distribution.
    """

    n_dimensions, n_coordinates = coordinates.shape
    if coordinates.size == 0:
        return np.empty(n_coordinates, dtype=nb.float64)

    if mean is None:
        mean = coordinate_mean(coordinates, mask=mask)

    if sigma_inv is None:
        if covariance is None:
            cov = coordinate_covariance(
                coordinates, mean=mean, mask=mask, dof=dof)
        else:
            cov = np.asarray(covariance)
        cov_inv = np.linalg.pinv(cov)
    else:
        cov_inv = np.asarray(sigma_inv)

    # So Numba doesn't barf
    numba_dummy = np.empty((n_dimensions, n_dimensions), dtype=nb.float64)
    offsets = np.empty(n_dimensions, dtype=nb.float64)

    variance = np.empty(n_coordinates, dtype=nb.float64)
    for k in range(n_coordinates):
        for i in range(n_dimensions):
            offsets[i] = mean[i] - coordinates[i, k]

        variance[k] = variance_from_offsets(offsets, numba_dummy,
                                            sigma_inv=cov_inv)

    return variance


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def check_edges(coordinates, reference, mask, threshold,
                algorithm=1):  # pragma: no cover
    """
    Determine whether a reference position is within a distribution "edge".

    The purpose of this function is to allow the resampling algorithm to
    determine whether a fit should be performed at a `reference` location
    given a sample distribution of `coordinates`.  If a fit is attempted too
    far from the mean sample distribution, it will lead to a misleading result
    which becomes more pronounced at higher fit orders.

    Therefore, the sample distribution is assigned an "edge" based on one of
    four definitions.  If the `reference` position is outside of this edge, a
    `False` value will be returned, and no fitting will occur.

    For all edge definition algorithms, the `threshold` parameter will
    determine the distance of the edge from the sample mean.  As `threshold`
    is increased, the edge approaches the sample mean resulting in a more
    severe clipping of fit locations away from the center of a distribution.
    If `threshold` = 0, no edge clipping will occur.

    Since the main engine of the resampling algorithm relies on :mod:`numba`,
    the edge `algorithm` should be supplied as an integer.  Please see the
    relevant function listed below for further details on how the "edge" is
    defined.

    +-----------+--------------------------------------+
    | algorithm | Function                             |
    +===========+======================================+
    |         1 | :func:`check_edge_with_distribution` |
    +-----------+--------------------------------------+
    |         2 | :func:`check_edge_with_ellipsoid`    |
    +-----------+--------------------------------------+
    |         3 | :func:`check_edge_with_box`          |
    +-----------+--------------------------------------+
    |         4 | :func:`check_edge_with_range`        |
    +-----------+--------------------------------------+

    Generally, algorithms are ordered from most to least robust, and slowest to
    fastest, so, the default (1) is considered the most robust (although
    slowest) of the available algorithms.

    When dealing with more than one dimension, the
    :func:`check_edge_with_distribution` algorithm is recommended as it
    accounts for the shape of the distribution.  If the sample distribution is
    unknown (as opposed to a set of uniformly spaced coordinates), there is a
    chance for some samples to be (or to approach) a collinear distribution.
    Attempting a fit at any location in a tangential direction away from the
    distribution would likely result in a very poor fit.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate to test.
    mask : numpy.ndarray of bool (n_samples,)
        A mask where `True` values indicate a sample should be included in
        the edge determination.
    threshold : numpy.ndarray (n_dimensions,)
        A threshold parameter determining how close an edge should be to the
        center of the distribution.  Higher values result in an edge closer to
        the sample mean.  A value should be provided for each dimension.  A
        zero value in any dimension will result in an infinite edge for that
        dimension.
    algorithm : int, optional
        Integer specifying which edge definition to use.  Please see above for
        the associated functions.  Invalid choices will disable edge checking.

    Returns
    -------
    inside : bool
        `True` if the reference coordinate is inside the edge of the sample
        distribution, and `False` otherwise.
    """

    for i in range(threshold.size):
        if threshold[i] != 0:
            break
    else:
        return True

    if algorithm == 1:
        return check_edge_with_distribution(
            coordinates, reference, mask, threshold)

    elif algorithm == 2:
        return check_edge_with_ellipsoid(
            coordinates, reference, mask, threshold)

    elif algorithm == 3:
        return check_edge_with_box(coordinates, reference, mask, threshold)

    elif algorithm == 4:
        return check_edge_with_range(coordinates, reference, mask, threshold)

    else:
        return True


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def check_edge_with_distribution(
        coordinates, reference, mask, threshold):  # pragma: no cover
    r"""
    Defines an edge based on statistical deviation from a sample distribution.

    Given a sample distribution (:math:`X`) of `coordinates`, the deviation of
    a `reference` coordinate :math:`X_{ref}` from the mean sample distribution
    :math:`\bar{X}` is given as:

    .. math::

            \sigma_{ref} = \sqrt{(X_{ref} - \bar{X})^{T} \Sigma^{-1}
                                 (X_{ref} - \bar{X})}

    where :math:`\Sigma` is the sample covariance of :math:`X`.  In this
    definition, the "edge" of the distribution is defined at
    :math:`\beta = 1 / threshold` so that reference locations where
    :math:`\sigma_{ref} \leq \beta` are considered inside the distribution
    edge.  For example, setting `threshold` = 2 will return a `False` value if
    :math:`\sigma_{ref} > 0.5`.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate.
    mask : numpy.ndarray of bool (n_samples,)
        A mask where `False` values exclude the corresponding sample from any
        distribution statistics.
    threshold : int or float
        The "edge" of the distribution is given by 1 / threshold.  If the
        deviation of the reference coordinate from the distribution mean is
        greater than the edge, a `False` value will be returned.  Setting
        threshold to zero results in an edge at infinity, i.e., all reference
        coordinates will be considered inside the distribution edge.

    Returns
    -------
    inside : bool
        `True` if `reference` is inside the distribution "edge" and `False`
        otherwise.
    """

    if array_sum(mask) == 0:
        return False

    return offset_variance(
        coordinates, reference, mask=mask, scale=threshold) <= 1


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def check_edge_with_ellipsoid(coordinates, reference, mask, threshold
                              ):  # pragma: no cover
    r"""
    Defines an ellipsoid edge around a coordinate distribution.

    Given a distribution (:math:`X`) of :math:`N` samples in :math:`K`
    dimensions, the center of mass for dimension :math:`k` is:

    .. math::

        \bar{X}_{k} = \frac{1}{N} \sum_{i=1}^{N}{X_{ik}}

    The ellipsoid center is at :math:`\bar{X}` with principle axes given by
    :math:`\beta`, where :math:`\beta = 1 - \text{threshold}`.  Note that
    the resampling algorithm scales all coordinates to a window parameter such
    that :math:`|X_k| \leq 1`, and the `threshold` parameter therefore defines
    a fraction of the window in the range :math:`0 < \text{threshold} < 1`.  If
    threshold[k] = 0 or 1, then no edge will be defined for dimension
    :math:`k`, and the ellipsoid definition will only apply over remaining
    dimensions (dimension :math:`k` will be ignored in all calculations).

    A reference coordinate (:math:`X_{ref}`) is considered inside the ellipsoid
    edge if:

    .. math::

        \sum_{k=1}^{K}{\frac{(X_{ref, k} - \bar{X}_k)^2}{\beta_k^2}} \leq 1

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate.
    mask : numpy.ndarray of bool (n_samples,)
        A mask where `False` values exclude the corresponding sample from the
        center-of-mass calculation.
    threshold : numpy.ndarray (n_dimensions,)
        Defines the principle axes (1 - threshold) of an ellipsoid centered on
        the coordinate center of mass in units of the resampling window
        parameter.  Must be in the range 0 < threshold < 1.

    Returns
    -------
    inside : bool
        `True` if `reference` is inside the distribution "edge" and `False`
        otherwise.
    """
    features, ndata = coordinates.shape
    n = array_sum(mask)
    if n == 0:
        return False

    beta = 1.0 - threshold
    offset = 0.0
    n2 = n * n
    for i in range(features):
        if threshold[i] <= 0 or threshold[i] >= 1:
            continue
        com = 0.0
        for k in range(ndata):
            if not mask[k]:
                continue
            com += coordinates[i, k] - reference[i]

        com /= beta[i]
        offset += com * com

    offset /= n2
    if offset > 1:
        return False
    else:
        return True


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def check_edge_with_box(coordinates, reference, mask, threshold
                        ):  # pragma: no cover
    r"""
    Defines a hyperrectangle edge around a coordinate distribution.

    Given a distribution (:math:`X`) of :math:`N` samples in :math:`K`
    dimensions, the center of mass for dimension :math:`k` is:

    .. math::

        \bar{X}_{k} = \frac{1}{N} \sum_{i=1}^{N}{X_{ik}}

    The hypercube center is at :math:`\bar{X}` and its width in each
    dimension :math:`k` is :math:`2\beta_k` where
    :math:`\beta = 1 - \text{threshold}`.  Note that the resampling algorithm
    scales all coordinates to a window parameter such that
    :math:`|X_k| \leq 1`, and the `threshold` parameter therefore defines a
    fraction of the window in the range :math:`0 < \text{threshold} < 1`.  If
    threshold[k] = 0 or 1, then no edge will be defined for dimension
    :math:`k`.

    If this definition the "edge" for dimension :math:`k` is at
    :math:`\bar{X}_k \pm \beta_k` and `reference` (:math:`X_{ref}`) is
    considered inside the edge if:

    .. math::

        | \bar{X}_k - X_{ref, k} | \leq \beta_k, \, \forall k

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate.
    mask : numpy.ndarray of bool (n_samples,)
        A mask where `False` values exclude the corresponding sample from the
        center-of-mass calculation.
    threshold : numpy.ndarray (n_dimensions,)
        Defines the half-width dimensions (1 - threshold) of the hyperrectangle
        centered on the coordinate center of mass in units of the resampling
        window parameter.  Must be in the range 0 < threshold < 1.

    Returns
    -------
    inside : bool
        `True` if `reference` is inside the distribution "edge" and `False`
        otherwise.
    """

    features, ndata = coordinates.shape
    n = array_sum(mask)
    if n == 0:
        return False

    beta = 1.0 - threshold
    for i in range(features):
        if threshold[i] <= 0 or threshold[i] >= 1:
            continue
        com = 0.0
        for k in range(ndata):
            if not mask[k]:
                continue
            com += coordinates[i, k] - reference[i]
        com /= n
        if abs(com) > beta[i]:
            return False
    else:
        return True


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def check_edge_with_range(coordinates, reference, mask, threshold
                          ):  # pragma: no cover
    r"""
    Defines an edge based on the range of coordinates in each dimension.

    Given a distribution of sample `coordinates` (:math:`X`), and a `reference`
    position (:math:`X_{ref}`) in :math:`K` dimensions, check for each
    dimension :math:`k`:

    .. math::

        \{-\beta_k, \beta_k\} \in
        [min(X_k - X_{ref, k}), max(X_k - X_{ref, k})]
        , \, \forall k

    where :math:`\beta = 1 - \text{threshold}`.  In order words, in each
    dimension :math:`k`, there must be at least one member of :math:`X_k` at
    a distance of :math:`\beta_k` from :math:`X_{ref, k}` for both the
    positive and negative directions along :math:`k`.

    Note that the resampling algorithm scales all coordinates to a window
    parameter such that :math:`|X_k| \leq 1`, and the `threshold` parameter
    therefore defines a fraction of the window in the range
    :math:`0 < \text{threshold} < 1`.  If threshold[k] < 0 or theshold > 1,
    then no check will be performed for dimension :math:`k`.

    Parameters
    ----------
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate.
    mask : numpy.ndarray of bool (n_samples,)
        A mask where `False` values exclude the corresponding sample from the
        range check.
    threshold : numpy.ndarray (n_dimensions,)
        Defines the threshold.  Must be in the range 0 < threshold < 1.

    Returns
    -------
    inside : bool
        `True` if `reference` is inside the distribution "edge" and `False`
        otherwise.
    """
    features, ndata = coordinates.shape
    n = array_sum(mask)
    if n == 0:
        return False

    neg_threshold = threshold * -1
    for i in range(features):
        left_found = False
        right_found = False
        if threshold[i] <= 0 or threshold[i] >= 1:
            continue
        for k in range(ndata):
            if not mask[k]:
                continue
            offset = coordinates[i, k] - reference[i]
            if offset < 0:
                if left_found:
                    continue
                else:
                    left_found = offset <= neg_threshold[i]
            elif offset > 0:
                if right_found:
                    continue
                else:
                    right_found = offset >= threshold[i]
            if left_found and right_found:
                break
        else:
            return False
    else:
        return True


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def check_orders(orders, coordinates, reference, algorithm=1, mask=None,
                 minimum_points=None, required=False, counts=-1
                 ):  # pragma: no cover
    r"""
    Checks the sample distribution is suitable for a polynomial fit order.

    For a polynomial fit to be successful at a given order, one needs to
    ensure that the sample distribution is suitable for such a fit.  At a
    minimum, there need to be enough samples to derive a fit.  However, unless
    the samples are known to be distributed in such a way where a fit is always
    possible (such as regularly spaced samples), there is the possibility that
    the fit may be under-determined.  For example, it is not possible to
    perform any other fit than the mean of sample values (order 0) if the
    samples all share the same coordinate.

    This problem is compounded by the use of :mod:`numba` which does not allow
    any exception handling.  If a fit fails at a single reference position,
    the whole algorithm will fail.

    There are several algorithms available, ordered in decreasing robustness,
    but increasing speed.  The chosen algorithm must be supplied using an
    integer label (because :mod:`numba`).  Please see the relevant function
    listed in the table below for a more detailed description of each.  The
    `order_algorithm` parameter is supplied to the main resampling algorithm
    during `__init__` and is used to select the relevant algorithm.

    +-----------+-------------------------------------+-----------------+
    | algorithm | Function                            | order_algorithm |
    +===========+=====================================+=================+
    |         1 | :func:`check_orders_with_bounds`    | 'bounded'       |
    +-----------+-------------------------------------+-----------------+
    |         2 | :func:`check_orders_without_bounds` | 'extrapolate'   |
    +-----------+-------------------------------------+-----------------+
    |         3 | :func:`check_orders_with_counts`    | 'counts'        |
    +-----------+-------------------------------------+-----------------+

    Generally, :func:`check_orders_with_bounds` is the most robust and ensures
    that a fit will not only succeed, but is not likely to deviate widely from
    the expected result.  The :func:`check_orders_without_bounds` function
    should also allow fits to succeed, but allow fits to be generated away from
    the sample distribution.  This may be desirable, for example, if one is
    resampling an image and wishes to perform fits close to the edge.
    Finally, :func:`check_orders_with_counts` is fast, but there is a decent
    possibility that the fit will fail if the user cannot guarantee that the
    samples are distributed appropriately.

    In rare cases, a distribution of collinear samples, not aligned along any
    dimensional axis may pass the `check_orders` test causing resampling
    algorithm to fail.  In this case, please consider using
    :func:`check_edge_with_distribution` to perform such a check.  This may
    be invoked by supplying `edge_algorithm='distribution'` during the
    initialization of the main resampling algorithm.

    Parameters
    ----------
    algorithm : int
        An integer specifying the order checking algorithm to use.  If an
        invalid option is supplied (not listed in the table above), the return
        value of -1 will abort any subsequent fitting.
    orders : numpy.ndarray of int
        The desired order of the fit as a (1,) or (n_dimensions,) array.  If
        only a single value is supplied, it will be applied over all
        dimensions.  This serves as an upper limit for the check.  If the
        samples are distributed in a way that allows for a fit to be performed
        using `orders`, the return value will also be `orders`.
    coordinates : numpy.ndarray (n_dimensions, n_samples)
        The coordinates of the sample distribution.  Not used by the `counts`
        algorithm, but must still be supplied as an array with the correct
        dimensions.
    reference : numpy.ndarray (n_dimensions,)
        The coordinates of the point at which to perform a fit.  Only required
        by the 'bounded' algorithm, but a value of the correct array shape must
        still be supplied.
    mask : numpy.ndarray of bool (n_samples,), optional
        An optional mask where `False` values indicate the associated sample
        should not be included in determining the maximum order.
    minimum_points : int, optional
        The minimum number of points required to perform a fit of the desired
        order, optionally passed in for speed if pre-calculated.  Only used by
        the 'counts' algorithm.
    required : bool, optional
        If required is `False`, the maximum available order given the
        distribution will be returned (up to a maximum of `orders`).  If
        required is `True`, and the maximum available order is less than
        `orders`, the first element of the return value will be set to -1,
        indicating the criteria was not met.
    counts : int, optional
        This is required by the 'counts' algorithm.  If `counts` < 0, it will
        be determined by sum(mask).  If counts is less than zero, and a mask
        is not supplied, not fit will be performed.

    Returns
    -------
    maximum_orders : numpy.ndarray
       An array of shape (1,) or (n_dimensions,) based on whether a single
       `orders` was passed in for all dimensions, or each dimension has a
       separate order requirement.  If `required` was set to `True`, and the
       sample distribution did not allow for the requested order, the first
       element will be set to -1.  Otherwise, if `required` was `False`, the
       maximum order for each dimension will be returned.  If a single `orders`
       was to be applied over all dimensions, the return value will also be
       of size 1, but contains the min(maximum_order) over all dimensions.
    """
    if algorithm == 1:  # check enough points either side
        return check_orders_with_bounds(orders, coordinates, reference,
                                        mask=mask, required=required)

    elif algorithm == 2:  # allow extrapolation if fitting outside sample span
        return check_orders_without_bounds(orders, coordinates, mask=mask,
                                           required=required)

    elif algorithm == 3:  # check enough points overall
        return check_orders_with_counts(orders, counts, mask=mask,
                                        minimum_points=minimum_points,
                                        n_dimensions=coordinates.shape[0],
                                        required=required)

    else:
        # Must be checked or don't fit.
        return np.full(orders.size, -1, dtype=nb.i8)


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def check_orders_with_bounds(orders, coordinates, reference, mask=None,
                             required=False):  # pragma: no cover
    r"""
    Checks maximum order for sample coordinates bounding a reference.

    Given the `coordinates` of a sample distribution (:math:`X`), a `reference`
    position (:math:`X_{ref}`), and the desired `orders` of fit (:math:`o`),
    returns the maximum available order of fit.

    For dimension :math:`k` define the sets of unique values:

    .. math::

        s_k^- = \{ x \in X_k |\, x < X_{ref, k} \} \\
        s_k^+ = \{ x \in X_k |\, x > X_{ref, k} \}

    The maximum order is then given as:

    .. math::

        o_k^{max} = min\{ |s_k^-|, |s_k^+|, o_k \}

    where :math:`|.|` represents the cardinality (size) of the set.

    For example, consider a 1-dimensional set of coordinates:

    .. math::

        X = [1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 6]

    and we wish to perform an order=3 polynomial fit at :math:`X_{ref}=2.5`.
    There are 4 unique values of :math:`X > X_{ref}` (:math:`\{3, 4, 5, 6\}`),
    but only 2 unique values of :math:`X < X_{ref}` (:math:`\{1, 2\}`).  The
    return value will be 2, indicating that only a 2nd order polynomial fit
    should be attempted.  If a 1st order polynomial fit was requested, the
    return value would be 1 since there are enough points less than and greater
    than the reference.

    Parameters
    ----------
    orders : numpy.ndarray of int
        The desired order of the fit as a (1,) or (n_dimensions,) array.  If
        only a single value is supplied, it will be applied over all
        dimensions.  This serves as an upper limit for the check.  If the
        samples are distributed in a way that allows for a fit to be performed
        using `orders`, the return value will also be `orders`.
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        The coordinates of the sample distribution.
    reference : numpy.ndarray (n_dimensions,)
        The reference coordinate.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An optional mask where `False` values indicate the associated sample
        should not be included in determining the maximum order.
    required : bool, optional
        If required is `False`, the maximum available order given the
        distribution will be returned (up to a maximum of `orders`).  If
        required is `True`, and the maximum available order is less than
        `orders`, the first element of the return value will be set to -1,
        indicating the criteria was not met.

    Returns
    -------
    maximum_orders : numpy.ndarray
       An array of shape (1,) or (n_dimensions,) based on whether a single
       `orders` was passed in for all dimensions, or each dimension has a
       separate order requirement.  If `required` was set to `True`, and the
       sample distribution did not allow for the requested order, the first
       element will be set to -1.  Otherwise, if `required` was `False`, the
       maximum order for each dimension will be returned.  If a single `orders`
       was to be applied over all dimensions, the return value will also be
       of size 1, but contains the min(maximum_order) over all dimensions.
    """

    n_dimensions = coordinates.shape[0]
    n_orders = orders.size
    symmetric = n_orders == 1
    order_out = np.empty(n_orders, dtype=nb.i8)
    for i in range(n_orders):
        order_out[i] = orders[i]

    for k in range(n_dimensions):
        idx = 0 if symmetric else k
        o = order_out[idx]
        o_max = check_orders_with_bounds_1d(
            o, coordinates[k], reference[k], mask=mask, required=required)
        if o_max < 0:
            order_out[0] = -1
            break
        if o_max < o:
            order_out[idx] = o_max

    return order_out


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def check_orders_with_bounds_1d(order, coordinates, reference, mask=None,
                                required=False):  # pragma: no cover
    r"""
    Support function for `check_orders_with_bounds`.

    Please see :func:`check_orders_with_bounds` for a full description of the
    algorithm.  This function performs the necessary calculations across a
    single dimension.

    Parameters
    ----------
    order : int
        The desired order of fit.
    coordinates : numpy.ndarray (n_coordinates,)
        The coordinates for 1-dimension of the sample distribution.
    reference : int or float
        The reference coordinate.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An optional mask where `False` values indicate the associated sample
        should not be included in determining the maximum order.
    required : bool, optional
        If required is `True`, and the maximum order is less than `order`,
        returns -1.  Otherwise, returns the maximum order.

    Returns
    -------
    max_order : int
        The maximum order given the 1-D distribution.  Will be set to -1
        if less than `order` and `required` is `True`.
    """
    if order == 0:
        return 0
    left = 0
    right = 0
    left_found = False
    right_found = False
    have_mask = mask is not None
    unique_left = np.empty(order)
    unique_right = np.empty(order)

    for i in range(coordinates.size):
        if have_mask and not mask[i]:
            continue
        offset = coordinates[i] - reference
        if offset < 0:
            if left_found:
                continue
            elif left == 0:
                unique_left[0] = offset
                left = 1
            else:
                for j in range(left):
                    if unique_left[j] == offset:
                        break
                else:
                    unique_left[left] = offset
                    left += 1
            if left >= order:
                left_found = True

        elif offset > 0:
            if right_found:
                continue
            elif right == 0:
                unique_right[0] = offset
                right = 1
            else:
                for j in range(right):
                    if unique_right[j] == offset:
                        break
                else:
                    unique_right[right] = offset
                    right += 1
            if right >= order:
                right_found = True

        if left_found and right_found:
            return order
    else:
        if required:
            return -1
        elif left < right:
            return left
        else:
            return right


@njit(nogil=False, cache=True, fastmath=_fast_flags, parallel=False)
def check_orders_without_bounds(orders, coordinates,
                                mask=None, required=False):  # pragma: no cover
    r"""
    Checks maximum order based on unique samples, irrespective of reference.

    Given the `coordinates` of a sample distribution (:math:`X`), and the
    desired `orders` of fit (:math:`o`), returns the maximum available order of
    fit.  Unlike :func:`check_orders_with_bounds`, the location of the fit
    is unimportant.  All that is required is for enough unique sample
    coordinates to be available for fitting.

    For dimension :math:`k`, the maximum fit order is given as:

    .. math::

        o_k^{max} = min\{ |X_k| - 1, o_k \}

    where :math:`|.|` represents the cardinality (size) of the set.

    For example, consider a 1-dimensional set of coordinates:

    .. math::

        X = [1, 1, 1, 2, 3, 4]

    The maximum order of fit would be 3 since there are 4 unique values
    (1, 2, 3, 4).  Therefore, when `orders` >= 3, the return value would be 3.
    For `orders` < 3, `orders` would be returned.

    If we had a 2-dimensional set of data:

    .. math::

        X = [(1, 0), (1, 0), (1, 0), (2, 1), (3, 1), (4, 1)]

    The maximum order of fit would be 3 in the first dimension, and 1 in the
    second.

    Parameters
    ----------
    orders : numpy.ndarray of int
        The desired order of the fit as a (1,) or (n_dimensions,) array.  If
        only a single value is supplied, it will be applied over all
        dimensions.  This serves as an upper limit for the check.  If the
        samples are distributed in a way that allows for a fit to be performed
        using `orders`, the return value will also be `orders`.
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        The coordinates of the sample distribution.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An optional mask where `False` values indicate the associated sample
        should not be included in determining the maximum order.
    required : bool, optional
        If required is `False`, the maximum available order given the
        distribution will be returned (up to a maximum of `orders`).  If
        required is `True`, and the maximum available order is less than
        `orders`, the first element of the return value will be set to -1,
        indicating the criteria was not met.

    Returns
    -------
    maximum_orders : numpy.ndarray
       An array of shape (1,) or (n_dimensions,) based on whether a single
       `orders` was passed in for all dimensions, or each dimension has a
       separate order requirement.  If `required` was set to `True`, and the
       sample distribution did not allow for the requested order, the first
       element will be set to -1.  Otherwise, if `required` was `False`, the
       maximum order for each dimension will be returned.  If a single `orders`
       was to be applied over all dimensions, the return value will also be
       of size 1, but contains the min(maximum_order) over all dimensions.
    """

    n_dimensions = coordinates.shape[0]
    n_orders = orders.size
    symmetric = n_orders == 1
    order_out = np.empty(orders.size, dtype=nb.i8)
    for i in range(n_orders):
        order_out[i] = orders[i]

    for k in range(n_dimensions):
        idx = 0 if symmetric else k
        o = order_out[idx]
        maximum_order = check_orders_without_bounds_1d(
            o, coordinates[k], mask=mask, required=required)
        if maximum_order < 0:
            order_out[0] = -1
            break
        if maximum_order < o:
            order_out[idx] = maximum_order

    return order_out


@njit(nogil=False, cache=True, fastmath=_fast_flags, parallel=False)
def check_orders_without_bounds_1d(order, coordinates, mask=None,
                                   required=False):  # pragma: no cover
    r"""
    Support function for `check_orders_without_bounds`.

    Please see :func:`check_orders_without_bounds` for a full description of
    the algorithm.  This function performs the necessary calculations across a
    single dimension.

    Parameters
    ----------
    order : int
        The desired order of fit.
    coordinates : numpy.ndarray (n_coordinates,)
        The coordinates for 1-dimension of the sample distribution.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An optional mask where `False` values indicate the associated sample
        should not be included in determining the maximum order.
    required : bool, optional
        If required is `True`, and the maximum order is less than `order`,
        returns -1.  Otherwise, returns the maximum order.

    Returns
    -------
    max_order : int
        The maximum order given the 1-D distribution.  Will be set to -1
        if less than `order` and `required` is `True`.
    """

    if order == 0:
        return 0
    max_order = -1
    unique_values = np.empty(order + 1, dtype=nb.float64)
    have_mask = mask is not None

    for i in range(coordinates.size):
        if have_mask and not mask[i]:
            continue
        value = coordinates[i]
        for j in range(max_order + 1):
            if unique_values[j] == value:
                break
        else:
            max_order += 1
            unique_values[max_order] = value
        if max_order >= order:
            return max_order
    else:
        if required:
            return -1
        else:
            return max_order


@njit(nogil=False, cache=True, fastmath=_fast_flags, parallel=False)
def check_orders_with_counts(orders, counts, mask=None, minimum_points=None,
                             n_dimensions=None, required=False
                             ):  # pragma: no cover
    r"""
    Checks maximum order based only on the number of samples.

    For :math:`N` samples of :math:`K` dimensional data, the minimum number of
    samples required to perform a polynomial fit with `orders` :math:`o` is:

    .. math::

        N_{min} = \prod_{k=1}^{K}{(o_k + 1)}

    if :math:`o_k = o_0, \, \forall k`, then it is possible to suggest a lower
    order over all dimensions in the case where :math:`N < N_{min}`.  This
    is given as:

    .. math::

        o_k^{max} = min\{ floor(N ^ {1 / K} - 1), o_k \}

    The suggested maximum order is returned by setting the `required` keyword
    to `False`.  If the orders vary between dimensions or `required` is `True`,
    the value of :math:`o_0` is set to -1 indicating a polynomial fit of the
    desired order is not possible.

    Parameters
    ----------
    orders : numpy.ndarray of int
        The desired order of the fit as a (1,) or (n_dimensions,) array.  If
        only a single value is supplied, it will be applied over all
        dimensions.  This serves as an upper limit for the check.  If the
        samples are distributed in a way that allows for a fit to be performed
        using `orders`, the return value will also be `orders`.
    counts : int
        The number of samples available for the fit.
    mask : numpy.ndarray of bool (n_coordinates,), optional
        An optional mask where `False` values indicate the associated sample
        should not be included in determining the maximum order.  The `counts`
        parameter will be ignored in favor of sum(mask).
    minimum_points : int, optional
        The minimum number of points required to perform a fit of the desired
        order, optionally passed in for speed.
    n_dimensions : int, optional
        If `orders` was supplied as an array of size 1, but should be applied
        over multiple dimensions, the number of dimensions should be supplied.
        Otherwise, the number of dimensions is taken to be equal to the number
        of orders.
    required : bool, optional
        If required is `False`, the maximum available order given the
        distribution will be returned (up to a maximum of `orders`).  If
        required is `True`, and the maximum available order is less than
        `orders`, the first element of the return value will be set to -1,
        indicating the criteria was not met.

    Returns
    -------
    maximum_orders : numpy.ndarray
       An array of shape (1,) or (n_dimensions,) based on whether a single
       `orders` was passed in for all dimensions, or each dimension has a
       separate order requirement.  If `required` was set to `True`, and the
       sample distribution did not allow for the requested order, the first
       element will be set to -1.  Unlike :func:`check_orders_with_bounds` or
       :func:`check_orders_without_bounds`, a suggested maximum order can only
       be returned by setting `required` to `False` if `orders` are equal
       in all dimensions.  Otherwise, it is impossible to know which dimension
       the order should be reduced for.  In this case, the first element of
       `maximum_orders` will be set to -1.
    """

    n_orders = orders.size
    if n_dimensions is None:
        n_dimensions = n_orders  # assume same number of dimensions

    symmetric = True
    first_order = orders[0]
    order_out = np.empty(n_orders, dtype=nb.i8)
    for i in range(n_orders):
        order_out[i] = orders[i]
        if symmetric and orders[i] != first_order:
            symmetric = False

    if minimum_points is None:
        minimum_points = 1
        for i in range(n_dimensions):
            order = orders[0] if symmetric else orders[i]
            minimum_points *= order + 1

    if counts < 0:
        counts = 0
        if mask is not None:
            for k in range(mask.size):
                counts += mask[k]
                if counts == minimum_points:  # only need to go up to here
                    break

    if counts >= minimum_points:
        return order_out
    elif required or not symmetric:
        # If the order is not symmetric, cannot recommend a new order since
        # we do not know which dimension to reduce.
        order_out[0] = -1
        return order_out
    else:
        limit = counts ** (1.0 / n_dimensions) - 1
        if limit < 0:
            order_out[0] = -1
        else:
            order_out[:] = nb.i8(limit)

    return order_out


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def apply_mask_to_set_arrays(mask, data, phi, error, weights,
                             counts=None):  # pragma: no cover
    """
    Set certain arrays to a fixed size based on a mask array.

    Parameters
    ----------
    mask : numpy.ndarray of bool (N,)
        Mask where `True` values indicate the associated element should be
        kept, and `False` will result in exclusion from the output arrays.
    data : numpy.ndarray (N,)
        The data array.
    phi : numpy.ndarray (n_terms, N)
        The polynomial terms of the fit equation.
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
    data_out, phi_out, error_out, weight_out : 4-tuple of numpy.ndarray.
       Resized arrays in which the last axis is of size `counts`.
    """

    if counts is None:
        counts = 0
        for i in range(mask.size):
            counts += mask[i]

    n_terms, n_data = phi.shape
    phi_out = np.empty((n_terms, int(counts)), dtype=nb.float64)
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
            for j in range(n_terms):
                phi_out[j, counter] = phi[j, i]
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

    return data_out, phi_out, error_out, weight_out


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def no_fit_solution(set_index, point_index,
                    fit_out, error_out, counts_out, weights_out,
                    distance_weights_out, rchi2_out, offset_variance_out,
                    get_error=True, get_counts=True,
                    get_weights=True, get_distance_weights=True,
                    get_rchi2=True, get_offset_variance=True,
                    cval=np.nan):  # pragma: no cover
    r"""
    Fill output arrays with set values on fit failure.

    On fit failure, the output arrays are filled with certain values indicating
    that a fit is not possible.  Count, and weight arrays contain zeros; the
    error, reduced chi-squared and offset variance are set to NaN; finally,
    the data array is set to `cval`, a user set float value.

    Parameters
    ----------
    set_index : int
        An integer representing the data set for which a fit cannot be
        performed.
    point_index : int
        An integer representing the index of the fit coordinate at which the
        fit cannot be performed.
    fit_out : numpy.ndarray (n_sets, n_coordinates)
        The output fit values.
    error_out : numpy.ndarray (n_sets, n_coordinates)
        The output error values on the fit.
    counts_out : numpy.ndarray (n_sets, n_coordinates)
        The number of samples used to create the fit.
    weights_out : numpy.ndarray (n_sets, n_coordinates)
        The sum of full weights applied to samples in the fit.
    distance_weights_out : numpy.ndarray (n_sets, n_coordinates)
        The sum of only the distance weights applied to samples in the fit.
    rchi2_out : numpy.ndarray (n_sets, n_coordinates)
        The reduced chi-squared statistic of the fit.
    offset_variance_out : numpy.ndarray (n_sets, n_coordinates)
        The variance as derived from the offset of the fit coordinate from the
        sample distribution.
    get_error : bool, optional
        If `False` do not update `error_out`.
    get_counts : bool, optional
        If 'False' do not update `counts_out`.
    get_weights : bool, optional
        If `False` do not update `weights_out`.
    get_distance_weights : bool, optional
        If `False` do not update `distance_weights_out`.
    get_rchi2 : bool, optional
        If `False` do not update `rchi2_out`.
    get_offset_variance : bool, optional
        If `False` do not update `offset_variance_out`.
    cval : float, optional
        The fill value for `data_out` on fit failure.

    Returns
    -------
    None
        All arrays are updated in-place.
    """

    fit_out[set_index, point_index] = cval
    if get_error:
        error_out[set_index, point_index] = np.nan
    if get_counts:
        counts_out[set_index, point_index] = 0
    if get_weights:
        weights_out[set_index, point_index] = 0.0
    if get_distance_weights:
        distance_weights_out[set_index, point_index] = 0.0
    if get_rchi2:
        rchi2_out[set_index, point_index] = np.nan
    if get_offset_variance:
        offset_variance_out[set_index, point_index] = np.nan


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_polynomial_fit(phi_samples, phi_point,
                         data, error, distance_weight, weight,
                         derivative_term_map=None,
                         calculate_variance=True,
                         calculate_rchi2=True,
                         calculate_derivative_mscp=True,
                         error_weighting=True,
                         estimate_covariance=False
                         ):  # pragma: no cover
    r"""
    Derive a polynomial fit from samples, then calculate fit at single point.

    The fit to the sample distribution is given as

    .. math::

        f(\Phi) = \hat{c} \cdot \Phi

    where :math:`\Phi` contains products of the sample coordinates for each
    coefficient term.  The coefficients :math:`\hat{c}` are solved for using
    least-squares fitting and then applied to calculate the fitted value at a
    single point :math:`f(\Phi_{fit})`.

    It is also possible to return an on the fit as a variance.  If a valid
    error is supplied, it will be propagated.  If no valid errors are
    available, they will be calculated from residuals on the fit.

    The reduced chi-squared (:math:`\chi^2`) statistic may also be calculated,
    but is only really meaningful if valid errors were supplied.  Otherwise,
    :math:`\chi^2 \equiv 1`.

    Finally, the covariance of gradients between dimensions may also be
    returned.  Note that these are the weighted mean of all sample gradients.

    Parameters
    ----------
    phi_samples : numpy.ndarray (n_terms, n_samples)
        The array of independent terms for each sample.
    phi_point : numpy.ndarray (n_terms,)
        The array containing the independent terms at the fitting point.
    data : numpy.ndarray (n_samples,)
        The array of sample values.
    error : numpy.ndarray (n_samples,)
        The array of error values for each sample.  Note that if errors are
        unavailable, an array of size 0 may be supplied.  If this is the case,
        and an error on the fit is required, it will be derived from the
        residuals of the fit from the data.  In addition, the reduced
        chi-squared statistic will always be 1.0 if derived from residuals.
    distance_weight : numpy.ndarray (n_samples,)
        The distance weighting factor (not including any error weighting)
        applied to each sample in the fit.
    weight : numpy.ndarray (n_samples,)
        The full weighting factor applied to each sample in the fit.
    derivative_term_map : numpy.ndarray, optional
        A mapping array for the determination of derivatives from the
        coefficients of the fit, and available terms in "phi".  The shape of
        the array is (n_dimensions, 3, n_derivative_terms).  This is only
        required if the gradient is required as an output.  For a full
        description of the derivative map, please see
        :func:`polynomial_derivative_map`.
    calculate_variance : bool, optional
        If `True`, calculate the variance on the fit.  The variance will be
        calculated irrespectively if a valid error was supplied, and the
        reduced chi-squared statistic is required as a return value.
    calculate_rchi2 : bool, optional
        If `True`, calculate the reduced chi-squared statistic of the fit.
        Note that valid errors must be supplied for this to be meaningful.
    calculate_derivative_mscp : bool, optional
        If `True`, calculate the covariance of the derivatives at the fit
        point.
    error_weighting : bool, optional
        If `True`, indicates that `weights` includes an error weighting
        factor of 1/sigma^2.  This allows for a slight speed increase when
        performing the fit as some mathematical terms will not need to be
        recalculated.
    estimate_covariance : bool, optional
        If `True`, uses :func:`estimated_covariance_matrix_inverse` instead
        of :func:`covariance_matrix_inverse` when determining the variance.
        This is suggested if the errors are not well-behaved.

    Returns
    -------
    fit_value, variance, rchi2, gradients : float, float, float, numpy.ndarray
        The value of the fit at the fit point.  The variance and reduced
        chi-squared will only be calculated if `calculate_variance` and
        `calculate_rchi2` are respectively set to `True`.  The `gradients`
        matrix is an (n_dimensions, n_dimensions) array where
        gradients[i, j] = dx_i * dx_j.
    """
    amat, beta = solve_amat_beta(phi_samples, data, weight)
    rank, coefficients = solve_coefficients(amat, beta)
    fit_value = fit_phi_value(phi_point, coefficients)

    error_valid = error.size != 0
    error_weighted_amat = amat if error_weighting else np.empty((0, 0))
    variance_required = calculate_variance or (error_valid and calculate_rchi2)
    e_inv_required = error_valid and (calculate_variance or calculate_rchi2)

    if error_valid:
        r_inv_required = calculate_rchi2
    else:
        r_inv_required = calculate_variance

    if r_inv_required:
        residuals = fit_residual(data, phi_samples, coefficients)
    else:
        residuals = data  # dummy allocation for Numba

    e_inv, r_inv = solve_inverse_covariance_matrices(
        phi_samples, error, residuals, distance_weight,
        error_weighted_amat=error_weighted_amat,
        rank=rank,
        calculate_error=e_inv_required,
        calculate_residual=r_inv_required,
        estimate_covariance=estimate_covariance)

    if variance_required:
        if error_valid:
            # Error propagation
            variance = fit_phi_variance(phi_point, e_inv)
        else:
            # Error estimate based on residuals
            variance = fit_phi_variance(phi_point, r_inv)
    else:
        variance = 0.0

    if calculate_derivative_mscp:
        if derivative_term_map is None:
            derivative_map = np.empty((0, 0, 0), dtype=nb.i8)
        else:
            derivative_map = np.asarray(derivative_term_map)
        gradient_mscp = derivative_mscp(
            coefficients, phi_samples, derivative_map, weight)
    else:
        gradient_mscp = r_inv  # dummy allocation for Numba

    if calculate_rchi2:
        if error_valid:
            rchi2 = fit_phi_variance(phi_point, r_inv) / variance
        else:
            rchi2 = 1.0  # since error was derived from residuals
    else:
        rchi2 = 0.0

    return fit_value, variance, rchi2, gradient_mscp


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def multivariate_gaussian(covariance, coordinates, center=None,
                          normalize=False):  # pragma: no cover
    r"""
    Return values of a multivariate Gaussian in K-dimensional coordinates.

    The density of a multivariate Gaussian is given as:

    .. math::

        f_{\mathbf X}(x_1, \ldots, x_k) = \frac
            {\exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1}(x - \mu) \right)}
            {\sqrt{(2 \pi)^K |\Sigma|}}

    where the `coordinates` :math:`X` are real :math:`K` dimensional vectors,
    :math:`\Sigma` is the covariance matrix with determinant :math:`|\Sigma|`,
    and the `center` of the distribution is given by :math:`\mu`.

    Note that by default, the factor :math:`{\sqrt{(2 \pi)^K |\Sigma|}}` is not
    applied unless `normalize` is `True`.

    Parameters
    ----------
    covariance : numpy.ndarray (n_dimensions, n_dimensions)
        The covariance matrix (:math:`\Sigma`).  Should be symmetric and
        positive definite.
    coordinates : numpy.ndarray (n_dimensions, n_coordinates)
        The coordinates at which to evaluate the multivariate Gaussian.
    center : numpy.ndarray (n_dimensions,), optional
        The center of the distribution.  If not supplied, the center is
        assumed to be zero in all dimensions.
    normalize : bool, optional
        If `True`, normalize by dividing by :math:`\sqrt{(2 \pi)^k |\Sigma|}`.

    Returns
    -------
    density : numpy.ndarray (n_coordinates,)
        The density evaluated at each coordinate.
    """
    n_dimensions, n_coordinates = coordinates.shape

    if center is None:
        cx = np.zeros(n_dimensions, dtype=nb.float64)
    else:
        cx = np.asarray(center, dtype=nb.float64)

    if n_dimensions == 0 or n_coordinates == 0:
        return np.full(n_coordinates, np.nan, dtype=nb.float64)

    covariance_inv = np.linalg.pinv(covariance)

    if normalize:
        norm = np.sqrt(
            ((2 * np.pi) ** n_dimensions) * np.linalg.det(covariance))
        if norm == 0:
            return np.full(n_coordinates, np.nan, dtype=nb.float64)
    else:
        norm = 1.0

    result = np.zeros(n_coordinates, dtype=nb.float64)
    for i in range(n_dimensions):
        xi = coordinates[i] - cx[i]
        for j in range(i, n_dimensions):
            if i == j:
                xj = xi
            else:
                xj = coordinates[j] - cx[j]
            cij = covariance_inv[i, j]
            for k in range(n_coordinates):
                value = xi[k] * cij * xj[k]
                if i == j:
                    value /= 2
                result[k] += value

    for k in range(n_coordinates):
        result[k] = math.exp(-result[k])
        if normalize:
            result[k] /= norm

    return result


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def scaled_adaptive_weight_matrix(sigma, rchi2, fixed=None
                                  ):  # pragma: no cover
    r"""
    Scales a Gaussian weighting kernel based on a prior fit.

    In the standard resampling algorithm, a polynomial fit may weight each
    sample (:math:`x`) according to its distance from the reference position at
    which the fit is derived (:math:`x_{ref}`) such that samples closer to the
    reference position have more influence on the fit than those that are
    farther.  The weighting function used is:

    .. math::

        w(x) = exp \left(
           -\sum_{k=1}^{K}{\frac{(x_{ref, k} - x_k)^2}{2 \sigma_k^2}}
                   \right)

    in :math:`K` dimensions where :math:`\sigma` (supplied to this function
    via `sigma`) is a scaling factor, equivalent to the standard deviation of
    a normal distribution.  Following a fit, it is also possible to generate
    a reduced chi-squared statistic (:math:`\chi_r^2`) which measures the
    "goodness" of fit.

    With this information we can rescale `sigma` in an attempt to get
    :math:`\chi_r^2 \rightarrow 1` i.e., get a good fit within noise
    limitations.  This function assumes that if :math:`\chi_r^2 < 1`, the
    samples have been over-fit, and therefore, the weighting function should be
    "widened" to allow more distant samples to have a stronger influence on the
    fit and subsequent :math:`\chi_r^2` calculation.  Likewise, if
    :math:`\chi_r^2 > 1`, this implies that the weighting function should be
    truncated so that the fit focuses more strongly on providing a good fit
    to nearby samples.  i.e., there is likely structure away from the fit
    location that cannot be modelled well by a polynomial of the given order.

    To accomplish this, the weighting kernel is rescaled such that:

    .. math::

        \chi_r^2 \prod_{k=1}^{K}{\sigma_{scaled, k}^2} =
            \prod_{k=1}^{K}{\sigma_k^2}

    The reason is that for a multivariate Gaussian:

    .. math::

        \int_{R^K} exp \left(
            -\frac{1}{2} (x - x_{ref})^T \Sigma^{-1} (x - x_{ref})
        \right)
        = (2 \pi)^{K/2} |\Sigma|^{1/2}

    where :math:`\sigma^2 = diag(\Sigma)`:

    .. math::

        |\Sigma| \propto \prod_{k=1}^{K}{\sigma_k^2} \propto \chi_r

    Note that in this specific implementation, the shape of the weighting
    kernel remains unchanged and only the overall size is allowed to vary.
    Therefore, a single scaling factor (:math:`\beta`) is applied over all
    dimensions such that:

    .. math::

        \sigma_{scaled, k} = \frac{\sigma_k}{\sqrt{\beta}}

    where

    .. math::

        \beta = \chi_r^{1 / K}

    To reduce subsequent calculations, a scaled :math:`\alpha` value is passed
    out instead of :math:`\sigma_{scaled}` where:

    .. math::

       \alpha = 2 \sigma^2

    Therefore, the final output value will be:

    .. math::

        \alpha_{scaled, k}^{-1} = \frac{\beta}{2 \sigma_k^2}

    Finally, scaling does not need to occur across all dimensions, and it is
    possible to fix the shape of the kernel in one or more dimensions by using
    the `fixed` parameter.  If this is the case:

    .. math::

        \beta = \chi_r^{\frac{1}{K - K_{fixed}}}

    where :math:`K_{fixed}` is the number dimensions in which scaling has been
    disabled.

    Parameters
    ----------
    sigma : numpy.ndarray (n_dimensions,)
        The standard deviations of the Gaussian for each dimensional component
        used for distance weighting of each sample in the initial fit.
    rchi2 : float
        The reduced chi-squared statistic of the fit.
    fixed : numpy.ndarray of bool (n_dimensions,), optional
        If supplied, `True` values indicate that the width of the Gaussian
        along the corresponding axis should not be altered in the output
        result.

    Returns
    -------
    inverse_alpha : numpy.ndarray (n_dimensions,)
        The scaled `sigma` values converted to the inverse `alpha` array,
        required by :func:`calculate_adaptive_distance_weights_scaled` to
        create a set of weighting factors.
    """

    n_dimensions = sigma.size
    scaled_inverse_alpha = np.empty(n_dimensions, dtype=nb.float64)

    if not np.isfinite(rchi2):
        scaled_inverse_alpha[0] = np.nan  # indicate failure and propagate
        return scaled_inverse_alpha

    for i in range(n_dimensions):
        scaled_inverse_alpha[i] = 0.5 / sigma[i] / sigma[i]

    if rchi2 <= 0:
        return scaled_inverse_alpha  # no change
    rchi = np.sqrt(rchi2)

    if fixed is None:
        scaling_factor = rchi ** (1 / n_dimensions)
        for i in range(n_dimensions):
            scaled_inverse_alpha[i] *= scaling_factor
        return scaled_inverse_alpha

    fix = np.asarray(fixed, dtype=nb.b1)  # for Numba hand-holding
    n_adapt = n_dimensions
    for i in range(n_dimensions):
        if fix[i]:
            n_adapt -= 1

    if n_adapt == 0:
        return scaled_inverse_alpha  # no change

    scaling_factor = rchi ** (1.0 / n_adapt)
    for i in range(n_dimensions):
        if not fix[i]:
            scaled_inverse_alpha[i] *= scaling_factor

    return scaled_inverse_alpha


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def scaled_adaptive_weight_matrices(sigma, rchi2_values, fixed=None
                                    ):  # pragma: no cover
    r"""
    Wrapper for `scaled_adaptive_weight_matrix` over multiple values.

    Please see :func:`scaled_adaptive_weight_matrix` for details on how the
    weighting kernel is modified using a single scaling factor.  This function
    performs the calculation for multiple scaling factors (:math:`\chi_r^2`).

    Parameters
    ----------
    sigma : numpy.ndarray (n_dimensions,)
        The standard deviations of the Gaussian for each dimensional component
        used for the distance weighting of each sample in the initial fit.
    rchi2_values : numpy.ndarray (n_data_sets, fit_shape)
        The reduced chi-squared statistics of the fit for each data set.  Here,
        `fit_shape` is an arbitrary array shape which depends upon the shape of
        the output fit coordinates defined by the user.
    fixed : numpy.ndarray of bool (n_dimensions,), optional
        If supplied, `True` values indicate that the width of the Gaussian
        along the corresponding axis should not be altered in the output
        result.

    Returns
    -------
    scaled_matrices : numpy.ndarray
        The scaled weighting kernel with shape
        (n_data_sets, fit_shape, 1, n_dimensions) where `fit_shape` is
        determined by the shape of the output fit coordinates supplied by the
        user, and `n_data_sets` is the number of data sets to be fit.  The
        third axis (of size 1), is a dummy dimension required for Numba to
        compile successfully.  The last dimension contains the new scaled
        inverse :math:`\alpha_{scaled,k}^{-1}` values as described in
        :func:`scaled_adaptive_weight_matrix`.
    """

    n = rchi2_values.size
    shape = rchi2_values.shape + (1, sigma.size)
    flat_shape = (n, 1, sigma.size)
    scaled_matrices = np.empty(shape).reshape(flat_shape)
    flat_rchi2 = rchi2_values.ravel()

    for i in range(n):
        scaled_matrices[i, 0] = scaled_adaptive_weight_matrix(
            sigma, flat_rchi2[i], fixed=fixed)

    return scaled_matrices.reshape(shape)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def shaped_adaptive_weight_matrix(sigma, rchi2, gradient_mscp,
                                  density=1.0,
                                  variance_offset=0.0,
                                  fixed=None,
                                  tolerance=None):  # pragma: no cover
    r"""
    Shape and scale the weighting kernel based on a prior fit.

    In the standard resampling algorithm, a polynomial fit may weight each
    sample coordinate (:math:`x`) according to its distance from the reference
    position at which the fit is derived (:math:`x_{ref}`) such that samples
    closer to the reference position have more influence on the fit than those
    that are farther.  The weighting function used is:

    .. math::

        W = exp(-\Delta x^T A^{-1} \Delta x)

    where :math:`{\Delta x}_k = x_{ref, k} - x_k` for dimension :math:`k`, and
    :math:`A` is a symmetric positive definite matrix defining the "shape" of
    the weighting kernel.  This effectively defines a multivariate Gaussian
    centered on the reference point where overall size, rotation, and stretch
    of each principle axis may be altered.  The goal of shaping the weighting
    kernel :math:`A`, is to produce a fit where the reduced chi-squared
    statistic of the fit equal to one (:math:`\chi_r^2 = 1`).  To derive the
    shape :math:`A`, an initial fit must have first been performed using a
    square diagonal matrix :math:`A_0`, whose diagonal elements are the square
    of the `sigma` parameter (:math:`diag(A_0) = 2 \sigma^2`).  It is then easy
    to define :math:`A_0^{-1}` in :math:`K` dimensions as:

    .. math::

        A_0^{-1} = \frac{1}{2}
            \begin{bmatrix}
                \frac{1}{\sigma_0^2} & 0 & \dots & 0 \\
                0 & \frac{1}{\sigma_1^2} & \ddots & \vdots \\
                \vdots & \ddots & \ddots & 0 \\
                0 & \dots & 0 & \frac{1}{\sigma_K^2}
            \end{bmatrix}

    Following (or during) the initial fit with :math:`A_0`, the mean
    square cross products of the derivatives evaluated at the sample
    coordinates should be calculated using :func:`derivative_mscp`, where the
    returned matrix has values:

    .. math::

        \bar{g}_{ij}^2 = \frac{\partial \bar{f}}{\partial X_i}
                         \frac{\partial \bar{f}}{\partial X_j}

    for dimensions :math:`X_i` and :math:`X_j` in K-dimensions, where
    :math:`\partial \bar{f} / \partial X_i` is the weighted mean of the partial
    derivatives over all samples in the fit with respect to :math:`X_i`, with
    weighting defined by :math:`W` (above) using :math:`A_0^{-1}`.

    The matrix :math:`\bar{g}^2` is only used to define the shape of new
    weighting kernel, not the overall size.  Therefore, it is normalized such
    that :math:`|\bar{g}^2| = 1`.

    We can then use singular value decomposition to factorize :math:`\bar{g}^2`
    into:

    .. math::

        \bar{g}^2 = U S V^T

    Here, the matrices :math:`U` and :math:`V^T` represent rotations (since
    :math:`|\bar{g}^2| > 0`), and the singular values (:math:`S`) can be
    thought of as the magnitudes of the semi-axes of an ellipsoid in
    K-dimensional space.  Naively, this provides us with a basis from which to
    determine the final "shape" matrix where:

    .. math::

        A^{-1} = \beta\, \bar{g}^2

    and :math:`\beta` is an as yet undetermined scaling factor representing
    the overall size of the new kernel.  The resulting weighting kernel has the
    shape of an ellipsoid with the smallest semi-axis oriented parallel to
    the gradient.  In other words, the new weighting will result in a fit that
    is less sensitive to distant samples in directions where the gradient is
    high, and more sensitive to distant samples in directions where the
    gradient is low.

    However, we must still keep in mind that our overall goal is to get
    :math:`\chi_r^2 \rightarrow 1` when a fit is performed using the new
    kernel :math:`A`.  For example, if :math:`\chi_r=1` in the initial fit,
    there is no need to modify the kernel, and if :math:`\chi_r < 1`, then we
    do not want to get an even better fit.

    Another factor to consider is that we cannot be completely confident in
    this new shape due to the distribution of samples.  If we are fitting at a
    point that is away from the center of the sample distribution, it is
    unadvisable to use a highly shaped kernel due to increased uncertainty in
    the mean partial derivatives.  Furthermore, even if we are fitting close to
    the center of the distribution, that does not mean that the derivative
    calculations were not skewed by a few nearby samples when fitting in a
    local depression of the sample density (for example, near the center of a
    donut-like distribution).

    We model our confidence (:math:`\gamma`) in the "shape" using a logistic
    function (see :func:`stretch_correction`) that factors in :math:`\chi_r^2`,
    a measure of the sample density profile (:math:`\rho`) at the fit
    coordinate, and from the deviation of the fit coordinate from the center of
    the sample distribution (:math:`\sigma_d`):

    .. math::

        \gamma = \frac{2}
                 {\left(
                     {1 + (2^{exp(\sigma)} - 1)e^{\rho (1 - \chi_r^2)}}
                 \right)^{1/exp(\sigma)}} - 1

    The density :math:`\rho` is calculated using :func:`relative_density` which
    sets :math:`\rho=1` when the samples are uniformly distributed,
    :math:`\rho > 1` when the distribution is concentrated on the fit
    coordinate, and :math:`0 < \rho < 1` when the fitting in a local depression
    of the distribution density.  The deviation (:math:`\sigma_d`) is
    calculated using :func:`offset_variance`.

    The confidence parameter has asymptotes at :math:`\gamma = \pm 1`, is
    equal to zero at :math:`\chi_r^2=1`, is positive for :math:`\chi_r^2 > 1`,
    and is negative for :math:`\chi_r^2 < 1`.  Also, the magnitude increases
    with :math:`\rho`, and decreases with :math:`\sigma_d`.  The final shape
    matrix is then defined as:

    .. math::

        A^{-1} = \beta\, U S^{\gamma} V^T

    Note that as :math:`\gamma \rightarrow 0`, the kernel approaches that of
    a spheroid.  For :math:`\chi_r^2 > 1`, the kernel approaches
    :math:`\bar{g}^2`.  When :math:`\chi_r^2 < 1`, the kernel effectively
    rotates so that the fit becomes increasingly sensitive to samples along
    the direction of the derivative.  The overall size of the kernel remains
    constant since :math:`|S^{\gamma}| = |\bar{g}^2| = 1`.

    Finally, the only remaining factor to calculate is the scaling factor
    :math:`\beta` which is given as:

    .. math::

        \beta = \left( \frac{\chi_r}{|A_0|} \right)^{1/K}

    Note that this scaling has the effect of setting

    .. math::

        \frac{|A_0|}{|A|} = \chi_r

    The user has the option of fixing the kernel in certain dimensions such
    that :math:`{A_0}_{k, k} = A_{k, k}`.  If this is the case, for any fixed
    dimension :math:`k` we set:

    .. math::

        {U S^{\gamma} V^T}_{k, k} = 1

        {U S^{\gamma} V^T}_{k, i \neq k}^2 = 0

        {U S^{\gamma} V^T}_{i \neq k, k}^2 = 0

    meaning the scaling is unaltered for dimensions :math:`k`, and no rotation
    will be applied to any other dimension with respect to :math:`k`.  Since
    the overall size must be controlled through fewer dimensions,
    :math:`\beta` must take the form of a diagonal matrix:

    .. math::

         diag(\beta)_{i \in fixed} = \frac{1}{2 \sigma_i^2}
         ,\,\,
         diag(\beta)_{i \notin fixed} =
             \left(
             \frac{\chi_r |A_0|}{|{U S^{\gamma} V^T}|}
             \prod_{i \in fixed}{2 \sigma_i^2}
             \right)^{1 / (K - K_{fixed})}

    Once again :math:`A^{-1} = \beta\, U S^{\gamma} V^T`.

    Parameters
    ----------
    sigma : numpy.ndarray (n_dimensions,)
        The standard deviations of the Gaussian for each dimensional component
        used for the distance weighting of each sample in the initial fit.
    rchi2 : float
        The reduced chi-squared statistic of the fit.
    gradient_mscp : numpy.ndarray (n_dimensions, n_dimensions)
        An array where gradient_mscp[i, j] = derivative[i] * derivative[j] in
        dimensions i and j.  Please see :func:`derivative_mscp` for further
        information.  Must be Hermitian and real-valued (symmetric).
    density : float, optional
        The local relative density of the samples around the fit coordinate.  A
        value of 1 represents uniform distribution.  Values greater than 1
        indicate clustering around the fitting point, and values less than 1
        indicate that samples are sparsely distributed around the fitting
        point.  Please see :func:`relative_density` for further information.
    variance_offset : float, optional
        The variance at the fit coordinate determined from the sample
        coordinate distribution. i.e., if a fit is performed at the center of
        the sample distribution, the variance is zero.  If done at 2-sigma from
        the sample distribution center, the variance is 4.
    fixed : numpy.ndarray of bool (n_dimensions,), optional
        If supplied, `True` values indicate that the width of the Gaussian
        along the corresponding axis should not be altered in the output
        result.
    tolerance : float, optional
        The threshold below which SVD values are considered zero when
        determining the matrix rank of `derivative_mscp`.  Please
        see :func:`numpy.linalg.matrix_rank` for further information.

    Returns
    -------
    shape_matrix : numpy.ndarray (n_dimensions, n_dimensions)
        A matrix defining the shape of the weighting kernel required by
        :func:`calculate_adaptive_distance_weights_shaped` to create a set of
        weighting factors.
    """

    n_dimensions = gradient_mscp.shape[0]
    if not np.isfinite(rchi2):
        return np.full((n_dimensions, n_dimensions), np.nan, dtype=nb.float64)

    n_adapt = n_dimensions
    if fixed is None:
        fast = True
        fix = np.empty(0, dtype=nb.b1)  # for Numba
    else:
        fix = np.asarray(fixed, dtype=nb.b1)
        fast = not np.any(fix)
        if not fast:
            for i in range(n_dimensions):
                n_adapt -= fix[i]

    if rchi2 <= 0 or n_adapt == 0:
        # In case of a bad value, exact fit, or nothing to be done.
        # Just return the same input sigma values (as an inverse alpha).
        alpha = 2 * sigma ** 2
        shape_matrix = np.empty((n_dimensions, n_dimensions), dtype=nb.float64)
        for i in range(n_dimensions):
            for j in range(i, n_dimensions):
                if i == j:
                    shape_matrix[i, j] = 1.0 / alpha[i]
                else:
                    shape_matrix[i, j] = 0.0
                    shape_matrix[j, i] = 0.0

        return shape_matrix

    rchi = np.sqrt(rchi2)

    # Define the shape of the matrix here (stretch and rotation).  It is
    # normalized since we are only interested in the shape at this point, not
    # the size.  If a shape cannot be determined, the solution is equal
    # to the scaled solution on a spheroid.
    if np.all(np.isfinite(gradient_mscp)):
        if np.linalg.matrix_rank(gradient_mscp,
                                 tol=tolerance) == n_dimensions:
            norm_factor = np.linalg.det(gradient_mscp)
        else:
            norm_factor = 0.0

        if norm_factor != 0 and np.isfinite(norm_factor):

            shape_matrix = gradient_mscp / (norm_factor ** (1 / n_dimensions))

            if np.all(np.isfinite(shape_matrix)):
                u, s, vh = np.linalg.svd(shape_matrix)

                # u and vh define the rotation, while s defines the stretch.
                # The stretch is reduced for low density/rchi2 and/or high
                # offset variance.

                correction = stretch_correction(
                    rchi, density, variance_offset)

                s **= correction
                shape_matrix = u @ np.diag(s) @ vh
            else:
                shape_matrix = np.eye(n_dimensions, dtype=nb.float64)
        else:
            shape_matrix = np.eye(n_dimensions, dtype=nb.float64)
    else:
        shape_matrix = np.eye(n_dimensions, dtype=nb.float64)

    # Now rescale according to the reduced chi-squared statistic in an attempt
    # to get rchi2 = 1.

    inverse_alpha = 0.5 / sigma ** 2
    if fast:
        determinant = 1.0
        for i in range(n_dimensions):
            determinant *= inverse_alpha[i]
        scale_factor = (rchi * determinant) ** (1.0 / n_dimensions)
        return shape_matrix * scale_factor

    initial_determinant = 1.0
    fixed_alpha_product = 1.0
    n_adapt = n_dimensions
    for i in range(n_dimensions):
        initial_determinant *= inverse_alpha[i]
        if fix[i]:
            fixed_alpha_product *= inverse_alpha[i]
            n_adapt -= 1
            shape_matrix[i, i] = 1.0
            for j in range(n_dimensions):
                if j != i:
                    shape_matrix[i, j] = 0.0
                    shape_matrix[j, i] = 0.0

    shaped_determinant = np.linalg.det(shape_matrix) * fixed_alpha_product
    scale = (rchi * initial_determinant
             / shaped_determinant) ** (1.0 / n_adapt)

    for i in range(n_dimensions):
        if fix[i]:
            shape_matrix[i, i] = inverse_alpha[i]
        else:
            for j in range(n_dimensions):
                if fix[j]:
                    continue
                shape_matrix[i, j] *= scale

    return shape_matrix


@njit(nogil=False, cache=True, fastmath=True, parallel=False)
def shaped_adaptive_weight_matrices(sigma, rchi2_values, gradient_mscp,
                                    density=None, variance_offsets=None,
                                    fixed=None):  # pragma: no cover
    r"""
    Wrapper for `shaped_adaptive_weight_matrix` over multiple values.

    Please see :func:`shaped_adaptive_weight_matrix` for details on how the
    weighting kernel is modified using a scale factor and measure of the
    derivatives of the fitting function.  This function performs the
    calculation for multiple scaling factors (:math:`\chi_r^2`) and
    derivative measures.

    Parameters
    ----------
    sigma : numpy.ndarray (n_dimensions,)
        The standard deviations of the Gaussian for each dimensional component
        used for the distance weighting of each sample in the initial fit.
    rchi2_values : numpy.ndarray (n_sets, shape)
        The reduced chi-squared statistics of the fit for each data set.  Here,
        `shape` is an arbitrary array shape which depends upon the shape of
        the output fit coordinates defined by the user.
    gradient_mscp : numpy.ndarray (n_sets, shape, n_dimensions, n_dimensions)
        An array where gradient_mscp[i, j] = derivative[i] * derivative[j] in
        dimensions i and j.  Please see :func:`derivative_mscp` for further
        information.  The last two dimensions must be Hermitian and real-valued
        (symmetric) for each fit set/coordinate.
    density : numpy.ndarray (n_sets, shape)
        The local relative density of the samples around the fit coordinate.  A
        value of 1 represents uniform distribution.  Values greater than 1
        indicate clustering around the fitting point, and values less than 1
        indicate that samples are sparsely distributed around the fitting
        point.  Please see :func:`relative_density` for further information.
    variance_offsets : numpy.ndarray (n_sets, shape)
        The variance at the fit coordinate determined from the sample
        coordinate distribution. i.e., if a fit is performed at the center of
        the sample distribution, the variance is zero.  If done at 2-sigma from
        the sample distribution center, the variance is 4.
    fixed : numpy.ndarray of bool (n_dimensions,), optional
        If supplied, `True` values indicate that the width of the Gaussian
        along the corresponding axis should not be altered in the output
        result.

    Returns
    -------
    shape_matrices : numpy.ndarray (n_sets, shape, n_dimensions, n_dimensions)
        Shape matrices defined for each set/coordinate.
    """

    shape = gradient_mscp.shape
    n = rchi2_values.size
    flat_shape = (n,) + shape[-2:]

    shape_matrices = np.empty(shape, dtype=nb.float64).reshape(flat_shape)

    flat_matrices = gradient_mscp.reshape(flat_shape)
    flat_rchi2 = rchi2_values.ravel()

    if variance_offsets is None:
        flat_offsets = np.zeros(flat_rchi2.size, dtype=nb.float64)
    else:
        flat_offsets = np.asarray(variance_offsets, dtype=nb.float64).ravel()

    if density is None:
        flat_density = np.ones(flat_rchi2.size, dtype=nb.float64)
    else:
        flat_density = np.asarray(density, dtype=nb.float64).ravel()

    for i in range(n):
        shape_matrices[i] = shaped_adaptive_weight_matrix(
            sigma, flat_rchi2[i], flat_matrices[i],
            density=flat_density[i],
            variance_offset=flat_offsets[i],
            fixed=fixed)

    return shape_matrices.reshape(shape)


@njit(fastmath=False, nogil=False, cache=True, parallel=False)
def stretch_correction(rchi2, density, variance_offset):  # pragma: no cover
    r"""
    A sigmoid function used by the "shaped" adaptive resampling algorithm.

    This sigmoid function is applied when determining the severity of stretch
    (:math:`s`) applied to principle axes of a rotated weighting kernel.  The
    correction term (:math:`\gamma`) is applied as:

    .. math::

        s_{corrected} = s^\gamma

    Since the stretch values are determined from the singular values of
    a normalized Hermitian matrix :math:`A` (see
    :func:`shaped_adaptive_weight_matrix`), where :math:`|A| \equiv 1`, then:

    .. math::

        \prod_{k=1}^{K}{s_k} = \prod_{k=1}^{K}{s_{corrected, k}} = 1

    in :math:`K` dimensions.  In other words, this does not affect the overall
    size (or volume) of the weighting kernel.

    The correction factor is calculated using :func:`half_max_sigmoid` using
    :math:`c=1`, lower and upper asymptotes as -1 and 1, such that the midpoint
    is fixed at zero when :math:`x=0`.  After making the necessary
    substitutions, the correction factor is given as:

    .. math::

        \gamma = \frac{2}
                 {\left( {1 + (2^{\nu} - 1)e^{B(1 - x)}} \right)^{1/\nu}} - 1

    We then set the rate as :math:`B = \rho` where :math:`\rho` is the
    `density` as determined by :func:`relative_density`, and the point of
    inflection as :math:`exp(\sigma)` where :math:`\sigma^2` is the
    `variance_offset` as determined by :func:`offset_variance`.  Finally,
    setting :math:`x = \chi_r^2` we arrive at:

    .. math::

        \gamma = \frac{2}
                 {\left(
                     {1 + (2^{exp(\sigma)} - 1)e^{\rho (1 - \chi_r^2)}}
                 \right)^{1/exp(\sigma)}} - 1

    As :math:`\gamma \rightarrow 0`, the resulting shape becomes more
    symmetrical.  This will be the case when the fitting point is away from
    the center of the sample distribution (high :math:`\sigma^2`), the fit
    occurs in a low density area (low :math:`\beta`), or
    :math:`\chi_r^2 \rightarrow 1`.

    It should be noted that :math:`s_{corrected} \rightarrow s` as
    :math:`\chi_r^2 \rightarrow \infty`.  However, in the range
    :math:`0 < \chi_r^2 < 1`, the correction factor is actually negative, with
    :math:`\gamma \rightarrow -1` as :math:`\chi_r^2 \rightarrow 0`.  This
    has the effect of effectively rotating the shape so that its major axis is
    perpendicular to the mean sample gradient rather than parallel.

    Parameters
    ----------
    rchi2 : int or float or numpy.ndarray (shape)
        The reduced chi-squared statistic of the initial fit.
    density : int or float or numpy.ndarray (shape)
        The local relative density at the fitting point (see
        :func:`relative_density`).
    variance_offset : int or float or numpy.ndarray (shape)
        The variance at the fitting with respect to the coordinate distribution
        of the sample used in the fit.  Please see :func:`offset_variance` for
        further information.

    Returns
    -------
    correction : numpy.ndarray
        The correction factor.

    Notes
    -----
    For high valued inflection points, :mod:`numpy` will set values in the
    denominator of the above equation to infinity, or 1, resulting in a
    misleading correction factor of :math:`\pm 1`.  In order to counter this,
    :func:`half_max_sigmoid` could not be used, and the calculation is done
    "by hand" with spurious values resulting in a correction factor of zero.
    This also requires some :mod:`numba` hackery resulting in the final output
    value being a `numpy.ndarray` in all cases.  When single valued inputs are
    supplied, the output will be a single-valued array of zero dimensions
    which should be suitable for subsequent calculations.
    """
    v = np.exp(np.sqrt(variance_offset))  # inflection point
    denominator = 1.0 + (np.exp(density * (1.0 - rchi2)) * (2.0 ** v - 1.0))
    denominator = np.asarray(denominator, dtype=nb.float64)
    shape = denominator.shape
    denominator = denominator.ravel()
    correction = (2.0 / (denominator ** (1.0 / v))) - 1

    for i in range(denominator.size):
        if not np.isfinite(denominator[i]):
            correction[i] = 0.0

    return correction.reshape(shape)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def sigmoid(x, factor=1.0, offset=0.0):  # pragma: no cover
    r"""
    Evaluate a scaled and shifted logistic function.

    The sigmoid function has the form:

    .. math::

        f(x) = \frac{1}{1 + e^{\beta (x - \alpha)}}

    where :math:`\beta` is the scaling `factor`, and :math:`\alpha` is an
    `offset` applied to `x`.

    Parameters
    ----------
    x : int or float or numpy.ndarray (shape)
        The independent variable.  If an array is supplied, must be the same
        shape as `factor` and `offset` (if both/either are also arrays).
    factor : int or float or numpy.ndarray (shape)
        The scaling factor applied to `x`.  If an array is supplied, must be
        the same shape as `x` and `offset` (if both/either are also arrays).
    offset : int or float or numpy.ndarray (shape)
        The offset to applied to `x`.  If an array is supplied, must be
        the same shape as `x` and `factor` (if both/either are also arrays).

    Returns
    -------
    result : float or numpy.ndarray (shape)
        The sigmoid function evaluated at `x`.
    """
    xx = (x - offset) * factor
    return 1.0 / (1.0 + np.exp(-xx))


@njit(fastmath=False, nogil=False, cache=True, parallel=False)
def logistic_curve(x, x0=0.0, k=1.0, a=0.0, c=1.0, q=1.0, b=1.0, v=1.0
                   ):  # pragma: no cover
    r"""
    Evaluate the generalized logistic function.

    The generalized logistic function is given as:

    .. math::

        f(x) = A + \frac{K - A}
                        {\left( C + Q e^{-B(x - x_0)} \right)^{1/\nu}}

    Taken from Wikipedia contributors. (2020, June 11). Generalised logistic
    function. In Wikipedia, The Free Encyclopedia.
    Retrieved 23:51, July 6, 2020, from
    https://en.wikipedia.org/w/index.php?title=Generalised_logistic_function&oldid=961965809

    Parameters
    ----------
    x : int or float or numpy.ndarray (shape)
        The independent variable.
    x0 : int or float or numpy.ndarray (shape), optional
        An offset applied to `x`.
    k : int or float or numpy.ndarray (shape), optional
        The upper asymptote when `c` is one.
    a : int or float or numpy.ndarray (shape), optional
        The lower asymptote.
    c : int or float or numpy.ndarray (shape), optional
        Typically takes a value of 1.  Otherwise, the upper asymptote is
        a + ((k - a) / c^(1/v)).
    q : int or float or numpy.ndarray (shape), optional
        Related to the value of f(0).
    b : int or float or numpy.ndarray (shape), optional
        The growth rate.
    v : int or float or numpy.ndarray (shape), optional
        Must be greater than zero.  Affects near which asymptote the maximum
        growth occurs.

    Returns
    -------
    result : float or numpy.ndarray (shape)
        The logistic function evaluated at `x`.
    """
    t = x - x0
    result = a + ((k - a) / (c + (q * np.exp(-b * t))) ** (1.0 / v))
    return result


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def richards_curve(x, q=1.0, a=0.0, k=1.0, b=1.0, x0=0.0):  # pragma: no cover
    r"""
    Evaluate a Richards' curve.

    The Richards' curve is a special case of the generalized logistic curve
    (see :func:`logistic_curve`) for :math:`c=1`, and :math:`v=q`:

    .. math::

        f(x) = A + \frac{K - A}
                {\left( 1 + Q e^{-B(x - x_0)} \right)^{1/Q}}

    Parameters
    ----------
    x : int or float or numpy.ndarray (shape)
        The independent variable.
    q : int or float or numpy.ndarray (shape), optional
        Related to the value of f(0).  Fixes the point of inflection.
    a : int or float or numpy.ndarray (shape), optional
        The lower asymptote.
    k : int or float or numpy.ndarray (shape), optional
        The upper asymptote.
    b : int or float or numpy.ndarray (shape), optional
        The growth rate.
    x0 : int or float or numpy.ndarray (shape), optional
        The value of `x` at which maximum growth occurs.

    Returns
    -------
    result : float or numpy.ndarray (shape)
        The Richards' curve evaluated at `x`.
    """
    return logistic_curve(x, x0=x0, a=a, k=k, c=1.0, q=q, b=b, v=q)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def half_max_sigmoid(x, x_half=0.0, k=1.0, a=0.0, c=1.0, q=1.0, b=1.0, v=1.0
                     ):  # pragma: no cover
    r"""
    Evaluate a special case of the logistic function where f(x0) = 0.5.

    The generalized logistic function is given as:

    .. math::

        f(x) = A + \frac{K - A}
                        {\left( C + Q e^{-B(x - x_0)} \right)^{1/\nu}}

    and may be evaluated with :func:`logistic_curve`.

    We can manipulate this function so that :math:`f(x_{half}) = (K + A) / 2`
    (the midpoint of the function) by setting the location of maximum growth
    (:math:`x_0`) to occur at:

    .. math::

        x_0 = x_{half} + \frac{1}{B}
                         \ln{\left( \frac{2^\nu - C}{Q} \right)}

    Since a logarithm is required, it is incumbent on the user to ensure that
    no logarithms are taken of any quantity :math:`\leq 0`, i.e.,
    :math:`(2^\nu - C) / Q > 0`.

    Parameters
    ----------
    x : int or float or numpy.ndarray (shape)
        The independent variable.
    x_half : int or float or numpy.ndarray (shape), optional
        The x value for which f(x) = 0.5.
    k : int or float or numpy.ndarray (shape), optional
        The upper asymptote when `c` is one.
    a : int or float or numpy.ndarray (shape), optional
        The lower asymptote.
    c : int or float or numpy.ndarray (shape), optional
        Typically takes a value of 1.  Otherwise, the upper asymptote is
        a + ((k - a) / c^(1/v)).
    q : int or float or numpy.ndarray (shape), optional
        Related to the value of f(0).  Fixes the point of inflection.  In this
        implementation, `q` is completely factored out after simplifying and
        does not have any affec
    b : int or float or numpy.ndarray (shape), optional
        The growth rate.
    v : int or float or numpy.ndarray (shape), optional
        Must be greater than zero.  Affects near which asymptote the maximum
        growth occurs (point of inflection).

    Returns
    -------
    result : float or numpy.ndarray
        The half-max sigmoid evaluated at `x`.
    """
    dx = np.log(((2 ** v) - c) / q) / b  # Note that q is irrelevant
    return logistic_curve(x, x0=x_half + dx, k=k, a=a, c=c, q=q, b=b, v=v)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_fits(sample_indices, sample_coordinates, sample_phi_terms,
               sample_data, sample_error, sample_mask,
               fit_coordinates, fit_phi_terms, order, alpha, adaptive_alpha,
               is_covar=False, mean_fit=False, cval=np.nan,
               fit_threshold=0.0, error_weighting=True,
               estimate_covariance=False, order_algorithm_idx=1,
               order_term_indices=None, derivative_term_map=None,
               derivative_term_indices=None, edge_algorithm_idx=1,
               edge_threshold=None, minimum_points=None,
               get_error=True, get_counts=True,
               get_weights=True, get_distance_weights=True, get_rchi2=True,
               get_cross_derivatives=True, get_offset_variance=True
               ):  # pragma: no cover
    r"""
    Solve all fits within one intersection block.

    This function is a wrapper for :func:`solve_fit` over all data sets and
    fit points.  The main computations here involve:

        1. Creating and populating the output arrays.
        2. Selecting the correct samples within the region of each fitting
           window.
        3. Calculating the full weighting factors for the fits.

    For further details on the actual fitting, please see :func:`solve_fit`.

    Parameters
    ----------
    sample_indices : numba.typed.List
        A list of 1-dimensional numpy.ndarray (dtype=int) of length n_fits.
        Each list element `sample_indices[i]`, contains the indices of samples
        within the "window" region of `fit_indices[i]`.
    sample_coordinates : numpy.ndarray (n_dimensions, n_samples)
        The independent coordinates for each sample in n_dimensions
    sample_phi_terms : numpy.ndarray (n_terms, n_samples)
        The polynomial terms of `sample_coordinates`.  Please see
        :func:`polynomial_terms` for further details.
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
    fit_phi_terms : numpy.ndarray (n_terms, n_fits)
        The polynomial terms of `fit_coordinates`.  Please see
        :func:`polynomial_terms` for further details.
    order : numpy.ndarray
        The desired order of the fit as a (1,) or (n_dimensions,) array.  If
        only a single value is supplied, it will be applied over all
        dimensions.
    alpha : numpy.ndarray (n_dimensions,)
        A distance weighting scaling factor per dimension.  The weighting
        kernel is applied equally to all sets and samples.  For further
        details, please see :func:`calculate_distance_weights`.  Will be
        overridden by `adaptive_alpha` if `adaptive_alpha.size > 0`.
    adaptive_alpha : numpy.ndarray
        Shape = (n_samples, n_sets, [1 or n_dimensions], n_dimensions).
        Defines a weighting kernel for each sample in each set.  The function
        :func:`calculate_adaptive_distance_weights_scaled` will be used for
        kernels of shape (1, n_dimensions), and
        :func:`calculate_adaptive_distance_weights_shaped` will be used for
        kernels of shape (n_dimensions, n_dimensions).  `adaptive_alpha` is
        a required parameter due to Numba constraints, and will override
        the `alpha` parameter unless it has a size of 0.  Therefore, to
        disable, please set the size of any dimension to zero.
    is_covar : bool, optional
        If `True`, indicates that `sample_data` contains covariance values
        that should be propagated through algorithm.  If this is the case,
        polynomial fitting is disabled, and a weighted variance is calculated
        instead.
    mean_fit : bool, optional
        If `True`, a weighted mean is performed instead of calculating a
        polynomial fit.
    cval : float, optional
        In a case that a fit is unable to be calculated at certain location,
        `cval` determines the fill value for the output `fit` array at those
        locations.
    fit_threshold : float, optional
        If fit_threshold is non-zero, perform a check on the goodness of the
        fit.  When the reduced-chi statistic is greater than
        abs(fit_threshold), the fit is determined to be a failure, and a
        replacement value is used. If `fit_threshold` < 0, failed fit values
        will be set to `cval`.  If `fit_threshold` > 0, failed fit values will
        be replaced by the weighted mean.
    error_weighting : bool, optional
        If `True`, weight the samples in the fit by the inverse variance
        (1 / sample_error^2) in addition to distance weighting.
    estimate_covariance : bool, optional
        If True, calculate the covariance of the fit coefficients using
        :func:`estimated_covariance_matrix_inverse`.  Otherwise, use
        :func:`covariance_matrix_inverse`.
    order_algorithm_idx : int, optional
        An integer specifying which polynomial order validation algorithm to
        use.  The default (1), will always be the more robust of all available
        options.  For further information, please see :func:`check_edges`.
    order_term_indices : numpy.ndarray (> max(order) + 1,), optional
        A 1-dimensional lookup array for use in determining the correct phi
        terms to use for a given polynomial order.  The order validation
        algorithm ensures a fit of the requested order is possible.  If not,
        and the orders are equal in all dimensions, it may also optionally
        return a suggested order.  In this case, `order_term_indices` is used
        to select the correct `sample_phi_terms` and `fit_phi_terms` for a
        given order (k), where terms are extracted via
        `phi[order_term_indices[k]:order_term_indices[k + 2]]`.
    derivative_term_map : numpy.ndarray, optional
        A mapping array for the determination of derivatives from the
        coefficients of the fit, and available terms in "phi".  The shape of
        the array is (n_dimensions, 3, n_derivative_terms).  This is only
        required if the gradient is required as an output.  For a full
        description of the derivative map, please see
        :func:`polynomial_derivative_map`.
    derivative_term_indices : numpy.ndarray (max(order) + 1,), optional
        If the fit order is allowed to vary, gives the indices in
        `derivative_term_map` for a given symmetrical order.  The correct
        `derivative_term_map` mapping for order k is given as
        `derivative_term_map[:, :, indices[k]:indices[k + 2]]`.
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
    minimum_points : int, optional
        Certain order validation algorithms check the number of available
        samples as a means to determine what order of fit is appropriate.
        If pre-calculated for the base `order`, it may be passed in here for
        a slight speed advantage.
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
    get_cross_derivatives : bool, optional
        If `True`, return the derivative mean-squared-cross-products of the
        samples for each of the fitted points.  See :func:`derivative_mscp`
        for further information.
    get_offset_variance : bool optional
        If `True`, return the offset of the fitting point from the sample
        distribution.  See :func:`offset_variance` for further information.

    Returns
    -------
    fit_results : 8-tuple of numpy.ndarray
        fit_results[0]: Fitted values.
        fit_results[1]: Error on the fit.
        fit_results[2]: Number of samples in each fit.
        fit_results[3]: Weight sums.
        fit_results[4]: Distance weight sums.
        fit_results[5]: Reduced chi-squared statistic.
        fit_results[6]: Derivative mean-squared-cross-products.
        fit_results[7]: Offset variances from the sample distribution center.

        All arrays except for fit_results[6] have the shape (n_sets, n_fits)
        or (0, 0) depending on whether `get_<name>` is `True` or `False`
        respectively.  The derivative MSCP is of shape
        (n_sets, n_fits, n_dimensions, n_dimensions) if requested, and
        (1, 0, 0, 0) otherwise.
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

    if get_cross_derivatives:
        cov_out = np.empty((n_sets, n_fits, n_dimensions, n_dimensions),
                           dtype=nb.float64)
    else:
        cov_out = np.empty((1, 0, 0, 0), dtype=nb.float64)

    if get_offset_variance:
        offset_variance_out = np.empty((n_sets, n_fits), dtype=nb.float64)
    else:
        offset_variance_out = np.empty((0, 0), dtype=nb.float64)

    if n_sets == 0 or n_fits == 0:  # pragma: no cover
        return (fit_out, error_out,
                counts_out, weights_out, distance_weights_out,
                rchi2_out, cov_out, offset_variance_out)

    adaptive_smoothing = adaptive_alpha.size > 0
    if adaptive_smoothing:
        shaped = adaptive_alpha.shape[-2] > 1
    else:
        shaped = False

    # For Numba compilation success
    dummy_fixed_weights = np.empty(0, dtype=nb.float64)
    dummy_adaptive_weights = np.empty((0, 0), dtype=nb.float64)

    for fit_index in range(len(sample_indices)):

        fit_coordinate = fit_coordinates[:, fit_index]
        fit_phi = fit_phi_terms[:, fit_index]

        # Arrays for all datasets within window region
        window_indices = sample_indices[fit_index]
        window_coordinates = sample_coordinates[:, window_indices]
        window_values = sample_data[:, window_indices]
        window_mask = sample_mask[:, window_indices]
        window_phi = sample_phi_terms[:, window_indices]

        if sample_error.shape[1] > 1:
            window_error = sample_error[:, window_indices]
        else:
            # Single values are expanded in apply_mask_to_set_arrays
            window_error = sample_error

        # Determine distance weighting
        if adaptive_smoothing:
            fixed_weights = dummy_fixed_weights
            if shaped:
                adaptive_weights = calculate_adaptive_distance_weights_shaped(
                    window_coordinates, fit_coordinate,
                    adaptive_alpha[window_indices])
            else:
                adaptive_weights = calculate_adaptive_distance_weights_scaled(
                    window_coordinates, fit_coordinate,
                    adaptive_alpha[window_indices])
        else:
            adaptive_weights = dummy_adaptive_weights
            fixed_weights = calculate_distance_weights(
                window_coordinates, fit_coordinate, alpha)

        for data_set in range(n_sets):

            if adaptive_smoothing:
                window_distance_weights = adaptive_weights[data_set]
            else:
                window_distance_weights = fixed_weights

            (fitted_value,
             fitted_error,
             counts,
             weightsum,
             distance_weight,
             rchi2,
             deriv_mscp,
             variance_offset
             ) = solve_fit(window_coordinates, window_phi,
                           window_values[data_set], window_error[data_set],
                           window_mask[data_set], window_distance_weights,
                           fit_coordinate, fit_phi, order,
                           is_covar=is_covar, fit_threshold=fit_threshold,
                           mean_fit=mean_fit, cval=cval,
                           error_weighting=error_weighting,
                           estimate_covariance=estimate_covariance,
                           order_algorithm_idx=order_algorithm_idx,
                           term_indices=order_term_indices,
                           derivative_term_map=derivative_term_map,
                           derivative_term_indices=derivative_term_indices,
                           edge_algorithm_idx=edge_algorithm_idx,
                           edge_threshold=edge_threshold,
                           minimum_points=minimum_points,
                           get_error=get_error, get_weights=get_weights,
                           get_distance_weights=get_distance_weights,
                           get_rchi2=get_rchi2,
                           get_cross_derivatives=get_cross_derivatives,
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

            if get_cross_derivatives and deriv_mscp.size != 0:
                cov_out[data_set, fit_index] = deriv_mscp

            if get_offset_variance:
                offset_variance_out[data_set, fit_index] = variance_offset

    return (fit_out, error_out,
            counts_out, weights_out, distance_weights_out,
            rchi2_out, cov_out, offset_variance_out)


@njit(fastmath=_fast_flags_all.difference({'nnan', 'ninf'}),
      nogil=False, cache=True, parallel=False)
def solve_fit(window_coordinates, window_phi, window_values, window_error,
              window_mask, window_distance_weights,
              fit_coordinate, fit_phi, order,
              is_covar=False, fit_threshold=0.0, mean_fit=False,
              cval=np.nan, error_weighting=True, estimate_covariance=False,
              order_algorithm_idx=1, term_indices=None,
              derivative_term_map=None, derivative_term_indices=None,
              edge_algorithm_idx=1, edge_threshold=None,
              minimum_points=None, get_error=True, get_weights=True,
              get_distance_weights=True, get_rchi2=True,
              get_cross_derivatives=True, get_offset_variance=True
              ):  # pragma: no cover
    r"""
    Solve for a fit at a single coordinate.

    Solves a polynomial fit of the form:

    .. math::

        f(\Phi) = \hat{c} \cdot \Phi

    where :math:`\hat{c}` are the derived polynomial coefficients for the
    :math:`\Phi` terms.  The :math:`\Phi` terms are derived from the
    independent values of the samples within the window region of the fit
    coordinate, and from the fit coordinates themselves (see
    :func:`polynomial_terms` for further details).

    The :math:`\Phi` terms are pre-calculated early in the resampling algorithm
    as this is a relatively cheap calculation, and we do not want to repeat
    the same calculation multiple times.  For example, if sample[1] is within
    the window region of point[1] and point[2], there should be no need
    to repeat the polynomial term calculation twice.  Initially, one might
    think that the actual coordinates could then be discarded, but there are a
    number of calculations that depend on the sample coordinates relative to
    the fitting points, which must therefore be dealt with "on-the-fly".

    EDGE CHECKING

    The first of the on-the-fly calculation is the "edge check".  Generally,
    polynomial fits are not well-defined away from the sample distribution
    from which they were derived.  This is especially true for higher order
    fits that may fit the sample distribution well, but start to deviate wildly
    when evaluated outside of the distribution.  The edge check step defines a
    border around the distribution, outside of which the fit will be aborted.
    There are a number of algorithms available which vary in robustness and
    speed.  Please see :func:`check_edges` for details on available algorithms.

    ORDER CHECKING

    The next step is to determine if it is possible to perform a polynomial
    fit of the given order.  For example, a 1-dimensional 2nd order polynomial
    fit can only be derived from a minimum of 3 samples.  Additionally, if
    some samples share the same coordinate, the fit becomes underdetermined, or
    if dealing with multidimensional data, one needs to ensure that the samples
    are not colinear.  If we also wish to propagate or derive valid errors, we
    should ensure the system is overdetermined.  There are a number of order
    checking algorithms which vary in robustness and speed.  Please see
    :func:`check_orders` for details on the available algorithms.

    There are two available actions if the samples fail the order check.  The
    first (default) is to simply abort fitting.  The second option is to
    lower the order of fit until the samples meet the order check requirements.
    This is only possible if the fit order is equal across all dimensions.  To
    allow for variable orders, set the `order_term_indices` (see parameter
    descriptions) to a valid value, and update `window_phi` and `fit_phi`
    accordingly.

    FITTING

    If the above checks pass, a fit can be attempted.  There are actually
    three types of fit that may be performed.  The first is the standard
    polynomial fit described above.  The second is a weighted mean which may
    explicitly be performed by setting `mean_fit` to `True`, or may be
    performed on-the-fly if the order was lowered to zero during the order
    check.  Finally, if `is_covar` was set to `True`, the `window_values` are
    considered covariances to propagate, and a fit will derived by propagating
    a weighted variance (this was created for the SOFIA HAWC+ pipeline).

    FINAL VALIDATION

    If a polynomial fit was performed, a final check may be performed to
    confirm that the solution does not deviate to significantly from the
    expected values.  This is done by evaluating the reduced chi-squared
    statistic of the fit (:math:`\chi_r^2`).  If
    :math:`\sqrt{\chi_r^2} > | \text{fit\_threshold} |`, the fit is not
    accepted, and is aborted if `fit_threshold` < 0, or set to the weighted
    mean of the samples if `fit_threshold` > 0.  No validation will be
    performed if `fit_threshold` is set to zero (default).  Note that
    `window_error` must be supplied in order for validation to be meaningful.

    Parameters
    ----------
    window_coordinates : numpy.ndarray (n_dimensions, n_samples)
        The independent coordinates within the window region of the fitting
        coordinate.
    window_phi : numpy.ndarray (n_terms, n_samples)
        The polynomial terms of `window_coordinates`.  Please see
        :func:`polynomial_terms` for further details.
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
    window_distance_weights : numpy.ndarray (n_samples,)
        The distance weighting factors applied to each sample in the fit.
    fit_coordinate : numpy.ndarray (n_dimensions,)
        The coordinate of the fitting point.
    fit_phi : numpy.ndarray (n_terms,)
        The polynomial of `fit_coordinate`.  Please see
        :func:`polynomial_terms` for further details.
    order : numpy.ndarray
        The desired order of the fit as a (1,) or (n_dimensions,) array.  If
        only a single value is supplied, it will be applied over all
        dimensions.
    is_covar : bool, optional
        If `True`, indicates that `window_values` contains covariance values
        that should be propagated through algorithm.  If this is the case,
        polynomial fitting is disabled, and a weighted variance is calculated
        instead.
    fit_threshold : float, optional
        If fit_threshold is non-zero, perform a check on the goodness of the
        fit.  When the reduced-chi statistic is greater than
        abs(fit_threshold), the fit is determined to be a failure, and a
        replacement value is used. If `fit_threshold` < 0, failed fit values
        will be set to `cval`.  If `fit_threshold` > 0, failed fit values
        will be replaced by the weighted mean.
    mean_fit : bool, optional
        If `True`, a weighted mean is performed instead of calculating a
        polynomial fit.
    cval : float, optional
        In a case that a fit is unable to be calculated at certain location,
        `cval` determines the returned fit value.
    error_weighting : bool, optional
        If `True`, weight the samples in the fit by the inverse variance
        (1 / window_error^2) in addition to distance weighting.
    estimate_covariance : bool, optional
        If `True`, when determining the error on the fit and reduced
        chi-squared, calculate the covariance of the fit coefficients using
        :func:`estimated_covariance_matrix_inverse`.  Otherwise, use
        :func:`covariance_matrix_inverse`.
    order_algorithm_idx : int, optional
        An integer specifying which polynomial order validation algorithm to
        use.  The default (1), will always be the more robust of all available
        options.  For further information, please see :func:`check_edges`.
    term_indices : numpy.ndarray (> max(order) + 1,), optional
        A 1-dimensional lookup array for use in determining the correct phi
        terms to use for a given polynomial order.  The order validation
        algorithm ensures a fit of the requested order is possible.  If not,
        and the orders are equal in all dimensions, it may also optionally
        return a suggested order.  In this case, `order_term_indices` is used
        to select the correct `window_phi` and `fit_phi` for a given order (k),
        where terms are extracted via
        `phi[order_term_indices[k]:order_term_indices[k+1]]`.
    derivative_term_map : numpy.ndarray, optional
        A mapping array for the determination of derivatives from the
        coefficients of the fit, and available terms in "phi".  The shape of
        the array is (n_dimensions, 3, n_derivative_terms).  This is only
        required if the gradient is required as an output.  For a full
        description of the derivative map, please see
        :func:`polynomial_derivative_map`.
    derivative_term_indices : numpy.ndarray (max(order) + 1,), optional
        If the fit order is allowed to vary, gives the indices in
        `derivative_term_map` for a given symmetrical order.  The correct
        `derivative_term_map` mapping for order k is given as
        `derivative_term_map[:, :, indices[k]:indices[k + 2]]`.
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
    minimum_points : int, optional
        Certain order validation algorithms check the number of available
        samples as a means to determine what order of fit is appropriate.
        If pre-calculated for the base `order`, it may be passed in here for
        a slight speed advantage.
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
    get_cross_derivatives : bool, optional
        If `True`, return the derivative mean-squared-cross-products of the
        samples in the fit.  See :func:`derivative_mscp` for further
        information.
    get_offset_variance : bool optional
        If `True`, return the offset of the fitting point from the sample
        distribution.  See :func:`offset_variance` for further information.

    Returns
    -------
    fit_result : 8-tuple
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
        fit_result[6]: Derivative mean-squared-cross-product (numpy.ndarray).
                       Set to shape (0, 0) on fit failure, and
                       (n_dimensions, n_dimensions) otherwise.
        fit_result[7]: Offset variance from the distribution center (float).
                       Set to NaN on fit failure.
    """

    # Determine whether to check fits and what to do with failures
    check_fit = fit_threshold != 0
    if check_fit:
        if fit_threshold < 0:
            replace_rejects = False
            fit_threshold *= -1
        else:
            replace_rejects = True
    else:
        replace_rejects = False

    # Switches determining what needs to be calculated
    rchi2_required = get_rchi2 or check_fit
    weightsum_required = (mean_fit or get_error or rchi2_required
                          or is_covar or get_weights)

    order_varies = term_indices is not None

    # need to update mask and counts for zero/bad weights
    counts = update_mask(window_distance_weights, window_mask)

    # The order is: fitted_value, fitted_error, counts,
    #               weight, distance_weight,
    #               rchi2, derivative_mscp, offset_variance
    failure_values = (cval, np.nan, counts, 0.0, 0.0, np.nan,
                      np.empty((0, 0), dtype=nb.float64), np.nan)

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

    # Validate order
    order = check_orders(order, window_coordinates, fit_coordinate,
                         order_algorithm_idx, mask=window_mask,
                         minimum_points=minimum_points,
                         required=not order_varies,
                         counts=counts)
    if order[0] == -1:
        return failure_values

    # Select the correct phi terms set in the case that orders vary
    # This only works for symmetric orders
    if derivative_term_map is None:
        derivative_map = np.empty((0, 0, 0), dtype=nb.i8)
    else:
        derivative_map = np.asarray(derivative_term_map, dtype=np.int64)

    if order_varies and term_indices is not None:
        o = order[0]  # Should be equal for all dimensions
        phi_term_indices = np.asarray(term_indices)
        i0, i1 = phi_term_indices[o: o + 2]
        fit_phi = fit_phi[i0: i1]
        window_phi = window_phi[i0: i1]

        if get_cross_derivatives and derivative_term_indices is not None:
            deriv_indices = np.asarray(derivative_term_indices)
            i0, i1 = deriv_indices[o: o + 2]
            derivative_map = np.asarray(derivative_map,
                                        dtype=nb.int64)[:, :, i0: i1]
        else:
            derivative_map = np.empty((0, 0, 0), dtype=nb.i8)

    # Remove masked values
    window_values, window_phi, window_error, window_distance_weights = \
        apply_mask_to_set_arrays(window_mask, window_values, window_phi,
                                 window_error, window_distance_weights,
                                 counts=counts)

    # If the order varies, and the suggested order is set to zero in all
    # dimensions, a mean fit should be performed.
    calculate_weightsum = weightsum_required
    calculate_mean = mean_fit
    if not calculate_mean and order_varies:
        for o in order:
            if o != 0:
                break
        else:
            calculate_mean = True
            calculate_weightsum = True
            get_cross_derivatives = False

    # Calculate fitting weights
    window_full_weights = calculate_fitting_weights(
        window_error, window_distance_weights, error_weighting=error_weighting)

    if calculate_weightsum:
        weightsum = array_sum(window_full_weights)
        if weightsum == 0:
            return failure_values
    else:
        weightsum = 0.0  # For Numba happiness

    if is_covar:
        # Propagate variance
        fitted_value = weighted_mean_variance(
            window_values, window_full_weights, weightsum=weightsum)

        if get_distance_weights:
            total_distance_weights = array_sum(window_distance_weights)
        else:
            total_distance_weights = 0.0

        if get_weights:
            total_weights = array_sum(window_full_weights)
        else:
            total_weights = 0.0

        return (fitted_value,
                np.nan,  # error
                counts,
                total_weights,
                total_distance_weights,
                np.nan,  # rchi2
                np.empty((0, 0), dtype=nb.float64),  # derivative_mscp
                np.nan  # offset_variance
                )

    if calculate_mean:  # i.e. symmetric order 0 (mean)
        fitted_value, fitted_variance, rchi2 = solve_mean_fit(
            window_values, window_error, window_full_weights,
            weightsum=weightsum,
            calculate_variance=get_error,
            calculate_rchi2=rchi2_required)

        deriv_mscp = np.empty((0, 0), dtype=nb.float64)

    else:  # solve with polynomial
        fitted_value, fitted_variance, rchi2, deriv_mscp = \
            solve_polynomial_fit(
                window_phi, fit_phi, window_values, window_error,
                window_distance_weights, window_full_weights,
                derivative_term_map=derivative_map,
                calculate_variance=get_error,
                calculate_rchi2=rchi2_required,
                calculate_derivative_mscp=get_cross_derivatives,
                error_weighting=error_weighting,
                estimate_covariance=estimate_covariance)

        # Check the fit didn't explode (optional)
        if check_fit:
            if math.sqrt(rchi2) > fit_threshold:
                if replace_rejects:  # Use a weighted mean instead
                    if not calculate_weightsum:
                        # Then it should be calculated now
                        weightsum = array_sum(window_full_weights)
                        if weightsum == 0:
                            return failure_values

                    fitted_value, fitted_variance, rchi2 = solve_mean_fit(
                        window_values, window_error, window_full_weights,
                        weightsum=weightsum,
                        calculate_variance=get_error,
                        calculate_rchi2=rchi2_required)
                else:
                    fitted_value = cval
                    fitted_variance = np.nan
                    rchi2 = np.nan
            else:
                fitted_value = cval
                fitted_variance = np.nan
                rchi2 = np.nan

    fitted_error = math.sqrt(fitted_variance) if get_error else np.nan

    if get_distance_weights:
        distance_weight = array_sum(window_distance_weights)
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
            deriv_mscp,
            variance_offset)


@nb.njit(fastmath=True, cache=True, nogil=False, parallel=False)
def fasttrapz(y, x):  # pragma: no cover
    r"""
    Fast 1-D integration using Trapezium method.

    Approximates the integration of a 1-D discrete valued function
    :math:`y_i = f(x_i)` with :math:`N` measurements as:

    .. math::

       \int_a^b f(x) \approx \frac{1}{2}
           \sum_{i=1}^{N}{ \left( y_{i - 1} + y_i \right)
                           \left( x_i - x_{i - 1} \right) }

    Parameters
    ----------
    y : numpy.ndarray (N,)
        Dependent variable
    x : numpy.ndarray (N,)
        Independent variable

    Returns
    -------
    area : float
        The integrated area
    """
    n = x.size
    area = 0.0
    x0 = x[0]
    y0 = y[0]
    for i in range(1, n):
        x1 = x[i]
        y1 = y[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
        x0 = x1
        y0 = y1

    return area


def convert_to_numba_list(thing):
    r"""
    Converts a Python iterable to a Numba list for use in jitted functions.

    Parameters
    ----------
    thing : iterable

    Returns
    -------
    numba.typed.List()
    """
    new_list = nb.typed.List()
    for x in thing:
        new_list.append(x)
    return new_list


def convert_to_list(thing):
    r"""
    Converts a Python iterable to a standard list suitable for jit functions.

    Parameters
    ----------
    thing : iterable

    Returns
    -------
    list
    """
    new_list = []
    for x in thing:
        new_list.append(x)
    return new_list


def get_object_size(obj):
    """
    Return the size of an object and all members in bytes.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bytes : int
    """
    blacklist = type, ModuleType, FunctionType
    if isinstance(obj, blacklist):
        raise TypeError('get_object_size() does not take argument of type: %s'
                        % type(obj))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, blacklist) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
