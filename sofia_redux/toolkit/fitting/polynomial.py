# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
import warnings

from astropy import log
import bottleneck as bn
from numba import njit
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from sofia_redux.toolkit.utilities.func \
    import faststack, taylor, remove_sample_nans
from sofia_redux.toolkit.stats.stats import meancomb
from sofia_redux.toolkit.utilities.base import Model

__all__ = ['polyexp', 'polysys', 'linear_equation', 'gaussj', 'poly1d',
           'polynd', 'zero_order_fit', 'linear_polyfit', 'gaussj_polyfit',
           'nonlinear_polyfit', 'polyfitnd', 'Polyfit', 'linear_vector_lstsq',
           'nonlinear_coefficients', 'nonlinear_evaluate', 'polyfit2d',
           'poly2d', 'polyinterp2d']


def polyexp(order, ndim=None, indexing='j'):
    """
    Returns exponents for given polynomial orders in arbitrary dimensions.

    Similar to `toolkit.resampling.resample_utils.polynomial_exponents`, but
    specialized for the polynomial fitting functions in `toolkit.fitting`.

    Parameters
    ----------
    order : int or array_like of int
        Polynomial order for which to generate exponents.  If an array
        will create full polynomial exponents over all len(order)
        dimensions.

    ndim : int, optional
        If set, return Taylor expansion for `ndim` dimensions for
        the given `order` if `order` is not an array.
    indexing : str, optional
        {'i', 'j'} If 'i', then if order = [nx, ny], exponents are
        ordered as [[y0, x0], [y1, x0], [yn, x0],..., [y1, x0]., ...
        if 'j', then if order = [nx, ny], exponents are ordered as
        [[x0, y0], [x1, y0], [xn, y0], ..., [x0, y1], ...

    Returns
    -------
    exponents : numpy.ndarray
        Polynomial exponents for the given order.  Will be of shape:

        order       ndim                   shape
        ----------  ----  ---------------------------------------------------
        int         None  (order+1,)
        array (n,)  None  (array[0]+1, array[1]+1, ..., array[n-1]+1)
        int         n     (ncoeffs, ndim) where ncoeffs is a Taylor expansion

    Examples
    --------
    >>> polyexp(3)
    array([0, 1, 2, 3])

    >>> polyexp([1, 2], indexing='i')
    array([[0, 0],
           [1, 0],
           [2, 0],
           [0, 1],
           [1, 1],
           [2, 1]])

    >>> polyexp([1, 2], indexing='j')
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1],
           [0, 2],
           [1, 2]])

    >>> polyexp(2, ndim=2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [0, 1],
           [1, 1],
           [0, 2]])
    """
    if hasattr(order, '__len__'):
        order = np.asarray(order, dtype=int)
        if order.ndim != 1:
            raise ValueError("order arrays must have 1 dimension")
        if indexing == 'j':
            order = np.flip(order)
        degree = [o + 1 for o in order]
        exponents = list(
            itertools.product(*map(lambda x: list(range(x)), degree)))
        exponents = np.array(exponents, dtype=int)
        exponents = np.flip(exponents, axis=-1)
    elif ndim is None:
        exponents = np.arange(int(order) + 1, dtype=int)
    else:
        exponents = np.array([list(e) for e in taylor(order, int(ndim))])
        exponents = np.flip(exponents, axis=-1)
    return exponents


def polysys(samples, order, exponents=None, error=None,
            product=None, ignorenans=True, mask=None, info=None):
    """
    Create a system of linear equations to solve n-D polynomials

    I've tried to be as efficient as possible, storing values that
    will be recalculated on subsequent iterations.

    Parameters
    ----------
    samples : array_like of float (ndim + 1, n_points)
        samples[0] should contain the independent values of the samples
        in the first dimension.  samples[-1] should contain the dependent
        values of the samples If solving for two features, samples[1]
        contains the independent values of the samples in the second
        dimension.  i.e. x = samples[0], y = samples[1], z = samples[2].
    order : int or array_like of int
        Either a scalar polynomial order to fit across all features or
        an array specifying the order to fit across each dimension.
    exponents : numpy.ndarray of (int or float) (n_exponents, ndimensions)
        If set will override `order`.
    error : float or array_like, optional
        (N,) error in dependent values
    product : numpy.ndarray of numpy.float64
        Pre-computed powers of the independent values in `v` where
        each dictionary element is of the form:

            dimension (int) -> exponent (int or float) -> numpy.ndarray

        such that:

            powers[1][3] = v[1] ** 3

        `powers` is updated if supplied.  Note that each power set is
        unique to the `v` and should be deleted if `v` changes.
    ignorenans : bool, optional
        If True, remove any sample points containing NaNs.
    mask : array_like of bool
        Mask indicating values to use (True) or ignore (False).
    info : dict, optional
        If supplied will be updated with exponents and product

    Returns
    -------
    alpha, beta : numpy.ndarray, numpy.ndarray
        A system of equations necessary to solve Ax=B where alpha (A) is
        of shape (coeffs, coeffs), beta (B) is of shape (coeffs,), and
        exponents contains the polynomial exponents used (coeffs, ndim)
    """

    if product is None:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("invalid samples features")
        ndim = samples.shape[0] - 1
        if exponents is None:
            exponents = polyexp(order, ndim=ndim)
        else:
            exponents = np.asarray(exponents)
        if exponents.shape[1] != ndim:
            raise ValueError("Exponents and samples features mismatch")
        product = np.empty(exponents.shape + (samples.shape[1],))
        for expi, dimi in np.ndindex(exponents.shape):
            product[expi, dimi] = samples[dimi] ** exponents[expi, dimi]
        product = np.prod(product, axis=1)
        if isinstance(info, dict):
            info['exponents'] = exponents

    if isinstance(info, dict):
        info['product'] = product

    if mask is None and ignorenans:
        mask = remove_sample_nans(samples, error, mask=True)
    elif mask is not None:
        mask = np.asarray(mask, dtype=bool)

    if error is not None and not isinstance(error, np.ndarray):
        error = np.asarray(error)

    if mask is not None and not mask.any():
        nc = product.shape[0]
        return np.full((nc, nc), np.nan), np.full(nc, np.nan)

    return linear_equation(product, samples[-1], error=error, mask=mask)


def linear_equation(design_matrix, values, error=None, mask=None):
    """
    Create a system of linear equations

    Parameters
    ----------
    design_matrix : numpy.ndarray
        (n_equations, n_sets, n_samples) or (n_sets, n_samples) of float.
    values : numpy.ndarray
        (n_parameters, n_samples) of float
    error : numpy.ndarray, optional
        (n_samples,) or () of float
    mask : numpy.ndarray, optional
        (n_samples,) of bool

    Returns
    -------
    alpha, beta : numpy.ndarray, numpy.ndarray
        A system of equations necessary to solve Ax=B where alpha (A) is
        of shape (coeffs, coeffs), beta (B) is of shape (coeffs,).
    """
    b = values
    amat = design_matrix
    inplace = False

    datavec = b.ndim == 2
    multi_equation = amat.ndim == 3

    if datavec and not multi_equation:
        # Then we're calculating the matrices of multiple
        # data sets with a single set of equations
        amat = np.repeat(amat[None], b.shape[0], axis=0)

    if error is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            b = b / error
            if datavec:
                amat /= error[:, None]
            else:
                amat = amat / error[None]
            inplace = True

    if mask is not None:
        if not mask.any():
            if datavec or multi_equation:
                shape = amat.shape[:2] + (amat.shape[1],)
            else:
                shape = amat.shape[1], amat.shape[1]
            return np.full(shape, np.nan), np.full(shape[:-1], np.nan)
        elif not mask.all():
            invalid = ~mask
            if datavec:
                amat[np.broadcast_to(invalid[:, None], amat.shape)] = 0.0
            else:
                amat[:, invalid] = 0.0

            if inplace:
                b[invalid] = np.nan
            else:
                b = b.copy()
                b[invalid] = np.nan

    if datavec or multi_equation:
        bslice = slice(None), None
        if not datavec:
            bslice += (None,)
        beta = bn.nansum(amat * b[bslice], axis=2)
        alpha = amat @ amat.swapaxes(1, 2)
    else:
        beta = bn.nansum(amat * b, axis=1)
        alpha = amat @ amat.T

    return alpha, beta


def gaussj(alpha, beta, invert=False, preserve=True):
    """
    Linear equation solution by Gauss-Jordan elimination and matrix inversion

    Parameters
    ----------
    alpha : array_like of float (N, N)
        Coefficient array where N is the number of unknown variables to
        be solved, and therefore is the number of linear equations.
    beta : array_like of float (N, M)
        Constant array containing the M right-hand side vectors
    invert : bool, optional
        If True, return A^-1 in addition to x.
    preserve : bool, optional
        If True, creates copies the input alpha and beta arrays.
        Otherwise, alpha and beta will be modified inplace if
        they are already arrays of type `numpy.float64`.

    Returns
    -------
    x [, inv_A)] : numpy.ndarray [, numpy.ndarray]
        The solution (x) to Ax=b (N, M).  If `invert` is True,
        then inv_A (N, N) will also be returned.

    Notes
    -----
    Created by:
        Liyun Wang, GSFC/ARC, November 10, 1994
    Created Python version:
        Dan Perera, USRA, April, 2019
    """
    if preserve:
        alpha = np.asarray(alpha).astype(float)
        beta = np.asarray(beta).astype(float)
    else:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

    n = alpha.shape[0]
    if alpha.ndim != 2 or alpha.shape[1] != n:
        raise ValueError("alpha must be a square matrix")
    elif beta.shape[0] != n:
        raise ValueError("beta is not compatible with alpha")
    isarr = beta.ndim != 1
    if not isarr:
        beta = beta[:, None]

    singular = False
    rowi = np.arange(n)
    pivots = np.empty(n, dtype=int)
    for i in range(n):
        # Find the last maximum row in column > diag
        maxrow = n - np.argmax(np.abs(alpha[i:, i][::-1])) - 1
        pivots[i] = maxrow

        if i != maxrow:
            alpha[[i, maxrow]] = alpha[[maxrow, i]]
            beta[[i, maxrow]] = beta[[maxrow, i]]

        if alpha[i, i] == 0:
            singular = True
            break

        pivinv = 1 / alpha[i, i]
        alpha[i, i] = 1
        alpha[i] *= pivinv
        beta[i] *= pivinv

        rowmask = rowi != i
        scale = alpha[rowmask, i][:, None]
        alpha[rowmask, i] = 0
        alpha[rowmask] -= scale * alpha[i]
        beta[rowmask] -= scale * beta[i]

    x = beta if isarr else beta[:, 0]
    if singular:
        x *= np.nan
        if invert:
            return x, alpha * np.nan

    if not invert:
        return x

    for col in range(n - 1, -1, -1):
        row = pivots[col]
        if col != row:
            alpha[:, [row, col]] = alpha[:, [col, row]]

    return x, alpha


def poly1d(x, coeffs, covar=None):
    """
    Evalulate polynomial coefficients at x

    Simple and quick when `polynd` is not necessary.

    Coefficients should be supplied in order of power such that:

        y = c[0] + c[1].x + c[2].x^2 + c[order].x^order

    Parameters
    ----------
    x : float or array_like of float
        Input 1D independent variable
    coeffs : array_like of float
        Polynomial coefficients
    covar : array_like of float
        If a 1D array is supplied, then it is assumed to be
        a the variance of the coefficients.  If a 2D array is
        supplied then assumed to be the covariance matrix.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        The fitted dependent variable and optionally, the variance
        if covar is supplied.
    """
    result = np.poly1d(np.flip(coeffs))(x)
    if covar is None:
        return result

    if not hasattr(x, '__len__'):
        isarr = False
        x = [x]
    else:
        isarr = True

    covar = np.asarray(covar, dtype=float)
    x = np.asarray(x, dtype=float)
    var = np.zeros(x.shape)
    if covar.ndim == 1:
        for i, cv in enumerate(covar):
            var += cv * ((x ** i) ** 2)
    elif covar.ndim == 2:
        xp_list, stored_xp = [], -1
        for i in range(covar.shape[0]):
            for j in range(covar.shape[1]):
                if j > stored_xp:
                    xp_list.append(x ** j)
                    stored_xp = j
                xi, xj = xp_list[i], xp_list[j]
                var += covar[i, j] * xi * xj
    else:
        raise ValueError("invalid covariance features")

    if not isarr:
        var = var[0]
    return result, var


def polynd(v, coefficients, exponents=None, covariance=None,
           product=None, info=None):
    """
    Evaluate a polynomial in multiple features

    Parameters
    ----------
    v : array_like of float (ndim, npoints)
        where v[dimension] contains the independent values for the given
        dimension at which to evaluate the coefficients.
    coefficients : array_like of float
        (ncoeffs,) or (full_poly_shape) array of polynomial coefficients.
        If `exponents` is not supplied, then a full set of polynomial
        coefficients will be generated based on the shape of coefficients.
        Otherwise, the number of coefficients should match
        exponents.shape[0].
    exponents : array_like of int, optional
        (ncoeffs, ndim) array of polynomial exponents.  If not supplied
        will be generated using `polynomial_exponents` based on the shape
        of `coefficients`.  Note that in this case we must return the
        full set of polynomial coefficients.  The dimensional order
        of the exponents should match the dimensional order of `v`.
    covariance : array_like of float (ncoeffs, ncoeffs), optional
        Covariance matrix.  `var` will be output in addition to `z`
        if supplied.
    product : numpy.ndarray of numpy.float64
        Pre-computed products the powers for each exponent.
    info : dict, optional
        If provided will be updated with exponents and product.

    Returns
    -------
    z, [var] : numpy.ndarray, [numpy.ndarray]
        z are coefficients evaluated at `v`.  If covar was provided
        then the variance is also returned.
    """
    v = np.asarray(v, dtype=float)

    if v.ndim < 2:
        raise ValueError("invalid v features")
    ndim = v.shape[0]
    coefficients = np.asarray(coefficients, dtype=float)

    dovar = covariance is not None
    if dovar:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2:
            raise ValueError("covar must be 2 dimensional (ncoeffs, ncoeffs)")

    if product is None:
        if exponents is None:
            order = [s - 1 for s in coefficients.shape]
            exponents = polyexp(order)
        else:
            exponents = np.asarray(exponents)

        if exponents.ndim != 2:
            raise ValueError("exponents must have shape (ncoeffs, ndim)")

        if exponents.ndim != 2 or exponents.shape[1] != ndim:
            raise ValueError("exponents and samples features mismatch")
        product = np.empty(exponents.shape + (v.shape[1:]))
        for expi, dimi in np.ndindex(exponents.shape):
            product[expi, dimi] = v[dimi] ** exponents[expi, dimi]
        product = np.prod(product, axis=1)

    if isinstance(info, dict):
        info['product'] = product
        if exponents is None:
            order = [s - 1 for s in coefficients.shape]
            exponents = polyexp(order)
        else:
            exponents = np.asarray(exponents)
        info['exponents'] = exponents

    parray = np.ones((1, product.ndim), dtype=int).ravel()
    parray[0] = -1
    result = np.sum(product * coefficients.T.ravel().reshape(parray), axis=0)

    if not dovar:
        return result
    elif not np.isfinite(covariance).any():
        return result, np.full(result.shape, np.nan)

    var = np.zeros(result.shape)
    for i in range(covariance.shape[0]):
        for j in range(covariance.shape[1]):
            var += covariance[i, j] * product[i] * product[j]

    return result, var


def zero_order_fit(data, error=None):
    """
    Calculate the zeroth order polynomial coefficients and covariance

    Basically an (optionally weighted) average.

    Parameters
    ----------
    data : array_like of (int or float)
    error : array_like of (int or float), optional

    Returns
    -------
    coefficients, covariance : numpy.ndarray, np.ndarray of float
        Polynomial coefficient 0 (1,) and variance (1,).
    """
    mean, mvar = meancomb(data, variance=error)
    return np.array([mean]), np.array([mvar])


def linear_polyfit(samples, order, exponents=None, error=1, mask=None,
                   covar=False, ignorenans=True, info=None,
                   product=None, **kwargs):
    """
    Fit a polynomial to data samples using linear least-squares.

    Creates a system of polynomial equations, solving Ax=B with linear
    least-squares.  Polynomial exponents are generated following the
    rules of `polynomial_exponents`.

    If the solution fails, all coefficients are set to NaN.

    Parameters
    ----------
    samples : array_like of float (ndim + 1, n_points)
        samples[0] should contain the independent values of the samples
        in the first dimension.  samples[-1] should contain the dependent
        values of the samples If solving for two features, samples[1]
        contains the independent values of the samples in the second
        dimension.  i.e. x = samples[0], y = samples[1], z = samples[2].
    order : int or array_like of int
        Either a scalar polynomial order to fit across all features or
        an array specifying the order to fit across each dimension.
    exponents : numpy.ndarray of (int or float) (n_coeffs, ndimensions)
        If set will override `order`.
    error : float or numpy.ndarray, optional
        (n_points,) error in z
    mask : array_like of bool
        array where True indicates a value to use in fitting and False
        is a value to ignore.  Overrides `ignorenans`.
    covar : bool, optional
        If True, return the covariance
    ignorenans : bool, optional
        If True, remove any sample points containing NaNs.
    product : numpy.ndarray of numpy.float64, optional
        Pre-calculated array to pass into `polysys`.  Overrides orders
        and exponents.
    info : dict, optional
        If provided will be updated with `product` and `exponents`
    kwargs : dict, optional
        Currently does nothing.  Just a place holder

    Returns
    -------
    coefficients, exponents, [covariance] : tuple of numpy.ndarray
        The polynomial coefficients (ncoeffs,),
        the polynomial exponents (ncoeffs, ndim), and
        (optionally) the covariance matrix (ncoeffs, ncoeffs)
    """
    _ = kwargs
    alpha, beta = polysys(
        samples, order, exponents=exponents, error=error,
        ignorenans=ignorenans, mask=mask, product=product, info=info)
    try:
        coeffs = np.linalg.solve(alpha, beta)
    except np.linalg.LinAlgError as err:
        log.debug("singular values encountered in matrix inversion: %s" % err)
        nc = alpha.shape[0]
        coeffs = np.full(nc, np.nan)
        return (coeffs, np.full((nc, nc), np.nan)) if covar else coeffs

    return (coeffs, np.linalg.inv(alpha)) if covar else coeffs


def gaussj_polyfit(samples, order, exponents=None, error=1, covar=False,
                   ignorenans=True, info=None, mask=None, product=None,
                   **kwargs):
    """
    Fit a polynomial to data samples using Gauss-Jordan elimination.

    Creates a system of polynomial equations, solving Ax=B using
    Gauss-Jordan elimination.  Polynomial exponents are generated
    following the rules of `polynomial_exponents`.

    If the solution fails, all coefficients are set to NaN.

    Parameters
    ----------
    samples : array_like of float (ndim + 1, n_points)
        samples[0] should contain the independent values of the samples
        in the first dimension.  samples[-1] should contain the dependent
        values of the samples If solving for two features, samples[1]
        contains the independent values of the samples in the second
        dimension.  i.e. x = samples[0], y = samples[1], z = samples[2].
    order : int or array_like of int
        Either a scalar polynomial order to fit across all features or
        an array specifying the order to fit across each dimension.
    exponents : numpy.ndarray of (int or float) (n_exponents, ndimensions)
        If set will override `order`.
    error : float or numpy.ndarray, optional
        (n_points,) error in z
    covar : bool, optional
        If True, return the covariance
    ignorenans : bool, optional
        If True, remove any sample points containing NaNs.
    product : numpy.ndarray of numpy.float64, optional
        Pre-calculated array to pass into `polysys`.  Overrides orders
        and exponents.
    info : dict, optional
        If provided will be updated with `product` and `exponents`
    mask : array_like of bool
        array where True indicates a value to use in fitting and False
        is a value to ignore.  Overrides `ignorenans`.
    kwargs : dict, optional
        Currently does nothing.  Just a place holder

    Returns
    -------
    coefficients, [covariance] : tuple of numpy.ndarray
        The polynomial coefficients (ncoeffs,), and (optionally)
        the covariance matrix (ncoeffs, ncoeffs)
    """
    _ = kwargs
    alpha, beta = polysys(
        samples, order, exponents=exponents, error=error,
        ignorenans=ignorenans, mask=mask, product=product, info=info)

    result = gaussj(alpha, beta, invert=covar, preserve=False)
    coeffs, covariance = result if covar else (result, None)
    return (coeffs, covariance) if covar else coeffs


def nonlinear_func(design_matrix, *coefficients):
    return np.asarray(coefficients) @ design_matrix


def nonlinear_coefficients(matrix, data, error=None, mask=None, **kwargs):

    datavec = data.ndim == 2
    ncoeffs = matrix.shape[0]
    nvec = data.shape[0] if datavec else 1
    coefficients = np.empty((nvec, ncoeffs))
    p0 = np.ones(ncoeffs)

    domask = mask is not None
    doerror = error is not None
    if not datavec:
        data = np.atleast_2d(data)
        if domask:
            mask = np.atleast_2d(mask)
        if doerror:
            error = np.atleast_2d(error)

    nsamples = mask.sum(axis=1) if domask else None

    for i in range(nvec):
        if domask:
            if nsamples[i] < ncoeffs:
                coefficients[i] = np.nan
                continue
            m = mask[i]
            s = matrix[:, m]
            v = data[i, m]
            e = error[i, m] if doerror else None
        else:
            s = matrix
            v = data[i]
            e = error[i] if doerror else None

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                coefficients[i], _ = curve_fit(
                    nonlinear_func, s, v, sigma=e, p0=p0, **kwargs)
            except RuntimeError:  # pragma: no cover
                coefficients[i] = np.nan
        p0.fill(1.0)

    return coefficients if datavec else coefficients[0]


def nonlinear_evaluate(matrix_in, data, matrix_out,
                       error=None, mask=None, **kwargs):
    coefficients = nonlinear_coefficients(
        matrix_in, data, error=error, mask=mask, **kwargs)
    return coefficients @ matrix_out


@njit
def linear_vector_solve(alpha, beta, matrix_out):  # pragma: no cover
    nc = alpha.shape[0]
    result = np.empty((beta.shape[0], matrix_out.shape[1]))
    for i in range(nc):
        coeffs = np.linalg.solve(alpha[i], beta[i])
        result[i] = coeffs @ matrix_out

    return result


def linear_vector_lstsq(alpha, beta, matrix_out):
    nc = alpha.shape[0]
    result = np.empty((beta.shape[0], matrix_out.shape[1]))

    for i in range(nc):
        coeffs = np.linalg.lstsq(alpha[i], beta[i], rcond=None)[0]
        result[i] = coeffs @ matrix_out

    return result


def linear_evaluate(matrix_in, data, matrix_out, error=None, mask=None,
                    allow_errors=True):
    alpha, beta = linear_equation(matrix_in, data, error=error, mask=mask)
    datavec = alpha.ndim == 3

    if not allow_errors:
        if datavec:
            return linear_vector_solve(alpha, beta, matrix_out)
        else:
            return np.linalg.solve(alpha, beta) @ matrix_out

    try:
        if datavec:
            result = linear_vector_solve(alpha, beta, matrix_out)
        else:
            result = np.linalg.solve(alpha, beta) @ matrix_out

    except np.linalg.LinAlgError:
        if datavec:
            nc = alpha.shape[0]
            result = np.empty((beta.shape[0], matrix_out.shape[1]))

            for i in range(nc):
                try:
                    coeffs = np.linalg.solve(alpha[i], beta[i])
                    result[i] = coeffs @ matrix_out
                except np.linalg.LinAlgError:
                    result[i] = np.nan
        else:
            result = np.full(matrix_out.shape[1], np.nan)

    return result


def gaussj_evaluate(matrix_in, data, matrix_out, error=None, mask=None):
    alpha, beta = linear_equation(matrix_in, data, error=error, mask=mask)
    datavec = alpha.ndim == 3
    if datavec:
        coeffs = np.empty(beta.shape)
        for i in range(alpha.shape[0]):
            coeffs[i] = gaussj(alpha[i], beta[i],
                               invert=False, preserve=False)
    else:
        coeffs = gaussj(alpha, beta, invert=False, preserve=False)
    return coeffs @ matrix_out


def nonlinear_polyfit(samples, order, exponents=None, error=1, product=None,
                      mask=None, covar=False, ignorenans=True, info=None,
                      **kwargs):
    """
    Solve for polynomial coefficients using non-linear least squares fit

    Parameters
    ----------
    samples : array_like of float (ndim + 1, n_points)
        samples[0] should contain the independent values of the samples
        in the first dimension.  samples[-1] should contain the dependent
        values of the samples If solving for two features, samples[1]
        contains the independent values of the samples in the second
        dimension.  i.e. x = samples[0], y = samples[1], z = samples[2].
    order : int or array_like of int
        Either a scalar polynomial order to fit across all features or
        an array specifying the order to fit across each dimension.
    exponents : array_like of (int or float) (n_exponents, ndimensions)
        If set will override `order`.
    error : float or array_like of float, optional
        (n_points,) error in z
    covar : bool, optional
        If True, return the covariance
    ignorenans : bool, optional
        If True, remove any sample points containing NaNs.
    info : dict, optional
        If provided will be updated with `product` and `exponents`.
        Note that `product` will always be None for nonlinear fitting.
    product : numpy.ndarray of numpy.float64, optional
        Not used.
    mask : array_like of bool
        array where True indicates a value to use in fitting and False
        is a value to ignore.  Overrides `ignorenans`.
    kwargs : dict, optional
        Additional keyword arguments to `scipy.optimize.curve_fit`

    Returns
    -------
    coefficients, exponents, [covariance] : tuple of numpy.ndarray
        The polynomial coefficients (ncoeffs,),
        the polynomial exponents (ncoeffs, ndim), and
        (optionally) the covariance matrix (ncoeffs, ncoeffs)
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim < 2:
        raise ValueError("Samples must have at least 1 feature")

    error = np.asarray(error, dtype=float)
    if error.shape == ():
        error = np.repeat(error, samples.shape[1])
    if product is not None:
        product = None

    if info is None:
        info = {}
    ndim = samples.shape[0] - 1
    if exponents is None:
        exponents = polyexp(order, ndim=ndim)
    else:
        exponents = np.asarray(exponents)
    if ndim == 1 and exponents.ndim == 1:
        exponents = exponents[:, None]
    elif exponents.ndim != 2:
        raise ValueError("Exponents must be a 1-D or 2-D array")
    elif exponents.shape[1] != ndim:
        raise ValueError(
            "Features mismatch: samples.shape[1] does not match "
            "exponents.shape[0]")
    info['exponents'] = exponents
    info['product'] = product

    # The model used for minimization.  Outputs dependent values based
    # on independent values and polynomial coefficients.
    def polymodel(independent_values, *coefficients):
        return polynd(independent_values, coefficients,
                      exponents=exponents)

    nc = exponents.shape[0]

    # data cannot contain any NaNs for curve_fit to work (np.inf is ok)
    nanmask = remove_sample_nans(samples, error, mask=True)
    if mask is None and ignorenans:
        mask = nanmask
    elif not ignorenans and not nanmask.all():  # a single NaN invalidates data
        coeffs = np.full(nc, np.nan)
        return (coeffs, np.full((nc, nc), np.nan)) if covar else coeffs
    s, e = samples[:, mask], error[mask]

    if s.shape[1] < nc:
        log.warning("insufficient non-NaN sample points")
        coeffs = np.full(nc, np.nan)
        return (coeffs, np.full((nc, nc), np.nan)) if covar else coeffs

    kwargs['p0'] = np.ones(nc)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        try:
            coeffs, covariance = curve_fit(
                polymodel, s[:-1], s[-1], sigma=e, **kwargs)
        except RuntimeError:  # pragma: no cover
            log.warning("least-squares minimization failed")
            coeffs = np.full(nc, np.nan)
            covariance = np.full((nc, nc), np.nan) if covar else None

    return (coeffs, covariance) if covar else coeffs


class Polyfit(Model):
    """
    Fits and evaluates polynomials in N-dimensions.

    Attributes
    ----------
    coefficients : numpy.ndarray (ncoeffs,)
        Polynomial fit coefficients
    covariance : numpy.ndarray (ncoeffs, ncoeffs)
        Polynomial fit covariance.  Will be None if the covariance matrix
        was not calculated.
    exponents : numpy.ndarray (ncoeffs, ndimensions)
        Polynomial exponents

    Examples
    --------
    >>> y, x = np.mgrid[:5, :5]
    >>> z = 1 + x * y + x ** 2
    >>> poly = Polyfit(x, y, z, 2)
    >>> assert np.allclose(poly(x, y), z)
    >>> print(poly.get_coefficients())
    [[ 1.  0.  1.]
     [ 0.  1.  0.]
     [-0.  0.  0.]]
    """
    def __init__(self, *args, error=1, mask=None, covar=True, stats=True,
                 robust=0.0, eps=0.01, maxiter=10, solver='gaussj',
                 set_exponents=False, ignorenans=True, **kwargs):

        self.exponents = None
        self.coefficients = None
        self._order = None
        self._solver = str(solver).lower().strip()
        self.fitter = None
        self._set_exponents = set_exponents
        self._interpolated_error = None
        self._product = None
        super().__init__(*args, error=error, mask=mask, covar=covar,
                         stats=stats, robust=robust, eps=eps,
                         maxiter=maxiter, ignorenans=ignorenans,
                         fit_kwargs=kwargs)

    def _parameters_string(self):
        """Returns a string suitable for displaying coefficients"""
        do_sigma = getattr(self.stats, 'sigma', None) is not None
        s = "\n    Exponents : Coefficients"
        s += "\n--------------------------------\n"
        for i, (ee, c) in enumerate(zip(self.exponents, self.coefficients)):
            s += "%s : %f" % (repr(tuple(ee)), c)
            if do_sigma:
                s += " +/- %f\n" % self.stats.sigma[i]
            else:
                s += '\n'
        return s

    def _parse_model_args(self):
        """Determines polynomial exponents and solving method"""
        order = self._model_args
        if hasattr(order, '__len__'):
            order = np.asarray(order)
            if self._set_exponents:
                if order.ndim != 2:
                    raise ValueError(
                        "set_exponents: Order must have 2 features")
                if order.shape[1] != self._ndim:
                    raise ValueError(
                        "set_exponents: Dimension 1 of order does not "
                        "match number of data features")
                exponents = order
                order = None
            else:
                if order.size != self._ndim:
                    raise ValueError(
                        "set_exponents: Order size does not match number of"
                        " features %i != %i" % (order.size, self._ndim))
                exponents = None
        else:
            exponents = None

        self.exponents = exponents
        self._order = order

        if self._solver == 'gaussj':
            self.fitter = gaussj_polyfit
        elif self._solver == 'linear':
            self.fitter = linear_polyfit
        elif self._solver == 'nonlinear':
            self.fitter = nonlinear_polyfit
        else:
            raise ValueError("Unknown solver %s" % repr(self._solver))

    def initial_fit(self):
        """Initial fits of polynomial to samples"""
        self.refit_mask(self._usermask, covar=self.covar)
        self._order = -1  # overridden by exponents

    def _fast_error(self):
        """Don't create the error unless asked for or already present

        This is for the errors of the samples only
        """
        if self._interpolated_error is None:
            if hasattr(self._error, '__len__'):
                self._interpolated_error = self._error
            else:
                self._interpolated_error = np.full(
                    self.mask.shape, float(self._error))
        return self._interpolated_error

    def refit_mask(self, mask, covar=False):
        """Place holder"""
        self.mask = mask
        doinfo = self._product is None or self.exponents is None
        info = {} if doinfo else None
        fit = self.fitter(
            self._samples, self._order, exponents=self.exponents,
            error=self._fast_error(), covar=covar, ignorenans=False,
            mask=self.mask, product=self._product, info=info,
            **self._fit_kwargs)

        if doinfo:
            self.exponents = info['exponents']
            self._product = info['product']

        if covar:
            self.coefficients, self.covariance = fit
        else:
            self.coefficients, self.covariance = fit, None

        self._nparam = self.coefficients.size
        self.success = not np.isnan(self.coefficients).any()
        if not self.success:
            self.covariance = None
        self._fit_statistics()

    def refit_data(self, *args, mask=None, error=None, covar=False):
        """
        Refit samples

        Parameters
        ----------
        args : tuple of array_like
          If one argument is supplied, it is assumed to be the dependent
          variable.  Otherwise, all independent and dependent variables
          should be supplied.  No other arguments should be supplied.
        mask : array_like of bool, optional
            New user mask.  If none is supplied then one will be created
            based on the `ignorenans`.
        error : float or array_like, optional
            New error.  If None is supplied, the old error will be used.
        covar : bool, optional
            If True, calculate the covariance.
        """
        if len(args) == 1:
            self._samples[-1] = args[0].ravel()
        elif len(args) > 1:
            self._samples = faststack(*args)
            self._product = None
        if mask is None:
            mask = remove_sample_nans(self._samples, error, mask=True)
        else:
            self._usermask = np.asarray(mask, dtype=bool)
            mask = self._usermask
        if error is not None:
            self._error = error.ravel()
            self._interpolated_error = None
        self.refit_mask(mask, covar=covar)

    def evaluate(self, independent_samples, dovar=False):
        """Evaluate the polynomial at independent values"""
        have_var = dovar and self.covariance is not None
        covariance = self.covariance if have_var else None
        fit = polynd(independent_samples, self.coefficients,
                     exponents=self.exponents, covariance=covariance)
        fit, var = fit if have_var else (fit, None)
        return (fit, var) if dovar else fit

    def get_coefficients(self, covar=False):
        """
        Return coefficients and covariance suitable for general `polynd` use.

        Notes
        -----
        The (row, col) ordering is reversed in this case since we wish
        to order coefficients in the same order as the data features,
        which the general case usage of `polynd` expects, but numpy
        does not.

        Parameters
        ----------
        covar : bool, optional
            If True, return the covariance matrix as well.

        Returns
        -------
        coefficients, [covariance] : numpy.ndarray, (numpy.ndarray or None)
        """
        exponents = np.flip(self.exponents, axis=1)
        cdim = np.max(exponents, axis=0).astype(int) + 1
        coeffs = np.zeros(cdim)
        for e, c in zip(exponents, self.coefficients):
            coeffs[tuple(e)] = c

        if self.covariance is None or not covar:
            return (coeffs, self.covariance.copy()) if covar else coeffs

        n = coeffs.size
        covariance = np.zeros((n, n))
        fac = np.roll(np.cumprod(cdim), 1)
        fac[0] = 1
        inds = np.sum(fac[None] * exponents, axis=1)

        i1, j1, i2, j2 = [], [], [], []
        for i, ci in enumerate(inds):
            for j, cj in enumerate(inds):
                i1.append(i)
                j1.append(j)
                i2.append(ci)
                j2.append(cj)
        covariance[i2, j2] = self.covariance[i1, j1]
        return coeffs, covariance


def polyfitnd(*args, error=1, mask=None, covar=False,
              solver='linear', set_exponents=False, model=False,
              stats=None, robust=0, eps=0.01, maxiter=10, **kwargs):
    r"""
    Fits polynomial coefficients to N-dimensional data.

    Parameters
    ----------
    args : N-tuple of array_like
        args[-1] : int or array_like of int
            The polynomial order for all dimensions, or the polynomial order
            for each dimension supplied in the same order as the independent
            values.
        args[-2] : array_like
            The dependent data values.
        args[:-2] : N-tuple of array_like
            The independent data values for each dimension.
    error : int or float or array_like, optional
        Error values for the dependent values.  If supplied as an array, the
        shape must match those supplied as arguments.
    mask : array_like of bool, optional
        If supplied, must match the shape of the input arguments.  `False`
        entries will be excluded from any fitting while `True` entries will
        be included in the fit.
    covar : bool, optional
        If `True`, calculate the covariance of the fit parameters.  Doing so
        will return the covariance in addition to the coefficients.
    solver : str, optional
        One of {'linear', 'nonlinear', 'gaussj'}.  Defines the algorithm by
        which to solve for x in the linear equation Ax=B as returned by
        `polysys`.
    set_exponents : bool, optional
        If `True`, indicates that the user has supplied their own set of
        polynomial terms in `args[-1]`.  For example, if 2 dimensional
        independent values were supplied in the (x, y) order,
        args[-1] = [[0, 0], [1, 2]] would solve for c_0 and c_1 in the
        equation: fit = c_0 + c_1.x.y^2.
    model : bool, optional
        If True, return the Polyfit model solution.  No other return values
        will occur such as the covariance or statistics.  However, covariance
        and statistics will still be calculated if `covar=True` or `stats=True`
        and can be accessed through those model attributes.
    stats : bool, optional
        If True, return the statistics on the fit as a Namespace.
    robust : int or float, optional
        If > 0, taken as the sigma threshold above which to identify outliers.
        Outliers are those identified as \|x_i - x_med\| / MAD > `robust`,
        where x is the residual of (data - fit) x_med is the median, MAD is the
        median absolute deviation defined as 1.482 * median(abs(x_i - x_med)).
        The fit will be iterated upon until the set of identified outliers does
        not change, or any change in the relative RMS is less than `eps`, or
        `maxiter` iterations have occurred.
    eps : int or float, optional
        Termination criterion of (\|delta_rms\| / rms) < eps for the robust
        outlier rejection iteration.
    maxiter : int, optional
        Maximum number of iterations for robust outlier rejection.
    kwargs : dict, optional
        Additional keyword arguments to pass into the Polyfit class during
        initialization.

    Returns
    -------
    (coefficients, [covariance], [stats]) or `Polyfit`
        The values return depend on the use of the `covariance`, `stats`, and
        `model` keyword values.

    Notes
    -----
    Order of Arguments and return value order:

    The arguments supplied must be provided as:

    (X0, [X1, X2, ..., XN], Y, order)

    where X0 denotes the independent values for the first dimension, and so on,
    up to dimension N.  Y are the dependent values, and order gives the
    polynomial order to fit to all, or each dimension.  All arrays (X and Y)
    must be of the same shape.

    A single value for order applies a redundancy controlled polynomial fit
    across all dimensions in which:

    c_i = 0 if sum(exponents_i) > order

    where c_i if the polynomial coefficient for term i.  If an array is given
    for the order argument, no such redundancy control will occur, and a
    polynomial coefficient will be calculated for each term.  i.e., for
    2-dimensional x,y data, order=2 would calculate coefficients for the
    following terms:

    C, x, x^2, y, x.y, y^2

    while order = [2, 2] would calculate coefficients for the full set of
    terms:

    C, x, x^2, y, x.y, x^2.y, y^2, x.y^2, x^2.y^2

    Alternatively, the user may define their own coefficients if
    `set_exponents=True`.  For example, order=[[0, 0], [1, 2]] would result
    in the terms:

    C, x.y^2

    The coefficients returned will be given as an array of N-dimensions with
    a size of order+1 in each dimension such that `coefficients[1, 2]` would
    give the coefficient for the x.y^2 term (if arguments were supplied in the
    (x, y) order).  However, if `set_exponents=True`, coefficients will be
    returned in the same order as the terms supplied (1-dimensional array).

    If covariance is also returned, it is an (NC x NC) array where NC if the
    number of coefficients.  Coefficients are ordered lexographically along
    each axis.  For example, in 2 dimensions each axis will consist of the
    following terms in the order:

    x^0.y^0, x^0.y^1, ..., x^0.y^order, x^1.y^0,..., x^order.y^order

    Examples
    --------
    >>> from sofia_redux.toolkit.fitting.polynomial import polyfitnd
    >>> import numpy as np
    >>> y, x = np.mgrid[:5, :5]
    >>> z = 1 + (2 * x * y) + (0.5 * x ** 2)  # f(x) = 1 + 2xy + 0.5x^2
    >>> coefficients = polyfitnd(x, y, z, 2)  # fit a 2nd order polynomial
    >>> print(coefficients.round(decimals=2) + 0)
    [[1.   0.   0.5]
     [0.   2.   0. ]
     [0.   0.   0. ]]
    """

    if stats is None:
        stats = model

    poly = Polyfit(*args, error=error, mask=mask, covar=covar,
                   solver=solver, stats=stats, set_exponents=set_exponents,
                   robust=robust, eps=eps, maxiter=maxiter, **kwargs)
    if model:
        return poly

    if set_exponents:
        # Don't create a regular array of coefficients since the user
        # could have used anything as exponents. Assume they know what
        # they are doing.
        coefficients, covariance = poly.coefficients, poly.covariance
    else:
        c = poly.get_coefficients(covar=covar)
        (coefficients, covariance) = c if covar else (c, None)

    if not covar and not stats:
        return coefficients

    result = (coefficients,)
    if covar:
        result = result + (covariance,)
    if stats:
        result = result + (poly.stats,)
    return result


def polyfit2d(x, y, z, kx=3, ky=3, full=False):
    """
    Least squares polynomial fit to a surface

    Parameters
    ----------
    x : array_like of float
        (shape1) x-coordinate independent interpolants
    y : array_like of float
        (shape1) y-coordinate independent interpolants
    z : array_like of float
        (shape1) dependent interpolant values
    kx : int, optional
        order of polynomial to fit in the x-direction
    ky : int, optional
        order of polynomial to fit in the y-direction
    full : bool, optional
        If True, will solve using the full polynomial matrix.  Otherwise,
        will use the upper-left triangle of the matrix.  See
        `polyinterp2d` for further details.  Note that if kx != ky, then
        the full matrix will be solved for.

    Returns
    -------
    numpy.ndarray
        (ky+1, kx+1) array of polynomial coefficients solveable by
        `poly2d`.
    """

    x, y, z = np.array(x), np.array(y), np.array(z)
    s = x.shape
    if y.shape != s or z.shape != s:
        log.error("incompatible array features")
        return
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    nx, ny = kx + 1, ky + 1
    full |= kx != ky
    loop = list(itertools.product(range(ny), range(nx)))
    if not full:
        loop2 = []
        for (j, i) in loop:
            if i <= (ky - j):
                loop2.append((j, i))
        loop = loop2
    a = np.zeros((x.size, len(loop)))
    for k, (j, i) in enumerate(loop):
        a[:, k] = (x ** i) * (y ** j)
    m, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    if full:
        coeffs = m.reshape((ny, nx))
    else:
        coeffs = np.zeros((ny, nx))
        for c, (j, i) in zip(m, loop):
            coeffs[j, i] = c
    return coeffs


def poly2d(x, y, coeffs):
    """
    Evaluate 2D polynomial coefficients

    Parameters
    ----------
    x : array_like of float
        (shape1) x-coordinate independent interpolants
    y : array_like of float
        (shape1) y-coordinate independent interpolants
    coeffs : numpy.ndarray
        (y_order + 1, x_order + 1) array of coefficients output by
        `polyfit2d`.

    Returns
    -------
    numpy.ndarray
        (shape1) polynomial coefficients evaluated at (y,x).
    """
    coeffs = np.array(coeffs)
    if coeffs.ndim != 2:
        log.error("coefficient array must be a 2D array")
        return
    x, y = np.array(x), np.array(y)
    s = x.shape
    if y.shape != s:
        log.error("incompatible coordinate array features")
        return
    ny, nx = coeffs.shape
    result = np.zeros(s)
    for c, (j, i) in zip(coeffs.ravel(),
                         itertools.product(range(ny), range(nx))):
        if c == 0:
            continue
        result += c * (x ** i) * (y ** j)
    return result


def polyinterp2d(x, y, z, xout, yout, kx=None, ky=None, order=3, full=False):
    """
    Interpolate 2D data using polynomial regression (global)

    Notes
    -----
    If full is False then only the upper-left triangle of polynomial
    coefficients will be calculated and applied.  For example, if
    order = 3:

    F(x,y) = a(0,0)    + a(0,1)x     + a(0,2)x^2   + a(0,3)x^3 +
             a(1,0)y   + a(1,1)x.y   + a(1,2)y.x^2 +
             a(2,0)y^2 + a(2,1)x.y^2 +
             a(3,0)y^3

    If full=True or kx != ky then for the order = 3:

    F(x,y) = a(0,0)    + a(0,1)x     + a(0,2)x^2     + a(0,3)x^3     +
             a(1,0)y   + a(1,1)y.x   + a(1,2)y.x^2   + a(1,3)y.x^3   +
             a(2,0)y^2 + a(2,1)y^2.x + a(2,2)y^2.x^2 + a(2,3)y^2.x^3 +
             a(3,0)y^3 + a(3,1)y^3.x + a(3,2)y^3.x^2 + a(3,3)y^3.x^3

    Parameters
    ----------
    x : array_like of float
        (shape1) x-coordinate independent interpolants
    y : array_like of float
        (shape1) y-coordinate independent interpolants
    z : array_like of float
        (shape1) dependent interpolant values
    xout : array_like of float
        (shape2) x-coordinates to interpolate to
    yout : array_like of float
        (shape2) y-coordinates to interpolate to
    kx : int, optional
        order of polynomial to fit in the x-direction
    ky : int, optional
        order of polynomial to fit in the y-direction
    order : int, optional
        order of polynomial to fit in both the x and y directions
        if not directly specified via `kx` or `ky`
    full : bool, optional
        If True, will solve using the full polynomial matrix.  Otherwise,
        will use the upper-left triangle of the matrix.  See above for
        further details.  Note that if kx != ky, the full matrix will be
        solved for.

    Returns
    -------
    numpy.ndarray
        (shape2) `z` interpolated to (xout, yout)
    """
    if kx is None:
        kx = order
    if ky is None:
        ky = order
    if not isinstance(kx, int) or kx < 0:
        log.error("kx must be a positive integer")
        return
    elif not isinstance(ky, int) or ky < 0:
        log.error("ky must be a positive integer")
        return
    coeffs = polyfit2d(x, y, z, kx=kx, ky=ky, full=full)
    if coeffs is None:
        return
    return poly2d(xout, yout, coeffs)
