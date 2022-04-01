# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
import numba as nb
import numpy as np

from sofia_redux.toolkit.fitting.polynomial import poly2d
from sofia_redux.toolkit.interpolate.interpolate import Interpolate
from sofia_redux.toolkit.resampling.resample_utils import (
    polynomial_exponents, polynomial_terms)
from sofia_redux.toolkit.image.utilities import (
    map_coordinates)


__all__ = ['warp_image', 'polywarp', 'polywarp_image',
           'is_homography_transform', 'full_transform', 'warp_terms',
           'estimate_polynomial_transform', 'warp_coordinates',
           'warp_array_elements', 'warp', 'PolynomialTransform']


def warp_image(data, xin, yin, xout, yout, order=3,
               interpolation_order=3, mode='constant', cval=np.nan,
               output_shape=None, missing_frac=0.5, extrapolate=True,
               get_transform=False):
    """
    Warp data using transformation defined by two sets of coordinates

    Parameters
    ----------
    data : np.ndarray
        input data (nrow, ncol)
    xin : array-like
        source x coordinate
    yin : array-like
        source y coordinate
    xout : array-like
        destination x coordinate
    yout : array-like
        destination y coordinate
    order : int, optional
        polynomial order if transform='polynomial'
    interpolation_order : int, optional
        order of interpolation for reconstruction.  must be in the
        range 0-5.
    cval : float, optional
        values outside input boundaries to be replaced with cval if
        mode='constant'.
    mode : str, optional
        Points outside input boundaries are filled according to the
        given mode.  cval
    output_shape : tuple of int
        (rows, cols) If None, the input shape is preserved
    missing_frac : float, optional
        value between 0 and 1.  1 = fully weighted by real values.
        0 = fully weighted by NaNs.  Any pixel weighted by less than
        this fraction will be replaced with cval.
    extrapolate : bool, optional
        If `False`, values outside of the rectangular range of `xout` and
        `yout` will be set to `cval`.
    get_transform : bool, optional
        If `True`, return the polynomial transform in addition to the results.

    Returns
    -------
    warped, [transform] : numpy.ndarray, [PolynomialTransform]
        The warped data of shape (nrow, ncol) and the polynomial transform
        if `get_transform` is `True`.
    """
    xi, yi = np.array(xin).ravel(), np.array(yin).ravel()
    xo, yo = np.array(xout).ravel(), np.array(yout).ravel()
    co = np.stack((yo, xo))
    ci = np.stack((yi, xi))
    nans = np.isnan(data)
    if not nans.any():
        dout, transform = warp(data.copy(), co, ci, order=order,
                               interpolation_order=interpolation_order,
                               mode=mode, cval=cval,
                               output_shape=output_shape,
                               get_transform=True)
        wout = (~np.isnan(dout)).astype(float)
    else:
        weights = (~nans).astype(float)
        dw = data.copy()
        dw[nans] = 0
        wout, transform = warp(weights, co, ci, order=order,
                               interpolation_order=interpolation_order,
                               mode=mode, cval=cval,
                               output_shape=output_shape,
                               get_transform=True)
        dout = warp(dw, co, ci, order=order,
                    interpolation_order=interpolation_order,
                    mode=mode, cval=cval, output_shape=output_shape)
        wout[np.isnan(wout)] = 0
        dividx = np.abs(wout) >= missing_frac
        dout[dividx] = dout[dividx] / wout[dividx]
    cutidx = np.abs(wout) < missing_frac
    dout[cutidx] = cval

    if not extrapolate:
        minx, maxx = int(np.ceil(np.min(xo))), int(np.floor(np.max(xo)))
        miny, maxy = int(np.ceil(np.min(yo))), int(np.floor(np.max(yo)))
        dout[:miny, :] = cval
        dout[:, :minx] = cval
        dout[maxy:, :] = cval
        dout[:, maxx:] = cval

    if get_transform:
        return dout, transform
    else:
        return dout


def polywarp(xi, yi, xo, yo, order=1):
    """
    Performs polynomial spatial warping

    Fit a function of the form
    xi[k] = sum over i and j from 0 to degree of: kx[i,j] * xo[k]^i * yo[k]^j
    yi[k] = sum over i and j from 0 to degree of: ky[i,j] * xo[k]^i * yo[k]^j

    Parameters
    ----------
    xi : array-like of float
        x coordinates to be fit as a function of xi, yi
    yi : array-like of float
        y coordinates to be fit as a function of xi, yi
    xo : array-like of float
        x independent coorindate.
    yo : array-like of float
        y independent coordinate
    order : int, optional
        The polynomial order to fit.  The number of coordinate pairs must
        greater than or equal to (order + 1)^2.

    Returns
    -------
    kx, ky : numpy.ndarray, numpy.ndarray
        Array coefficients for xi, yi as a function of (xo,yo).
        shape = (degree+1, degree+1)

    Notes
    -----
    xi, yi, xo, and yo must all have the same length
    Taken from https://github.com/tvwenger/polywarp/blob/master/Polywarp.py
    """
    if len(xo) != len(yo) or len(xo) != len(xi) or len(xo) != len(yi):
        raise ValueError("length of xo, yo, xi, and yi must be the same")
    if len(xo) < (order + 1) ** 2:
        raise ValueError("length of transformation arrays must be greater "
                         "than (degree+1)**2")

    xo = np.array(xo)
    yo = np.array(yo)
    xi = np.array(xi)
    yi = np.array(yi)
    x = np.array([xi, yi])
    u = np.array([xo, yo])
    ut = np.zeros([(order + 1) ** 2, len(xo)])
    u2i = np.zeros(order + 1)
    for i in range(len(xo)):
        u2i[0] = 1.0
        zz = u[1, i]
        for j in range(1, order + 1):
            u2i[j] = u2i[j - 1] * zz
        ut[0: order + 1, i] = u2i
        for j in range(1, order + 1):
            ut[j * (order + 1): j * (order + 1) + order + 1, i] = \
                u2i * u[0, i] ** j
    uu = ut.T
    kk = np.dot(np.linalg.inv(np.dot(ut, uu).T).T, ut)
    kx = np.dot(kk, x[0, :].T).reshape(order + 1, order + 1)
    ky = np.dot(kk, x[1, :].T).reshape(order + 1, order + 1)
    return kx, ky


def polywarp_image(image, x0, y0, x1, y1, order=1, method='cubic',
                   cubic=None, mode='constant', cval=np.nan,
                   ignorenans=True, get_transform=False):
    """
    Warp an image by mapping 2 coordinate sets with a polynomial transform.

    Represents the transform from one set of coordinates to another with a
    polynomial fit, then applies that transform to an image.  After the
    coordinate warping coefficients have been determined, values are derived
    with one of the interpolation methods available in the `Interpolate`
    class.

    This is a wrapper for the `polywarp` function originally converted from
    IDL, which does not fit into the standard API of `toolkit`.

    Parameters
    ----------
    image : array_like
        The 2 dimensional image array
    x0 : array_like
        The x-coordinates defining the first set of coordinates.
    y0 : array_like
        The y-coordinates defining the first set of coordinates.
    x1 : array_like
        The x-coordinates defining the second set of coordinates.
    y1 : array_like
        The y-coordinates defining the second set of coordinates.
    order : int, optional
        The polynomial order to fit.  Coordinate sets must have a length of
        at least (order + 1)^2.
    method : str, optional
        The interpolation method to use.  Please see
        `toolkit.interpolate.Interpolate`.
    cubic : float, optional
        The cubic parameter if cubic interpolation is used.  Please see
        `toolkit.interpolate.Interpolate`.
    mode : str, optional
        Defines edge handling during interpolation.  Please see
        `toolkit.interpolate.Interpolate`.
    cval : int or float, optional
        If `mode=constant`, defines the value of points outside the boundary.
    ignorenans : bool, optional
        If True, do not include NaN values during interpolation.
    get_transform : bool, optional
        if True, return the transformation object as the second element
        of a 2-tuple in the return value
    Returns
    -------
    warped_image : numpy.ndarray of numpy.float64
        The warped image with the same shape as `image`.
    """

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Data must be a 2-D array")

    kx, ky = polywarp(x0, y0, x1, y1, order=order)
    yg, xg = np.arange(image.shape[0]), np.arange(image.shape[1])
    x, y = np.meshgrid(xg, yg)
    xi = poly2d(x, y, kx.T)
    yi = poly2d(x, y, ky.T)
    interpolator = Interpolate(xg, yg, image, method=method, cval=cval,
                               cubic=cubic, ignorenans=ignorenans, mode=mode)
    result = interpolator(xi, yi)
    return (result, interpolator) if get_transform else result


@nb.njit(cache=True, parallel=False, fastmath=True)
def is_homography_transform(transform, n_dimensions):  # pragma: no cover
    """
    Check if a transform is homographic.

    Parameters
    ----------
    transform : numpy.ndarray (float)
        An array of shape (>=n_dimensions, >=n_dimensions).
    n_dimensions : int
        The number of coordinate dimensions to which the transform will be
        applied.

    Returns
    -------
    homographic : bool
    """
    if transform.shape[0] <= n_dimensions:
        return False
    if transform[n_dimensions, n_dimensions] != 1:
        return True
    for i in range(n_dimensions):
        if transform[n_dimensions, i] != 0 or transform[i, n_dimensions] != 0:
            return True
    return False


@nb.njit(cache=True, parallel=False, fastmath=False)
def full_transform(coordinates, transform):  # pragma: no cover
    """
    Apply a metric transform to the supplied coordinates.

    A standard correlation matrix transform, or homography (planar) transform
    may be applied.  For 2-dimensions, the homography transform is given as::

        | x' |     |x|   |h11 h12 h13| |x|
        | y' | = H |y| = |h21 h22 h23| |y|
        | 1  |     |1|   |h31 h32 h33| |1|

    where H is the transform matrix.  The h13 and h23 elements represent a
    translation, and the h31 and h32 represent relative scaling.  The h33
    element provides and overall scaling factor to the results.  For the
    standard transformation h11, h12, h31, h32 are zero, and h33 is 1.
    Alternatively, the standard transform may be invoked by supplying a matrix
    of shape (n_dimensions, n_dimensions).

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        An array of shape (n_dimensions, shape) where shape may be arbitrary.
    transform : numpy.ndarray (float)
        The transform array of shape (n_dimensions + 1, n_dimensions + 1) to
        fully represent a homography transform.

    Returns
    -------
    transformed_coordinates : numpy.ndarray (float)
        The `coordinates` transformed by `transform` with the same shape as
        `coordinates`.  Note however, that if a one-dimensional `coordinates`
        was supplied, the output will be of shape (1, coordinates.size).
    """
    coordinates = np.atleast_2d(coordinates)
    n_dimensions = coordinates.shape[0]
    n_coordinates = coordinates[0].size
    homographic = is_homography_transform(transform, n_dimensions)
    o = np.zeros((n_dimensions, n_coordinates), dtype=nb.float64)

    # Flatten the input coordinates
    c = np.empty((n_dimensions, n_coordinates), dtype=nb.float64)
    for dim in range(n_dimensions):
        c[dim] = coordinates[dim].ravel()

    if homographic:
        scale_x = transform[n_dimensions, :n_dimensions]
        scale_add = transform[n_dimensions, n_dimensions]
        offset = transform[:n_dimensions, n_dimensions]
    else:
        scale_x = np.empty(0, dtype=nb.float64)
        scale_add = 0.0
        offset = np.empty(0, dtype=nb.float64)

    # Need to do it this way round for precision
    for k in range(n_coordinates):
        for i in range(n_dimensions):
            scaling = 0.0
            xt = 0.0
            for j in range(n_dimensions):
                xj = c[j, k]
                if xj == 0:
                    continue
                xt += transform[i, j] * xj
                if homographic:
                    scaling += scale_x[j] * xj
            if homographic:
                xt += offset[i]
                xt /= scaling + scale_add

            o[i, k] = xt

    # Reshape and set to float
    output = np.empty_like(coordinates, dtype=nb.float64)
    for i in range(n_dimensions):
        dimension_line = output[i].flat
        for k in range(n_coordinates):
            dimension_line[k] = o[i, k]
    return output


@nb.njit(cache=True, parallel=False, fastmath=False)
def warp_terms(terms, coefficients):  # pragma: no cover
    """
    Apply coefficients to polynomial terms.

    Parameters
    ----------
    terms : numpy.ndarray (float)
        The polynomial terms of shape (n_coefficients, n).
    coefficients : numpy.ndarray (float)
        The polynomial coefficients of shape (n_dimensions, n_coefficients,).

    Returns
    -------
    values : numpy.ndarray (float)
        The fitted values of shape (n_dimensions, n).
    """
    n_coefficients, n = terms.shape
    n_dimensions = coefficients.shape[0]
    warped = np.empty((n_dimensions, n), dtype=nb.float64)
    for dimension in range(n_dimensions):
        c_line = coefficients[dimension]
        for i in range(n):
            fit = 0.0
            for j in range(n_coefficients):
                fit += c_line[j] * terms[j, i]

            warped[dimension, i] = fit

    return warped


def estimate_polynomial_transform(source, destination, order=2,
                                  get_exponents=False):
    """
    Estimate the polynomial transform for (x, y) coordinates.

    Parameters
    ----------
    source : numpy.ndarray
        The source coordinates of shape (n_dimensions, n).
    destination : numpy.ndarray
        The destination coordinates of shape (n_dimensions, n).
    order : int, optional
        The polynomial order (number of coefficients is order + 1).
    get_exponents : bool, optional
        If `True`, return the polynomial exponents used to derive the
        coefficients.

    Returns
    -------
    coefficients, [exponents] : numpy.ndarray (float), numpy.ndarray (float)
        The derived polynomial coefficients of shape (n_dimensions, n_coeffs),
        and the optionally returned exponents of shape
        (n_coeffs, n_dimensions).
    """
    exponents = polynomial_exponents(order, ndim=source.shape[0])
    n_coefficients, n_dimensions = exponents.shape
    n = source[0].size

    amat = np.zeros((n * n_dimensions,
                     (n_coefficients * n_dimensions) + 1), dtype=float)

    coordinates = np.empty((n_dimensions, n), dtype=float)
    for i in range(n_dimensions):
        start_row = i * n
        end_row = start_row + n
        coordinates[i] = source[i].ravel()
        amat[start_row:end_row, -1] = destination[i].ravel()

    terms = polynomial_terms(coordinates, exponents).T
    for i in range(n_dimensions):
        start_row = i * n
        end_row = start_row + n
        start_col = i * n_coefficients
        end_col = start_col + n_coefficients
        amat[start_row:end_row, start_col:end_col] = terms

    _, _, v = np.linalg.svd(amat)
    # solution is right singular vector that corresponds to smallest
    # singular value
    coefficients = -v[-1, :-1] / v[-1, -1]
    coefficients = coefficients.reshape(
        (n_dimensions, n_coefficients))

    if get_exponents:
        return coefficients, exponents
    else:
        return coefficients


def warp_coordinates(coordinates, source, destination, order=2):
    """
    Apply the warping between two sets of coordinates to another.

    Parameters
    ----------
    coordinates : numpy.ndarray
        The coordinates to warp of shape (n_dimensions, shape,).
    source : numpy.ndarray
        The reference source coordinates of shape (n_dimensions, shape2,).
        Used in conjunction with `destination` to define the warping transform.
    destination : numpy.ndarray
        The reference destination coordinates of shape (n_dimensions, shape2,).
        Used in conjunction with `source` to define the warping transform.
    order : int, optional
        The order of polynomial used to model the transformation between
        `source` and `destination`.

    Returns
    -------
    warped_coordinates : numpy.ndarray (float)
        The `coordinates` warped by estimating the transform between
        `source` and `destination`.
    """
    coefficients, exponents = estimate_polynomial_transform(
        source, destination, order=order, get_exponents=True)
    terms = polynomial_terms(coordinates, exponents)
    warped_coordinates = warp_terms(terms, coefficients)
    return warped_coordinates


def warp_array_elements(source, destination, shape, order=2,
                        get_transform=False):
    """
    Warp the indices of an array with a given shape using a polynomial.

    Note that all dimensions should be supplied using the Numpy (y, x) ordering
    convention.

    Parameters
    ----------
    source : numpy.ndarray
        The reference source coordinates of shape (n_dimensions, shape2,).
        Used in conjunction with `destination` to define the warping transform.
    destination : numpy.ndarray
        The reference destination coordinates of shape (n_dimensions, shape2,).
        Used in conjunction with `source` to define the warping transform.
    shape : tuple (int)
        The shape of the output array with len(shape) equal to n_dimensions.
    order : int, optional
        The order of polynomial used to model the transformation between
        `source` and `destination`.
    get_transform : bool, optional
        If `True`, return the polynomial transform in addition to the results.

    Returns
    -------
    warped_coordinates, [transform] : np.ndarray (float), PolynomialTransform
        The warped array indices of shape (n_dimensions, shape,).
    """
    n_dimensions = len(shape)
    coordinates = np.empty((n_dimensions,) + tuple(shape), dtype=float)
    # In (y, x) order
    indices = np.indices(shape, dtype=float).reshape(n_dimensions, -1)
    transform = PolynomialTransform(source, destination, order=order)
    warped = transform(indices)
    for i in range(n_dimensions):
        coordinates[i].flat = warped[i]

    if get_transform:
        return coordinates, transform
    else:
        return coordinates


def warp(data, source, destination, order=2,
         interpolation_order=3, mode='constant', output_shape=None,
         cval=np.nan, clip=True, get_transform=False, threshold=0.5):
    """
    Warp an n-dimensional image according to a given coordinate transform.

    Parameters
    ----------
    data : numpy.ndarray
        The data to warp of with n_dimensions of shape (shape,).  The data
        must not contain any NaN values.
    source : numpy.ndarray
        The input coordinates of shape (n_dimensions, shape,).  Dimensions
        should be ordered using the Numpy convention (y, x).
    destination : numpy.ndarray
        The warped coordinates of shape (n_dimensions, shape,).  Dimensions
        should be ordered using the Numpy convention (y, x).
    order : int, optional
        The order of polynomial coefficients used to model the warping.
    interpolation_order : int, optional
        The order of interpolation.
    mode : str, optional
        May take values of {'constant', 'edge', 'symmetric', 'reflect',
        'wrap'}.  Points outside the boundaries of the input are filled
        according to the given mode.  Modes match the behavior of
        :func:`np.pad`.
    output_shape : tuple (int), optional
        Shape of the output array generated.  By default the shape of the
        input array is preserved.  Dimensions should be ordered using the
        Numpy convention (y, x).
    cval : float, optional
        Used in conjunction with the 'constant' mode, and is the value set
        outside the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input array.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    get_transform : bool, optional
        If `True`, return the polynomial transform in addition to the results.

    Returns
    -------
    warped : numpy.ndarray (float)
        The warped data of shape (shape,).
    """
    data = data.astype(float)
    if output_shape is None:
        output_shape = data.shape

    warped_coordinates, transform = warp_array_elements(
        source, destination, output_shape, order=order, get_transform=True)

    warped = map_coordinates(
        data, warped_coordinates, order=interpolation_order, mode=mode,
        cval=cval, clip=clip, threshold=threshold)

    if get_transform:
        return warped, transform
    else:
        return warped


class PolynomialTransform(ABC):

    def __init__(self, source=None, destination=None, order=2):
        """
        Initialize a polynomial transform.

        Parameters
        ----------
        source : numpy.ndarray, optional
            The source coordinates from which to base the transform of shape
            (n_dimensions, shape,)
        destination : numpy.ndarray, optional
            The destination coordinates from which to base the transform of
            shape (n_dimensions, shape,).
        order : int, optional
            The order of polynomial fit.
        """
        self.coefficients = None
        self.exponents = None
        self.inverse_coefficients = None
        self.order = None
        self.estimate_transform(source, destination, order=order)

    @property
    def ndim(self):
        """
        Return the number of dimensions in the fit.

        Returns
        -------
        int
        """
        if self.coefficients is None:
            return 0
        return self.coefficients.shape[0]

    @property
    def n_coeffs(self):
        """
        Return the number of coefficients for the fit.

        Returns
        -------
        int
        """
        if self.coefficients is None:
            return 0
        return self.coefficients.shape[1]

    def estimate_transform(self, source=None, destination=None, order=2):
        """
        Estimate the transform given source and destination coordinates.

        Parameters
        ----------
        source : numpy.ndarray, optional
            The source coordinates from which to base the transform of shape
            (n_dimensions, shape,)
        destination : numpy.ndarray, optional
            The destination coordinates from which to base the transform of
            shape (n_dimensions, shape,).
        order : int, optional
            The order of polynomial fit.

        Returns
        -------
        None
        """
        self.order = order
        self.coefficients = None
        self.exponents = None
        self.inverse_coefficients = None
        if source is None or destination is None:
            return

        source = np.asarray(source)
        destination = np.asarray(destination)

        if source.ndim == 1:
            source = source[None]
        if destination.ndim == 1:
            destination = destination[None]
        if source.shape != destination.shape:
            raise ValueError(f"Source and destination coordinates do not "
                             f"have the same shape {source.shape} != "
                             f"{destination.shape}.")

        coefficients, exponents = estimate_polynomial_transform(
            source, destination, order=order, get_exponents=True)
        self.coefficients = coefficients
        self.exponents = exponents
        self.inverse_coefficients = estimate_polynomial_transform(
            destination, source, order=order, get_exponents=False)

    def transform(self, coordinates, inverse=False):
        """
        Transform a given set of coordinates using the stored polynomial.

        Parameters
        ----------
        coordinates : numpy.ndarray or float
            The coordinates to transform of shape (n_dimensions, shape,),
            (n_dimensions,) or a float if one-dimensional.
        inverse : bool, optional
            If `True`, perform the inverse transform instead.

        Returns
        -------
        warped_coordinates : numpy.ndarray (float)
        """
        if self.coefficients is None:
            raise ValueError("No polynomial fit has been determined.")

        if inverse:
            coefficients = self.inverse_coefficients
        else:
            coefficients = self.coefficients

        coordinates = np.asarray(coordinates)
        if coordinates.ndim == 0:
            if self.ndim == 1:
                singular_value = True
            else:
                raise ValueError(f"Incompatible input dimensions of "
                                 f"{coordinates.shape} for {self.ndim}-D fit.")
        elif coordinates.ndim == 1:
            if self.ndim > 1:
                if coordinates.size == self.ndim:
                    coordinates = coordinates[:, None]
                    singular_value = True
                else:
                    raise ValueError(f"Incompatible input dimensions of "
                                     f"{coordinates.shape} for {self.ndim}-D "
                                     f"fit.")
            else:
                singular_value = coordinates.shape == ()
        elif coordinates.shape[0] != self.ndim:
            raise ValueError(f"Incompatible input dimensions of "
                             f"{coordinates.shape} for {self.ndim}-D "
                             f"fit.")
        else:
            singular_value = False

        coordinates = np.atleast_2d(coordinates)
        ndim = coordinates.shape[0]

        if coordinates.ndim > 2:
            flat_coordinates = np.empty(
                (coordinates.shape[0], coordinates[0].size), dtype=float)
            for i in range(ndim):
                flat_coordinates[i] = coordinates[i].ravel()
            reshape = True
        else:
            flat_coordinates = coordinates
            reshape = False

        terms = polynomial_terms(flat_coordinates, self.exponents)
        warped_coordinates = warp_terms(terms, coefficients)
        if reshape:
            warped = np.empty(coordinates.shape, dtype=float)
            for i in range(ndim):
                warped[i].flat = warped_coordinates[i]
        else:
            warped = warped_coordinates

        if singular_value:
            if self.ndim > 1:
                return warped[:, 0]
            else:
                return warped.ravel()[0]
        else:
            return warped

    def __call__(self, coordinates, inverse=False):
        """
        Transform a given set of coordinates using the stored polynomial.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates to transform of shape (n_dimensions, shape,).
        inverse : bool, optional
            If `True`, perform the inverse transform instead.

        Returns
        -------
        warped_coordinates : numpy.ndarray (float)
        """
        return self.transform(coordinates, inverse=inverse)
