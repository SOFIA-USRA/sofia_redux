# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from skimage.transform import estimate_transform, warp

from sofia_redux.toolkit.fitting.polynomial import poly2d
from sofia_redux.toolkit.interpolate.interpolate import Interpolate


__all__ = ['warp_image', 'polywarp', 'polywarp_image']


def warp_image(data, xin, yin, xout, yout, order=3,
               transform='polynomial', interpolation_order=3,
               mode='constant', cval=np.nan, output_shape=None,
               get_transform=False, missing_frac=0.5,
               extrapolate=True):
    """
    Warp data using transformation defined by two sets of coordinates

    Parameters
    ----------
    data : np.ndarray
        input data (nrow, ncol)
    xin : array-like
        source x coordinate
    yin : array-like
        source y coorindate
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
    transform : str, optional
        See scikit.image.transform for all available transform types
    get_transform : bool, optional
        if True, return the transformation object as the second element
        of a 2-tuple in the return value
    missing_frac : float, optional
        value between 0 and 1.  1 = fully weighted by real values.
        0 = fully weighted by NaNs.  Any pixel weighted by less than
        this fraction will be replaced with cval.

    Returns
    -------
    numpy.ndarray
        data (nrow, ncol)
        data, transform (if get_transform=True)
    """

    xi, yi = np.array(xin).ravel(), np.array(yin).ravel()
    xo, yo = np.array(xout).ravel(), np.array(yout).ravel()
    minx, maxx = int(np.ceil(np.min(xo))), int(np.floor(np.max(xo)))
    miny, maxy = int(np.ceil(np.min(yo))), int(np.floor(np.max(yo)))

    if transform != 'polynomial':
        # Add corner pins to allow extrapolation
        xc = [0, 0, data.shape[1] - 1, data.shape[1] - 1]
        yc = [0, data.shape[1] - 1, 0, data.shape[0] - 1]
        xi, yi = np.append(xi, xc), np.append(yi, yc)
        xo, yo = np.append(xo, xc), np.append(yo, yc)

    co = np.stack((xo, yo), axis=1)
    ci = np.stack((xi, yi), axis=1)
    if transform == 'polynomial':
        # This is the inverse transform (out -> in)
        coordinate_transform = estimate_transform(
            'polynomial', co, ci, order=order)
        func = coordinate_transform
    else:
        coordinate_transform = estimate_transform(transform, co, ci)
        func = coordinate_transform

    nans = np.isnan(data)
    if not nans.any():
        dout = warp(data.copy(), func, order=interpolation_order,
                    mode=mode, cval=cval, preserve_range=True,
                    output_shape=output_shape)
        wout = (~np.isnan(dout)).astype(float)
    else:
        weights = (~nans).astype(float)
        dw = data.copy()
        dw[nans] = 0
        wout = warp(weights, func, order=interpolation_order,
                    mode=mode, cval=cval, preserve_range=True,
                    output_shape=output_shape)
        dout = warp(dw, func, order=interpolation_order,
                    mode=mode, cval=cval, preserve_range=True,
                    output_shape=output_shape)
        wout[np.isnan(wout)] = 0
        dividx = np.abs(wout) >= missing_frac
        dout[dividx] = dout[dividx] / wout[dividx]
    cutidx = np.abs(wout) < missing_frac
    dout[cutidx] = cval

    if not extrapolate:
        dout[:miny, :] = cval
        dout[:, :minx] = cval
        dout[maxy:, :] = cval
        dout[:, maxx:] = cval

    if get_transform:
        return dout, func
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
