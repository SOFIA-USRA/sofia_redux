# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
import math
import warnings

from astropy import log
import numba as nb
import numpy as np
from scipy import interpolate
from scipy.ndimage import interpolation
from scipy.spatial.qhull import Delaunay

from sofia_redux.toolkit.utilities.func import nantrim, to_array_shape, stack


__all__ = ['line_shift', 'interpolate_nans', 'spline', 'sincinterp',
           'interp_1d_point', 'interp_1d_point_with_error',
           'interp_error_1d', 'interp_error_nd', 'interp_error',
           'Interpolate', 'tabinv', 'findidx']


def line_shift(y, offset, order=3, missing=np.nan):
    """
    Shift an equally spaced array of data values by an offset

    Required because no Python standard interpolation algorithms
    allow for nan values in the input data without slowing down
    the processing by several orders of magnitude.

    Parameters
    ----------
    y : array_like
        equally spaced input data
    offset : offset
        offset to shift_image data.  Units are the input data spacing
    order : int
        values must be 2-5.
        2-5: spline interpolation of the same order
    missing : int or float
        numpy.nan values are treated as missing and will be ignored
        during the fit.  In the output array, any missing values
        will be replaced by `missing`.
    Returns
    -------
    array_like
        numpy.ndarray.  Will be of numpy.float32 if y was int type
    """
    intype = type(y[0])
    if order == 0:
        offset = int(offset)
        result = interpolation.shift(np.float64(y), offset,
                                     order=0, cval=missing)
        if not (np.isnan(result).any() and np.issubdtype(intype, np.integer)):
            result = intype(result)
        return result

    mask = ~np.isnan(y)
    weights = np.float64(mask)
    result = np.empty_like(weights)
    result.fill(missing)
    ny = int(weights.sum())
    order = order if order < ny else int(ny - 1)
    if order <= 0:
        return result
    yind = np.arange(len(y))
    tck = interpolate.splrep(yind[mask], y[mask], s=0, k=order)
    xout = yind - offset
    valid = (xout >= min(yind[mask])) & (xout <= max(yind[mask]))
    if ~np.any(valid):
        return result
    yout = interpolate.splev(xout[valid], tck)
    result[yind[valid]] = yout
    if not np.issubdtype(intype, np.integer):
        return intype(result)
    else:
        return np.float32(result)


def interpolate_nans(x, y, xout, missing=np.nan, order=3, width=1, tck=False):
    """
    Interpolate values containing NaNs

    Parameters
    ----------
    x : np.array
        independent variable
    y : np.array
        dependent variable
    xout : np.array
        output locations
    missing : int or float, optional
        value to fill when a value cannot be determined
    width : spacing between x values.  Used to determine minimum order
    tck : bool, optional
        if True, returns an array containing the vector limits of knots,
        B-spline coefficients, and the degree of the spline.
    order : int
        spline order

    Returns
    -------
    np.array or tuple
        y interpolated onto xout or tck
    """

    itype = np.asarray(y).dtype
    if not tck:
        result = np.full((len(xout),), missing, dtype=itype)
    else:
        result = [], [], []
    if order < 1 or order > 5:
        log.error("order must be between 1 and 5")
        return result

    mask = ~np.isnan(y)
    nvalid = mask.sum()
    if not np.any(mask):
        return result

    order = order if order < nvalid else nvalid - 1
    if order <= 0:
        return result

    idxvalid = np.arange(len(y))[mask]
    xvalid = x[idxvalid]
    yvalid = y[idxvalid]
    sortidx = np.array(
        [x for _, x in sorted(zip(xvalid, np.arange(nvalid)))])
    sortx = xvalid[sortidx]

    lower = np.array(list(
        map(lambda z: ((xvalid < z) & (xvalid >= z - order * width)).sum(),
            sortx)))
    upper = np.array(list(
        map(lambda z: ((xvalid > z) & (xvalid <= z + order * width)).sum(),
            sortx)))

    # we do not want to fit areas that are not bounded by valid data
    # so define regions that can be fitted successfully using a determined
    # order.  single values will just have to be plonked into a cell
    singles = np.where((lower == 0) & (upper == 0))[0]
    for loner in singles:
        lidx = sortidx[loner]
        if not tck:
            result[idxvalid[lidx]] = yvalid[lidx]
        else:
            result[0].append(xvalid[lidx])
            result[1].append(yvalid[lidx])

    if len(lower) <= 1 or len(upper) <= 1:  # pragma: no cover
        return result

    starts = (lower == 0) & (upper != 0)
    starts[:-1][starts[:-1] & starts[1:]] = False
    stops = (lower != 0) & (upper == 0)
    stops[1:][stops[1:] & stops[:-1]] = False
    box_lower = np.where(starts)[0]
    box_upper = np.where(stops)[0]

    for bl in box_lower:
        bu = box_upper[box_upper > bl].min()
        inds = sortidx[bl: bu + 1]
        outidx = (xout >= xvalid[inds].min()) & (xout <= xvalid[inds].max())
        if not np.any(outidx):
            continue
        fx = xvalid[inds]
        fy = yvalid[inds]
        fidx = fx.argsort()
        fx = fx[fidx]
        fy = fy[fidx]
        maxorder = (lower[bl: bu + 1] + upper[bl: bu + 1]).max()
        maxorder = order if maxorder > order else maxorder
        box_tck = interpolate.splrep(fx, fy, k=maxorder, s=0)

        if tck:
            result[0].extend(box_tck[0])
            result[1].extend(box_tck[1])
            result[2].append(box_tck[2])
        else:
            result[outidx] = interpolate.splev(xout[outidx], box_tck)

    return result


def spline(x, y, xout, sigma=1.0):
    """
    Perform cubic spline (tensioned) interpolation

    Replicates IDL spline function.  There are no standard Python
    Libraries that do this.  This function fits the input points
    exactly allowing flexibility between the points where the
    "tension" of the fit is determined by sigma.

    Parameters
    ----------
    x : array_like of float (N,)
        Independent values.  Values MUST be monotonically increasing
    y : array_like of float (N,)
        Dependent values.
    xout : float or array_like of float (M,)
        New Independent values
    sigma : float, optional
        The amount of "tension" that is applied to the curve.  The
        default value is 1.0.  If sigma is close to 0, (e.g., 0.01),
        then effectively there is a cubic spline fit.  If sigma is
        large, (e.g., greater than 10), then the fit will be like a
        polynomial interpolation

    Returns
    -------
    yout : float or numpy.ndarray of float (M,)
        `y` interpolated at `xout`.

    Notes
    -------
    Author:	Walter W. Jones, Naval Research Laboratory, Sept 26, 1976.
    Reviewer: Sidney Prahl, Texas Instruments.
    Adapted for IDL: DMS, March, 1983.
    CT, RSI, July 2003: Added double precision support and DOUBLE keyword,
    use vector math to speed up the loops.
    CT, RSI, August 2003: Must have at least 3 points.
    Adapted for Python: Dan Perera (USRA), April, 2019
    """
    if sigma < 0.001:
        sigma = 0.001
    x, y = np.array(x).astype(float), np.array(y).astype(float)
    n = x.size
    if x.ndim != 1:
        raise ValueError("x array must have 1 dimension")
    elif y.ndim != 1:
        raise ValueError("y array must have 1 dimension")
    elif n < 3:
        raise ValueError("require at least 3 elements in x and y")
    elif y.size != n:
        raise ValueError("x and y shape mismatch")

    isarr = hasattr(xout, '__len__')
    if not isarr:
        xout = [xout]
    xout = np.array(xout).astype(float)
    if xout.ndim != 1:
        raise ValueError("xout must have 1 dimension")

    yp = np.zeros(2 * n)  # storage

    delx1 = x[1] - x[0]
    right = delx1 < 0
    dx1 = (y[1] - y[0]) / delx1
    delx2 = x[2] - x[1]
    delx12 = x[2] - x[0]

    c1 = -(delx12 + delx1) / delx12 / delx1
    c2 = delx12 / delx1 / delx2
    c3 = -delx1 / delx12 / delx2

    nm1 = n - 1

    slpp1 = c1 * y[0] + c2 * y[1] + c3 * y[2]
    deln = x[nm1] - x[nm1 - 1]
    delnm1 = x[nm1 - 1] - x[nm1 - 2]
    delnn = x[nm1] - x[nm1 - 2]
    c1 = (delnn + deln) / delnn / deln
    c2 = -delnn / deln / delnm1
    c3 = deln / delnn / delnm1
    slppn = c3 * y[nm1 - 2] + c2 * y[nm1 - 1] + c1 * y[nm1]

    sigmap = sigma * nm1 / (x[nm1] - x[0])
    dels = sigmap * delx1
    exps = np.exp(dels)
    sinhs = 0.5 * (exps - (1.0 / exps))
    sinhin = 1.0 / (delx1 * sinhs)
    diag1 = sinhin * (dels * 0.5 * (exps + (1.0 / exps)) - sinhs)
    diagin = 1.0 / diag1
    yp[0] = diagin * (dx1 - slpp1)
    spdiag = sinhin * (sinhs - dels)
    yp[n] = diagin * spdiag

    delx2 = np.diff(x)  # x[1:] - x[:-1]
    dx2 = np.diff(y) / delx2  # (y[1:] - y[:-1]) / delx2
    dels = sigmap * delx2
    exps = np.exp(dels)
    sinhs = 0.5 * (exps - (1.0 / exps))
    sinhin = 1.0 / (delx2 * sinhs)
    diag2 = sinhin * (dels * (0.5 * (exps + (1.0 / exps))) - sinhs)
    diag2[1:] += diag2[:-1]
    diag2[0] = 0
    dx2nm1 = dx2[nm1 - 1]  # need to save this to calculate yp[nm1]
    dx2[1:] -= dx2[:-1]
    dx2[0] = 0
    spdiag = sinhin * (sinhs - dels)

    # Need to do an iterative loop for this part
    for i in range(1, nm1):
        diagin = 1.0 / (diag2[i] - spdiag[i - 1] * yp[i + n - 1])
        yp[i] = diagin * (dx2[i] - spdiag[i - 1] * yp[i - 1])
        yp[i + n] = diagin * spdiag[i]

    diagin = 1.0 / (diag1 - spdiag[nm1 - 1] * yp[n + nm1 - 1])
    yp[nm1] = diagin * (slppn - dx2nm1 - spdiag[nm1 - 1] * yp[nm1 - 1])
    for i in range(n - 2, -1, -1):
        yp[i] -= yp[i + n] * yp[i + 1]

    # find subscript where x[subs] > xout(j) > xx[subs-1]
    subs1 = np.digitize(xout, x[1:nm1], right=right)
    subs = subs1 + 1

    s = x[nm1] - x[0]
    sigmap = sigma * nm1 / s
    del1 = xout - x[subs1]
    del2 = x[subs] - xout
    dels = x[subs] - x[subs1]
    exps1 = np.exp(sigmap * del1)
    sinhd1 = 0.5 * (exps1 - (1.0 / exps1))
    exps = np.exp(sigmap * del2)
    sinhd2 = 0.5 * (exps - (1.0 / exps))
    exps *= exps1
    sinhs = 0.5 * (exps - (1.0 / exps))
    spl = (yp[subs] * sinhd1 + yp[subs1] * sinhd2) / sinhs
    spl += ((y[subs] - yp[subs]) * del1 + (y[subs1] - yp[subs1]) * del2) / dels

    return spl if isarr else spl[0]


def sincinterp(x, y, xout, dampfac=3.25, ksize=21,
               skipsort=False, cval=np.nan):
    """
    Perform a sinc interpolation on a data set

    Parameters
    ----------
    x : array_like of (int or float)
        (shape1,) The independent values of the data
    y : array_like of (int or float)
        (shape1,) The dependent values of the data
    xout : array_like of (int or float) or (int or float)
        (shape2,) The new independent values of the data
    dampfac : int or float, optional
        damping factor for sinc interpolation
    ksize : float or int, optional
        Kernel size used for interpolation.  This should be a positive
        odd integer.  If an even value is passed, then the kernel size
        will be increased by one.  Float values will be floored.
    skipsort : bool, optional
        By default, x and y data are sorted by the x value.  This can
        be a fairly intensive operation, so if you know the data is
        already sorted, then set `skipsort` to skip sorting.  Note, this
        is dangerous as no error will be reported.  Be sure you have
        a sorted `x` before attempting.
    cval: float, optional
        Value to fill in requested interpolation points outside the
        data range.

    Returns
    -------
    numpy.ndarray of float
        (shape2,) The new dependent values of the data
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 1:
        raise ValueError("ksize must be a positive (odd) integer")

    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    if x.shape != y.shape:
        raise ValueError("x and y array shape mismatch")

    isarr = hasattr(xout, '__len__')
    if not isarr:
        xout = [xout]
    xout = np.array(xout).astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    yout = np.full(xout.shape, cval)
    if mask.sum() < 2:  # cannot interpolate
        return yout

    x, y = x[mask], y[mask]
    if not skipsort:
        xsort = np.argsort(x)
        x, y = x[xsort], y[xsort]
    xx = tabinv(x, xout)
    ix = xx.astype(int)

    # Values directly on points
    fx = xx - ix
    onpoint = fx == 0
    if onpoint.any():
        yout[onpoint] = np.take(y, ix[onpoint])

    # use sinc interpolation for the points having fractional values
    if not onpoint.all():
        dx = fx[~onpoint]
        kernel = np.arange(ksize) - (ksize // 2)
        kx = np.resize(kernel, (dx.size, kernel.size)).T
        lobe = kx.copy()
        kx = kx - dx
        sinc = np.exp(-(kx / dampfac) ** 2) * np.sinc(kx)
        lobe = np.clip(lobe + ix[~onpoint], 0, x.size - 1)
        vals = np.take(y, lobe)
        yout[~onpoint] = np.sum(sinc * vals, axis=0)

    if not isarr:
        yout = yout[0]

    return yout


@nb.njit(fastmath=True, cache=True, nogil=True)
def interp_1d_point(x, y, xout):  # pragma: no cover
    """
    Perform linear interpolation at a single point. `x` must be monotonically
    increasing or decreasing containing unique values.

    Superfast compared to anything else including `numpy.interp`.

    Parameters
    ----------
    x : array_like of float
        (N,) array of independent values.  Must be monotonically
        increasing or decreasing.
    y : array_like of float
        (N,) array of dependent values.
    xout : float
        New independent value.

    Returns
    -------
    yout : float
        The new dependent values at `xout`.
    """
    # Check for direction
    if x[0] > x[-1]:  # this is ok - just a view, not copy
        x = x[::-1]
        y = y[::-1]

    if x[0] > xout:
        return np.nan
    elif x[-1] < xout:
        return np.nan

    if x[0] == xout:
        return y[0]

    right = -1
    for i in range(1, x.size):
        if x[i] >= xout:
            right = i
            break

    rx = x[right]
    if rx == xout:
        return y[right]
    left = right - 1

    lx = x[left]
    ly = y[left]
    ry = y[right]
    dy = ry - ly

    weight = (xout - lx) / (rx - lx)
    return ly + (weight * dy)


@nb.njit(fastmath=True, cache=True, nogil=True)
def interp_1d_point_with_error(x, y, error, xout):  # pragma: no cover
    """
    Perform linear interpolation at a single point with error propagation.
    `x` must be monotonically increasing or decreasing containing unique
    values.

    Superfast compared to anything else including `numpy.interp`.

    Parameters
    ----------
    x : array_like of float
        (N,) array of independent values.  Must be monotonically
        increasing or decreasing.
    y : array_like of float
        (N,) array of dependent values.
    error : array_like of float
        (N,) array of errors in dependent values.
    xout : float
        New independent value.

    Returns
    -------
    yout, eout : float, float
        The new dependent values and error at `xout`.
    """
    # Check for direction
    if x[0] > x[-1]:  # this is ok - just a view, not copy
        x = x[::-1]
        y = y[::-1]
        error = error[::-1]

    if x[0] > xout:
        return np.nan, np.nan
    elif x[-1] < xout:
        return np.nan, np.nan

    if x[0] == xout:
        return y[0], error[0]

    right = -1
    for i in range(1, x.size):
        if x[i] >= xout:
            right = i
            break

    rx = x[right]
    if rx == xout:
        return y[right], error[right]
    left = right - 1

    lx = x[left]
    ly = y[left]
    ry = y[right]
    dy = ry - ly
    weight = (xout - lx) / (rx - lx)
    yout = ly + (weight * dy)

    lv = error[left]
    lv *= lv
    rv = error[right]
    rv *= rv

    vsum = lv + rv
    lv += vsum * weight * weight
    right_weight = 1.0 - weight
    rv += vsum * right_weight * right_weight
    if lv < rv:
        eout = math.sqrt(lv)
    else:
        eout = math.sqrt(rv)

    return yout, eout


def interp_error_1d(x, error, xout, cval=np.nan):
    """
    Perform linear interpolation of errors

    Parameters
    ----------
    x : array_like of float
        (N,) array of independent values.
    error : array_like of float
        (N,) array of errors in dependent values.
    xout : int or float or (array_like of (int or float))
        New independent values.  Either a scalar or array of shape (M,)
    cval : float, optional
        Value to fill in requested interpolation points outside the
        data range.

    Returns
    -------
    float or numpy.ndarray
        The new dependent error values at `xout`.  Will be either a
        float or array of shape (M,) and numpy.float64 type depending
        on the supplied `xout`.
    """
    isarr = hasattr(xout, '__len__')
    if not isarr:
        xout = [xout]

    x, error, xout = np.array(x), np.array(error), np.array(xout)
    shape, oshape = x.shape, xout.shape

    if error.shape != shape:
        raise ValueError("x and error shape mismatch")

    var = error ** 2
    error = np.full(oshape, float(cval))

    # determine index of xout on x
    idx = findidx(x, xout, left=np.nan, right=np.nan)
    mask = np.isfinite(idx)
    if not mask.any():
        return error

    xout, eout = xout[mask], np.empty(mask.sum())
    left = np.floor(idx[mask]).astype(int)
    right = np.ceil(idx[mask]).astype(int)

    # where no interpolation is required, just copy the values over
    pmask = left == right
    eout[pmask] = var[left][pmask]

    # now look at the points where interpolation is required
    pmask = np.logical_not(pmask, out=pmask)
    if pmask.any():
        left, right, xout = left[pmask], right[pmask], xout[pmask]
        vsum = var[left] + var[right]
        weights = (xout - x[left]) / (x[right] - x[left])
        vleft = var[left] + ((weights ** 2) * vsum)
        vright = var[right] + (((1 - weights) ** 2) * vsum)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # take the smallest (not closest) value
            eout[pmask] = np.clip(vleft, 0, vright)

    error[mask] = np.sqrt(eout)
    if not isarr:
        error = error[0]

    return error


def interp_error_nd(points, error, xi, cval=np.nan):
    """Propagate errors using Delaunay triangulation in N-dimensions

    Uses `interp_error_1d` to propagate errors in 1-dimension following
    linear interpolation, or propagates errors in N-dimensions when Delaunay
    triangulation has been used for linear interpolation.

    This is not accurate for any other type of interpolation other than the
    types mentioned above.

    Parameters
    ----------
    points : numpy.ndarray (nsamples, ndim) or Delaunay triangulation
        Coordinates of points or the pre-computed scipy.spatial.Delaunay
        triangulation.
    error : float or numpy.ndarray  (nsamples,)
        Error at each point.
    xi : numpy.ndarray (npoints, ndim)
        Points at which to interpolate data.
    cval : float, optional
        Value to fill in requested interpolation points outside the
        data range.

    Returns
    -------
    numpy.ndarray (npoints,)
    """
    tri = points if isinstance(points, Delaunay) else Delaunay(points)
    ndim = tri.points.shape[1]
    simplices = tri.find_simplex(xi)
    mask = simplices != -1
    result = np.full(xi.shape[0], cval)
    if not mask.any():
        return result

    vertices = tri.vertices[simplices[mask]]

    if not hasattr(error, '__len__'):
        error = to_array_shape(float(error), np.max(vertices) + 1)

    # Find points that do not require interpolation
    vpoints = tri.points[vertices]
    xi_find = xi[mask]
    same_points = np.all(xi_find[:, None] == vpoints, axis=2)
    same = same_points.any(axis=1)
    if same.any():
        values = result[mask]
        values[same] = error[vertices[same_points]]
        result[mask] = values
        mask[same] = False
        if not mask.any():
            return result
        vinds = tri.vertices[simplices[mask]]

    else:
        vinds = vertices

    var = error ** 2
    var = var[vinds]

    transform = tri.transform[simplices[mask]]
    b = np.einsum('ijk,ik->ij',
                  transform[:, :ndim, :ndim],
                  xi[mask] - transform[:, ndim, :])

    weights = np.c_[b, 1 - b.sum(axis=1)]
    v0 = var + (weights ** 2) * var
    v1 = var + ((1 - weights) ** 2) * var
    result[mask] = np.sqrt(np.min([v0, v1], axis=0).sum(axis=1))
    return result


def interp_error(points, error, xi, cval=np.nan):
    """Propagate errors using linear interpolation in N-dimensions

    Allows linear interpolation of errors over 1 or multiple dimensions
    using `interp_error_1d` or `interp_error_nd`.  Note, error propagation
    is only valid for linear interpolation in 1-dimension or linear
    interpolation through Delaunay triangulation in N-dimensions.

    Parameters
    ----------
    points : numpy.ndarray (nsamples, ndim) or Delaunay triangulation
        Coordinates of points or the pre-computed scipy.spatial.Delaunay
        triangulation.
    error : float or numpy.ndarray  (nsamples,)
        Error at each point.
    xi : numpy.ndarray (npoints, ndim)
        Points at which to interpolate data.
    cval : float, optional
        Value to fill in requested interpolation points outside the
        data range.

    Returns
    -------
    numpy.ndarray (npoints,)
    """
    if isinstance(points, Delaunay):
        return interp_error_nd(points, error, xi, cval=cval)

    points = np.asarray(points)
    if points.ndim == 1:
        return interp_error_1d(points, error, xi, cval=cval)
    elif points.ndim == 2:
        if points.shape[1] == 1:
            return interp_error_1d(points[:, 0], error, xi, cval=cval)
        else:
            return interp_error_nd(points, error, xi, cval=cval)
    else:
        raise ValueError(
            "points must be a 1 (npoints,) or 2 (npoints, ndim) array "
            "unless supplying the Delaunay triangulation")


class Interpolate(object):
    """
    Fast interpolation on a regular grid

    Much like scipy.interpolate.RegularGridInterpolator except better.
    Allows for cubic interpolation, omission of grid coordinates,
    and NaN handling.

    Parameters
    ----------
    args : array_like or tuple of array_like
        Either a single array whose coordinates will be determined
        by the features, or arrays of independent values followed
        by the dependent values.
    method : str, optional
        One of {'linear', 'cubic', 'nearest'}
    cubic : float, optional
        Defines the value of "a" below for the fast approximation of
        the bicubic interpolator.  a = -0.5 produces third order
        convergence with respect to the sampling interval.  The
        convolution weights are defined as::

            w(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1 for |x|<=1,

                   a|x|^3 - 5a|x|^2 + 8a|x| - 4a for 1<|x|<2,

                   0 otherwise

        If method is None, setting cubic to a float will result in
        cubic interpolation using the above weightings in each
        dimension.  The default value for `cubic` is -0.5.
    mode : str, optional
        One of {'nearest', 'reflect', 'mirror', 'wrap', 'constant'}
        which determines how edge conditions are handled when the
        interpolation kernel overlaps a border.  Valid values and
        behaviours are as follows:

        - 'nearest' (a a a a | a b c d | d d d d):
          The input is extended by replicating the last pixel

        - 'reflect' (d c b a | a b c d | d c b a):
          The input is extended by reflecting about the edge of
          the last pixel.

        - 'mirror' (d c b | a b c d | c b a):
          The input is extended by reflecting about the center of
          last pixel.

        - 'wrap' (a b c d | a b c d | a b c d):
          The input is extended by wrapping around the opposite
          edge.

        - `constant` (k k k k | a b c d | k k k k):
           The input is extended by filling all values beyond the
           edge with the same constant value defined by the `cval`
           parameter.

    cval : float, optional
        The value used to fill values when `mode` is 'constant'.
    ignorenans : bool, optional
        If True, NaNs will be ignored in all calculations where possible.
    """
    def __init__(self, *args, method=None, cval=np.nan, cubic=None,
                 ignorenans=True, error=None, mode='constant'):

        self._valid_modes = [
            'nearest', 'reflect', 'mirror', 'wrap', 'constant']
        self._valid_methods = ['linear', 'cubic', 'nearest']

        if method is None:
            method = 'linear' if cubic is None else 'cubic'
        if method not in self._valid_methods:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        if method == 'cubic' and cubic is None:
            cubic = -0.5
        self.cubic = cubic

        if mode not in self._valid_modes:
            raise ValueError("Edge mode '%s' is not available" % mode)
        self.mode = mode
        self._product = np.nanprod if ignorenans else np.prod
        self._sum = np.nansum if ignorenans else np.sum
        self._min = np.nanmin if ignorenans else np.min
        self.cval = cval
        self.values = None
        self.variance = None
        self.do_error = False
        self.ndim = None
        self.grid = ()
        self.dgrid = ()
        self.parse_arguments(*args, error=error)

    def parse_arguments(self, *args, error=None):
        """
        Parse initialization arguments.

        Parameters
        ----------
        args : array_like or tuple of array_like
            Either a single array whose coordinates will be determined
            by the features, or arrays of independent values followed
            by the dependent values.
        error : numpy.ndarray, optional
            The associated error values for the data.

        Returns
        -------
        None
        """
        nargs = len(args)
        if nargs == 1:
            values = np.asarray(args[0])
            points = tuple(np.arange(side) for side in values.shape[::-1])
        else:
            values, points = np.asarray(args[-1]), args[:-1][::-1]
            if len(points) != values.ndim:
                raise ValueError(
                    "There are %d grid arrays, but values has %d features"
                    % (len(points), values.ndim))

        self.ndim = values.ndim
        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        fill_value_dtype = np.asarray(self.cval).dtype
        if (hasattr(values, 'dtype') and not
                np.can_cast(fill_value_dtype, values.dtype,
                            casting='same_kind')):
            raise ValueError("cval must be  of a type compatible with values")

        self.grid = self.dgrid = ()
        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d grid points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
            g = np.asarray(p)
            d = np.empty(g.shape)
            d[:-1] = np.diff(g)
            d[-1] = d[-2]
            self.grid += g,
            self.dgrid += d,

        self.set_values_and_error(values, error=error)

    def set_values_and_error(self, values, error=None):
        """
        Set new interpolating values and error.

        Parameters
        ----------
        values : numpy.ndarray (int or float)
            The new values to set.  Must be the same shape as the interpolation
            grid.
        error : numpy.ndarray (int or float), optional
            Optional error values to set.

        Returns
        -------
        None
        """
        self.set_values(values)
        if error is not None:
            self.variance = np.full_like(self.values, np.nan)
            self.variance[(slice(0, -1),) * values.ndim] = np.asarray(
                error ** 2)
            self.do_error = True
        else:
            self.variance = None
            self.do_error = False

    def set_values(self, values):
        """
        Reset the interpolation values only.

        Parameters
        ----------
        values : numpy.ndarray, optional
            The new values to set.  Must be the same shape as the interpolation
            grid.

        Returns
        -------
        None
        """
        self.values = np.full([s + 1 for s in values.shape], self.cval,
                              dtype=values.dtype)
        self.values[(slice(0, -1),) * values.ndim] = values

    def __call__(self, *args, method=None, cubic=None, mode=None):
        """
        Interpolation at coordinates.

        Parameters
        ----------
        args : array_like or tuple of array_like
            The coordinates to sample the gridded data at.  The order of
            features follow the 'xy' rather than 'ij' convention.  For
            example, arguments for two-dimensional data should be supplied
            in the order (x, y).
        method : str
            The method of interpolation to perform.  One of {'linear', 'cubic',
            'nearest'}.
        """
        if method is None:
            method = self.method if cubic is None else 'cubic'
        if method not in self._valid_methods:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method

        if self.method == 'cubic':
            if cubic is not None:
                self.cubic = cubic
            elif self.cubic is None:
                self.cubic = -0.5

        if mode is None:
            mode = self.mode
        if mode not in self._valid_modes:
            raise ValueError("Edge mode %s is not available" % mode)
        self.mode = mode

        if len(args) != self.ndim:
            raise ValueError(
                "require %i input arguments for %i-D data" %
                (self.ndim, self.ndim))

        shape_in = np.asarray(args[0]).shape
        xi = stack(*args, copy=False)[::-1]

        indices, norm_distances = self._find_indices(xi)
        result = None
        if method == 'linear':
            result = self._evaluate_linear(indices, norm_distances)
        elif method == 'nearest':
            result = self._evaluate_nearest(indices, norm_distances)
        elif method == 'cubic':
            result = self._evaluate_cubic(indices, norm_distances)

        return result.reshape(shape_in)

    @staticmethod
    def cubic_weights(distances, a=-0.5):
        d = np.abs(np.asarray(distances))
        w = np.zeros(d.shape)
        i = d <= 1
        w[i] = ((a + 2) * d[i] ** 3) - ((a + 3) * d[i] ** 2) + 1
        i = (d > 1) & (d < 2)
        w[i] = (a * d[i] ** 3) - (5 * a * d[i] ** 2) + (8 * a * d[i]) - (4 * a)
        return w

    def _evaluate_cubic(self, indices, norm_distances, cubic=-0.5):
        indset = itertools.product(*indices)
        weights = itertools.product(*[self.cubic_weights(d, a=cubic)
                                      for d in norm_distances])
        s = indices[0].shape[0] ** self.ndim, indices[0].shape[1]
        values = np.empty(s, dtype=self.values.dtype)
        for vals, inds, wts in zip(values, indset, weights):
            vals[:] = self.values[inds] * self._product(wts, axis=0)

        return self._sum(values, axis=0)

    def _evaluate_linear(self, indices, norm_distances):
        indset = itertools.product(*indices)
        weights = itertools.product(*[abs(1 - d) for d in norm_distances])
        s = indices[0].shape[0] ** self.ndim, indices[0].shape[1]
        values = np.empty(s, dtype=self.values.dtype)

        for vals, inds, wts in zip(values, indset, weights):
            vals[:] = self.values[inds] * self._product(wts, axis=0)
        return self._sum(values, axis=0)

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = []
        select_points = np.arange(indices[0].shape[1])
        for i, yi in zip(indices, norm_distances):
            idx = np.argmin(yi, axis=0)
            idx_res.append(i[idx, select_points])
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant interpolants for xi
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []

        dx = 2 if self.method == 'cubic' else 1
        offsets = np.arange(-dx, dx)

        # iterate through features
        for x, grid, dgrid in zip(xi, self.grid, self.dgrid):
            highlim = grid.size - 1
            i = np.searchsorted(grid, x)
            l0 = i == 0
            h0 = i == grid.size
            i = i[:, None] + offsets[None]
            inds = np.clip(i, 0, highlim)
            d = (x[:, None] - grid[inds]) / dgrid[inds]
            d = abs(d[:, dx]) % 1
            d[l0] = 0
            d[h0] = 1
            d = abs(d[:, None] + offsets)

            low = i < 0
            high = i > highlim

            if self.mode == 'nearest':
                pass

            elif self.mode == 'reflect':
                inds[low] = abs(i[low])
                inds[high] = 2 * highlim - i[high]

            elif self.mode == 'mirror':
                inds[low] = abs(i[low] + 1)
                inds[high] = 2 * highlim - i[high] + 1

            elif self.mode == 'wrap':
                inds[low] = highlim + i[low] + 1
                inds[high] = i[high] - highlim - 1

            elif self.mode == 'constant':
                low |= l0[:, None]
                high |= h0[:, None]
                inds[low] = -1
                inds[high] = -1

            else:
                raise ValueError("unknown edge mode (%s)" % self.mode)

            indices.append(inds.T)
            norm_distances.append(d.T)

        return indices, norm_distances


def tabinv(array, xvals, missing=None, fast=True):
    """
    Find the effective index of a function value in an ordered vector with
    NaN handling.

    Parameters
    ----------
    array : array_like
        The array to be searched.  Should be monotonic increasing or
        decreasing.
    xvals : the function value(s) whose effective index is sought
    missing : float or int, optional
        Value to return if outside the limits.  Default uses constant
        value of the first or last value in the array.  Only pertinent
        if `fast` is True.
    fast : bool, optional
        If False, uses `findidx` to check if x in monotonic and does
        not allow extrapolation beyond the limits of `array`.  NaNs
        will break monotonic check if not limited to padding on the
        edges of `array`.  If fast is True, np.interp is used without
        any form of error checking.

    Returns
    -------
    numpy.ndarray
        The effective index or indices of `array`

    Examples
    --------
    >>> from sofia_redux.toolkit.interpolate.interpolate import tabinv
    >>> import numpy as np
    >>> import pytest
    >>> x = [np.nan, np.nan, 1, 2, np.nan, 3, np.nan, np.nan]
    >>> tabinv(x, 1.5)
    2.5
    >>> with pytest.raises(ValueError):
    ...     tabinv(x, 1.5, fast=False)
    ...
    >>> x = [np.nan, np.nan, 1, 2, 3, np.nan, np.nan]
    >>> tabinv(x, 1.5, fast=False)
    2.5
    """
    array = np.asarray(array, dtype=float)
    if array.ndim != 1:
        raise ValueError("Array must have 1-dimension")
    if fast:
        return np.interp(xvals, array, np.arange(len(array)),
                         left=missing, right=missing)
    else:
        idx = nantrim(array, 2, bounds=True)
        return findidx(array[idx[0]: idx[1]], xvals) + idx[0]


def findidx(ix, ox, left=0, right=None):
    """
    Finds the effective index of a function value in an ordered array.

    Formerly tabinv.  findidx will abort if xi is not monotonic.
    Equality of neighboring values in xi is allowed but results may
    not be unique.  This requirement may mean that xi padded with
    zeroes could cause findidx to abort.

    A binary search is used to find the values xi[i] and xi[i+1] where
    xi[i] < xo < xi[i+1].  Output (ieff) is then computed using linear
    interpolation between i and i+1::

        ieff = i + (xo - xi[i]) / (xi[i+1] - xi[i])

    Let n = number of elements in xi::
        if (x < xi[0]) or (x > xi[n-1]) then xo = NaN

    Parameters
    ----------
    ix : array_like of float
        (N,) The array to be searched, must be monotonic increasing or
        decreasing.
    ox : (array_like of float) or float
         (M,) The function value(s) whose effective index is sought.
    left : float, optional
        Value to return for ox < min(ix).  default is 0
    right : float, optional
        Value to return for ox > max(ix).  default is len(ix) - 1

    Returns
    -------
    float or numpy.ndarray
        (M,) The effective index or indices of xo in xi.  Note that
        output type will be float and will need to changed to
        integer in order to be used for indexing another array.
    """
    ix = np.asarray(ix)
    n = ix.size

    if np.isnan(ix).any():
        raise ValueError("NaNs detected: cannot test for mononcity")

    if hasattr(ox, '__len__'):
        ox = np.array(ox)
        shape = ox.shape
        ox = ox.ravel()
    else:
        ox = np.array([ox])
        shape = None

    if n <= 1:
        return 0.0 if shape is None else np.zeros(shape)

    # Initialize binary search area and compute number of divisions needed
    ileft, iright = np.zeros((2, ox.size), dtype=int)
    ndivisions = int(np.log10(n) / np.log10(2) + 1)

    # Test for monotonicity
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        i = (ix - np.roll(ix, 1))[1:]
        a = i >= 0
        if a.sum() == (n - 1):  # test for increasing array
            np.add(iright, n - 1, out=iright)
        else:  # test for decreasing array
            a = i <= 0
            if a.sum() == (n - 1):
                np.add(ileft, n - 1, out=ileft)
            else:
                raise ValueError("xi is not monotonic")

        # Perform binary search by successively dividing search
        # interval in half
        for i in range(ndivisions):
            idiv = (ileft + iright) // 2
            xval = ix[idiv]
            less, greater = ox <= xval, ox > xval
            np.multiply(ileft, less, out=ileft)
            np.add(ileft, idiv * greater, out=ileft)
            np.multiply(iright, greater, out=iright)
            np.add(iright, idiv * less, out=iright)

        # Interpolate between interval of width = 1
        # value on left and right sides
        xleft, xright = ix[ileft], ix[iright]
        iszero = xleft == xright
        ieff = ((xright - ox) * ileft).astype(float)
        np.add(ieff, (ox - xleft) * iright, out=ieff)
        np.add(ieff, iszero * ileft, out=ieff)
        np.divide(ieff, (xright - xleft + iszero), out=ieff)

        # do not allow extrapolation beyond ends
        np.clip(ieff, 0, n - 1, out=ieff)
        past_edge = ox < np.nanmin(ix)
        ieff[past_edge] = left
        past_edge[...] = ox > np.nanmax(ix)
        if right is None:
            ieff[past_edge] = n - 1
        else:
            ieff[past_edge] = right

        return ieff[0] if shape is None else np.reshape(ieff, shape)
