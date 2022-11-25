# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numba as nb
import numpy as np

from sofia_redux.instruments.exes.tortcoord import tortcoord
from sofia_redux.toolkit.image.utilities import map_coordinates

__all__ = ['tort']


def tort(data, header, variance=None, skew=False,
         order=3, missing=np.nan, badfrac=0.5, get_illum=False,
         interpolation_method='spline'):
    """
    Correct image for optical distortion.

    Calls `exes.tortcoord` to calculate undistorted coordinates from
    raw pixel coordinates and distortion parameters in the header.  The
    image is then interpolated onto the undistorted coordinates.  If
    the variance is provided, it is also interpolated onto undistorted
    coordinates.

    Parameters
    ----------
    data : numpy.ndarray
        [nframe, nspec, nspat] cube or [nspec, nspat] image.
    header : fits.Header
        FITS header with distortion parameters set.
    variance : numpy.ndarray, optional
        Variance array matching `data` dimensions.
    skew : bool, optional
        If True, correct for echelon slit skewing.
    order : int, optional
        Order of interpolation (0-5).  0 is nearest value, 1 is
        linear interpolation, 3 is cubic interpolation.
    missing : float, optional
        Value to use in output frames for missing data.
    badfrac : float, optional
        The allowable fraction of NaN-valued points used to generate an
        interpolated value.  If this fraction is greater than `badfrac`,
        it will be replaced by `missing`.  Note that all NaNs are replaced by
        zero before interpolation.
    get_illum : bool, optional
        If set, an undistorted illumination mask will be returned.
    interpolation_method : {'spline', 'convolution'}, optional
        If 'convolution', a separable parametric cubic convolution algorithm
        is applied, replicating the original IDL interpolation method for the
        EXES pipeline.  If 'spline', the interpolation is performed via
        piecewise splines of the specified order.

    Returns
    -------
    tortdata, [tortvar], [tortillum] : numpy.ndarray or tuple of numpy.ndarray
        The corrected data and optional variance and illumination arrays
        (if supplied and/or requested).  Will be of shape
        [nframe, nspec, nspat] if nframe > 1; [nspec, nspat] otherwise.
    """
    # get original data dimensions
    ndim = data.ndim

    # make sure data has 3 dimensions and matches variance
    # also, make a flat illumination array
    data_ok, data, variance, illum = _check_data(
        data, header, variance=variance)
    if not data_ok:
        raise ValueError('Data cannot be corrected.')

    # convert raw coordinates to undistorted coordinates
    # (including echelon slit skewing if desired)
    u, v = tortcoord(header, skew=skew)
    tortdata = np.empty(data.shape, dtype=np.float64)

    dovar = variance is not None
    tortvar = np.empty_like(tortdata) if dovar else None
    tortillum = np.empty_like(tortdata) if get_illum else None

    if interpolation_method == 'spline':
        map_function = map_coordinates
    else:
        map_function = _cubic_convolution

    for i in range(data.shape[0]):
        tortdata[i] = map_function(data[i], np.array([v, u]),
                                   order=order, cval=missing,
                                   threshold=badfrac)
        if dovar:
            # same as data
            tortvar[i] = map_function(variance[i], np.array([v, u]),
                                      order=order, cval=missing,
                                      threshold=badfrac)
        if get_illum:
            # missing value should be -1, but it may need NaN handling
            # to match data
            if np.isnan(missing):
                cval = np.nan
            else:
                cval = -1
            tortillum[i] = map_function(illum[i].astype(float),
                                        np.array([v, u]),
                                        order=order, cval=cval,
                                        threshold=badfrac, clip=False)
            tortillum[i][np.isnan(tortillum[i])] = -1

    if ndim == 2:
        tortdata = tortdata[0]
        if dovar:
            tortvar = tortvar[0]
        if get_illum:
            tortillum = tortillum[0]

    if not dovar and not get_illum:
        return tortdata

    result = tortdata,
    if dovar:
        result = result + (tortvar,)

    if get_illum:
        result = result + (np.round(tortillum).astype(int),)

    return result


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def _cubic_convolution(data, coords, alpha=-0.5, cval=np.nan,
                       threshold=0.5, order=3, clip=True):  # pragma: no cover
    """
    Separable 2D parametric cubic convolution interpolation.

    Directly replicates behavior of the IDL interpolate function, used in
    the original version of this pipeline.

    Following a formulation in:
    Reichenbach, Stephen E. and Geng, Frank,
    "Two-Dimensional Cubic Convolution" (2003). CSE Journal Articles. 15.

    data : numpy.ndarray
        The data array to map with n_dimensions dimensions with a size of N.
    coordinates : numpy.ndarray
        The coordinates at which `data` is evaluated.  Must be of shape
        (n_dimensions, N). Dimensions are ordered using the Numpy (y, x)
        convention.
    order : int, optional
        Included for compatibility with map_coordinates inputs; not used
        in this function.
    cval : float, optional
        The value to set for missing data.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    threshold : float, optional
        Points with total weight >= `threshold` are considered valid;
        others will be set to cval in the output.

    Returns
    -------
    mapped_coordinates : numpy.ndarray
        The result of transforming the data. The shape of the output is
        derived from that of `coordinates` by dropping the first axis.
    """
    ny, nx = data.shape
    result = np.full((ny, nx), cval, dtype=nb.float64)
    minval = np.nanmin(data)
    maxval = np.nanmax(data)

    v, u = coords.astype(nb.float64)
    knots = [-3, -2, -1, 0, 1, 2, 3]
    for i in range(ny):
        for j in range(nx):
            # check for extrapolated point
            if ((v[i, j] <= 0 or v[i, j] >= ny - 1)
                    or (u[i, j] <= 0 or u[i, j] >= nx - 1)):
                continue

            cv = int(round(v[i, j]))
            cu = int(round(u[i, j]))

            r_s = 0
            sum_weight = 0
            for yk in knots:
                # check/pad boundaries
                yi = cv + yk
                if yi < -1 or yi >= ny + 1:
                    continue
                if yi < 0:
                    yi = 0
                elif yi >= ny:
                    yi = ny - 1

                # kernel function in y
                y = abs(v[i, j] - (cv + yk))
                if y <= 1:
                    f0y = 2 * y ** 3 - 3 * y ** 2 + 1
                    f1y = y ** 3 - y ** 2
                elif y <= 2:
                    f0y = 0
                    f1y = y**3 - 5 * y**2 + 8 * y - 4
                else:
                    f0y = 0
                    f1y = 0

                for xk in knots:
                    # check/pad boundaries
                    xi = cu + xk
                    if xi < -1 or xi >= nx + 1:
                        continue
                    if xi < 0:
                        xi = 0
                    elif xi >= nx:
                        xi = nx - 1

                    # data at knot in input image
                    p_m = data[yi, xi]
                    if np.isnan(p_m):
                        continue

                    # kernel function in x
                    x = abs(u[i, j] - (cu + xk))
                    if x <= 1:
                        f0x = 2 * x**3 - 3 * x**2 + 1
                        f1x = x**3 - x**2
                    elif x <= 2:
                        f0x = 0
                        f1x = x**3 - 5 * x**2 + 8 * x - 4
                    else:
                        f0x = 0
                        f1x = 0

                    f_s = (f0x + alpha * f1x) * (f0y + alpha * f1y)
                    r_s += p_m * f_s
                    sum_weight += f_s

            if sum_weight > threshold:
                value = r_s * 1 / sum_weight
                if clip:
                    if value < minval:
                        value = minval
                    elif value > maxval:
                        value = maxval
                result[i, j] = value

    return result


def _check_data(data, header, variance=None):
    """Check input data dimensions."""
    hnx = header['NSPAT']
    hny = header['NSPEC']

    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[None]
    nframes, ny, nx = shape = data.shape
    illum = np.ones(shape, dtype=int)

    if (ny, nx) != (hny, hnx):
        log.error("Data (y, x) dimensions do not match header: "
                  "%s != %s" % (repr((ny, nx)), repr((hny, hnx))))
        return False, data, variance, illum

    if variance is not None:
        if variance.ndim == 2:
            variance = variance[None]
        variance = np.asarray(variance, dtype=float)
        if variance.shape != shape:
            log.error("Variance and data shape mismatch: "
                      "%s != %s" % (repr(variance.shape), repr(shape)))
            return False, data, variance, illum

    return True, data, variance, illum
