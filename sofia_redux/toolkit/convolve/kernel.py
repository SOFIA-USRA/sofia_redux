# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy import ndimage

from sofia_redux.toolkit.convolve.base import ConvolveBase
from sofia_redux.toolkit.convolve.filter import savgol, savgol_windows

__all__ = ['apply_ndkernel', 'convolve', 'savitzky_golay', 'KernelConvolve',
           'BoxConvolve', 'SavgolConvolve']


def apply_ndkernel(data, kernel, axes=None, normalize=True,
                   is_error=False, **kwargs):
    """
    Apply a kernel over multiple features

    Parameters
    ----------
    data : array_like (shape)
    kernel :  array_like
        Must have 1 dimension or the same number of features as "data"
    axes : array_like of int (ndim)
    normalize : bool, optional
        If True, normalize the kernel
    is_error : bool, optional
        If True, assumes input data are error values and propagates
        accordingly
    kwargs : dict, optional
        Optional keywords to pass into scipy.ndimage.convolve1d or
        scipy.ndimage.convolve.

    Returns
    -------
    convolved : numpy.ndarray (shape)
        "data" convolved with "kernel".
    """
    data = np.asarray(data, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    by_axis = kernel.ndim == 1 or data.ndim == 1
    if not by_axis and (data.ndim != kernel.ndim):
        raise ValueError("kernel must have 1 dimension or the same "
                         "number of features as the input sample")
    if by_axis and axes is None:
        axes = np.arange(data.ndim)

    result = data ** 2 if is_error else data

    if normalize:
        ksum = np.abs(kernel).sum()
        k = kernel / ksum if ksum != 0 else kernel
    else:
        k = kernel

    if is_error:
        k = k ** 2

    if not by_axis:
        result = ndimage.convolve(data, k, **kwargs)
        return np.sqrt(result) if is_error else result

    for axis in axes:
        result = ndimage.convolve1d(result, k, axis=axis, **kwargs)
    return np.sqrt(result) if is_error else result


class KernelConvolve(ConvolveBase):
    """Generic convolution with a kernel"""
    def __init__(self, *args, error=1, mask=None, stats=True,
                 do_error=True, robust=0, eps=0.01, maxiter=100,
                 normalize=True, ignorenans=True, **kwargs):

        self._kernel = np.asarray(args[-1]).astype(float)
        self._normalize = normalize
        super().__init__(*args, error=error, mask=mask,
                         do_error=do_error, stats=stats,
                         robust=robust, eps=eps, maxiter=maxiter,
                         ignorenans=ignorenans, **kwargs)

    def _convolve(self, cleaned):
        self._result = apply_ndkernel(
            cleaned, self._kernel, axes=self._axes,
            normalize=self._normalize, **self._fit_kwargs).ravel()
        if self.do_error:
            self._interpolated_error = apply_ndkernel(
                self.error, self._kernel, axes=self._axes, is_error=True,
                normalize=self._normalize, **self._fit_kwargs).ravel()


class BoxConvolve(KernelConvolve):
    """Convolution with a box kernel (mean)"""
    def __init__(self, *args, error=1, mask=None, stats=True,
                 do_error=True, robust=0, eps=0.01, maxiter=100,
                 normalize=True, ignorenans=True, **kwargs):

        args = args[:-1] + (np.ones(args[-1]),)

        super().__init__(*args, error=error, mask=mask,
                         do_error=do_error, stats=stats,
                         robust=robust, eps=eps, maxiter=maxiter,
                         normalize=normalize, ignorenans=ignorenans,
                         **kwargs)


class SavgolConvolve(ConvolveBase):
    """Convolve using Savitzky-Golay filter"""
    def __init__(self, *args, error=1, mask=None, stats=True,
                 do_error=True, robust=0, eps=0.01, maxiter=100,
                 scale=False, order=2, ignorenans=True,
                 **kwargs):

        self._scaled = scale
        self._order = order
        self._window = None

        super().__init__(*args, error=error, mask=mask,
                         do_error=do_error, stats=stats,
                         robust=robust, eps=eps, maxiter=maxiter,
                         ignorenans=ignorenans, **kwargs)

    def _parse_model_args(self):
        z = self.reshape(self._samples[-1], copy=False)
        self._order, self._window = savgol_windows(
            self._order, self._model_args,
            *(tuple(self._samples[:-1]) + (z,)), scale=self._scaled)

    def _convolve(self, cleaned):
        self._result = savgol(
            cleaned, self._window, order=self._order, axes=self._axes,
            check=False, **self._fit_kwargs).ravel()
        if self.do_error:
            self._interpolated_error = savgol(
                self.error, self._window, order=self._order, is_error=True,
                axes=self._axes, check=False, **self._fit_kwargs).ravel()


def convolve(*args, error=None, mask=None, stats=False, do_error=None,
             robust=0, eps=0.01, maxiter=100, normalize=True,
             ignorenans=True, **kwargs):
    """
    Convolve an N-dimensional array with a user defined kernel or fixed box.

    A wrapper function for the `KernelConvolve`.

    Parameters
    ----------
    args : [samples], data, kernel
        "data" must be an n-dimensional array of size (shape,).  Optionally,
        coordinate values for each data element may be provided for each
        dimension ("samples").  If not provided, they will be determined to
        be regularly spaced at unit intervals.  "samples" must be provided
        for each dimension (x, y, z, ... order), where each array matches the
        data (shape,).  Finally, a kernel of the same dimensions as "data"
        must be provided following the "data" argument.  If "kernel" is an
        integer, a box kernel will be applied with a width of "kernel" over
        each dimension.
    error : float or int or array, optional
        A scalar value containing the error for each data point or an array
        of values may be provided and propagated through the convolution
        and interpolation.  If provided, and "do_error" is `None` or True, the
        returned result will be a tuple of the form (convolved_result, error).
    mask : array of bool, optional
        If provided must match data (shape,).  `False` values indicate data
        locations that should be excluded from convolution calculations.
        `False` values will be replaced by linear interpolated values.
    stats : bool, optional
        If `True`, return statistics on the convolution/interpolation in the
        returned result.
    do_error : bool, optional
        If `True`, and `error` is not `None`, return the propagated error in
        addition to the result.
    robust : int or float, optional
        If non-zero, will perform iterations of the convolution, masking out
        data points if the abs(residual/error) is greater than "robust".
        Iteration will continue until the delta_sigma/sigma ratio is less than
        "eps".
    eps : float, optional
        The precision limit for use in the "robust" iteration process.
    maxiter : int, optional
        The maximum number of iterations to attempt in "robust" iteration.
    normalize : bool, optional
        If True, the kernel will be normalized such that sum(kernel) = 1
        prior to convolution.
    ignorenans : bool, optional
        If True, NaNs will be masked out during convolution and interpolation.
        Otherwise, they will propagate through all calculations.
    kwargs : dict, optional
        Optional keyword arguments to pass into BoxConvolve or KernelConvolve
        parent class.

    Returns
    -------
    convolved_result, [error], [statistics] : array or tuple of array
        The convolved result, and optional propagated error (if "error" is
        provided and "do_error" is not "False") and statistics (if "stats"
        is `True`.  The output shape for the result and error will be the
        same as the input data.
    """
    if hasattr(args[-1], '__len__'):
        convolver = KernelConvolve
    else:
        convolver = BoxConvolve

    if error is None:
        do_error = False
    elif do_error is None:
        do_error = error is not None

    c = convolver(*args, error=error,
                  mask=mask, stats=stats,
                  do_error=do_error, robust=robust, eps=eps,
                  maxiter=maxiter, normalize=normalize,
                  ignorenans=ignorenans, **kwargs)

    result = c.result
    if not do_error and not stats:
        return result
    result = result,
    if do_error:
        result = result + (c.error,)
    if stats:
        result = result + (c.stats,)
    return result


def savitzky_golay(*args, order=2, error=1, mask=None, do_error=False,
                   robust=0, eps=0.01, maxiter=100, model=False,
                   scale=False, ignorenans=True, **kwargs):
    """
    Apply a least-squares (Savitzky-Golay) polynomial filter

    Convenience wrapper for toolkit.convolve.kernel.SavgolConvolve

    Any NaN or False "mask" values will be interpolated over before
    filtering.  Error will be propagated correctly and will account
    for interpolation over masked/NaN values or outliers removed during
    "robust" iteration.

    Arguments are supplied in the following order:
    [x0, x1, ... xn], values, window, [optional parameters]

    where x are dimension coordinates supplied in the same order
    as the axes of values.  Therefore, if you supplied arguments
    in the order x, y, z then you should expect the zeroth axis
    of z to relate to the x-coordinates and the first axis of z
    to relate to the y-coordinates.  This is contrary to standard
    numpy (row (y), col (x)) ordering.  For consistency with numpy,
    one would supply the arguments in the y, x, z order.

    The x0, x1,... independent values do not need to be supplied,
    and instead, the features of values will be used to generate
    a regular grid of coordinates.  Be aware however that the
    independent variables are used to rescale window widths if
    "scale" is set to True, and also used for linear interpolation
    over masked, NaN, or identified outliers (if "robust").

    Parameters
    ----------
    args : (ndim + 2)-tuple or 2-tuple

         - args[-1] contains the filtering window width.  This
           should be a positive integer greater than order and less
           than the size of the image.

         - args[-2] contains the dependent variables as an array
           of shape (shape).  If independent variables were supplied,
           (shape) must match across args[:ndim + 1].  Otherwise, the
           dimension and shape is determined by args[-2].

         - args[:ndim] contain the independent variables.  These are
           used for interpolation purposes or "scale" and should be of
           size (shape).  However, they do not need to be specified if
           interpolation is being carried out on a regular grid.  In this
           case, the user should just supply the dependent variables as an
           array of the correct dimension and shape.

    order : int, optional
        Order of polynomial to use in designing the filter.  Higher
        orders produce sharper features.
    error : float or array_like of float (shape), optional
        Dependent variable errors to propagate
    mask : array_like or bool (shape), optional
        Mask indicating good (True) and bad (False) samples.  Bad values
        will not be included in the fit and will be replaced with linear
        interpolation prior to filtering.
    do_error : bool, optional
        If True, return the propagated error in addition to the filtered
        data.
    robust : int or float, optional
        If > 0, taken as the sigma threshold above which to identify
        outliers.  Outliers are those identified as
        abs(x_i - x_med) / MAD > "robust"
        where x is the residual of (data - fit) x_med is the median, MAD
        is the median absolute deviation defined as
        1.482 * median(abs(x_i - x_med)).  The fit will be iterated upon
        until the set of identified outliers does not change, or any
        change in the relative RMS is less than "eps", or "maxiter"
        iterations have occured.
    eps : float, optional
        The limit of fractional check in RMS if "robust" is > 0
    maxiter : int, optional
        The maximum number iterations if "robust" is > 0
    model : bool, optional
        If set, return the fitting model with lots of nice little
        methods and statistics to play with.
    scale : bool, optional
        If True, scale "window" to the average spacing between samples
        over each dimension.  Note that this replaces "width" in the
        old IDL version.  This option should not be used if working in
        multiple non-orthogonal features, as average spacing per
        dimension is taken as the average separation between ordered
        dimensional coordinates.
    ignorenans : bool, optional
        If True (default and highly recommended), NaNs will be removed
        and interpolated over.
    kwargs : dict, optional.
        Optional keywords and values that would be passed into
        scipy.signal.savgol_filter.

    Returns
    -------
    filtered_data : numpy.ndarray of numpy.float64 (shape) or Model
        The filtered data or an instance of toolkit.base.Model.


    Notes
    -----
    Edge conditions are handled differently according to whether the
    data is one dimensional or not.  If it is, then extrapolation is
    permitted.  However, extrapolation is not permitted in features
    of > 2.  In these cases, everything should work as expected
    unless the corner samples are NaN.  Interpolation is not possible
    and NaNs will be applied to the filtered array according to
    width/2 of the window centered on the corner NaN.

    Also note, the IDL version "robustsg" was only 1-dimensional and
    savitzky_golay will perform exactly the same in this case.

    Examples
    --------
    Generate coefficients for cubic-polynomial with a window of 5

    >>> import numpy as np
    >>> x = np.arange(5)
    >>> y = [0, 0, 1, 0, 0]
    >>> filtered = savitzky_golay(x, y, 5, order=2)
    >>> assert np.allclose(filtered, [-3/35, 12/35, 17/35, 12/35, -3/35])

    2-dimensional example

    >>> (y, x), z = np.mgrid[:11, :11], np.zeros((11, 11))
    >>> z[5, 5] = 1
    >>> x = x / 2

    Coordinates are inferred if not supplied

    >>> filtered1 = savitzky_golay(x, y, z, 5, order=2)
    >>> filtered2 = savitzky_golay(z, 5, order=2)
    >>> assert np.allclose(filtered1, filtered2)
    >>> filtered3 = savitzky_golay(x, y, z, 5, order=2, scale=True)
    >>> print(np.sum(filtered1 != 0))  # window = (5, 5) x, y pixels
    25
    >>> print(np.sum(filtered3 != 0))  # window = (11, 5) x, y, pixels
    55

    Note that in the last case, scale scaled the x window width
    to 11 pixels since the average spacing in x is 0.5.  i.e. a
    window width of 1 is equal to 2 x-pixels.  Therefore, the window
    is expanded to 2 * 5 = 10, and then an extra pixel is added on
    to ensure that the filter is centered (odd number of pixels).
    """
    sg = SavgolConvolve(*args, order=order, error=error, mask=mask,
                        do_error=do_error, robust=robust, eps=eps,
                        maxiter=maxiter, stats=model, scale=scale,
                        ignorenans=ignorenans, **kwargs)
    if model:
        return sg
    return (sg.result, sg.error) if do_error else sg.result
