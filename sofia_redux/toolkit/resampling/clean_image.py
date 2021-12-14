# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from .resample_polynomial import ResamplePolynomial

__all__ = ['clean_image']


def clean_image(image, error=None, mask=None, window=None, order=1,
                fix_order=True, robust=None, negthresh=None, mode=None,
                leaf_size=None, **kwargs):
    """
    Uses `ResamplePolynomial` to correct NaNs in image and/or supplied in mask.

    Parameters
    ----------
    image : array_like of float
        (M, N) array of data values.  NaN values will automatically be
        considered 'bad' and replaced.
    error : array_like of float, optional
        (M, N) array of error (1-sigma) values
        associated with the `data` array.  `error` will be used to
        weight fits and also be propagated to the output error values.
    mask : array_like of bool, optional
        (M, N) array of bool where True indicates a valid data point
        that can be included the fitting and False indicates data points
        that should be excluded from the fit and replaced.
        Masked points will be reflected in the output counts array.
    window : array_like or float or int, optional
        (2,) array or single float value specifying the maximum
        euclidean distance of a data sample from a resampling point such
        that it can be included in a local fit.  `window` may be declared
        for each feature.  For example, when fitting 2-dimensional (x, y)
        data, a window of 1.0 would create a circular fitting window
        around each resampling point, whereas a window of (1.0, 0.5)
        would create an eliptical fitting window with a semi-major axis
        of 1.0 in x and semi-minor axis of 0.5 in y.  If not supplied,
        `window` is calculated based on an estimate of the median
        population density of the data for each feature.
    order : array_like or int, optional
        (2,) array or single integer value specifying the
        polynomial fit order for each feature.
    fix_order : bool, optional
        In order for local polynomial fitting to occur, the default
        requirement is that nsamples >= (order + 1) ** 2,
        where nsamples is the number of data samples within `window`.
        If `fix_order` is True and this condition is not met, then
        local fitting will be aborted for that point and a value of
        `cval` will be returned instead.  If `fix_order` is False,
        then `order` will be reduced to the maximum value where this
        condition can be met.  NOTE: this is only available if
        `order` is symmetrical. i.e. it was passed in as a single
        integer to be applied across both x and y.  Otherwise, it is
        unclear as to which dimension order should be reduced to meet
        the condition.
    robust : float, optional
        Specifies an outlier rejection threshold for `data`.
        A data point is identified as an outlier if::

            |x_i - x_med|/MAD > robust,

        where x_med is the median,
        and MAD is the Median Absolute Deviation defined as::

            1.482 * median(|x_i - x_med|).

    negthresh : float, optional
        Specifies a negative value rejection threshold such that
        data < (-stddev(data) * negthresh) will be excluded from
        the fit.
    mode : str, optional
        The type of check to perform on whether the sample distribution
        for each resampling point is adequate to derive a polynomial fit.
        Depending on `order` and `fix_order`, if the distribution does
        not meet the criteria for `mode`, either the fit will be aborted,
        returning a value of `cval` or the fit order will be reduced.
        Available modes are:

            - 'edges': Require that there are `order` samples in both
              the negative and positive directions of each feature
              from the resampling point.
            - 'counts': Require that there are (order + 1) ** 2
              samples within the `window` of each resampling point.
            - 'extrapolate': Attempt to fit regardless of the sample
              distribution.

        Note that 'edges' is the most robust mode as it ensures
        that no singular values will be encountered during the
        least-squares fitting of polynomial coefficients.
    leaf_size : int, optional
        Number of points at which to switch to brute-force during the
        ball tree query algorithm.  See `sklearn.neighbours.BallTree`
        for further details.

    Returns
    -------
    cleaned_image, [variance_out], [counts] : n_tuple of numpy.ndarray (M, N)
        See `sofia_redux.toolkit.resampling.ResamplePolynomial`
    """
    if mask is None:
        mask = np.isfinite(image)
    else:
        mask = np.asarray(mask, dtype=bool)
    missing = np.logical_not(mask)
    nfind = missing.sum()
    corrected = np.asarray(image).astype('float')

    if nfind == 0:
        return corrected

    shape = corrected.shape[-2:]
    ygrid, xgrid = np.mgrid[:shape[0], :shape[1]]
    if error is not None:
        error = np.asarray(error).ravel()

    resampler = ResamplePolynomial([xgrid.ravel(), ygrid.ravel()],
                                   corrected.ravel(),
                                   error=error, mask=mask.ravel(),
                                   order=order, window=window,
                                   fix_order=fix_order,
                                   robust=robust, negthresh=negthresh,
                                   leaf_size=leaf_size)

    corrected[missing] = resampler([xgrid[missing], ygrid[missing]],
                                   order_algorithm=mode, **kwargs)
    return corrected
