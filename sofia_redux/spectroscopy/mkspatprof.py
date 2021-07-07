# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
import numpy as np
from scipy.ndimage import gaussian_filter

from sofia_redux.toolkit.fitting.polynomial import polyfitnd, poly1d

__all__ = ['mkspatprof']


def mkspatprof(rectimg, atran=None, atmosthresh=None,
               bgsub=True, orders=None, ndeg=4, robust=5.0,
               smooth_sigma=1.0, return_fit_profile=False):
    """
    Construct average spatial profiles.

    Each order should already be resampled onto a uniform grid.
    For each order, the median background is subtracted on a column by
    column basis.  The median spatial profile is then created.  If the
    user passes atran and atmosthresh, then pixels that have atmospheric
    transmission below atmosthresh are ignored. The median spatial
    profile is then used to normalize the image. 2D polynomial coefficients
    are then derived on a row by row basis, and used to construct a
    2D spatial map.

    Outputs from this function (median profile and spatial map)
    are required for all further reduction steps (setting apertures,
    tracing continua, and extracting spectra).  The value for rectimg
    comes from the output of `sofia_redux.spectroscopy.rectify`.
    Atmospheric data (atran) and all other parameters are instrument
    dependent.

    Parameters
    ----------
    rectimg : dict
        As returned by `sofia_redux.spectroscopy.rectify` with integer keys
        indicating the order. The values are dictionaries with keys as
        follows:

            ``"image"``
                numpy.ndarray (ns, nw)
                Rectified image array (required)
            ``"wave"``
                numpy.ndarray (nw,)
                Wave coordinates along image axis=1 (required)
            ``"spatial"``
                numpy.ndarray (ns,)
                Spatial coordinates along image axis=0 (required)
            ``"mask"``
                numpy.ndarray (ns, nw)
            ``"pixsum"``
                numpy.ndarray (ns, nw)
            ``"variance"``
                numpy.ndarray (ns, nw)

    atran : numpy.ndarray, optional
        (2, Ndata) array where atran[0, :] gives the wavelengths for
        atmospheric transmission and atran[1, :] gives the atmospheric
        transmission.
    atmosthresh : float, optional
        The transmission (0 -> 1) below which data are ignored.
    bgsub : bool, optional
        If True, the median background level is subtracted from the profile.
        It may be useful to set this to False when using a mode with a
        short slit.
    orders : array_like of int, optional
        List or order numbers, ordered from the bottom to the top of the
        image.  If not provided, order numbers will be derived from a
        sorted list of all order numbers in the order_mask.
    ndeg : int, optional
        The degree of the polynomial row fits.
    robust : float, optional
        The robust threshold for polynomial row fits.
    smooth_sigma : float, optional
        If greater than 0, the fit profile will be smoothed by a Gaussian
        with this width.
    return_fit_profile : bool, optional
        If set, a fit profile with dimensions (ns,nw) will be returned
        as well as the median profile. This spatial map is required
        for optimal extraction steps.

    Returns
    -------
    median_profile : dict
        order (int) -> profile (numpy.ndarray)
            (n_spatial, 2) spatial profile where profile[:, 0] = spatial
            coordinate and profile[:, 1] = median spatial profile.
    fit_profile : dict, optional
            order (int) -> profile (numpy.ndarray)
            (n_spatial, n_wave) profile from fit coefficients.
    """
    if orders is None:
        orders = np.unique(list(rectimg.keys())).astype(int)
    else:
        orders = np.unique(orders).astype(int)

    do_atran = atran is not None and atmosthresh is not None
    if do_atran:
        atran = np.array(atran)
        if atran.ndim != 2:
            log.error("Invalid atran shape.")
            return

    result = {}
    fit_profile = {}
    for order in orders:
        rectified = rectimg.get(order)
        if rectified is None:
            log.warning(f"Order {order} is missing from rectimg.")
            continue

        image = rectified.get('image')
        if image is None:
            log.warning(f"Order {order} is missing image key.")
            continue

        wave = rectified.get('wave')
        if wave is None:
            log.warning(f"Order {order} is missing wave key.")
            continue

        spatial = rectified.get('spatial')
        if spatial is None:
            log.warning(f"Order {order} is missing spatial key.")
            continue

        image = np.array(image)
        if bgsub:
            # subtract background
            image -= np.nanmedian(image, axis=0)
        xgrid = rectified['wave'].copy()

        if do_atran:
            # mask out atmosphere
            atmos_mask = np.interp(
                xgrid, atran[0], atran[1], left=0, right=0) < atmosthresh
            image[:, atmos_mask] = np.nan

        # Compute median spatial profile and normalize the image
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            profile = np.nanmedian(image, axis=1)
            profile /= np.nansum(np.abs(profile))
            result[order] = profile.copy()

            # Normalize the image
            norms = np.nanmedian(image / np.array([profile]).T, axis=0)
            image /= np.array([norms])

        # Make spatial profile coefficients for each row
        spatmap = np.zeros_like(image)
        for idx, imgrow in enumerate(image):
            good = np.isfinite(imgrow)
            coeff = polyfitnd(xgrid[good], imgrow[good],
                              ndeg, robust=robust)
            spatmap[idx, :] = poly1d(xgrid, coeff)

        # Smooth a little if desired, along the columns
        if smooth_sigma is not None and smooth_sigma > 0:
            spatmap = gaussian_filter(spatmap, (smooth_sigma, 0.0),
                                      mode='nearest')

        fit_profile[order] = spatmap

    if return_fit_profile:
        result = result, fit_profile
    return result
