# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.stats import sigma_clipped_stats
import bottleneck as bn
import numpy as np
from scipy.ndimage import filters

__all__ = ['fixpix', 'maskbp']


def _test_stamp(stamp, test_median, test_sigma, sign=1):
    _, _, sig = sigma_clipped_stats(stamp, sigma=5)
    pixval = stamp[2, 2]

    # mask hot pixel
    stamp[2, 2] = np.nan

    # median nearest 8 pixels
    med8 = bn.nanmedian(stamp[1:4, 1:4])
    stamp[1:4, 1:4] = np.nan

    # median next 16 pixels out
    med16 = bn.nanmedian(stamp)

    # decide from the noise if we are on a feature
    # or gaussian background
    # todo - see if magic numbers can be sourced
    if sig > test_sigma:
        sig = max(test_sigma, np.sqrt(5 + .21 * np.abs(med8)))

    # heuristic for blocking pixel
    if sign > 0:
        if ((med8 + 2 * med16) / 3 - test_median) > (2 * sig):
            if pixval > (2 * med8 - test_median + 3 * sig):
                # pixel is bad
                return True, med8
        elif (pixval - (med8 + 2 * med16) / 3) > (5 * sig):
            # pixel is bad
            return True, med8
    else:
        if ((med8 + 2 * med16) / 3 - test_median) < (-2 * sig):
            if pixval < (2 * med8 - test_median - 3 * sig):
                # pixel is bad
                return True, med8
        elif (pixval - (med8 + 2 * med16) / 3) < (-5 * sig):
            # pixel is bad
            return True, med8

    # pixel is good
    return False, pixval


def fixpix(data, max_iter=5):
    """
    Identify hot and cold pixels in a data array.

    Parameters
    ----------
    data : numpy.ndarray
        Image array.
    max_iter : int, optional
        Number of iterations to perform.

    Returns
    -------
    mask : numpy.ndarray of int16
        Mask array with bad pixels marked (1 = bad, 0 = good).
    """
    mask = np.zeros(data.shape, dtype=np.int16)

    # median of image
    medimg = np.nanmedian(data)
    log.debug(f'Median value: {medimg:.2f}')

    # local noise in image with 5x5 box filter
    sigma = filters.generic_filter(data, bn.nanstd, size=5,
                                   mode='constant', cval=np.nan)

    # stats for noise value
    medsig, _, sigsig = sigma_clipped_stats(sigma, sigma=5)
    test_limit = medsig + 2 * sigsig
    log.debug(f'Median noise: {medsig:.2f} +/- {sigsig:.2f}')

    # iteratively find hot and cold pixels
    niter = 0
    new_badpix = True
    nhot = 0
    ncold = 0
    corrected_image = data.copy()
    padded = np.pad(corrected_image, 2, mode='reflect')
    while niter < max_iter and new_badpix:
        new_badpix = False

        # 5x5 box filter: max is hot pix, min is cold pix
        hot = filters.maximum_filter(corrected_image, size=5, mode='mirror')
        cold = filters.minimum_filter(corrected_image, size=5, mode='mirror')

        # check surrounding area for each hot pixel to determine
        # if it's source-like or bad pixel-like
        idx = np.where(corrected_image == hot)
        for y, x in zip(idx[0], idx[1]):
            stamp = padded[y:y + 5, x:x + 5].copy()
            mark_bad, replace = _test_stamp(stamp, medimg, test_limit)
            if mark_bad:
                log.debug(f'Replace hot x,y={x},{y} value '
                          f'{corrected_image[y,x]} with {replace}')
                corrected_image[y, x] = replace
                mask[y, x] = 1
                nhot += 1
                new_badpix = True

        # same for cold pixel
        idx = np.where(corrected_image == cold)
        for y, x in zip(idx[0], idx[1]):
            stamp = padded[y:y + 5, x:x + 5].copy()
            mark_bad, replace = _test_stamp(stamp, medimg, test_limit,
                                            sign=-1)
            if mark_bad:
                log.debug(f'Replace cold x,y={x},{y} value '
                          f'{corrected_image[y,x]} with {replace}')
                corrected_image[y, x] = replace
                mask[y, x] = 1
                ncold += 1
                new_badpix = True

        niter += 1
        log.debug(f'Iteration {niter}: total {nhot} hot, {ncold} cold')

    log.info(f'Found {nhot} hot pixels and {ncold} cold pixels')
    return mask


def maskbp(hdul, cval=None, max_iter=5):
    """
    Mask hot and cold bad pixels.

    Parameters
    ----------
    hdul : fits.HDUList
        Input data.  Should have FLUX, ERROR, and BADMASK extensions.
    cval : float, optional
        Constant value to replace bad pixels with.  If not
        provided, bad pixels will not be replaced.
    max_iter : int, optional
        Number of iterations of bad pixel finding to perform.

    Returns
    -------
    fits.HDUList
        Masked data. BADMASK extension is updated with the new
        bad pixels.  FLUX and ERROR extensions may be updated to
        replace the bad pixels, if `cval` is not None.
    """
    data = hdul['FLUX'].data
    error = hdul['ERROR'].data
    mask = hdul['BADMASK'].data

    # mark outlier pixels
    mask |= fixpix(data, max_iter=max_iter)

    # mask input data if desired
    if cval is not None:
        data[mask == 1] = cval
        error[mask == 1] = cval

    return hdul
