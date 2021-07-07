# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.fitting.polynomial import poly1d, polyfitnd
from sofia_redux.toolkit.utilities.func import nantrim
from sofia_redux.toolkit.stats import meancomb
import warnings

__all__ = ['col_subbg', 'extspec']


def col_subbg(col_arc, col_image, col_var, col_apmask, col_mask, bgorder):
    """
    Fit background to a single column.

    Parameters
    ----------
    col_arc : numpy.ndarray of float
        (nrow,) spatial coordinates
    col_image : numpy.ndarray of float
        (nrow,) column image values
    col_var : numpy.ndarray of float
        (nrow,) variance of image column
    col_apmask : numpy.ndarray of float
        (nrow,) background mask where 0 indicates a background pixel
    col_mask : numpy.ndarray of bool
        (nrow,) True indicates a valid image pixel, False indicates a
        bad pixel.
    bgorder : int
        Order of polynomial to fit to the background

    Returns
    -------
    corrected_image, corrected_var, coefficients : 3-tuple of numpy.ndarray
        The column with background subtracted (nrow,), the corrected
        variance (nrow,) and the polynomial coefficients fit to the
        background (bgorder + 1,).
    """
    region = np.array(np.isnan(col_apmask))
    if region.sum() < (bgorder + 2):
        log.error("Not enough background points found.")
        return
    bgmodel = polyfitnd(col_arc[region], col_image[region], bgorder,
                        error=np.sqrt(col_var[region]), robust=4,
                        mask=col_mask[region], covar=True, model=True)
    if not bgmodel.success:
        log.error("Polynomial fit failed.")
        return
    bgfit, bgvar = bgmodel(col_arc, dovar=True)
    return col_image - bgfit, col_var + bgvar, bgmodel.coefficients


def col_fixdata(fix, col_image, col_var, col_bits, col_mask, model_profile,
                scale_coeffs, threshold=5.0):
    """
    Replace bad pixels in a single column.

    Parameters
    ----------
    fix : numpy.ndarray of int
        Indices of data in the column to fix.
    col_image : numpy.ndarray of float
        (nrow,) column image values
    col_var : numpy.ndarray of float
        (nrow,) variance of image column
    col_bits : numpy.ndarray of int
        (nrow,) bit mask for image column
    col_mask : numpy.ndarray of bool
        (nrow,) boolean bad pixel mask for image column
    model_profile : numpy.ndarray of float
        (nrow,) spatial model for the image column
    scale_coeffs : array-like of float
        Coefficients for scaling the model to the flux.
    threshold : float
        Robust threshold for scaling the model to the
        variance image.
    """

    # Replace bad pixels with the scaled model values
    col_image[fix] = poly1d(
        model_profile[fix], scale_coeffs)

    # Scale the profile to the variance and replace
    absprof = np.abs(model_profile)
    var_model = polyfitnd(absprof, col_var, 1,
                          model=True, robust=threshold)
    col_var[fix] = var_model.stats.fit[fix]

    # set 2 in the bitmask
    col_bits[fix] += 2

    # set False in the bool mask
    col_mask[fix] = False


def extspec(rectimg, profile=None, spatial_map=None,
            optimal=False, sub_background=True, fix_bad=False,
            bgorder=2, threshold=5.0, sum_function=None,
            error_function=None):
    """
    Extracts spectra from a rectified spectral image.

    For each column (wavelength) in each order:

       a. (optional) Remove background using spatial regions
          identified in the aperture mask.
       b. (optional) Use the spatial model in spatial_map or profile to
          remove bad pixels.
       c. Extract the spectral value for that column using either
          the standard or optimal extraction algorithm.

    Standard Extraction:
    If bad pixels were identified via the spatial model, they are
    replaced by the value of the spatial model at that wavelength.
    The spectral value for a given aperture is then taken as the sum
    of all pixels lying on the aperture.

    Optimal Extraction:
    The spectral value for each aperture is taken by taking a mean of
    the column weighted by the spatial model, for pixels identified as
    the aperture radius in the aperture mask.  Bad pixels are ignored.

    Parameters
    ----------
    rectimg : dict
        Rectified image data with aperture definitions.  The aperture
        mask is as produced by the `sofia_redux.spectroscopy.mkapmask`
        function.

        Structure is:
            order (int) -> dict
                image -> numpy.ndarray (ns, nw)
                    Rectified order image.
                variance -> numpy.ndarray (ns, nw)
                    Rectified variance for image.
                wave -> numpy.ndarray (nw,)
                    Wavelength coordinates.
                spatial -> numpy.ndarray (ns,)
                    Spatial coordinates.
                mask -> numpy.ndarray (ns, nw)
                    Boolean bad pixel mask (True = good).
                bitmask -> numpy.ndarray (ns, nw)
                    Bit-set mask. 1=nonlinear pixel.
                apmask -> numpy.ndarray (ns, nw)
                    Aperture mask.
                apsign -> list of {1, -1}, optional
                    Aperture signs for each aperture.  Must match the
                    number of apertures in the aperture mask if provided.
                    All aperture signs are assumed positive if not provided.

    profile : dict, optional
        Median spatial profile for each order.

        Structure is:
            order (int) -> numpy.ndarray (ns,)

    spatial_map : dict, optional
        2D spatial map for each order.  If provided, `profile` is ignored.

        Structure is:
            order (int) -> numpy.ndarray (ns, nw)

    optimal : bool, optional
        If set, optimal extraction is used.
    sub_background : bool, optional
        If set, background regions identified in the aperture mask
        are fit and subtracted from each column.  If set, `profile`
        or `spatial_map` must be provided.
    fix_bad : bool, optional
        If set, bad pixels will be fixed in the 2D spectral image.
        For standard extraction, the fixed pixels will be used to calculate
        the extracted flux.  For optimal extraction, the fixed pixels
        will be ignored.
    bgorder : int, optional
        Background fitting order.
    threshold : float, optional
        Robust threshold for background fits, in number of sigma.
    sum_function : function, optional
        Function to use as sum over aperture, when optimal=False.
        Default is np.nansum(flux * weights). The provided function
        should accept one two array arguments (flux and weights).
    error_function : function, optional
        Function to use for error propagation, given variance input,
        when optimal=False.  Default is np.sqrt(np.nansum(flux * weights)).
        The provided function should accept two array arguments
        (variance and weights).

    Returns
    -------
    spectra : dict
        order (int) -> dict
            spectra : numpy.ndarray of float (n_apertures, 4, n_spec)
                Each order contains a single array where:
                    array[aperture, 0] = wavelength
                    array[aperture, 1] = flux
                    array[aperture, 2] = error
                    array[aperture, 3] = bit-set mask
    """
    # todo -- add checks for input structure
    use_model = False
    use_profile = False
    if optimal or fix_bad:
        if spatial_map is None:
            if profile is None:
                log.error("Optimal or fix_bad requires "
                          "spatial map or median profile.")
                return
            else:
                use_profile = True
        use_model = True
    orders = np.unique(list(rectimg.keys())).astype(int)
    if use_profile:
        log.debug('Using median profile for extraction.')
    elif use_model:
        log.debug('Using spatial map for extraction.')
    else:
        log.debug('No profile or spatial map used for extraction.')

    result = {}
    # Loop through each order
    for orderi, order in enumerate(orders):

        image = rectimg[order]['image']
        variance = rectimg[order]['variance']
        wave = rectimg[order]['wave']
        space = rectimg[order]['spatial']
        nwave = wave.size
        nspace = space.size

        if 'mask' in rectimg[order]:
            mask = rectimg[order]['mask']
        else:
            mask = np.full((nspace, nwave), True)
        if 'bitmask' in rectimg[order]:
            bitmask = rectimg[order]['bitmask']
        else:
            bitmask = np.zeros((nspace, nwave), dtype=np.int)

        # get aperture mask, as generated by mkapmask
        apmask = rectimg[order]['apmask']
        nap = int(np.nanmax(np.abs(apmask)))

        # assign aperture signs
        apsign = rectimg[order].get('apsign', None)
        if apsign is None or len(apsign) == 0:
            apsign = [1.0] * nap
        elif len(apsign) != nap:
            log.error('Mismatch between aperture signs and aperture mask.')
            return

        # check for background regions
        if sub_background and not np.any(np.isnan(apmask)):
            log.warning('No background regions found; '
                        'not subtracting background.')
            sub_background = False

        oflux = np.full((nap, nwave), np.nan)
        oerr = np.full((nap, nwave), np.nan)
        obits = np.full((nap, nwave), 0)

        # initialize necessary arrays and variables
        scale_coeffs, model_profile, goodpix = None, None, None

        for wavei in range(nwave):
            col_image = image[:, wavei]
            col_mask = mask[:, wavei]
            col_var = variance[:, wavei]
            col_bits = bitmask[:, wavei]
            col_apmask = apmask[:, wavei].copy()

            if sub_background:
                bgresult = col_subbg(
                    space, col_image, col_var, col_apmask, col_mask, bgorder)
                if bgresult is None:
                    msg = "Background fit failed at order=%i" % order
                    msg += " column=%i" % wavei
                    msg += " wavelength=%f" % wave[wavei]
                    log.warning(msg)
                else:
                    col_image, col_var, c = bgresult

            if use_model:
                # Scale the profile model and identify bad pixels
                if use_profile:
                    model_profile = profile[order]
                else:
                    model_profile = spatial_map[order][:, wavei]
                scale_model = polyfitnd(model_profile, col_image, 1,
                                        model=True, robust=threshold,
                                        mask=col_mask)
                goodpix = scale_model.mask
                scale_coeffs = scale_model.coefficients

                # fix bad pixels in image if desired
                if fix_bad:
                    fix = ~goodpix
                    if fix.any():
                        col_fixdata(fix, col_image, col_var,
                                    col_bits, col_mask,
                                    model_profile, scale_coeffs, threshold)

            # get aperture masks in a more useful form,
            # avoiding float and NaN equality comparisons
            col_apmask[np.isnan(col_apmask)] = 0
            psfmask = np.abs(col_apmask)
            inner_mask = np.zeros_like(col_apmask, dtype=int)
            inner_idx = col_apmask < 0
            inner_mask[inner_idx] = np.abs(col_apmask[inner_idx]).astype(int)

            # Begin extraction
            for api in range(nap):
                zap = (psfmask > api) & (psfmask <= (api + 1))
                sign = apsign[api]

                if optimal:
                    # Optimal extraction

                    # Enforce profile positivity
                    pos_profile = np.clip(model_profile.copy() * sign,
                                          0, np.inf)

                    # Sum over PSF, accounting for pixels partially on aperture
                    weight = psfmask[zap] - api
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        apsum = np.nansum(pos_profile[zap] * weight)
                    if apsum == 0:
                        pos_profile *= np.nan
                    else:
                        pos_profile /= apsum

                    # Scale data values and calculate mean over
                    # aperture radius only, ignoring fractional and bad pixels
                    zinner = (inner_mask == (api + 1))
                    idx = zinner & (pos_profile != 0) & goodpix
                    if idx.any():
                        scale_image = col_image[idx] / pos_profile[idx]
                        scale_var = col_var[idx] / (pos_profile[idx] ** 2)
                        mean, mvar = meancomb(scale_image, variance=scale_var)
                        oflux[api, wavei] = sign * mean
                        oerr[api, wavei] = np.sqrt(mvar)
                    else:
                        oflux[api, wavei] = np.nan
                        oerr[api, wavei] = np.nan
                        obits[api, wavei] += 8
                        msg = "Optimal extraction failed at order=%i " % order
                        msg += "aperture=%i " % api
                        msg += "column=%i wavelength=%f" % (wavei, wave[wavei])
                        log.warning(msg)
                else:
                    # Sum accounting for pixels partially on aperture
                    weight = psfmask[zap] - api
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        if sum_function is None:
                            oflux[api, wavei] = \
                                sign * np.nansum(col_image[zap] * weight)
                        else:
                            oflux[api, wavei] = \
                                sign * sum_function(col_image[zap], weight)
                        if error_function is None:
                            oerr[api, wavei] = \
                                np.sqrt(np.nansum(col_var[zap]
                                                  * (weight ** 2)))
                        else:
                            oerr[api, wavei] = error_function(col_var[zap],
                                                              weight)

                    # set bitmask for any fixed pixels
                    if np.any(col_bits[zap] > 1):
                        obits[api, wavei] += 2

                # Set bitmask for linearity
                if 1 in col_bits:
                    obits[api, wavei] += 1

                # Correct sign in image
                col_image[zap] *= sign

            # store modified image and variance back to input
            image[:, wavei] = col_image
            variance[:, wavei] = col_var

        # Store results
        nanstrip = nantrim(wave, 2)
        order_array = np.zeros((nap, 4, nanstrip.sum()))
        for api in range(nap):
            order_array[api, 0] = wave[nanstrip]
            order_array[api, 1] = oflux[api, nanstrip]
            order_array[api, 2] = oerr[api, nanstrip]
            order_array[api, 3] = np.mod(obits[api, nanstrip], 256)
        result[order] = order_array

    return result
