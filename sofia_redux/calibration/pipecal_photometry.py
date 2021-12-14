# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fit a source and perform aperture photometry."""

import warnings

from astropy import log
import numpy as np
import photutils

from sofia_redux.calibration.pipecal_fitpeak import pipecal_fitpeak
from sofia_redux.calibration.pipecal_error import PipeCalError

__all__ = ['pipecal_photometry']


def pipecal_photometry(image, variance, srcpos=None,
                       fitsize=138, stampsize='auto', fwhm=5.0,
                       profile='moffat', aprad=12.0,
                       skyrad=None, runits='Me/s',
                       stamp_center=True, allow_badfit=False):

    """
    Perform aperture photometry and profile fits on image data.

    Procedure:
        1. Take a small stamp image near the source position and fit a
           profile to it to get a centroid position of the source. The
           fit is performed by pipecal_fitpeak.
        2. Refit a larger subimage (set by `fitsize`) with pipecal_fitpeak to
           get fit parameters with associated errors.
        3. Call photutils to perform aperture photometry at the centroid
           position.
        4. Return calculated values in list of dictionaries.

    Parameters
    ----------
    image : array
        2D image array containing an object to perform photometry on.
    variance : array
        2D variance array corresponding to image values.
    srcpos : array-like, optional
          Initial guess at source position (x,y), zero-indexed.
          Defaults to the center of the image if not provided.
    fitsize : int, optional
        Size of subimage to fit.
    stampsize : 'auto', int, or float, optional
        Size of initial stamp, for peak location.  If 'auto', the
        stamp will be derived from the fitsize.
    fwhm : float, optional
        Initial guess at PSF FWHM.
    profile : {'moffat', 'gaussian', 'lorentzian'}, optional
        Type of profile to fit to image.
    aprad : float, optional
        Aperture radius for aperture photometry.
    skyrad : array-like, optional
        Sky radii (inner, outer) for aperture photometry. Defaults
        to [15.,25.] if not provided.
    runits : string, optional
        Raw data units, before flux calibration. Used in
        comment strings for some output keywords.
    stamp_center : bool, optional
        If True, the initial centroid position will be adjusted by
        the centroid in the stamp fit. This is usually desirable,
        but may sometimes result in the fit being pulled away from
        the intended target, in the case of crowded fields.  If
        False, the starting source position is not adjusted by
        the centroid in the stamp fit before doing the full image
        profile fit.
    allow_badfit : bool, optional
        If False, failed profile fits will raise PipeCalErrors.
        If True, an error will be logged, but an exception will not be
        raised, and aperture photometry will still be attempted at
        the initial source position.

    Returns
    -------
    phot : list
        List containing photometric measurements values. Each value
        stored in a dictionary with fields 'key', 'value', and
        'comment'. If the value has an associated error, it is stored
        in 'value' as a two-element array, where the first element is
        the value and the second element is the error.

    Raises
    ------
    PipeCalError
        For any improperly set parameters or for failed fit.
    """

    # Image dimensions:
    if not isinstance(image, np.ndarray):
        msg = 'Invalid image type'
        log.error(msg)
        raise PipeCalError(msg)
    elif len(image.shape) != 2:
        msg = 'Image must be 2d array'
        log.error(msg)
        raise PipeCalError(msg)
    else:
        max_row, max_col = image.shape

    if not isinstance(variance, np.ndarray):
        msg = 'Invalid variance type'
        log.error(msg)
        raise PipeCalError(msg)
    elif len(variance.shape) != 2:
        msg = 'Variance must be 2d array'
        log.error(msg)
        raise PipeCalError(msg)
    elif variance.shape != image.shape:
        msg = 'Variance must be same shape as image.'
        log.error(msg)
        raise PipeCalError(msg)

    # Check defaults:
    if srcpos is None:
        srcpos = [max_col / 2, max_row / 2]
    try:
        fitsize = int(fitsize)
    except (ValueError, TypeError):
        fitsize = 138
    try:
        fwhm = float(fwhm)
    except (ValueError, TypeError):
        fwhm = 5.0
    try:
        aprad = float(aprad)
    except (ValueError, TypeError):
        aprad = 12.0

    runits = str(runits).strip()
    profile = str(profile).strip().lower()
    if profile.lower() not in ['moffat', 'gaussian', 'lorentzian']:
        msg = 'Invalid profile selection'
        log.error(msg)
        raise PipeCalError(msg)

    if skyrad is None:
        skyrad = [15., 25.]
    elif not hasattr(skyrad, '__len__') or len(skyrad) != 2:
        msg = 'Invalid sky radius'
        log.error(msg)
        raise PipeCalError(msg)

    # Set fit parameter defaults
    if profile == 'gaussian':
        factor = 2. * np.sqrt(2. * np.log(2.))
    elif profile == 'lorentzian':
        factor = 2.0
    else:
        factor = 2.0

    # Take stamp
    # Size is FWHM * 5, but no smaller than 20
    # and no larger than fitsize
    if stampsize == 'auto':
        stampsize = min(max(fwhm * 5.0, 20), fitsize)
    else:
        try:
            stampsize = float(stampsize)
        except (ValueError, TypeError):
            msg = 'Invalid stampsize. Should be "auto" or int/float.'
            log.error(msg)
            raise PipeCalError(msg)
    log.debug('Full fit size: {}'.format(fitsize))
    log.debug('Initial stamp size: {}'.format(stampsize))

    # column min, the left side of the stamp, must be between 0 and
    # twice the stampsize from the right edge of the image
    left = int(max(0, min(srcpos[0] - stampsize,
                          max_col - 2 * stampsize)))

    # column max, the right side of the stamp, must be before the
    # right side of the image and is based on column min
    right = int(min(max_col - 1, max(srcpos[0] + stampsize - 1,
                                     left + 20)))

    # row min, the bottom side of the stamp, must be between 0
    # and twice the stampsize from the top of the image
    bottom = int(max(0, min(srcpos[1] - stampsize,
                            max_row - 2 * stampsize)))

    # row max, the top side of the stamp, must be under the top side
    # of the image and is based on row min
    top = int(min(max_row - 1, max(srcpos[1] + stampsize - 1,
                                   bottom + 20)))

    # Select out the stamp
    # Since Python is row-major:
    stamp = image[bottom:top + 1, left:right + 1]
    max_row_stamp, max_col_stamp = stamp.shape

    # check for reasonable data
    if np.sum(np.isfinite(stamp)) < 3:
        msg = 'Stamp image is empty'
        log.error(msg)
        raise PipeCalError(msg) from None

    # Find the location of the star relative to the stamp size
    col_cen = srcpos[0] - left
    row_cen = srcpos[1] - bottom

    # Starting parameter
    # baseline, peak, col-cent, row-cent,
    # col-width, row-width, rotation angle, power law index
    est = dict()
    est['baseline'] = np.nanmedian(stamp[np.isfinite(stamp)])
    est['dpeak'] = np.nanmax(stamp[np.isfinite(stamp)])
    est['col_mean'] = col_cen
    est['row_mean'] = row_cen
    est['col_sigma'] = fwhm / factor
    est['row_sigma'] = fwhm / factor
    est['theta'] = 0.
    est['beta'] = 3.

    # Set up boundaries
    bounds = dict()
    bounds['baseline'] = [-np.inf, np.inf]
    bounds['dpeak'] = [-np.inf, np.inf]
    bounds['col_mean'] = [0, max_col_stamp]
    bounds['row_mean'] = [0, max_row_stamp]
    bounds['col_sigma'] = [0, 2 * stampsize / factor]
    bounds['row_sigma'] = [0, 2 * stampsize / factor]
    bounds['beta'] = [2, 6]
    bounds['theta'] = [-np.pi / 2., np.pi / 2.]

    # Fit peak
    try:
        fitpars, sigma, bestnorm = \
            pipecal_fitpeak(stamp, estimates=est, bounds=bounds,
                            profile=profile)
    except (RuntimeError, ValueError):
        msg = 'Unable to fit stamp'
        log.error(msg)
        if allow_badfit:
            fitpars = est.copy()
        else:
            raise PipeCalError(msg) from None

    log.debug(f'Stamp parameters: {fitpars}')

    # update center position for fit image from stamp, if desired
    # Not updating is sometimes helpful in crowded or
    # high-background fields.
    if stamp_center:
        col_cen = int(round(fitpars['col_mean'])) + left
        row_cen = int(round(fitpars['row_mean'])) + bottom
    else:
        col_cen += left
        row_cen += bottom

    # Refit with larger image (usually 138x138) pixels,
    # centered on the location of the star found in the stamp,
    # using the variance array to get error estimates
    ncut = fitsize // 2
    left = int(max(0, min(col_cen - ncut,
                          max_col - 2 * ncut)))
    right = int(min(max_col - 1,
                    max(col_cen + ncut - 1, left + 20)))
    bottom = int(max(0, min(row_cen - ncut,
                            max_row - 2 * ncut)))
    top = int(min(max_row - 1,
                  max(row_cen + ncut - 1, bottom + 20)))

    # Select out larger image
    subim = image[bottom:top + 1, left:right + 1]
    subvar = variance[bottom:top + 1, left:right + 1]
    max_row_stamp, max_col_stamp = subim.shape

    # Select out larger variance and convert to error
    # Replace NaNs, infs, negative values with very small value
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        subvar[(subvar <= 0) | (~np.isfinite(subvar))] = 1e-15
    suberr = np.sqrt(subvar)

    # Shift the centroid and bounds for the new subimage
    col_cen -= left
    row_cen -= bottom
    est = fitpars
    est['col_mean'] = col_cen
    est['row_mean'] = row_cen
    bounds['col_mean'] = [0, max_col_stamp]
    bounds['row_mean'] = [0, max_row_stamp]
    bounds['col_sigma'] = [0, 2 * stampsize / factor]
    bounds['row_sigma'] = [0, 2 * stampsize / factor]

    try:
        fitpars, sigma, bestnorm = \
            pipecal_fitpeak(subim, estimates=est, bounds=bounds,
                            error=suberr, profile=profile)
    except (RuntimeError, ValueError):
        msg = 'Unable to fit subimage'
        log.error(msg)
        badfit = True
        if allow_badfit:
            fitpars = est.copy()
            sigma = est.copy()
        else:
            raise PipeCalError(msg) from None
    else:
        badfit = False

    log.debug(f'Fit parameters: {fitpars}')

    # Check for a valid fit
    num_finite = sum([np.isfinite(fitpars[i]) for i in fitpars])
    if badfit or len(fitpars) < 7 or num_finite != len(fitpars):
        log.debug('Invalid fit')
        # Invalid fit
        # Set all values except centroid and peak to zero
        for key in fitpars:
            fitpars[key] = 0.
            sigma[key] = 0.
        fitpars['col_mean'] = col_cen
        fitpars['row_mean'] = row_cen
        try:
            fitpars['dpeak'] = subim[int(np.round(row_cen)),
                                     int(np.round(col_cen))]
        except IndexError:  # pragma: no cover
            # catch for bad center definition
            pass

        factor = 1.
        factor_err = 0.
        pfactor = 1.
        pfactor_err = 0.
    else:
        # Valid fit

        # Calculate factors to account for power law
        if profile == 'moffat' and fitpars['beta'] != 0:
            # FWHM factor
            n = fitpars['beta']
            dn = sigma['beta']
            n_1 = 1 / n
            factor = 2 * np.sqrt(2**n_1 - 1)
            # Error on f is (df/dn) * error on n
            factor_err = (dn * (np.log(2) * 2**n_1)
                          / (n**2 * np.sqrt(2**n_1 - 1)))
            # Profile flux factor
            pfactor = fitpars['beta'] - 1
            pfactor_err = sigma['beta']

        else:
            factor_err = 0.
            pfactor = 1.
            pfactor_err = 0

    # Get values, calculate errors from fit parameters
    # 0 = baseline
    baseline = fitpars['baseline']
    baseline_err = sigma['baseline']

    # 1 = dpeak
    peak = fitpars['dpeak']
    peak_err = sigma['dpeak']

    # 2 = x_sigma
    fwhmx = fitpars['col_sigma'] * factor
    fwhmx_err = np.sqrt(factor**2 * sigma['col_sigma']**2
                        + fitpars['col_sigma']**2 * factor_err**2)
    # 3 = y_sigma
    fwhmy = fitpars['row_sigma'] * factor
    fwhmy_err = np.sqrt(factor**2 * sigma['row_sigma']**2
                        + fitpars['row_sigma']**2 * factor_err**2)

    # 4 = x_mean; shift back to full image CS
    starx = fitpars['col_mean'] + left
    starx_err = sigma['col_mean']

    # 5 = y_mean; shift back to full image CS
    stary = fitpars['row_mean'] + bottom
    stary_err = sigma['row_mean']

    # Check the rotation angle, convert to degrees
    # 6 = theta
    angle = fitpars['theta'] * 180. / np.pi
    if fwhmx < fwhmy:
        if angle < 0:
            angle += 90.
        else:
            angle -= 90.
    if angle < 0:
        angle += 180.
    angle_err = sigma['theta'] * 180. / np.pi

    # 7 = beta
    try:
        plaw = fitpars['beta']
        plaw_err = sigma['beta']
    except KeyError:
        plaw = 0.
        plaw_err = 0.

    # Propagate the error using:
    # var(a*b) = (var(a)*b**2) + (var(b)*a**2)
    # var(a/b) = (var(a)/b**2) + (var(b)*a**2/b**4)
    profile_flux = np.pi * (fitpars['dpeak']
                            * fitpars['col_sigma']
                            * fitpars['row_sigma']) / pfactor
    flux_err_1 = (sigma['dpeak']**2
                  * fitpars['col_sigma']**2
                  * fitpars['row_sigma']**2 / pfactor**2)
    flux_err_2 = (sigma['col_sigma']**2
                  * fitpars['dpeak']**2
                  * fitpars['row_sigma']**2 / pfactor**2)
    flux_err_3 = (sigma['row_sigma']**2
                  * fitpars['dpeak']**2
                  * fitpars['col_sigma']**2 / pfactor**2)
    flux_err_4 = (pfactor_err**2
                  * fitpars['dpeak']**2
                  * fitpars['col_sigma']**2
                  * fitpars['row_sigma']**2 / pfactor**4)

    profile_flux_err = np.pi * np.sqrt(flux_err_1 + flux_err_2
                                       + flux_err_3 + flux_err_4)

    # Do aperture photometry

    # set up source and sky apertures
    src_aper = photutils.CircularAperture(
        [starx, stary], r=aprad)
    try:
        sky_aper = photutils.CircularAnnulus(
            [starx, stary], r_in=skyrad[0], r_out=skyrad[1])
    except ValueError:
        # invalid sky radii passed: ignore sky extraction
        sky_aper = None

    # make a mask to ignore NaN values
    mask = np.zeros_like(image, dtype=bool)
    mask[~np.isfinite(image)] = True

    # extract source flux
    raw_table = photutils.aperture_photometry(
        image, src_aper, mask=mask)

    # extract variance on the source flux
    var_table = photutils.aperture_photometry(
        variance, src_aper, mask=mask)

    # extract sky background flux
    if sky_aper is not None:
        bg_table = photutils.aperture_photometry(
            image, sky_aper, mask=mask)

        # extract variance on the sky flux
        varbg_table = photutils.aperture_photometry(
            variance, sky_aper, mask=mask)

        # scale background flux to source aperture area
        sky_area = sky_aper.area
        src_area = src_aper.area
        bg_sum = bg_table['aperture_sum'] * src_area / sky_area
        varsky = varbg_table['aperture_sum'] * (src_area / sky_area)**2

        # store sky value as average per pixel
        sky = float(bg_sum / sky_area)
        sky_err = np.sqrt(float(varsky / sky_area ** 2))

        # get the standard deviation of the flux in the
        # background region
        sky_mask = sky_aper.to_mask(method='center')
        sky_flux = sky_mask.multiply(image)
        try:
            sky_std = np.nanstd(sky_flux[sky_flux != 0])
        except (TypeError, ValueError, IndexError):  # pragma: no cover
            sky_std = 0.0
    else:
        bg_sum = 0.0
        varsky = 0.0
        sky = 0.0
        sky_err = 0.0
        sky_std = 0.0

    # subtract background from source flux
    if not np.isfinite(float(bg_sum)):
        bg_sum = 0.0
    final_sum = raw_table['aperture_sum'] - bg_sum

    # variance is summed over the aperture
    varflux = var_table['aperture_sum']

    # propagate sky error to subtracted source error
    varflux += varsky

    # final source flux and associated error
    flux = float(final_sum)
    flux_err = np.sqrt(float(varflux))

    if not np.isfinite(flux):
        flux = 0.
    if not np.isfinite(sky):
        sky = 0.
    if not np.isfinite(flux_err):
        flux_err = 0.
    if not np.isfinite(sky_err):
        sky_err = 0.

    # Compile results
    phot = list()
    phot.append({'key': 'PHOTPROF',
                 'value': profile.upper(),
                 'comment': 'Profile fit type for photometry'})
    phot.append({'key': 'PHOTIMSZ',
                 'value': min(max_row_stamp, max_col_stamp),
                 'comment': 'Sub-image size for photometry'})
    phot.append({'key': 'PHOTAPER',
                 'value': aprad,
                 'comment': 'Aperture radius for photometry'})
    phot.append({'key': 'PHOTSKAP',
                 'value': '{0}, {1}'.format(skyrad[0], skyrad[1]),
                 'comment': 'Sky aperture radii for photometry (pix)'})
    phot.append({'key': 'STCENTX',
                 'value': [starx, starx_err],
                 'comment': 'Source centroid x-value (pix)'})
    phot.append({'key': 'STCENTY',
                 'value': [stary, stary_err],
                 'comment': 'Source centroid y-value (pix)'})
    phot.append({'key': 'STPEAK',
                 'value': [peak, peak_err],
                 'comment': 'Source fit peak value ({0})'.format(runits)})
    phot.append({'key': 'STBKG',
                 'value': [baseline, baseline_err],
                 'comment': 'Source fit background '
                            'level ({0})'.format(runits)})
    phot.append({'key': 'STFWHMX',
                 'value': [fwhmy, fwhmy_err],
                 'comment': 'Source fit FWHM in x-direction (pix)'})
    phot.append({'key': 'STFWHMY',
                 'value': [fwhmx, fwhmx_err],
                 'comment': 'Source fit FWHM in y-direction (pix)'})
    phot.append({'key': 'STANGLE',
                 'value': [angle, angle_err],
                 'comment': 'Source fit position angle (deg)'})
    phot.append({'key': 'STPWLAW',
                 'value': [plaw, plaw_err],
                 'comment': 'Source fit power law index'})
    phot.append({'key': 'STPRFLX',
                 'value': [profile_flux, profile_flux_err],
                 'comment': 'Source flux from profile ({0})'.format(runits)})
    phot.append({'key': 'STAPFLX',
                 'value': [flux, flux_err],
                 'comment': ('Source flux from aper phot '
                             '({0})'.format(runits))})
    phot.append({'key': 'STAPSKY',
                 'value': [sky, sky_err],
                 'comment': ('Sky flux from photometry '
                             '({0}/pix)'.format(runits))})
    phot.append({'key': 'STAPSSTD',
                 'value': sky_std,
                 'comment': ('Sky flux std dev '
                             '({0}/pix)'.format(runits))})
    return phot
