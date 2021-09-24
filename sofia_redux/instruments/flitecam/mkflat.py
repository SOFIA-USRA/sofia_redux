# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
from photutils import detect_sources
from photutils import detect_threshold

from sofia_redux.instruments.forcast.hdmerge import hdmerge
from sofia_redux.toolkit.image.combine import combine_images
from sofia_redux.toolkit.utilities.fits import hdinsert, set_log_level

__all__ = ['mkflat']


def mkflat(infiles, method='median', weighted=True, robust=True,
           sigma=5, maxiters=None, psf_fwhm=6.0, obj_sigma=2.0):
    """
    Make a flat from dithered sky images.

    The process is:

    - Directly combine all images into a draft flat.
    - Divide input images by the draft flat.
    - Use the resulting gain-corrected images to identify and
      mask any sources.
    - Scale all images to the median value of all the masked data.
    - Combine scaled, masked images into a final flat, propagating
      errors.

    Parameters
    ----------
    infiles : list of fits.HDUList
        Input data.  Should have FLUX, ERROR, and BADMASK extensions.
    method : {'mean', 'median', 'sum'}, optional
        Combination function to use.
    weighted : bool, optional
        If True and method is 'mean', the input variance will be used
        to weight the mean combination.  Ignored if variance is not
        provided.
    robust : bool, optional
        If True, the threshold and maxiters parameters will be used
        to reject outliers before combination.  Outlier rejection is
        performed via `astropy.stats.sigma_clip`.
    sigma : float, optional
        The number of standard deviations for clipping; passed to
        sigma_clip.
    maxiters : int or None, optional
        The maximum number of clipping iterations to perform; passed to
        sigma_clip
    psf_fwhm : float, optional
        Expected FWHM of sources for masking. Used to smooth the image
        before detecting objects.
    obj_sigma : float, optional
        The number of standard deviations above the background, used
         in detecting sources.

    Returns
    -------
    fits.HDUList
        Normalized flat field correction image, with FLAT, FLAT_ERROR, and
        FLAT_BADMASK extensions.  Array dimensions match input.
    """
    shape = infiles[0][0].data.shape
    flat_mask = np.zeros(shape, dtype=int)

    data_list, var_list, header_list = list(), list(), list()
    for hdul in infiles:
        data_list.append(hdul['FLUX'].data.copy())
        var_list.append(hdul['ERROR'].data.copy() ** 2)
        header_list.append(hdul[0].header)

    # combine images for a draft flat, with high robust rejection
    if len(data_list) == 1:
        # just directly use the data as a flat
        flat_data, flat_var = data_list[0], var_list[0]
    else:
        draft_method = 'median'
        draft_flat, draft_flat_var = combine_images(
            data_list, variance=var_list,
            method=draft_method, weighted=weighted,
            robust=robust, sigma=obj_sigma, maxiters=maxiters)
        draft_flat /= np.nanmedian(draft_flat)

        # get kernel to smooth image by the beam
        gsigma = psf_fwhm * gaussian_fwhm_to_sigma
        gsize = int(gsigma * 2)
        kernel = Gaussian2DKernel(gsigma, x_size=gsize, y_size=gsize)
        kernel.normalize()

        # draft correct and mask sources in all images
        masked_data = []
        masked_var = []
        scale_to = None
        for i, data in enumerate(data_list):
            # draft gain correction
            with np.errstate(invalid='ignore'):
                gain_corr = data / draft_flat

            # replace any NaNs with nearby non-NaN,
            # for smoother source detection
            nanval = np.isnan(gain_corr)
            idx = np.where(~nanval, np.arange(nanval.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            gain_corr = gain_corr[np.arange(idx.shape[0])[:, None], idx]

            # clipped background threshold image with low
            # detection threshold
            with set_log_level('CRITICAL'):
                threshold = detect_threshold(gain_corr, nsigma=obj_sigma)

                # detect any 5 connected pixels above the threshold
                segmented = detect_sources(gain_corr, threshold,
                                           npixels=5, filter_kernel=kernel)
            if segmented is not None:
                obj_mask = (segmented.data != 0)
            else:
                obj_mask = np.full(flat_mask.shape, False)

            # add objects to mask
            if np.any(obj_mask):
                flat_mask[obj_mask] += 1

            # masked the data
            mdata = data_list[i].copy()
            mvar = var_list[i].copy()
            if np.any(obj_mask):
                mdata[obj_mask] = np.nan
                mvar[obj_mask] = np.nan

            masked_data.append(mdata)
            masked_var.append(mvar)

        # scale data to masked median value
        scale_to = np.nanmedian(masked_data)
        for d, v in zip(masked_data, masked_var):
            scale_from = np.nanmedian(d)
            d *= scale_to / scale_from
            v *= (scale_to / scale_from) ** 2

        # make flat from masked, scaled data
        flat_data, flat_var = combine_images(
            masked_data, variance=masked_var, method=method, weighted=weighted,
            robust=robust, sigma=sigma, maxiters=maxiters)

    # normalize flat
    norm = np.nanmedian(flat_data)
    if np.allclose(norm, 0) or not np.isfinite(norm):
        raise ValueError('No valid flat data')

    flat_data /= norm
    flat_err = np.sqrt(flat_var) / norm

    flat_header = hdmerge(header_list)
    hdinsert(flat_header, 'EXTNAME', 'FLAT', comment='Extension name')
    hdinsert(flat_header, 'PRODTYPE', 'normalized_gain',
             comment='Product type')
    hdinsert(flat_header, 'FLATNORM', norm,
             comment='Flat normalization value')
    flat = fits.HDUList([fits.PrimaryHDU(data=flat_data, header=flat_header),
                         fits.ImageHDU(data=flat_err, name='FLAT_ERROR'),
                         fits.ImageHDU(data=flat_mask, name='FLAT_BADMASK')])

    return flat
