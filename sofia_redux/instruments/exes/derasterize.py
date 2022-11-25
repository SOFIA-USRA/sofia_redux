# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes.readraw import readraw
from sofia_redux.toolkit.fitting.polynomial import polyfitnd
from sofia_redux.toolkit.utilities.fits import set_log_level

__all__ = ['derasterize']


def derasterize(data, header, dark_data=None, dark_header=None, overlap=32):
    """
    Read and recombine a rasterized flat file.

    When background fluxes are high, raster flats are taken in subarray
    chunks to avoid saturating the detector.  These chunks are stored in
    a custom format in the raw FITS file and need special handling to
    be reassembled into a full-frame flat for use with the science data.

    The procedure is:

       1. Divide the input cube into detector stripes.
       2. Call `readraw` for each stripe to coadd raw readouts.
       3. Use overlap regions to fit and correct for gain and offset
          differences between stripes.
       4. Assemble corrected stripes into a full 1024 x 1024 array.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, nspec, nspat].
    header : fits.Header
        Header of FITS file. May be updated in place.
    dark_data : numpy.ndarray, optional
        3D raw raster dark data [nframe, nspec, nspat].  Must match the
        input data if provided.
    dark_header : fits.Header, optional
        Header of dark FITS file.
    overlap : int, optional
        The size of the overlap region in each detector stripe.

    Returns
    -------
    deraster, variance, mask : 3-tuple of numpy.ndarray
       The derasterized and coadded data, variance, and good data mask.
       The deraster and variance arrays have shape (nframes, ny, nx).
       The mask has shape (ny, nx) and Boolean data type, where True
       indicates a good pixel; False indicates a bad pixel.
    """

    # check for dark data
    if dark_data is not None:
        if dark_data.shape != data.shape:
            message = f'Dark data has wrong dimensions {dark_data.shape}'
            raise RuntimeError(message)
        if dark_header is None:
            message = 'Dark header must be provided with dark data'
            raise RuntimeError(message)

    # nx is 1024 + 8 reference pix
    # ny is subarray stripe + overlap pixels
    # nz is number of stripes x nframes per pattern x npattern
    nz, ny, nx = data.shape
    full_size = 1024

    # stripe size
    ns = ny - overlap
    nstripe = full_size // ns
    nframe = nz // nstripe
    if full_size % ns != 0:
        message = (f'Specified overlap of {overlap} rows '
                   f'does not match data dimensions {data.shape}.')
        raise RuntimeError(message)
    if nz % nstripe != 0:
        message = (f'Number of stripes ({nstripe}) does not '
                   f'match data dimensions {data.shape}.')
        raise RuntimeError(message)

    deraster = np.zeros((1, full_size, full_size))
    variance = np.zeros_like(deraster)
    lin_mask = np.full((full_size, full_size), True)

    # some fudge values for subarray location and edge effects
    fit_buffer = 5
    bottom_buffer = 2
    offset_fudge = 2

    last_overlap = None
    for i in range(nstripe):
        log.info('')
        log.info(f'Stripe {i + 1}')
        log.info('')
        zstart = i * nframe
        zstop = zstart + nframe
        raw_frames = data[zstart:zstop]

        # 2nd and subsequent subarray seem to be offset by a couple
        # pixels in y, and the bottom row or two look bad
        if i == nstripe - 1:
            # last stripe: use offset, clip the bottom,
            # don't go beyond top row
            fudge = offset_fudge
            bottom = bottom_buffer
            top = 0
            raw_ystart = full_size - ns - overlap - fudge
        elif i > 0:
            # middle stripes: use offset, clip the bottom,
            # expand the top into the overlap
            fudge = offset_fudge
            bottom = bottom_buffer
            top = bottom_buffer
            raw_ystart = i * ns - fudge
        else:
            # first stripe: no offset, don't clip the bottom,
            # expand the top into the overlap
            fudge = 0
            bottom = 0
            top = bottom_buffer
            raw_ystart = 0

        # index into the full output array
        ystart = i * ns - fudge + bottom
        ystop = i * ns + ns - fudge + top

        # set the ectpat to extract the correct region of the bad pixel mask
        raw_header = header.copy()
        raw_ystop = ns + overlap
        ectpat = f'0 1 {raw_ystart // 2} {raw_ystop // 2} 0 {nx}'
        raw_header['ECTPAT'] = ectpat
        raw_header['FRAMETIM'] *= full_size / ny

        # coadd frames with simple destructive mode, no linearity,
        # tossing first 2 ints
        log.info('Reading flat frames')
        coadd, var, lin = readraw(raw_frames, raw_header, algorithm=0,
                                  do_lincor=False, toss_nint=2)
        coadd = np.squeeze(coadd)
        var = np.squeeze(var)

        # directly subtract dark if available
        if dark_data is not None:
            raw_dark = dark_data[zstart:zstop]
            raw_header = dark_header.copy()
            raw_header['ECTPAT'] = ectpat
            raw_header['FRAMETIM'] *= full_size / ny
            log.info('Subtracting dark frames')
            with set_log_level('WARNING'):
                coadd_dark, _, _ = readraw(raw_dark, raw_header, algorithm=0,
                                           do_lincor=False, toss_nint=2)
            coadd -= np.squeeze(coadd_dark)

        if i == 0:
            # no correction for first stripe
            corrected_coadd = coadd
            last_overlap = corrected_coadd[
                ns + fit_buffer:ns + overlap - fit_buffer,
                fit_buffer:full_size - fit_buffer]
        else:
            # overlap with previous is at bottom of stripe array
            if i != nstripe - 1:
                overlap_section = coadd[
                    fit_buffer:overlap - fit_buffer,
                    fit_buffer:full_size - fit_buffer]
            else:
                # doubled for last stripe
                overlap_section = coadd[
                    fit_buffer + overlap: 2 * overlap - fit_buffer,
                    fit_buffer:full_size - fit_buffer]

            # derive fit gain/offset from overlap
            diff = last_overlap - overlap_section
            coeff = polyfitnd(overlap_section.ravel(), diff.ravel(),
                              1, robust=6.0)

            # correct new stripe to previous
            correction = coadd * coeff[1] + coeff[0]
            corrected_coadd = coadd + correction

            # keep top section for overlap with next stripe
            last_overlap = corrected_coadd[
                ns + fit_buffer:ns + overlap - fit_buffer,
                fit_buffer:full_size - fit_buffer]

        # place stripe in output flat
        if i < nstripe - 1:
            # all but last stripe: overlap is on top
            deraster[0, ystart:ystop] = corrected_coadd[bottom:ns + top]
            variance[0, ystart:ystop] = var[bottom:ns + top]
            lin_mask[ystart:ystop] = lin[bottom:ns + top]
        else:
            # last stripe: overlap is on bottom
            deraster[0, ystart:ystop] = corrected_coadd[overlap + bottom:]
            variance[0, ystart:ystop] = var[overlap + bottom:]
            lin_mask[ystart:ystop] = lin[overlap + bottom:]

    # scale factor for subarray
    deraster *= full_size / ny
    variance *= (full_size / ny) ** 2

    # set header keywords for future steps
    header['DATASRC'] = 'CALIBRATION'
    header['OBSTYPE'] = 'FLAT'
    header['DETSEC'] = '[1:1024,1:1024]'
    header['NSPAT'] = full_size
    header['NSPEC'] = full_size

    return deraster, variance, lin_mask
