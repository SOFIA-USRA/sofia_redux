# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import hdinsert

__all__ = ['backsub']


def backsub(hdul, bgfile=None, method='flatnorm'):
    """
    Correct flux data for background level.

    To correct pixels individually, a background image must be
    provided with a FLUX extension and (optionally) an ERROR extension.
    The error is propagated assuming that the input data is independent
    of the sky data. Delete the extension or set the error plane to
    zero-values to skip error propagation.

    Otherwise, a single background level can be determined from the
    median of each image, or from a header value derived from a
    sky flat file, stored in the FLATNORM keyword in the primary header
    of the input data.  In this case, no error is propagated.

    Parameters
    ----------
    hdul : fits.HDUList
        Input data. Should have FLUX, ERROR, FLAT, and FLAT_ERROR
        extensions.
    bgfile : fits.HDUList, optional
        Background image. Should have FLUX and ERROR extensions.
    method : {'flatnorm', 'median'}, optional
        Method for background value determination, if background image
        is not provided.  If 'flatnorm', the header keyword FLATNORM
        will be subtracted from the data. If 'median', the image median
        will be subtracted from the data. Ignored if `bgfile` is not
        None.

    Returns
    -------
    fits.HDUList
        Corrected data, with updated FLUX and ERROR extensions.
    """
    # input data
    header = hdul[0].header
    flux = hdul['FLUX'].data
    var = hdul['ERROR'].data ** 2

    if bgfile is not None:
        bgdata = bgfile['FLUX'].data
        if 'ERROR' in bgfile:
            bgvar = bgfile['ERROR'].data ** 2
        else:
            bgvar = 0.0

        corrected_flux = flux - bgdata
        corrected_var = var + bgvar

        # record method and value
        hdinsert(header, 'BGSOURCE',
                 bgfile[0].header.get('FILENAME', 'UNKNOWN'),
                 'Background value source')
        hdinsert(header, 'BGVALUE', np.nanmedian(bgdata),
                 'Median background value')
    else:
        if method == 'flatnorm':
            bgval = hdul[0].header.get('FLATNORM', None)
            if bgval is None:
                log.warning('FLATNORM keyword is missing; background '
                            'will not be corrected.')
                bgval = 0
            hdinsert(header, 'BGSOURCE', 'FLATNORM keyword',
                     'Background value source')
        else:
            bgval = np.nanmedian(flux)
            hdinsert(header, 'BGSOURCE', 'Image median',
                     'Background value source')

        corrected_flux = flux - bgval
        corrected_var = var
        hdinsert(header, 'BGVALUE', bgval, 'Median background value')

    # make output hdul
    corrected = fits.HDUList()
    expected = ['FLUX', 'ERROR', 'BADMASK', 'EXPOSURE']
    for hdu in hdul:
        if hdu.name in expected:
            corrected.append(hdu)

    corrected['FLUX'].data = corrected_flux
    corrected['ERROR'].data = np.sqrt(corrected_var)

    return corrected
