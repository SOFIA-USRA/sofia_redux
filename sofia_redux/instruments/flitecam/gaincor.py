# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import hdinsert

__all__ = ['gaincor']


def gaincor(hdul):
    """
    Correct flux data for gain variations.

    The normalized flat field and associated error should have been
    previously generated and attached to the input.

    Flat errors are propagated to the output error plane, assuming
    that flat and flux are statistically independent.  Note that this
    may not be a correct assumption for all FLITECAM data.  For example,
    sky files corrected with a master sky flat generated from those
    files will have correlated pixels.  Error estimates in cases like
    these should be treated as suspect.

    Parameters
    ----------
    hdul : fits.HDUList
        Input data.  Should have FLUX, ERROR, FLAT, and FLAT_ERROR
        extensions.

    Returns
    -------
    fits.HDUList
        Corrected data, with updated FLUX and ERROR extensions.
        The FLAT and FLAT_ERROR extensions are dropped.
    """
    # input data
    flux = hdul['FLUX'].data
    var = hdul['ERROR'].data ** 2
    flat = hdul['FLAT'].data
    if 'FLAT_ERROR' in hdul:
        flat_var = hdul['FLAT_ERROR'].data ** 2
    else:
        flat_var = np.zeros_like(flat)
    flat_hdr = hdul['FLAT'].header
    flat_norm = flat_hdr.get('FLATNORM', 1.0)
    flat_obs = flat_hdr.get('ASSC_OBS', 'UNKNOWN')

    # check for zeros
    flat[flat == 0] = np.nan

    # divide by flat
    corrected_flux = flux / flat

    # propagate error, assuming flat and flux are statistically
    # independent
    corrected_var = var / flat ** 2 + flat_var * flux ** 2 / flat ** 4

    # make output hdul with only science extensions
    corrected = fits.HDUList()
    expected = ['FLUX', 'ERROR', 'BADMASK', 'EXPOSURE']
    for hdu in hdul:
        if hdu.name in expected:
            corrected.append(hdu)

    corrected['FLUX'].data = corrected_flux
    corrected['ERROR'].data = np.sqrt(corrected_var)

    # add some flat values to output header
    header = corrected[0].header
    hdinsert(header, 'FLATNORM', flat_norm, 'Flat normalization value')
    hdinsert(header, 'FLAT_OBS', flat_obs, 'Flat OBS-IDs')

    return corrected
