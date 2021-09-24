# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

__all__ = ['calcvar']


def calcvar(data, header):
    """
    Calculate read and poisson noise from raw FLITECAM images.

    The variance is calculated using the procedure outlined by
    Vacca et al. (2004). As such this procedure is appropriate
    for data read using M/CDS (Multiple / correlated double
    sampling; Fowler sampling). The standard deviation (error) can
    then be calculated by taking the square root of the variance array.

    Parameters
    ----------
    data : numpy.ndarray
        Linearity corrected image array.
    header : astropy.io.fits.Header
        Raw image header.

    Returns
    -------
    numpy.ndarray
        Image array with the same dimensions as the input raw image
        containing the calculated variance.
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        log.error("Must provide valid data array")
        return

    if not isinstance(header, fits.header.Header):
        log.error("Must provide valid header")
        return

    # FLITECAM constants, provided by instrument PI
    gain = 7.1
    rms_readnoise = 43.0

    # time keywords from header
    itime = header.get('ITIME')
    coadds = header.get('COADDS')
    ndr = header.get('NDR')
    readtime = header.get('TABLE_MS')

    if None in [itime, coadds, ndr, readtime]:
        log.error("Missing time keywords in header")
        return

    readtime /= 1000.
    datavar = np.abs(data) / (gain * ndr * coadds**2 * itime**2)
    crtn = 1. - (readtime * (ndr**2 - 1.) / (3. * itime * ndr))
    readnoise_var = (2. * rms_readnoise**2) / (ndr * coadds
                                               * gain**2 * itime**2.)

    var = datavar * crtn + readnoise_var

    return var
