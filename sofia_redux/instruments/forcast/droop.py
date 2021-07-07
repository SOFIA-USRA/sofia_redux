# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io.fits.header import Header
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap

from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('Droop')

__all__ = ['droop']


def droop(data, header=None, frac=None, variance=None):
    """
    Corrects droop electronic signal

    The FORCAST arrays and readout electronics exhibit a linear
    response offset caused by he presence of a signal on the array.
    This effect is called "droop" since the result is a reduced
    signal that is proportional to the total signal in the 15 other
    pixels in row read from the multiplexer simultaneously with
    that pixel.  The droop correction removes the droop offset by
    multiplying each pixel by a value derived from the sum of
    every 16th pixel in the same row all multiplied by an empirically
    determined offset fraction: droopfrac = 0.0035.

    Parameters
    ----------
    data : numpy.ndarray
        Input dara array (nimage, nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        Input FITS header; will be updated with a HISTORY message
    frac : float, optional
        Channel suppression correction factor.  If this keyword
        is not set, the configuration file will be checked for the
        key FRACDROOP.  If not found, the default value of 0.0035
        will be used.
    variance : numpy.ndarray, optional
        Variance array (nimage, nrow, ncol) to update in parallel
        with the data array.  Note that, for now, it is assumed that
        there is no error in the droop correction.  If variance is
        passed, it will be returned unmodified.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The droop corrected array (nimage, nrow, ncol) or (nrow, ncol)
        propagated variance (nimage, nrow, ncol) or (nrow, ncol)
    """
    if not isinstance(header, Header):
        header = Header()
        addhist(header, 'Created header')

    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        addhist(header, 'Droop not corrected (invalid data)')
        log.error('invalid data')
        return

    dovar = variance is not None and variance.shape == data.shape
    var = variance.copy() if dovar else None
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (invalid variance)')
        log.error('invalid variance')

    if frac is None:
        frac = getpar(header, 'FRACDROOP', dtype=float, default=0.0035,
                      comment='fraction for droop correction')

    minval = getpar(header, 'MINDROOP', dtype=float, default=0, warn=True,
                    comment='minimum value for droop correction')
    maxval = getpar(header, 'MAXDROOP', dtype=float, default=-65535, warn=True,
                    comment='maximum value for droop correction')
    nreadouts = getpar(header, 'NRODROOP', dtype=int, default=16, warn=True,
                       comment='number of rows for droop correction')

    # Put into cast friendly format
    ndim = len(data.shape)
    corr = np.array([data.copy()]) if ndim == 2 else data.copy()
    corr = corr.astype('float64')

    # for each row, add a correction to each chunk of <nreadouts> pixels
    # based on a fraction of their sum
    nsets = int(data.shape[-1] / nreadouts)  # readouts per row
    for i in range(nsets):
        x1, x2 = i * nreadouts, (i + 1) * nreadouts
        section = corr[:, :, x1: x2]
        section += frac * np.nansum(section, axis=2, keepdims=True)

    # Ensure corrected values lie within (minval, maxval)
    finite = ~np.isnan(corr)
    low, high = finite.copy(), finite.copy()
    low[low] &= corr[low] < minval
    high[high] &= corr[high] > maxval
    corr[low] = minval
    corr[high] = maxval

    # Put back to original format
    corr = corr.astype(data.dtype)
    if ndim == 2:
        corr = corr[0]

    addhist(header, 'Applied channel suppression (droop) correction')
    addhist(header, 'Channel suppression correction factor %f' % frac)

    return corr, var
