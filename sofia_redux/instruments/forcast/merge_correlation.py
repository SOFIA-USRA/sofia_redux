# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap
from sofia_redux.toolkit.image.adjust import register_image

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.merge_shift import merge_shift
from sofia_redux.instruments.forcast.readmode import readmode

addhist = add_history_wrap('Merge')

__all__ = ['merge_correlation']


def merge_correlation(data, header, variance=None, maxshift=999999999.,
                      normmap=None, upsample=100, maxregister=16,
                      resize=True):
    """
    Merge an image using a correlation algorithm

    Add each frame of the data to a 2-d summation frame in a manner
    appropriate to the current reduction scheme, then average by the
    number of frames

    Parameters
    ----------
    data : numpy.ndarray
        Data to be merged i.e. frame with target images (nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        FITS header of the new input data file
    maxshift : float, optional
        Maximum possible value of the shift
    variance : numpy.ndarray, optional
        Propagate provided variance.  Must match shape of data array
        (nrow, ncol).
    normmap : numpy.ndarray, optional
        Array to hold the normalization map
    upsample : int, optional
        Determines the fractional pixel accuracy of the registration
        algorithm.  An upsample factor of 100 will result in
        registration accurate to one 100th of a pixel.
    maxregister : int or float or array-like, optional
        The maximum pixel shift allowed in each dimension applied
        during the registration algorithm.  Order of dimensions is
        (x, y).  The initial chop and nod estimates will be retrieved
        from the header.  A maximum correlation will be searched for
        in the range (chop_or_nod +/- maxregister).  Set to None to
        perform a maximum correlation search over the entire image.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Merged image i.e. frame with images of object at chop and nod
        positions merged.
    """
    # Initial sanity checks
    if not isinstance(header, fits.header.Header):
        log.error("invalid header")
        return

    var = variance.copy() if isinstance(variance, np.ndarray) else None
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, "merge not applied (invalid data)")
        log.error("invalid data - must by a 2-d array")
        return
    elif np.isnan(data).all():
        addhist(header, "merge not applied (invalid data)")
        log.error("data are all NaN")
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("invalid variance")

    border = getpar(
        header, 'BORDER', dtype=int, default=128,
        comment='additional border pixels')
    if border * 2 >= data.shape[0] or border * 2 >= data.shape[1]:
        addhist(header, "merge not applied (invalid border)")
        log.error("border %s is too large for data shape %s" %
                  (border, repr(data.shape)))
        return

    posdata = data.copy()
    # can't use NaNs with this method
    posdata[np.isnan(posdata)] = np.nanmedian(posdata)
    if border > 0:
        log.info('Removing {} pixel border from '
                 'consideration'.format(border))
        posdata = posdata[border: -border, border: -border]
    header_shift = imgshift_header(header, dither=False)

    chop = np.array([header_shift['chopx'], header_shift['chopy']])
    if chop[0] != 0 or chop[1] != 0 or maxregister is None:
        chop = register_image(posdata, -posdata, upsample=upsample,
                              maxshift=maxregister, shift0=chop)

    nod = np.array([header_shift['nodx'], header_shift['nody']])
    if (nod[0] != 0 or nod[1] != 0 or maxregister is None) \
            and np.sqrt(np.sum(nod ** 2)) < maxshift:
        nod = register_image(posdata, -posdata, upsample=upsample,
                             maxshift=maxregister, shift0=nod)

    # Read mode from the header
    chopnod = [chop[0], chop[1], nod[0], nod[1]]
    mode = readmode(header)
    return merge_shift(
        data, chopnod, header=header, variance=var, resize=resize,
        normmap=normmap, maxshift=maxshift, nmc=mode == 'NMC')
