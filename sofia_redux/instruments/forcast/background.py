# Licensed under a 3-clause BSD style license - see LICENSE.rst

import statistics

import numpy as np

from sofia_redux.toolkit.utilities.fits import hdinsert, href, kref

__all__ = ['background', 'mode']


def mode(data):
    """
    Return the most common data point from discrete or nominal data

    If there is not exactly one most common value, the minimum value
    (of the most common values) will be returned.

    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    The most common value

    Raises
    ------
    ValueError
        If data is empty
    """
    d = np.array(data).ravel()
    result = min(statistics.multimode(d))
    return result


def background(data, section, header=None, stat=None, mask=None):
    """
    Calculate the background of the image

    Takes median or mode of good pixels in selected region

    Parameters
    ----------
    data : numpy.ndarray
        (nimage, ncol, nrow)
    section : tuple of (int or float) or list of (int or float)
        Data section to use.  Should have the format (X0, Y0, XDIM, YDIM)
        where X0, Y0 is the center of the section and XDIM, YDIM is the
        dimension of the section.
    header : astropy.io.fits.header.Header, optional
        Input header array to be updated with NLINSLEV (containing the
        background levels) and HISTORY keyworks
    stat : str, optional
        Statistic to use to calculate the background level ('median' or
        'mode')
    mask : numpy.ndarray, optional
        (col, row) Illumination mask to indicate regions of the image to
        use in calculating the image (True = good)

    Returns
    -------
    numpy.ndarray
        (nimage) containing background levels for each input frame
    """

    if len(section) != 4:
        raise ValueError(
            "section should have the format (x0, y0, xdim, ydim)")

    ndim = len(data.shape)
    if ndim not in [2, 3]:
        return

    box_x = section[0] - section[2] / 2.0, section[0] + section[2] / 2.0
    box_y = section[1] - section[3] / 2.0, section[1] + section[3] / 2.0
    if ndim == 2:
        nplanes, (ny, nx) = 1, data.shape
    else:
        nplanes, ny, nx = data.shape

    if not isinstance(mask, np.ndarray):
        mask = np.full((ny, nx), True)
    mask = mask.astype(bool)

    y, x = np.meshgrid(*map(np.arange, mask.shape), indexing='ij')
    mask *= (x > box_x[0]) & (x < box_x[1]) & (y > box_y[0]) & (y < box_y[1])
    if not mask.any():
        return

    func = mode if stat == 'mode' else np.nanmedian
    if ndim == 2:
        result = [func(data[mask])]
    else:
        result = tuple(map(lambda u: func(data[u, :, :][mask.astype(bool)]),
                           range(nplanes)))

    if header is not None:
        nlinslev = ','.join(str(x) for x in result)
        hdinsert(header, 'NLINSLEV', '[%s]' % nlinslev,
                 comment='Signal level of the background',
                 refkey=kref)

        history = 'Background: level is (%s) ' % nlinslev
        history += 'using section defined by NLINSLEV keyword'
        hdinsert(header, 'HISTORY', history, refkey=href)

    return np.array(result)
