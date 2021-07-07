# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import bitset
from sofia_redux.toolkit.interpolate import findidx
import numpy as np

__all__ = ['interpflagspec']


def interpflagspec(x, y, xout, nbits=8, cval=0):
    """
    Performs a linear interpolation on a bit-set flag array

    Parameters
    ----------
    x : array_like of float
        (N,) independent values of spectrum
    y : array_like of int
        (N,) dependent values of the bit-set flag array
    xout : (array_like of float) or float
        (M,) new independent values of spectrum
    nbits : int, optional
        The number of bits to scan through.  The assumption is that
        they are sequential starting with the first bit.
    cval : float, optional
        Value to fill in requested interpolation points outside the
        data range.

    Returns
    -------
    numpy.ndarray (int)
        (M,) new dependent values of the bit-set flag array at `xout`.
    """

    if not hasattr(xout, '__len__'):
        isarr = False
        xout = [xout]
    else:
        isarr = True

    x, y, xout = np.array(x), np.array(y), np.array(xout)
    if x.shape != y.shape:
        raise ValueError("x and y array shape mismatch")
    try:
        y = y.astype(int)
    except (ValueError, TypeError):
        raise ValueError("y must be convertable to %s" % int)

    mask = np.isfinite(x)
    np.logical_and(mask, np.isfinite(y), out=mask)
    if not mask.any():
        return np.full(xout.shape, cval)
    x, y = x[mask], y[mask]

    idx = findidx(x, xout, left=np.nan, right=np.nan)
    mask = np.isfinite(idx)
    nvalid = mask.sum()
    if nvalid == 0:
        return np.full(xout.shape, cval)

    xout, yout = xout[mask], np.zeros(nvalid, dtype=int)
    left = np.floor(idx[mask]).astype(int)
    right = np.ceil(idx[mask]).astype(int)

    on_point = left == right  # points where no interpolation is required
    ipoints = ~on_point  # points where interpolation is required

    left, center, right = left[ipoints], left[on_point], right[ipoints]
    m = (xout[ipoints] - x[left]) / (x[right] - x[left])
    tmp = np.empty(nvalid, dtype=int)
    for bit in range(nbits):
        bset = bitset(y, np.array([bit]), skip_checks=True)
        tmp[:] = 0
        if np.any(on_point):
            tmp[on_point] = bset[center]
        if np.any(ipoints):
            dy = bset[right] - bset[left]
            bset = bset[left] + (m * dy)
            tmp[ipoints] = np.ceil(bset, out=bset).astype(int)
        yout += tmp * (2 ** bit)

    np.mod(yout, 256, out=yout)

    if not isarr:
        return yout[0]
    else:
        result = np.full(mask.shape, cval)
        result[mask] = yout
        return result
