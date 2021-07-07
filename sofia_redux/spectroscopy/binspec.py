# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import fasttrapz
from scipy.interpolate import interp1d

__all__ = ['binspec']


def binspec(x, y, delta, xout=None, lmin=None, lmax=None, average=False):
    """
    Bin a spectrum between lmin and lmax with bins delta wide

    Output wavelength bins are calculated from input lmin/lmax/dl.
    Flux values at edges of bins are interpolated, and values
    between the edges are summed.  If desired, the summed flux is
    divided by the bin size.  If either lmin or lmax is None, and
    xout is provided, the bins are assumed to be uneven and may have
    gaps.

    Parameters
    ----------
    x : array_like of (int or float)
        Independent variable
    y : array_like of (int or float)
        Dependent variable
    delta : int or float or (array_like of (int or float))
        Bin width.  May be a scalar value or 1-D array (N_out,) matching
        the output data array size of the second dimension.
    xout : int or float or array_like of (int or float), optional
        Output locations
    lmin : int or float, optional
        Minimum value of independent variable
    lmax : int or float, optional
        Maximum value of independent variable
    average : bool, optional
        If True, average the y over the bin (to conserve flux)

    Returns
    -------
    numpy.ndarray
        (2, N) where [0, :] = x out, and [1, :] = y out
    """
    try:
        x = np.array(x).astype(float)
    except (ValueError, TypeError):
        log.error("Invalid input x data type")
        return
    try:
        y = np.array(y).astype(float)
    except (ValueError, TypeError):
        log.error("Invalid input y data type")
        return
    if x.size < 2:
        log.error("At least two input points are required")
        return
    if x.shape != y.shape:
        log.error("X and Y arrays have different dimensions")
        return
    if xout is not None:
        try:
            if not hasattr(xout, '__len__'):
                xout = [xout]
            xout = np.array(xout).astype(float)
        except (ValueError, TypeError):
            log.error("Invalid output x data type")
            return
    try:
        if not hasattr(delta, '__len__'):
            delta = [delta]
        delta = np.array(delta).astype(float)
        ndelta = delta.size
    except (ValueError, TypeError):
        log.error("Invalid delta data type")
        return

    if (lmin is None or lmax is None) and xout is not None:
        # Assume we have a set of output central wavelengths to match
        nxout = xout.size
        if ndelta != nxout:
            if hasattr(delta, '__len__'):
                delta = delta[0]
            delta = np.full(len(xout), delta)
        xledg = xout - delta / 2
        xhedg = xout + delta / 2
    elif ndelta == 1:
        # single value for delta (constant dl)
        lmin = x.min() if lmin is None else lmin
        lmax = x.max() if lmax is None else lmax
        nedgs = int((lmax - lmin) / delta) + 1
        xledg = np.arange(nedgs) * delta + lmin
        xhedg = xledg.copy() + delta
        xout = xledg + delta / 2
        nxout = xout.size
        delta = np.full(nxout, delta[0])
    else:
        lmin = x.min() if lmin is None else lmin
        lmax = x.max() if lmax is None else lmax
        nbins = ndelta

        xlow = lmin
        idx = 0
        xledg = []
        xhedg = []
        while idx < nbins:
            xupp = xlow + delta[idx]
            if xupp <= lmax:
                xledg.append(xlow)
                xhedg.append(xupp)
            else:  # pragma: no cover
                # this clause is covered in tests,
                # but coverage doesn't count it
                break
            xlow = xupp
            idx += 1
        xledg = np.array(xledg)
        xhedg = np.array(xhedg)

        xout = (xledg + xhedg) / 2
        nxout = xout.size
        delta = xhedg - xledg

    nout = len(xout)
    yout = np.zeros(nxout) if nxout > nout else np.zeros(nout)
    # interpolate values at edges of bins
    xedgarr = list(xledg)
    xedgarr.append(xhedg[-1])
    finterp = interp1d(x, y, fill_value='extrapolate')
    yedgarr = finterp(xedgarr)

    for i, (lower, upper) in enumerate(zip(xledg, xhedg)):
        idx = np.array((x > lower) & (x <= upper))

        area = fasttrapz(y[idx], x[idx])
        # add on edges
        if idx.sum() != 0:
            iidx = np.nonzero(idx)[0]
            i0, i1 = iidx[0], iidx[-1]
            x0, x1 = x[i0], x[i1]
            y0, y1 = y[i0], y[i1]
            if lower < x0:
                area += (x0 - lower) * (yedgarr[i] + y0) / 2.0
            if upper > x1:
                area += (upper - x1) * (yedgarr[i + 1] + y1) / 2.0

        yout[i] = area

    if average:
        yout /= delta
    return np.stack((xout, yout))
