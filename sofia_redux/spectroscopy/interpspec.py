# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.interpolate import interp_error

__all__ = ['interpspec']


def interpspec(x, y, xout, error=None, leavenans=False, cval=np.nan):
    """
    Perform a linear interpolation and propagate errors

    Points outside the ix range are set to NaN.  NaNs are removed
    from both ix and ox.  NaNs are also removed in the y output
    unless leavenans is set to True.

    Parameters
    ----------
    x : array_like of float
        (N,) Independent values of spectrum
    y : array_like of float
        (N,) Dependent values of spectrum
    xout : array_like of float
        (M,) New independent values of spectrum
    error : array_like of float, optional
        (N,) Error values of the spectrum to propagate
    leavenans : bool, optional
        If True, leave NaNs in the input spectra
    cval : float, optional
        Value to fill in requested interpolation points outside the
        data range.

    Returns
    -------
    numpy.ndarary
        (M,) New dependent values of the spectrum
    """
    if not isinstance(cval, (int, float)):
        log.error("Constant value cval must be a float")
        return

    x, y = np.array(x), np.array(y)
    shape = x.shape
    if y.shape != shape:
        log.error("X and Y array shape mismatch")
        return
    doerr = error is not None
    if doerr:
        error = np.array(error)
        if error.shape != shape:
            log.error("X and error shape mismatch")
            return

    if not leavenans:
        mask = np.isfinite(x)
        np.logical_and(mask, np.isfinite(y), out=mask)
        x, y = x[mask], y[mask]
        if doerr:
            error = error[mask]

    yout = np.interp(xout, x, y, left=cval, right=cval)
    if not doerr:
        return yout
    return yout, interp_error(x, error, xout, cval=cval)
