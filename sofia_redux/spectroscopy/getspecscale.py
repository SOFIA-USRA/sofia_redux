# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
import numpy as np

from sofia_redux.toolkit.stats import medcomb

__all__ = ['getspecscale']


def getspecscale(stack, refidx=None):
    """
    Determines the scale factors for a _stack of spectra

    Returns an array of scale factors which when multiplied into each
    spectrum, produces a spectrum whose flux level is about that of
    the median of all the spectra.  If `refidx` is given, then the
    spectra are scaled to the spectrum _stack.

    Parameters
    ----------
    stack : array_like of (int or float)
        (n_spectra, n_wave)  An array of spectra from which to determine
        scale factors.
    refidx : int, optional
        If supplied, the _stack is scaled to the flux level of this
        spectrum.

    Returns
    -------
    numpy.ndarray of float
        (n_spectra,) array of scale factors
    """
    stack = np.array(stack).astype(float)
    if stack.ndim < 2 or stack.ndim > 3:
        log.error("Invalid _stack input")
        return
    nspec = stack.shape[0]

    # If refidx is not provided, use the median of the _stack
    refspec = medcomb(stack, axis=0)[0] if refidx is None else stack[refidx]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        normspec = refspec[None] / stack
        zi = np.isfinite(normspec)
        if not zi.any():
            scales = np.full((nspec,), 1.0)
        else:
            if stack.ndim > 2:
                scales = np.nanmedian(normspec, axis=(1, 2))
            else:
                scales = np.nanmedian(normspec, axis=1)
        scales[~np.isfinite(scales)] = 1.0

    return scales
