# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.stats import medcomb
from sofia_redux.toolkit.convolve.kernel import savitzky_golay

__all__ = ['speccor']


def speccor(stack, fwidth=4, err_stack=None, refspec=None, select=None,
            window=11, info=None):
    """
    Correct a _stack of spectra for shape differences

    Returns an array of spectra corrected to the reference spectrum

    Parameters
    ----------
    stack : array_like of (int or float)
        (n_spectra, n_wave) array of spectra to be corrected.
    fwidth : (int or float), optional
        FFT filter width
    err_stack : array_like of (int or float)
        (n_spectra, n_wave) array of errors corresponding to the _stack.
        If given, the corrected error array is given as a second output.
    refspec : array_like of (int or float)
        (n_wave,) reference spectrum that the spectra in the _stack
        are corrected to.
    select : array_like of (int or bool)
        (n_spectra,) array denoting which spectra to use to determine
        the reference spectrum.  However, all the spectra in the _stack
        are scaled to the reference spectrum (True = good, False = bad).
    window : int, optional
        Positive odd integer defining the width of the Savitzky-Golay used
        to smooth each spectra before FFT.
    info : dict, optional
        If supplied will be updated with mask and corrections

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        (n_spectra, n_wave) array of spectra corrected to the reference
        spectrum.  If `err_stack` is supplied, it will be returned as
        a second array.
    """
    stack = np.array(stack)
    if stack.ndim < 2:
        log.error("Invalid _stack input")
        return
    if err_stack is not None:
        err_stack = np.array(err_stack)
        if err_stack.shape[:2] != stack.shape[:2]:
            log.error("Error _stack does not match input _stack")
            return
        doerr = True
    else:
        doerr = False
    n_spec, n_wave = stack.shape[:2]
    if select is None:
        goodspec = np.arange(n_spec)
    else:
        goodspec = np.where(select)[0]

    goodpix = np.all(np.isfinite(stack), axis=0)
    npoints = goodpix.sum()

    if not goodpix.all():
        badidx = np.arange(n_spec), np.argwhere(~goodpix)
        stack[badidx] = np.nan

    # Get shape reference spectrum
    if refspec is not None:
        rspec = refspec
    else:
        rspec, _ = medcomb(stack[goodspec], axis=0)

    # Create filter
    ffilter = np.empty(npoints)
    mid, mod2 = np.divmod(npoints, 2)[:2]
    ffilter[:mid + mod2] = np.arange(mid + mod2)
    ffilter[mid + mod2:] = np.arange(mid)[::-1]
    ffilter = 1 / (1 + (ffilter / fwidth) ** 10)

    # Filter the reference spectrum
    ref_low_freq = np.fft.ifft(np.fft.fft(rspec[goodpix]) * ffilter).real

    update = isinstance(info, dict)
    if update:
        info['corrections'] = np.zeros((n_spec, npoints))
        info['correction_mask'] = goodpix
    x = np.arange(n_wave).astype(float)
    for i, spectrum in enumerate(stack):
        # Smooth to remove bad pixels that interfere with the FFT
        spec_sg = savitzky_golay(
            x, spectrum, window, robust=5, eps=0.1)
        low_freq = np.fft.ifft(np.fft.fft(spec_sg[goodpix]) * ffilter).real
        correction = ref_low_freq / low_freq
        stack[i, goodpix] *= correction
        if update:
            info['corrections'][i] = correction
        if doerr:
            err_stack[i, goodpix] *= correction

    if doerr:
        return stack, err_stack
    else:
        return stack
