# Licensed under a 3-clause BSD style license - see LICENSE.rst

from warnings import catch_warnings, simplefilter

from astropy import log
from astropy.io.fits.header import Header
import numpy as np
from scipy import fftpack
from scipy.signal import medfilt

from sofia_redux.toolkit.utilities.fits import add_history_wrap

from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('JBClean')

__all__ = ['jbfft', 'jbmedian', 'jbclean']


def _nanmedian(*args, **kwargs):
    """Wrap nanmedian to ignore errors."""
    with catch_warnings():
        simplefilter('ignore')
        return np.nanmedian(*args, **kwargs)


def jbfft(data, bar_spacing=16):
    """
    Remove jailbars with FFT

    Interpolate over NaNs in the direction of the jailbars
    (y-direction).  Filters in the x-direction using FFT to
    remove jailbars.  NaNs are replaced following jailbar
    removal.  Other functions such as maskinterp should be
    used to replace NaNs after jailbar removal.

    Notes
    -----
    Filtering in the frequency domain inserts an amplitude
    error in the output of the order (jailbar amplitude / 16).
    Not noticable if jailbars are not significant, but still
    noticable.  Someone with more FFT knowledge than me should
    try and fix this.  Thankfully 'MEDIAN' is the default jailbar
    removal method.

    Parameters
    ----------
    data : np.ndarray
        (nrow, ncol)
    bar_spacing : int, optional
        known jailbar spacing between columns

    Returns
    -------
    numpy.ndarray
        Clean data array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray):
        log.error("data is not %s" % np.ndarray)
        return data
    mask = np.isnan(data)
    if mask.all():
        log.warning("data are all NaN")
        return data
    masked = data.copy()
    if mask.any():
        # interpolate over NaNs in along bar direction
        y = np.arange(data.shape[0])
        for x in range(data.shape[1]):
            d, m = masked[:, x], mask[:, x]
            if m.all():
                continue
            masked[:, x] = np.interp(y, y[~m], d[~m])
        # If any NaNs are left, just default to median
        masked[np.isnan(masked)] = _nanmedian(masked)

    # Create mask in Fourier space
    nbars = data.shape[1] // bar_spacing
    jailbar_mask = np.full(data.shape, np.complex(1.0))
    for idx in range(nbars - 1):
        jailbar_mask[:, (idx + 1) * bar_spacing] = 0

    fft_data = fftpack.fft(masked)
    fft_filtered = fft_data * jailbar_mask
    filtered = np.abs(fftpack.ifft(fft_filtered))

    # put NaNs back in
    filtered[mask] = np.nan
    return filtered


def jbmedian(data, width=4, bar_spacing=16):
    """
    Remove jailbars using the median of correlated columns

    Replace the jailbar patter with the median of correlated columns
    (every 16).  Large scale variations are removed by subtracting
    an image smoothed with a 16-pixel box

    Parameters
    ----------
    data : numpy.ndarray
        (nrow, ncol) input data array
    bar_spacing : int, optional
        known jailbar spacing between columns
    width : int, optional
        median filtering will be applied along rows using a kernel
        width of `bar_spacing + width`.  An additional pixel will be
        added to make the kernel odd sized if necessary.

    Returns
    -------
    numpy.ndarray
        (nrow, ncol) cleaned data array
    """
    if not isinstance(data, np.ndarray):
        log.error("data is not %s" % np.ndarray)
        return data
    if np.isnan(data).all():
        log.warning("data are all NaN")
        return data

    kw = bar_spacing + width
    kw += (kw % 2) == 0
    mid = kw // 2
    wrapped = np.empty((data.shape[0], data.shape[1] + kw - 1), data.dtype)
    wrapped[:, :mid] = data.copy()[:, -mid:]
    wrapped[:, mid: -mid] = data.copy()[:, :]
    wrapped[:, -mid:] = data.copy()[:, :mid]
    filtered = medfilt(wrapped, kernel_size=(1, kw))
    filtered = filtered[:, mid: -mid]
    jailbar = data - filtered

    nbars = data.shape[1] // bar_spacing
    indices = np.arange(nbars) * bar_spacing
    for row in jailbar:
        for offset in range(bar_spacing):
            row[indices + offset] = _nanmedian(row[indices + offset])

    # only happens if data.shape % bar_spacing != 0 (shouldn't happen)
    # it's here for completeness
    missed = data.shape[0] - (nbars * bar_spacing)
    if missed > 0:
        for row in jailbar:
            for offset in range(missed):
                idx = offset + 1
                row[-idx] = _nanmedian(row[indices + bar_spacing - idx])

    return data - jailbar


def jbclean(data, header=None, variance=None, bar_spacing=16, width=4):
    """
    Removes "jailbar" artifacts from images

    Filter the input data with a 16x16 box and remove the filtered
    image from the input image.  Then, the jil bar is most of the
    features found in the subtracted image.  We use this image to
    calcular the jail bar using median.  This function checks the
    configuration file (dripconf.txt) for a preferred method.  If
    JBCLEAN is set to fft, it does a Fourier transform of the image,
    to mask the 16-pixel periodic jailbar pattern.  This median
    method appears to work better than the fft method in most cases
    and is recommended.  If JBCLEAN is set to n, no jailbar correction
    is performed.  Note that if JBCLEAN=fft, this function is called
    by sofia_redux.instruments.forcast.clean on raw frames;
    if JBCLEAN=median, this function is called by
    sofia_redux.instruments.forcast.stack, on the chop/nod subtracted frames.

    Note: for now, it is assumed that the jailbar has no error, so the
    variance, if passes, is not modified.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        Input header; will be updated with HISTORY message
    variance : numpy.ndarray, optional
        Variance array (nrow, ncol) to update in parallel with
        the data array
    bar_spacing : int, optional
        known jailbar spacing between columns
    width : int, optional
        Only applies if JBCLEAN=median.  See `jbmedian` for details

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Jailbar-cleaned array (nimage, nrow, ncol) or (nrow, ncol)
        Propagated variance (nimage, nrow, ncol) or (nrow, ncol)
    """
    if not isinstance(header, Header):
        header = Header()
        addhist(header, 'Invalid header')

    var = variance.copy() if isinstance(variance, np.ndarray) else None
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, 'Did not clean jailbar (Invalid data)')
        log.error("must provide a valid 2D %s" % np.ndarray)
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("Variance must match data: %s" % type(variance))
        return

    jbcleaned = data.copy()
    jbmethod = getpar(header, 'JBCLEAN', default=None, dtype=str,
                      comment="Jail bar cleaning algorithm")
    jbmethod = jbmethod.upper().strip()
    if jbmethod == 'FFT':
        jbcleaned = jbfft(jbcleaned, bar_spacing=bar_spacing)
    elif jbmethod == 'MEDIAN':
        jbcleaned = jbmedian(jbcleaned, bar_spacing=bar_spacing, width=width)
    else:
        addhist(header, 'Jailbars not cleaned (invalid method)')
        log.error("jailbar method %s unrecognized" % jbmethod)
        return

    return jbcleaned, var
