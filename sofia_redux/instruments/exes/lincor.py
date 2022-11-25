# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes.utils import get_detsec, get_reset_dark
from sofia_redux.toolkit.utilities.fits import getdata
from sofia_redux.toolkit.utilities.func import goodfile


__all__ = ['lincor']


def lincor(data, header):
    """
    Correct raw readout frames for detector nonlinearity.

    Uses the linearity coefficients file stored in header['LINFILE']
    to apply a correction to each raw readout frame.  Each frame in the
    file corresponds to a polynomial correction coefficient for each pixel
    in the frame.

    This algorithm subtracts the data from a bias level obtained from
    an input dark file, calculates the polynomial correction from the
    coefficients file, then applies it to any pixel between the upper
    and lower limits.  The upper limit for each pixel is set in the
    first plane of the coefficients file.  The lower limit is set in
    header['LO_LIM'].

    Parameters
    ----------
    data : numpy.ndarray
        [nframe, nspec, nspat] input data array.
    header : fits.Header
        Input FITS header array.

    Returns
    -------
    corrected_data, mask : numpy.ndarray, numpy.ndarray
        The corrected data (numpy.float64) and mask (bool) where True
        indicates a good pixel, False indicates bad.  Both arrays are
        the same shape as `data`.
    """
    data = _check_data(data)
    minframe, maxframe, coeffs = _get_linearity_coefficients(header)
    bias = get_reset_dark(header)
    corrected_data, mask = _apply_correction(
        data, header, bias, minframe, maxframe, coeffs)

    return corrected_data, mask


def _get_linearity_coefficients(header):
    """Get linearity coefficients from a file indicated in the header."""
    nx = header['NSPAT']
    ny = header['NSPEC']
    linearity_file = str(header.get('LINFILE', 'UNKNOWN')).strip()

    if not goodfile(linearity_file, verbose=True):
        raise ValueError("Could not read linearity file: %s" %
                         linearity_file)
    coeffs = getdata(linearity_file)
    if coeffs.ndim != 3 or coeffs.shape[2] < nx or coeffs.shape[1] < ny:
        raise ValueError("Linearity coefficients too small for data "
                         "%s; the data is %s." %
                         (repr(coeffs.shape[1:]), repr((ny, nx))))

    xstart, xstop, ystart, ystop = get_detsec(header)
    minframe = np.round(coeffs[0, ystart:ystop, xstart:xstop]).astype(int)
    maxframe = np.round(coeffs[1, ystart:ystop, xstart:xstop]).astype(int)
    coeffs = coeffs[2:, ystart:ystop, xstart:xstop].astype(float)
    return minframe, maxframe, coeffs


def _apply_correction(data, header, bias, minframe, maxframe, coefficients):
    """Apply the linearity correction to the data."""
    upper_limit = float(header.get('UP_LIM', np.inf))
    lower_limit = float(header.get('LO_LIM', -np.inf))
    minframe = np.clip(minframe, lower_limit, None)
    maxframe = np.clip(maxframe, None, upper_limit)

    # mark bad pixels in mask
    warned = False
    mask = np.full(data.shape, False)
    corrected_data = np.empty(data.shape, dtype=float)
    for frame in range(data.shape[0]):

        # subtract the bias from the frame
        dataframe = bias - data[frame]

        # Find correctable points
        correct = (dataframe > minframe) & (dataframe < maxframe)

        mask[frame] |= dataframe > maxframe

        if not warned:
            nbad = mask[frame].sum()
            if nbad > (0.01 * data.shape[1] * data.shape[2]):
                log.warning("Many pixels are uncorrectable. "
                            "Likely nonlinear data")
                warned = True

        if correct.any():
            correction = coefficients[-1][correct].copy()
            subframe = dataframe[correct].copy()
            for j in range(coefficients.shape[0] - 2, -1, -1):
                correction *= subframe
                correction += coefficients[j][correct]

            idx = (correction < 1e-3) | (correction > 1)
            correction[idx] = 1.0
            subframe /= correction
            dataframe[correct] = subframe

        # Add bias back into frame and store
        corrected_data[frame] = bias - dataframe

    # return mask indicating good pixels
    mask = ~mask

    return corrected_data, mask


def _check_data(data):
    """Check input data for expected dimensions."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError("Data must be a 3-D cube (nframe, nspec, nspat)")
    return data
