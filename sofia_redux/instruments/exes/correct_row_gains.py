# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.toolkit.fitting.polynomial import polyfitnd

__all__ = ['correct_row_gains']


def correct_row_gains(data):
    """
    Correct odd/even row gain offsets.

    Changes are additive only, so associated variance is not propagated.

    The procedure is:

         1. Split data into odd and even rows.
         2. Subtract odd data from even data.
         3. Fit the difference with a 1st order polynomial to
            derive gain offsets.
         4. Use the fit coefficients to correct the odd rows to
            the even rows.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, nspec, nspat].

    Returns
    -------
    corrected_data : numpy.ndarray
       The corrected data.
    """
    corrected_data = data.copy()

    even_rows = data[:, :-1:2, :]
    odd_rows = data[:, 1::2, :]

    # fit row difference
    diff = even_rows - odd_rows
    coeff = polyfitnd(even_rows.ravel(), diff.ravel(), 1, robust=6.0)
    if np.any(np.isnan(coeff)):
        log.warning('Fit failed; not correcting row gains.')
        return data

    # correct odd rows to even
    corrected_data[:, 1::2, :] += odd_rows * coeff[1] + coeff[0]

    return corrected_data
