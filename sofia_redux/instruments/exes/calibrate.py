# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils

__all__ = ['calibrate']


def calibrate(data, header, flat, variance, flat_var=None):
    """
    Calibrate spectral image to physical units.

    Each frame in the input data cube is multiplied by the flat
    frame, which has been normalized by the black-body function.
    This has the effect of both correcting the science frame for
    instrumental response and calibrating it to intensity units.

    Parameters
    ----------
    data : numpy.ndarray
        Data cube of shape [nframe, nspec, nspat] or image [nspec, nspat].
    header : fits.Header
        Header associated with the input data.
    flat : numpy.ndarray
        Flat image of shape [nspec, nspat].
    variance : numpy.ndarray
        Variance array matching data shape.
    flat_var : numpy.ndarray, optional
        Flat variance array of shape (nspec, nspat).  If provided, it
        is propagated into the output variance.

    Returns
    -------
    data, variance : numpy.ndarray, numpy.ndarray
        The calibrated data and updated variance.
    """

    nx = header['NSPAT']
    ny = header['NSPEC']

    try:
        nz = utils.check_data_dimensions(data=data, nx=nx, ny=ny)
    except RuntimeError:
        log.error(f'Data has wrong dimensions {data.shape}. '
                  f'Not applying flat')
        return data, variance

    try:
        utils.check_variance_dimensions(variance, nx, ny, nz)
    except RuntimeError:
        log.error(f'Variance has wrong dimensions {data.shape}. '
                  f'Not applying flat')
        return data, variance
    else:
        try:
            utils.check_variance_dimensions(flat_var, nx, ny, 1)
        except RuntimeError:
            log.warning(f'Flat variance has wrong dimensions '
                        f'{flat_var.shape}')
            flat_var = np.zeros((ny, nx))

    # Loop over frames
    cal_data = data * flat
    variance = variance * flat ** 2 + flat_var * data ** 2

    return cal_data, variance
