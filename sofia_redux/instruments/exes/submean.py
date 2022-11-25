# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils

__all__ = ['submean']


def submean(data, header, flat, illum, order_mask):
    """
    Subtract residual sky background from nod-on-slit data by
    removing the mean value at each wavelength.

    For each input frame, the mean value at each spectral column is
    calculated, using the flat to weight the values, then is subtracted
    from the data in the column.  Input data is assumed to be distortion
    corrected and rotated to align the spectral axis with the x-axis.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, nspec, nspat]
    header : fits.Header
        FITS header associated with the data.
    flat : numpy.ndarray
        2D processed flat [nspec, nspat], as produced by `exes.makeflat`.
    illum : numpy.ndarray
        2D array [nspec, nspat] indicating illuminated regions of
        the frame. 1=illuminated, 0=unilluminated, -1=pixel that
        does not correspond to any region in the raw frame.
    order_mask : numpy.ndarray
        2D array [nspat,nspec] indicating the order number for every
        pixel in the image. For pixels outsize illuminated orders,
        the value in the order_mask is NaN.

    Returns
    -------
    corrected: numpy.ndarray
         Returns the corrected 3D data cube [nspat,nspec,nframe].
    """

    nx = header.get('NSPAT')
    ny = header.get('NSPEC')

    try:
        nz = utils.check_data_dimensions(data=data, nx=nx, ny=ny)
    except RuntimeError:
        log.error(f'Data has wrong dimensions {data.shape}. '
                  f'Not applying flat')
        return data

    if illum is None:
        illum = np.ones((ny, nx))
    else:
        if illum.ndim != 2 or illum.shape != (ny, nx):
            log.error(f'Illum array has wrong dimensions '
                      f'{illum.shape}. Not correcting background.')
            return data

    # Get good data from flat/illumination mask
    good = (illum == 1) & (flat > 0)

    # Loop over frames
    log.info('Subtracting mean from each spectral point')
    corrected = data.copy()
    for i in range(nz):
        d = data[i]

        frame_good = good & ~np.isnan(d)
        avg = _multi_order_avg(d, flat, frame_good, order_mask)

        # Note:
        # We are not propagating variance, assumption is that there
        # is no error in the average value
        corrected[i] = data[i] - avg

    return corrected


def _multi_order_avg(data, flat, good, order_mask):

    # make flat weights without dividing by zero
    # note: flat is inverted, so high value = low illumination
    # and should be low weight
    weight_frame = flat.copy()
    weight_frame[~good] = 1.
    weight_frame = 1 / weight_frame ** 2
    weight_frame[~good] = 0.

    # do a weighted average over y in each order
    avg = np.zeros_like(data)
    n_order = np.nanmax(order_mask)
    for j in range(n_order):
        order_idx = (order_mask == j + 1)
        idx = good & order_idx
        if idx.sum() == 0:
            continue

        masked_data = np.ma.MaskedArray(data, mask=~idx)
        row = np.ma.average(masked_data, weights=weight_frame, axis=0)
        row = np.ma.filled(row, fill_value=0)

        avg_order = np.zeros_like(data)
        avg_order[:] = row
        avg[order_idx] = avg_order[order_idx]

    return avg
