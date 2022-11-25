# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils

__all__ = ['cirrus']


def cirrus(data, header, abeams, bbeams, flat):
    """
    Correct nod-off-slit data for residual sky noise.

    Remove sky noise by subtracting (a + by) * B + (c + dy) / flat from A,
    where B is the sky nod array, A is the source array, and a, b, c, d
    are chosen to minimize the sum of (A - B)^2.

    Note that sky noise depends on y because clouds can vary during an
    array readout.

    This algorithm removes the average continuum along the slit.
    If this is not desired, background subtraction should be done during
    spectral extraction instead.

    Parameters
    ----------
    data : numpy.ndarray
        Data cube of shape (nframe, nspec, nspat) or image (nspec, nspat).
    header : fits.Header
        Header associated with the input data.
    abeams : `list` of int
        List of frame indices corresponding to A frames in the data cube.
    bbeams : `list` of int
        List of frame indices corresponding to B frames in the data cube.
    flat : numpy.ndarray
        Flat image of shape (nspec, nspat).

    Returns
    -------
    data : numpy.ndarray
        The corrected data.
    """
    nx = header['NSPAT']
    ny = header['NSPEC']

    try:
        nz = utils.check_data_dimensions(data=data, nx=nx, ny=ny)
    except RuntimeError:
        log.error(f'Data has wrong dimensions {data.shape}. '
                  f'Not correcting background.')
        return data

    try:
        _check_beams(abeams, bbeams, nz)
    except RuntimeError:
        log.error('A and B beams must be specified and number '
                  'must match. Not correcting background.')
        return data

    log.info('Subtracting sky fluctuations in cirrus')
    log.warning('This algorithm removes average continuum along slit. '
                'Use background subtraction during spectral extraction '
                'if this is not desired.')

    # Make index array to get distance from center
    y, x = np.mgrid[0:ny, 0:nx]
    xny2 = (ny + 1) / 2
    dy = (y - xny2) / ny

    # Initialize
    alpha = np.zeros((4, 4))
    beta = np.zeros(4)
    corrected = np.zeros((nz, ny, nx))
    xpar = np.zeros((4, ny, nx))

    # Get correctable points from flat
    z = flat > 0
    if np.sum(z) == 0:
        log.error('No good points in flat.')
        return data

    for k in range(int(nz / 2)):

        # Get A and B data
        adata = data[abeams[k]]
        bdata = data[bbeams[k]]

        # Find sky noise parameters
        # Set to zero where flat is bad (so no contribution to sum)
        xpar[0][z] = bdata[z]
        xpar[1][z] = bdata[z] * dy[z]
        xpar[2][z] = 1 / flat[z]
        xpar[3][z] = dy[z] / flat[z]

        for i in range(4):
            for j in range(4):
                alpha[i, j] = np.nansum(xpar[i] * xpar[j])
            beta[i] = np.nansum((adata - bdata) * xpar[i])

        try:
            inv_alpha = np.linalg.inv(alpha)
        except np.linalg.LinAlgError:
            log.warning('Could not find sky parameters. '
                        'Not correcting background.')
            return data
        alpha = inv_alpha

        a = np.zeros(4)
        for i in range(4):
            a[i] = np.nansum(beta * alpha[i])

        log.info(f'Sky noise parameters for pair {k+1}: ')
        log.info(a)

        # Correct A beam
        adata[z] = (adata[z] - (a[0] + a[1] * dy[z]) * bdata[z]
                    - (a[2] + a[3] * dy[z]) / flat[z])
        corrected[abeams[k]] = adata
        corrected[bbeams[k]] = bdata

    return corrected


def _check_beams(abeams, bbeams, nz):
    if (len(abeams) == 0
            or len(bbeams) == 0
            or len(abeams) != len(bbeams)
            or len(abeams) != nz // 2):
        raise RuntimeError
