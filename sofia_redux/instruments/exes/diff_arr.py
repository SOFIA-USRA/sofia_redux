# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Subtract sky frames or nod pairs.
"""

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils

__all__ = ['diff_arr']


def diff_arr(data, header, abeams, bbeams, variance, mask=None, dark=None,
             black_dark=False):
    """
    Subtract sky frames, nod pairs, or a dark frame.

    If header['INSTMODE'] is 'MAP', then the B beams are assumed to be
    sky frames.  They are averaged, and the resulting frame is
    subtracted from each designated A frame.  Otherwise, B beams are
    subtracted pair-wise from neighboring A beams.

    The output array has a number of frames equal to the number of input
    A nods. If an input bad pixel mask is provided, pixels are marked bad
    in the output mask if there were any bad pixels at that location in
    the subtracted frames, i.e. the input A and B masks are or-combined.

    If a dark frame is provided and `black_dark` is True, then the dark is
    subtracted from identified B nods instead of performing A-B subtraction.
    This is intended to provide clean sky frames for calibration purposes.
    In this case, the output data has a number of frames equal to the number
    of input B nods.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, nspec, nspat].
    header : fits.Header
        Header of FITS file.
    abeams : array-like of int
        Index numbers of A frames in data cube.
    bbeams : array-like of int
        Index numbers of B frames in data cube.
    variance : numpy.ndarray
        3D variance cube [nframe, nspec, nspat].
    mask : numpy.ndarray, optional
        Bad pixel array [nspec, nspat] indicating valid data
        (True=good, False=bad).
    dark : numpy.ndarray, optional
        If provided, and `black_dark` is True, this array is subtracted
        from the B nods instead of performing nod-subtraction.
    black_dark : bool, optional
        If True, the `dark` array is subtracted instead of performing
        nod subtraction.

    Returns
    -------
    diff_data, diff_var, diff_mask : 3-tuple of numpy.ndarray
        The differenced data, associated variance, and propagated bad
        pixel mask. All have dimensions [ndiff, nspec, nspat]. In the mask,
        True=good, False=bad.
    """

    nx = header['NSPAT']
    ny = header['NSPEC']

    try:
        nz = utils.check_data_dimensions(data=data, nx=nx, ny=ny)
    except RuntimeError:
        log.error(f'Data has wrong dimensions {data.shape}. '
                  f'Not subtracting frames.')
        return data, variance, mask

    try:
        do_var = utils.check_variance_dimensions(variance, nx, ny, nz)
    except RuntimeError:
        log.error(f'Variance has wrong dimensions {variance.shape}. '
                  f'Not subtracting frames.')
        return data, variance, mask

    try:
        b_info = _check_beams(abeams, bbeams, header, data, variance,
                              do_var, nz, black_dark, mask)
    except RuntimeError:
        log.error('A and B beams must be specified. '
                  'Not subtracting frames.')
        return data, variance, mask

    diff_data, diff_var, diff_mask = _apply_beams(
        data, variance, header, abeams, bbeams,
        nx, ny, do_var, b_info, black_dark,
        dark, mask)

    diff_data, diff_var = _replace_small_values(diff_data, diff_var,
                                                header, do_var)

    return diff_data, diff_var, diff_mask


def _check_beams(abeams, bbeams, header, data, variance, do_var,
                 nz, black_dark, mask):
    """Check input data and beam identification."""
    if len(abeams) == 0 or len(bbeams) == 0:
        raise RuntimeError
    instmode = str(header.get('INSTMODE', 'UNKNOWN')).strip().upper()
    if (not black_dark
            and instmode != 'MAP'
            and (len(abeams) != len(bbeams) or len(abeams) != nz // 2)):
        raise RuntimeError

    b_data, b_var, b_mask = None, None, None
    b_average, b_var_average = None, None
    if instmode == 'MAP':
        # Combine B beams
        b_data = data[bbeams]
        b_average = np.nanmean(b_data, axis=0)
        if do_var:
            b_var = variance[bbeams]
            b_var_average = np.nanmean(b_var, axis=0)
        if mask is not None:
            b_mask = np.any(mask[bbeams], axis=0).astype(int)
    b = {'data': b_data, 'data_avg': b_average,
         'var': b_var, 'var_avg': b_var_average,
         'mask': b_mask}
    return b


def _apply_beams(data, variance, header, abeams, bbeams, nx, ny, do_var,
                 b_info, black_dark, dark, mask):
    """Subtract specified beams."""
    if black_dark:
        nz = len(bbeams)
    else:
        nz = len(abeams)
    do_mask = (mask is not None)
    diff_data = np.zeros((nz, ny, nx), dtype=float)
    diff_var = np.zeros_like(diff_data)
    diff_mask = np.zeros((nz, ny, nx), dtype=int)
    b_var, b_mask = None, None
    for i in range(nz):
        if header['INSTMODE'] == 'MAP' and not black_dark:
            b_data = b_info['data_avg']
            if do_var:
                b_var = b_info['var_avg']
            if do_mask:
                b_mask = b_info['mask']
        else:
            b_data = data[bbeams[i]]
            if do_var:
                b_var = variance[bbeams[i]]
            if do_mask:
                b_mask = mask[bbeams[i]]

        if black_dark:
            # use B beams only for dark subtraction:
            # used for making sky spectra
            diff_data[i] = b_data - dark
            if do_var:
                diff_var[i] = b_var
            if do_mask:
                diff_mask[i] = b_mask
        else:
            diff_data[i] = data[abeams[i]] - b_data

            # Note: source code does this differently, taking 2x the
            # minimum of the a and b variances
            if do_var:
                diff_var[i] = variance[abeams[i]] + b_var
            if do_mask:
                diff_mask[i] = np.any([mask[abeams[i]], b_mask],
                                      axis=0).astype(int)

    return diff_data, diff_var, diff_mask


def _replace_small_values(diff_data, diff_var, header, do_var):
    """Replace small values with a reasonable minimum value."""
    gain = float(header['PAGAIN'])
    beamtime = float(header['BEAMTIME'])
    data_min = 1 / (beamtime * np.abs(gain))
    diff_data[diff_data == 0] = data_min
    if do_var:
        var_min = 16. / (beamtime * gain)**2
        diff_var[diff_var < var_min] = var_min
    return diff_data, diff_var
