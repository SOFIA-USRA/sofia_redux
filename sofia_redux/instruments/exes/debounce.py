# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import tort as et

__all__ = ['debounce']


def debounce(data, header, abeams, bbeams, flat,
             good_data=None, illum=None,
             variance=None, spectral=False):
    """
    Correct for optics shifts between nods (bounces).

    Each nod pair is undistorted, then the B nod is shifted slightly
    in the spatial direction and differenced with the A nod.  The
    shift direction that results in a minimum squared difference
    (summed over the spectral direction) is used as the bounce
    direction.

    The amplitude of the shift is controlled by the
    bounce parameter (hdr['BOUNCE']), which should be set to a
    number whose absolute value is between 0 and 1 (typically 0.1).
    If the bounce parameter is set to a positive number, only the
    above shift (the first-derivative bounce) will be corrected.

    If the bounce parameter is set to a negative value (e.g. -0.1), the
    debouncing algorithm will also attempt to fix the
    second-derivative bounce by smoothing the A or B nod slightly;
    the amount of smoothing is also controlled by the absolute value
    of the bounce parameter.

    Note that if the observed source is too near the edge of the order,
    it may confuse the debouncing algorithm; in this case, it is usually
    best to turn debouncing off (i.e. set the bounce parameter to 0).

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, nspec, nspat].
    header : fits.Header
        Header of FITS file. May be updated in place.
    abeams : array-like of int
        Index numbers of A frames in data cube.
    bbeams : array-like of int
        Index numbers of B frames in data cube.
    flat : numpy.ndarray
        2D processed flat, as produced by makeflat [nspec, nspat].
    good_data : numpy.ndarray, optional
        Bad pixel array matching data shape indicating valid data
        (True=good, False=bad).
    illum : numpy.ndarray, optional
        2D array of size [nspec, nspat] indicating illuminated regions of
        the frame. 1 = illuminated, 0 = unilluminated, -1 = pixel that
        does not correspond to any region in the raw frame (after
        undistortion).
    variance : numpy.ndarray, optional
        3D array of size [nframe, nspec, nspat] of variance planes
        corresponding to data.
    spectral : bool, optional
        If True, debounce is applied in the spectral direction instead
        of the spatial.

    Returns
    -------
    corrected : numpy.ndarray
        Corrected 3D data cube
    """
    try:
        params = _check_inputs(data, header, abeams, bbeams, flat, good_data,
                               illum, variance, spectral)
    except ValueError as msg:
        log.error(msg)
        return data

    if _bounce_confusion(params):
        log.warning('Nodded source may confuse debounce')

    _check_nonzero_data(params)
    _check_neighbors(params)
    _check_nonzero_data(params)

    success = _derive_bounce_for_pairs(params)
    if not success:
        return data

    log.info(f"First derivative bounce parameters: "
             f"{params['first_deriv_shift']}")
    if params['bounce'] < 0:
        log.info(f"Second derivative bounce parameters: "
                 f"{params['second_deriv_shift']}")

    if params['nzero'] > 0:
        if params['modes']['scan']:
            log.info(f"{params['nzero']} pairs exceeded bounce limit")
        else:
            log.info(f"Setting weight = 0 for {params['nzero']} pairs")

    return params['bounce_data']


def _check_inputs(data, header, a_beams, b_beams, flat, data_mask,
                  illum, variance, spectral):
    """
    Check input shape and values.

    Later functions in this step take the output params dictionary as
    input and add to it or update it as output.

    Parameters
    ----------
    data : numpy.ndarray
    header : fits.Header
    a_beams: array-like of int
    b_beams: array-like of int
    flat : numpy.ndarray
    data_mask : numpy.ndarray
    illum : numpy.ndarray
    variance : numpy.ndarray
    spectral : bool

    Returns
    -------
    params : dict
        Contains all input data, reformatted as needed.
        Keys are:

            - 'data': data array
            - 'variance': variance array
            - 'header': header object
            - 'a_beams': list of A beam indices
            - 'b_beams': list of B beam indices
            - 'flat': flat array
            - 'data_mask': mask array, ny x nx
            - 'frame_mask': mask array, nz
            - 'illum': illumination array
            - 'spectral': spectral direction flag
            - 'modes': dict with 'scan', 'nod', 'crossdisp' keys
            - 'do_var': flag to indicate variance processing
            - 'nz': number of frames
            - 'ny': data shape, y-direction
            - 'nx': data shape, x-direction
            - 'bounce': bounce amplitude parameter
            - 'skip_small': flag to indicate small bounces should be skipped

    """
    if header['BOUNCE'] == 0:
        raise ValueError('Bounce factor = 0. Cannot apply debounce.')

    try:
        nz = _check_data_dimensions(data, header)
    except RuntimeError:
        message = (f'Data has wrong dimensions '
                   f'{data.shape}.')
        raise ValueError(message)

    try:
        do_var = _check_variance_dimensions(variance, header, nz)
    except RuntimeError:
        message = (f'Variance has wrong dimensions '
                   f'{variance.shape}.')
        raise ValueError(message)

    try:
        _check_beams(a_beams, b_beams, nz)
    except RuntimeError:
        message = ('A and B beams must be specified and numbers '
                   'must match.')
        raise ValueError(message)

    try:
        data_mask, frame_mask = _check_masks(data_mask, data)
    except RuntimeError:
        message = f'Provided mask has wrong dimensions {data_mask.shape}.'
        raise ValueError(message)

    if illum is None:
        illum = np.ones_like(data[0])

    modes = _check_obsmode(header)

    params = {'data': data, 'variance': variance, 'header': header,
              'a_beams': a_beams, 'b_beams': b_beams, 'flat': flat,
              'data_mask': data_mask, 'frame_mask': frame_mask, 'illum': illum,
              'spectral': spectral, 'modes': modes, 'do_var': do_var,
              'nz': nz, 'ny': header['NSPEC'], 'nx': header['NSPAT'],
              'bounce': header['BOUNCE'], 'skip_small': True}
    return params


def _check_data_dimensions(value, header):
    """Check input data dimensions."""
    if value.ndim <= 2:
        nz = 1
    else:
        nz = value.shape[0]

    if (value.ndim not in [2, 3]
            or value.shape[-1] != header['NSPAT']
            or value.shape[1] != header['NSPEC']):
        raise RuntimeError

    return nz


def _check_variance_dimensions(variance, header, nz):
    """Check input variance dimensions."""
    if variance is None:
        return False
    if variance.ndim <= 2:
        vnz = 1
    else:
        vnz = variance.shape[0]

    if (nz != vnz
            or variance.shape[-1] != header['NSPAT']
            or variance.shape[1] != header['NSPEC']):
        raise RuntimeError

    return True


def _check_beams(abeams, bbeams, nz):
    """Check input beam definitions."""
    if (len(abeams) == 0
            or len(bbeams) == 0
            or len(abeams) != len(bbeams)
            or len(abeams) != nz // 2):
        raise RuntimeError


def _check_masks(mask, data):
    """Check and reformat good data masks."""
    if mask is None:
        mask = np.full((data.shape), True)
    elif mask.shape != data.shape:
        raise RuntimeError
    frame_mask = np.any(mask, axis=(1, 2))
    data_mask = np.all(mask[frame_mask], axis=0)

    return data_mask, frame_mask


def _check_obsmode(header):
    """Check and flag instrument configuration and mode."""
    obsmode = str(header['INSTMODE']).upper()
    instcfg = str(header['INSTCFG']).upper()

    modes = dict()
    modes['scan'] = obsmode in ['MAP', 'FILEMAP']
    modes['nod'] = obsmode in ['NOD_ON_SLIT', 'NOD_OFF_SLIT']
    modes['crossdisp'] = instcfg in ['HIGH_MED', 'HIGH_LOW']
    return modes


def _bounce_confusion(params):
    """Check if debounce is likely to be confused by nodded sources."""
    modes = params['modes']
    header = params['header']
    if modes['nod']:
        nodamp = float(header['NODAMP'])
        pltscl = float(header['PLTSCALE'])
        spacing = float(header['SPACING'])
        nbelow = float(header['NBELOW'])

        checks = [int(header[key]) != -9999
                  for key in ['NODAMP', 'PLTSCALE', 'SPACING', 'NBELOW']]
        if all(checks):
            distance = nodamp / pltscl
            slit_length = spacing - nbelow
            if 0.6 * slit_length < distance < slit_length:
                return True
    return False


def _check_nonzero_data(params):
    """Check for valid, nonzero data."""
    data = params['data']
    frame_mask = params['frame_mask']
    good = (data != 0) & np.isfinite(data)
    data_nonzero = good[frame_mask].all(axis=0)
    params['data_nonzero'] = data_nonzero.astype(bool)


def _check_neighbors(params):
    """Check for valid neighbors."""
    nx = params['nx']
    ny = params['ny']
    data_mask = params['data_mask']
    flat = params['flat']
    data_nonzero = params['data_nonzero']
    crossdisp = params['modes']['crossdisp']
    spectral = params['spectral']

    y, x = np.indices((ny, nx))

    if (crossdisp and not spectral) or (spectral and not crossdisp):
        params['direction'] = 'x'
        axis = 1
        bounds = (y - 1 >= 0) & (y + 1 < ny)
    else:
        params['direction'] = 'y'
        axis = 0
        bounds = (x - 1 >= 0) & (x + 1 < nx)

    data_check = ((_shift_2d_array(data_mask, -1, axis))
                  & (_shift_2d_array(data_mask, 1, axis))
                  & data_mask)
    flat_check = ((_shift_2d_array(flat, -1, axis) != 0)
                  & (_shift_2d_array(flat, 1, axis) != 0)
                  & (flat != 0))
    ok = data_nonzero & bounds & data_check & flat_check
    if ok.sum() == 0:
        raise ValueError('No good pixels in data array. '
                         'Not applying debounce')
    params['ok_idx'] = ok


def _calc_bounce(params, subselect, a_coefficients, b_coeffients,
                 second_deriv=False, bounce=None):
    """Calculate the bounce variance for given coefficients."""
    nx = params['nx']
    ny = params['ny']
    if bounce is None:
        abs_bounce = np.abs(params['bounce'])
    else:
        abs_bounce = bounce
    direction = params['direction']
    ok_idx = params['ok_idx']
    a_data = params['a_data']
    b_data = params['b_data']
    header = params['header']
    illum = params['illum']

    var = list()

    for a_coeff, b_coeff in zip(a_coefficients, b_coeffients):
        shift_a = np.zeros((ny, nx), dtype=float)
        shift_b = np.zeros_like(shift_a, dtype=float)
        if second_deriv:
            a_sub_diff = (subselect['am1'][ok_idx]
                          - 2 * a_data[ok_idx]
                          + subselect['ap1'][ok_idx])

            b_sub_diff = (subselect['bm1'][ok_idx]
                          - 2 * b_data[ok_idx]
                          + subselect['bp1'][ok_idx])
        else:
            a_sub_diff = (subselect['ap1'][ok_idx]
                          - subselect['am1'][ok_idx])
            b_sub_diff = (subselect['bp1'][ok_idx]
                          - subselect['bm1'][ok_idx])

        a_factor = a_coeff * abs_bounce * a_sub_diff
        shift_a[ok_idx] = a_data[ok_idx] + a_factor

        b_factor = b_coeff * abs_bounce * b_sub_diff
        shift_b[ok_idx] = b_data[ok_idx] + b_factor

        diff = shift_a - shift_b
        diff = et.tort(diff, header, order=1, skew=True)
        diff[illum != 1] = 0.

        # Sum over order (wavelength direction) to calculate a variance
        if direction == 'x':
            # Sum over y (XD)
            total = np.nansum(diff, axis=0)
        else:
            # Sum over x (LS)
            total = np.nansum(diff, axis=1)
        var.append(np.nansum(total ** 2))

    return np.array(var)


def _derive_bounce_for_pairs(params):
    """Derive and apply the bounce correction for each nod pair."""
    abs_bounce = np.abs(params['bounce'])
    data = params['data']
    bounce_data = data.copy()
    scan = params['modes']['scan']
    variance = params['variance']
    ok_idx = params['ok_idx']
    first_deriv_shift = np.zeros_like(params['a_beams'], dtype=float)
    second_deriv_shift = np.zeros_like(params['a_beams'], dtype=float)
    frame_mask = params['frame_mask']
    nzero = 0
    skip_small = params['skip_small']

    for i in range(len(params['a_beams'])):

        a_frame = params['a_beams'][i]
        b_frame = params['b_beams'][i]
        # Check both A and B are good frames
        if not (frame_mask[a_frame]
                and frame_mask[b_frame]):  # pragma: no cover
            continue

        subselect = _subselect_data(params, a_frame, b_frame)

        # Fix first derivative bounce by shifting B beams
        # calculate variances of A-B for three bounce parameters
        a_par = [0.0, 0.25, 0.0]  # u, v, w
        b_par = [-0.5, 0.25, 0.5]
        var = _calc_bounce(params, subselect, a_par, b_par, second_deriv=False)
        a, b = _var_application(var)

        if a > 0:
            bounce_amp = 0.5 * abs_bounce * b / a
        else:
            log.warning(f"Can't find best 1st derivative "
                        f"bounce for pair {i + 1}")
            if not scan:
                # Marks these frames as bad
                frame_mask[a_frame] = False
                frame_mask[b_frame] = False
                if frame_mask.sum() == 0:
                    log.error('No good frames remaining. '
                              'Not applying debounce')
                    return False
                else:
                    first_deriv_shift[i] = 0
                    nzero += 1
            else:
                first_deriv_shift[i] = 0
                nzero += 1
            continue

        if np.abs(bounce_amp) > 2 * abs_bounce:
            if not scan:
                # Mark frames as bad
                frame_mask[a_frame] = False
                frame_mask[b_frame] = False
                if frame_mask.sum() == 0:
                    log.error('No good frames remaining. '
                              'Not applying debounce')
                    return False
                else:
                    first_deriv_shift[0] = 0
                    nzero += 1
                continue
            else:
                first_deriv_shift[i] = 0
                nzero += 1
                continue

        # Skip recalculation/shifting for small bounce_amp
        if not skip_small or np.abs(bounce_amp) >= abs_bounce / 100.:
            # Skip recalculation for scans
            if scan:  # pragma: no cover
                b = 0
            else:
                # Calculate bounce for minimum variance
                a_par = [0.25, 0.0, 0.0]
                b_par = [0.25, 0.5, 1.0]
                var = _calc_bounce(params, subselect, a_par, b_par,
                                   bounce=bounce_amp)
                a, b = _var_application(var)

            if a > 0 and np.abs(b / a) < 2:
                bounce_amp = bounce_amp * (1 + b / (2 * a))
            elif np.abs(bounce_amp) >= 0.1 * abs_bounce:
                log.warning(f"Can't recalculate 1st derivative "
                            f"bounce for pair {i + 1}")
                if not scan:
                    # Mark frame as bad
                    frame_mask[a_frame] = False
                    frame_mask[b_frame] = False
                    if frame_mask.sum() == 0:
                        log.error('No good frames remaining. '
                                  'Not applying debounce')
                        return False
                    else:
                        first_deriv_shift[i] = 0
                        nzero += 1
                else:  # pragma: no cover
                    # shouldn't be reachable
                    first_deriv_shift[i] = 0
                    nzero += 1

                continue  # pragma: no cover

            # Shift b_data
            first_deriv_shift[i] = bounce_amp
            shift_b = data[b_frame].copy()
            shift = 0.5 * bounce_amp * (subselect['bp1'][ok_idx]
                                        - subselect['bm1'][ok_idx])
            shift_b[ok_idx] = shift_b[ok_idx] + shift
            bounce_data[b_frame] = shift_b

            # Do the same for variance plane
            if params['do_var']:
                shift = ((0.5 * bounce_amp) ** 2
                         * (subselect['bp1'][ok_idx]
                            - subselect['bm1'][ok_idx]))
                variance[b_frame][ok_idx] = variance[b_frame][ok_idx] + shift

        # Fix 2nd derivative bounce by smoothing arr or brr
        if params['bounce'] > 0:
            continue

        a_par = [1.0, 0.5, 0.0]
        b_par = [0.0, 0.5, 1.0]
        var = _calc_bounce(params, subselect, a_par, b_par,
                           second_deriv=True, bounce=abs_bounce)
        a, b = _var_application(var)

        # Calculate bounce for minimum variance
        if a > 0:
            bounce_amp = 0.5 * abs_bounce * b / a
        else:
            log.warning(f"Can't find best 2nd derivative bounce "
                        f"for pair {i + 1}")
            if not scan:
                # Mark frames as bad
                frame_mask[a_frame] = False
                frame_mask[b_frame] = False
                if frame_mask.sum() == 0:
                    log.error('No good frames remaining. Not applying '
                              'debounce')
                    return False
                else:
                    second_deriv_shift[i] = 0
                    nzero += 1
            else:
                second_deriv_shift[i] = 0
                nzero += 1
            continue

        if np.abs(bounce_amp) < abs_bounce / 100 and skip_small:
            # correction is small, don't bother
            second_deriv_shift[i] = 0
            continue
        elif np.abs(bounce_amp) > 4 * abs_bounce:
            if not scan:
                # Mark frames as bad
                frame_mask[a_frame] = False
                frame_mask[b_frame] = False
                if frame_mask.sum() == 0:
                    log.error('No good frames remaining. Not applying '
                              'debounce')
                    return False
                else:
                    second_deriv_shift[i] = 0
                    nzero += 1
            else:
                second_deriv_shift[i] = 0
                nzero += 1
            continue

        # Skip recalculation for scans
        recalc = False
        if not scan:
            recalc = True
            # Calculate bounce for minimum variance
            if bounce_amp < 0:
                a_par = [1.0, 1.5, 2.0]
                b_par = [1.0, 0.5, 0.0]
            else:
                a_par = [1.0, 0.5, 0.0]
                b_par = [1.0, 1.5, 2.0]
            var = _calc_bounce(params, subselect, a_par, b_par,
                               second_deriv=True, bounce=np.abs(bounce_amp))
            a, b = _var_application(var)

        if recalc and a > 0 and np.abs(b / a) < 2:
            bounce_amp = bounce_amp * (1 + 0.5 * b / a)
            second_deriv_shift[i] = bounce_amp
        elif np.abs(bounce_amp) < 0.1 * abs_bounce:
            second_deriv_shift[i] = bounce_amp
        else:
            log.warning(f"Can't recalculate 2nd derivative bounce "
                        f"for pair {i + 1}")
            if not scan:
                # Mark frames as bad
                frame_mask[a_frame] = False
                frame_mask[b_frame] = False
                if frame_mask.sum() == 0:
                    log.error('No good frames remaining. Not applying '
                              'debounce')
                    return False
                else:
                    second_deriv_shift[i] = 0
                    nzero += 1
            else:
                second_deriv_shift[i] = 0
                nzero += 1
            continue

        # Smooth a or b data
        if bounce_amp < 0:
            smooth = data[a_frame].copy()
            factor = (subselect['am1'][ok_idx]
                      - 2 * smooth[ok_idx]
                      + subselect['ap1'][ok_idx])
            smooth[ok_idx] = smooth[ok_idx] + np.abs(bounce_amp) * factor
            bounce_data[a_frame] = smooth

            # Do the same for variance
            if params['do_var']:
                smooth_v = variance[a_frame]
                factor = (subselect['vam1'][ok_idx]
                          + 4 * variance[a_frame][ok_idx]
                          + subselect['vap1'][ok_idx])
                smooth_v[ok_idx] = smooth_v[ok_idx] + bounce_amp ** 2 * factor
                variance[a_frame] = smooth_v
        else:
            smooth = data[b_frame].copy()
            factor = (subselect['bm1'][ok_idx]
                      - 2 * smooth[ok_idx]
                      + subselect['bp1'][ok_idx])
            smooth[ok_idx] = smooth[ok_idx] + np.abs(bounce_amp) * factor
            bounce_data[b_frame] = smooth

            # Do the same for variance
            if params['do_var']:
                smooth_v = variance[b_frame]
                factor = (subselect['vbm1'][ok_idx]
                          + 4 * variance[b_frame][ok_idx]
                          + subselect['vbp1'][ok_idx])
                smooth_v[ok_idx] = smooth_v[ok_idx] + bounce_amp ** 2 * factor
                variance[b_frame] = smooth_v

    params['first_deriv_shift'] = first_deriv_shift
    params['second_deriv_shift'] = second_deriv_shift
    params['nzero'] = nzero
    params['bounce_data'] = bounce_data
    return True


def _var_application(var):
    """Calculate paramters from variance."""
    a = var[0] - 2 * var[1] + var[2]
    b = var[0] - var[2]
    return a, b


def _subselect_data(params, a_frame, b_frame):
    """Shift data plus and minus 1."""
    data = params['data'].copy()
    variance = params['variance']
    params['a_data'] = params['data'][a_frame].copy()
    params['b_data'] = params['data'][b_frame].copy()
    results = dict()

    if params['direction'] == 'x':
        axis = 1
    else:
        axis = 0

    frames = {'a': a_frame, 'b': b_frame}
    shifts = {'p': -1, 'm': 1}
    for f, frame in frames.items():
        for s, shift in shifts.items():
            arr = _shift_2d_array(data[frame], shift, axis)
            results[f'{f}{s}1'] = arr
            if params['do_var']:
                arr = _shift_2d_array(variance[frame], shift, axis)
                results[f'v{f}{s}1'] = arr

    return results


def _shift_2d_array(data, n, axis):
    """Apply a shift to 2D data along a specified axis."""
    if n == 0:
        return data
    shifted = np.roll(data, n, axis=axis)
    if axis == 0:
        # Shift up/down
        if n >= 0:
            # Shift down
            fill = data[0, :]
            shifted[:n, :] = np.expand_dims(fill, axis=axis)
        else:
            # Shift up
            fill = data[-1, :]
            shifted[n:, :] = np.expand_dims(fill, axis=axis)
    else:
        # Shift left/right
        if n >= 0:
            # Shift right
            fill = data[:, 0]
            shifted[:, :n] = np.expand_dims(fill, axis=axis)
        else:
            # Shift left
            fill = data[:, -1]
            shifted[:, n:] = np.expand_dims(fill, axis=axis)
    return shifted
