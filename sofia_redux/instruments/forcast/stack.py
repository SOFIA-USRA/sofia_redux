# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert, kref

from sofia_redux.instruments.forcast.background import background
from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.jbclean import jbclean
from sofia_redux.instruments.forcast.readmode import readmode
from sofia_redux.instruments.forcast.read_section import read_section

addhist = add_history_wrap('Stack')

__all__ = ['add_stacks', 'background_scale', 'stack_c2nc2',
           'stack_map', 'stack_c3d', 'stack_cm',
           'stack_stare', 'convert_to_electrons',
           'subtract_background', 'stack']


def add_stacks(data, header, variance=None):
    """
    Add data frames together at the same stack (position)

    The number of frames returned in the output will be equal
    to floor(nframes / nstacks) if OTMODE is AD (All Destructive).
    For SUR mode, the original data will be returned.

    Parameters
    ----------
    data : numpy.ndarray
        3d data array to sum (nframes, nrow, ncol)
    header : astropy.io.fits.header.Header
        FITS header
    variance : numpy.ndarray, optional
        3d variance array to propagate (nframes, nrow, ncol)

    Returns
    -------
    2-tuple
        summed data array (npos, nrow, ncol)
        propagate variance array (npos, nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        msg = "stack failed (invalid header at add_stacks)"
        log.error(msg)
        return

    if not isinstance(data, np.ndarray) or len(data.shape) != 3:
        msg = "stack failed (invalid data at add_stacks)"
        addhist(header, msg)
        log.error(msg)
        return

    nframes = data.shape[0]
    nstacks = getpar(header, 'OTSTACKS', dtype=int, default=None)
    if nstacks is None:
        msg = "stack failed (invalid OTSTACKS %s)" % nstacks
        addhist(header, msg)
        log.error(msg)
        return
    posdata = data.copy()
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    posvar = variance.copy() if dovar else None

    mode = getpar(header, 'OTMODE', dtype=str).upper().strip()
    if mode == 'AD':  # (All Destructive)
        if nstacks <= 1:
            return posdata, posvar

        if nframes % nstacks > 0:
            log.warning(
                "sum_frames (AD) - nframes is not a multiple of nstacks")
            log.warning("ignoring last frames")

        npos = int(data.shape[0] / nstacks)
        dout = np.full((npos, data.shape[-2], data.shape[-1]), np.nan)
        vout = dout.copy() if dovar else None
        for i in range(npos):
            i0, i1 = i * nstacks, (i + 1) * nstacks
            dout[i] = np.nansum(posdata[i0: i1], axis=0)
            if dovar:
                vout[i] = np.nansum(posvar[i0: i1], axis=0)
        return dout, vout
    else:
        msg = 'stack failed (invalid OTMODE %s)' % mode
        addhist(header, msg)
        log.error(msg)
        return


def background_scale(data, header, mask=None):
    """
    Return frame scale levels

    Parameters
    ----------
    data : numpy.ndarray
        input data array (nframe, nrow, ncol)
    header : astropy.io.fits.header.Header
    mask : numpy.ndarray, optional
        mask to pass into sofia_redux.instruments.forcast.background
        (nrow, ncol).  Defines pixels to include in background
        calculation (True=include)

    Returns
    -------
    numpy.array
        scale factors for each frame or None if not applied
    """
    bgscale = getpar(header, 'BGSCALE', default=False, dtype=bool,
                     comment='background scale')
    if not bgscale:
        return
    addhist(header, 'All frames scaled to BG level of first chop position')
    section = read_section(*data.shape[-2:])
    return background(data, section, mask=mask)


def stack_c2nc2(data, header, variance=None, bglevel=None, extra=None):
    """
    Run the stacking algorithm on C2NC2 data

    Calculates the chop-subtracted frames.  For frame i (in steps of 2),
    the resulting chop-subtracted frame would be:

        chop = frame[i] - (frame[i+1] * bgscale[i]/bgscale[i+1])

    chop frames are then summed

    Parameters
    ----------
    data : numpy.ndarray
        (npos, nrow, ncol)
    header : astropy.io.fits.header.Header
        FITS header to update with HISTORY messages
    variance : numpy.ndarray, optional
        variance array to propagate (npos, nrow, ncol)
    bglevel : array_like, optional
        should be of length npos
    extra : dict, optional
        If set will be updated with:
            chopsub -> numpy.ndarray (npos / 2, nrow, ncol)
                chop-subtracted data
            chopsub_var -> numpy.ndarray (npos / 2, nrow, ncol)
                propagated chop-subtracted variance

    Returns
    -------
    2-tuple
        numpy.ndarray : The stacked data array (nrow, ncol)
        numpy.ndarray : The propagated variance array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 3:
        msg = "stack failed (invalid data at stack_c2nc2)"
        addhist(header, msg)
        log.error(msg)
        return
    chopsub = data.copy()
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance at stack_c2nc2)"
        addhist(header, msg)
        log.warning(msg)
    var = variance.copy() if dovar else None

    on = np.array([i % 2 == 0 for i in range(data.shape[0])]).astype(bool)
    scale = [1] * data.shape[0] if bglevel is None else bglevel
    scale = np.array(scale)
    if len(scale) != data.shape[0]:
        msg = 'variance not propagated (invalid background at stack_c2nc2)'
        addhist(header, msg)
        log.error(msg)
        return

    chopscale = np.array([[scale[on] / scale[~on]]]).T
    for val in chopscale:
        addhist(header, 'Scaling for frame 2: %f' % val)
        log.info('Scaling for frame 2: %f' % val)
    chopsub = chopsub[on] - (chopsub[~on] * chopscale)
    if dovar:
        var = var[on] + (var[~on] * (chopscale ** 2))
    if isinstance(extra, dict):
        extra['chopsub'] = chopsub

    return np.nansum(chopsub, axis=0), np.nansum(var, axis=0)


def stack_map(data, header, variance=None, bglevel=None, extra=None):
    """
    Run the stacking algorithm on MAP (Mapping mode) data

    Calculates the chop and nod-subtracted frames.  For each frame in a
    set of 4 using scale s, the algorithm would be:

        1. chop1 = frame1 - (frame2 * s1/s2)
        2. chop2 = frame3 - (frame4 * s3/s4)
        3. nod = chop1 - (chop2 * s1/s3)
        4. result = sum(nods from each set of 4)
        5. if NODBEAM = 'B', multiply by -1

    If the number of frames is not divisible by four, frames at the
    end will be clipped from any reductions.  e.g., in the above
    algorithm, if there were 10 frames, only frames 1-8 would be
    included.  i.e. steps 1-3 would be run on frames 1-4, then on
    frames 5-8, and then the sum would be returned.

    Parameters
    ----------
    data : numpy.ndarray
        (npos, nrow, ncol)
    header : astropy.io.fits.header.Header
        FITS header to update with HISTORY messages
    variance : numpy.ndarray, optional
        variance array to propagate (npos, nrow, ncol)
    bglevel : array_like, optional
        should be of length npos
    extra : dict, optional
        If set will be updated with:
            chopsub -> numpy.ndarray (npos / 2, nrow, ncol)
                chop-subtracted data
            nodsub -> numpy.ndarray (npos / 4, nrow, ncol)
                nod-subtracted data

    Returns
    -------
    2-tuple
        numpy.ndarray : The stacked data array (nrow, ncol)
        numpy.ndarray : The propagated variance array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 3 or \
            data.shape[0] < 4:
        msg = "stack failed (invalid data at stack_map)"
        addhist(header, msg)
        log.error(msg)
        return
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance at stack_map)"
        addhist(header, msg)
        log.warning(msg)
    var = variance.copy() if dovar else None

    # Clip additional frames
    d = data.copy()
    additional_frames = data.shape[0] % 4
    if additional_frames > 0:
        msg = "invalid number of frames (ignoring last %s for stack_map)"
        addhist(header, msg)
        log.warning(msg)
        d = d[:-additional_frames]
        if dovar:
            var = var[:-additional_frames]
        if bglevel is not None:
            bglevel = bglevel[:-additional_frames]

    # Get ratio of chop scales
    on = np.array([i % 2 == 0 for i in range(d.shape[0])]).astype(bool)
    scale = [1] * d.shape[0] if bglevel is None else bglevel
    scale = np.array(scale)
    if len(scale) != d.shape[0]:
        msg = 'stack failed (invalid background at stack_map)'
        addhist(header, msg)
        log.error(msg)
        return

    # Begin with the chops
    chopscale = np.array([[scale[on] / scale[~on]]]).T
    chopsub = d[on] - (d[~on] * chopscale)
    chopsub_var = var[on] + (var[~on] * (chopscale ** 2)) if dovar else None

    # Get ratio of nod scales
    chopon_scale = scale[on]
    mid = len(on) // 2
    nodon = on[:mid]

    # Apply nod scaling to chop-subtracted data data
    nodscale = np.array([[chopon_scale[nodon] / chopon_scale[~nodon]]]).T
    chopsub[~nodon] *= nodscale

    # Get nod subtraction
    nodsub = chopsub[nodon] - chopsub[~nodon]
    if dovar:
        chopsub_var[~nodon] *= nodscale ** 2
        nodsub_var = chopsub_var[nodon] + chopsub_var[~nodon]
    else:
        nodsub_var = None

    # Additional reporting to the user and header
    for i in range(d.shape[0] // 4):
        scales = chopscale[i * 2], chopscale[i * 2 + 1], nodscale[i]
        if bglevel is not None:
            addhist(header, 'Scaling factors for frames '
                            '2,3,4: %f,%f,%f' % scales)
            log.info('Scaling factors for frames '
                     '2,3,4: %f,%f,%f' % scales)

    if isinstance(extra, dict):
        extra['chopsub'] = chopsub
        extra['nodsub'] = nodsub

    # Check whether data is A or B beam, multiply by -1 if B
    sign = -1 if getpar(header, 'NODBEAM') == 'B' else 1
    return np.nansum(nodsub, axis=0) * sign, np.nansum(nodsub_var, axis=0)


def stack_c3d(data, header, variance=None, extra=None):
    """
    Run the stacking algorithm on C3D data (3 position chop with dither)

    result = frame1 - frame2 - frame3

    Parameters
    ----------
    data : numpy.ndarray
        (3, nrow, ncol)
    header : astropy.io.fits.header.Header
        FITS header to update with HISTORY messages
    variance : numpy.ndarray, optional
        variance array to propagate (3, nrow, ncol)
    extra : dict, optional
        If set will be updated with:
            chopsub -> numpy.ndarray (nrow, ncol)
                chop-subtracted data (same as output data in this case)

    Returns
    -------
    2-tuple
        numpy.ndarray : The stacked data array (nrow, ncol)
        numpy.ndarray : The propagated variance array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 3 or \
            data.shape[0] != 3:
        msg = "stack failed (invalid data at stack_c3d)"
        addhist(header, msg)
        log.error(msg)
        return
    chopsub = data.copy()
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance at stack_c3d)"
        addhist(header, msg)
        log.warning(msg)
    var = variance.copy() if dovar else None

    chopsub[1:] *= -1
    chopsub = np.nansum(chopsub, axis=0)
    var = np.nansum(var, axis=0)
    if isinstance(extra, dict):
        extra['chopsub'] = chopsub.copy()

    return chopsub, var


def stack_cm(data, header, variance=None, extra=None):
    """
    Run the stacking algorithm on CM data (multi-position chop)

    result = frame1 - frame2 - frame3 - ... - frameN

    Parameters
    ----------
    data : numpy.ndarray
        (nframe, nrow, ncol)
    header : astropy.io.fits.header.Header
        FITS header to update with HISTORY messages
    variance : numpy.ndarray, optional
        variance array to propagate (nframe, nrow, ncol)
    extra : dict, optional
        If set will be updated with:
            chopsub -> numpy.ndarray (nrow, ncol)
                chop-subtracted data (same as output data in this case)

    Returns
    -------
    2-tuple
        numpy.ndarray : The stacked data array (nrow, ncol)
        numpy.ndarray : The propagated variance array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 3 or \
            len(data.shape) < 2:
        msg = "stack failed (invalid data at stack_cm)"
        addhist(header, msg)
        log.error(msg)
        return
    chopsub = data.copy()
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance at stack_cm)"
        addhist(header, msg)
        log.warning(msg)
    var = variance.copy() if dovar else None

    # allow subtraction on all frames if not found in header
    nchop = getpar(
        header, 'CHPNPOS', dtype=int, default=data.shape[0], warn=True)
    if nchop != data.shape[0]:
        msg = "data shape mismatch with CHPNPOS (stack_cm) "
        msg += "- ignoring last frames"
        addhist(header, msg)
        log.warning(msg)
        chopsub = chopsub[:nchop]
        if dovar:
            var = var[:nchop]

    chopsub[1:] *= -1
    chopsub = np.nansum(chopsub, axis=0)
    var = np.nansum(var, axis=0)
    if isinstance(extra, dict):
        extra['chopsub'] = chopsub.copy()

    return chopsub, var


def stack_stare(data, header, variance=None):
    """
    Run the stacking algorithm on STARE data

    result = median of all frames

    The variance is approximated as (pi/2) times the variance of a mean
    operation.

    Parameters
    ----------
    data : numpy.ndarray
        (nframes, nrow, ncol)
    header : astropy.io.fits.header.Header
        FITS header to update with HISTORY messages
    variance : numpy.ndarray, optional
        variance array to propagate (nframes, nrow, ncol)

    Returns
    -------
    2-tuple
        numpy.ndarray : The stacked data array (nrow, ncol)
        numpy.ndarray : The propagated variance array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 3:
        msg = "stack failed (invalid data at stack_stare)"
        addhist(header, msg)
        log.error(msg)
        return

    # if only one plane, return input
    if data.shape[0] == 1:
        return data, variance

    result = data.copy()
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance at stack_stare)"
        addhist(header, msg)
        log.warning(msg)
    var = variance.copy() if dovar else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = np.nanmedian(result, axis=0)
    if dovar:
        weight = np.sum(~np.isnan(var), axis=0)
        nzi = weight != 0
        var = np.nansum(var, axis=0)
        var[nzi] *= 0.5 * np.pi / (weight[nzi] ** 2)
        var[~nzi] = np.nan

    return result, var


def convert_to_electrons(data, header, variance=None):
    """
    Convert data to mega-electrons per seconds

    The following keywords must exist in the header
    in order to calculate the conversion
    factor:

        - FRMRATE: frame rate (Hz)
        - EPERADU: electrons/adu

    Parameters
    ----------
    data : numpy.ndarray
        data to scale
    header : astropy.io.fits.header.Header
        FITS header
    variance : numpy.ndarray, optional
        variance to propagate

    Returns
    -------
    2-tuple
        numpy.ndarray : data in units of mega-electrons per second
        numpy.ndarray : propagated variance
    """
    vals = []
    for key in ['FRMRATE', 'EPERADU']:
        vals.append(getpar(header, key, dtype=float, default=None, warn=True))
    if None in vals:
        msg = "Could not convert to electrons (invalid FRMRATE/EPERADU)"
        addhist(header, msg)
        log.error(msg)
        return
    factor = 1e-6 * vals[0] * vals[1]
    data *= factor
    if variance is not None:
        variance *= factor ** 2
    addhist(header, 'Counts converted to millions of e/s using')
    addhist(header, 'FRMRATE=%s and EPERADU=%s' % (vals[0], vals[1]))
    return data, variance


def subtract_background(data, header=None, mask=None, stat='mode'):
    """
    Remove background from data

    Parameters
    ----------
    data : numpy.ndarray
        input data (nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        FITS header to update with HISTORY messages
    mask : numpy.ndarray, optional
        mask defining which pixels to use for background calculation.
        True=Use.
    stat : {'mode', 'median'}
        Statistic to use in calculating the residual background.

    Returns
    -------
    numpy.ndarray
        background subtracted data (nrow, ncol)
    """
    if header is None:
        header = fits.header.Header()
    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        msg = "background not subtracted (invalid data)"
        addhist(header, msg)
        log.warning(msg)
        return
    if stat not in ['mode', 'median']:
        log.warning('Unrecognized background statistic; using stat=mode.')
        stat = 'mode'
    result = data.copy()
    ndim = len(data.shape)
    if ndim == 2:
        result = np.array([result])

    if not getpar(header, 'BGSUB', dtype=bool, default=False,
                  comment='residual background subtracted after stack'):
        return
    section = read_section(data.shape[-1], data.shape[-2])
    bglevel = background(data, section, stat=stat, mask=mask)
    if bglevel is None:
        msg = "could not subtract background (invalid background)"
        addhist(header, msg)
        log.warning(msg)
        return
    for frame, sub in zip(result, bglevel):
        msg = "Removing BG Level: %s" % sub
        addhist(header, msg)
        log.info(msg)
        frame -= sub
    if ndim == 2:
        result = result[0]
    return result


def stack(data, header, variance=None, mask=None, extra=None, stat='mode'):
    """
    Subtracts chop/nod pairs to remove astronomical/telescope background

    This function uses sofia_redux.instruments.forcast.readmode
    to read the chop/nod mode from the header; this determines how the
    frames are subtracted.  If the BGSUB keyword is set to True in the
    configuration file, then any residual background level will be subtracted
    from the stacked data.  If the BGSCALE keyword is set to True in the
    configuration file, the individual frames will be multiplicatively
    scaled to the same level before subtraction.  If the JBCLEAN keyword
    is set to median, sofia_redux.instruments.forcast.jbclean will be
    called on the stacked data.  At the end of the function, the units
    will be converted to mega-electrons per second, using FRMRATE and
    EPERADU keywords in the header.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nimage, nrow, ncol)
    header : astropy.io.fits.header.Header
        Input FITS header; will be updated with HISTORY message
    variance : numpy.ndarray, optional
        Variance array (nimage, nrow, ncol) to update in parallel
        with the data array.
    mask : numpy.ndarray, optional
        (col, row) Illumination mask to indicate regions of the image to
        use in calculating the background (True = good)
    extra : dict, optional
        If provided, will be updated with:
            posdata -> numpy.ndarray: summed stacks
            chopsub -> numpy.ndarray: chop-subtracted data
            nodsub -> numpy.ndarray: nod-subtracted data
    stat : {'mode', 'median'}
        Statistic to use in calculating the residual background.

    Returns
    -------
    2-tuple
        - numpy.ndarray (nrow, ncol) stacked data
        - numpy.ndarray (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
        msg = "Invalid image header"
        addhist(header, msg)
        log.error(msg)
        return
    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        msg = 'Stack was not applied (Invalid data)'
        addhist(header, msg)
        log.error(msg)
        return

    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    if variance is not None and not dovar:
        msg = "Not propagating variance (invalid variance)"
        addhist(header, msg)
        log.warning(msg)
    var = variance.copy() if dovar else None

    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != data.shape[-2:]:
            msg = "background mask invalid"
            addhist(header, msg)
            log.warning(msg)
            mask = None
        else:
            mask = mask.astype(bool)

    # Get everything 3-D (nframes, nrow, ncol)
    ndim = len(data.shape)
    if ndim == 2:
        posdata = np.array([data.copy()])
        if dovar:
            var = np.array([var])
    else:
        posdata = data.copy()

    # Add frames
    result = add_stacks(posdata, header, variance=var)
    if result is None:
        log.error("stack failed")
        return
    posdata, var = result
    if isinstance(extra, dict):
        extra['posdata'] = posdata

    bglevel = background_scale(posdata, header, mask=mask)
    mode = readmode(header)

    # run appropriate method
    c2_modes = ['C2', 'C2NC2']
    map_modes = ['NAS', 'NOS', 'C2NC4', 'NXCAC', 'C2N', 'NMC',
                 'NPC', 'NPCCAS', 'NPCNAS', 'C2ND', 'SLITSCAN',
                 'MAP']

    if mode in c2_modes:
        result = stack_c2nc2(posdata, header, variance=var,
                             bglevel=bglevel, extra=extra)
    elif mode in map_modes:
        result = stack_map(posdata, header, variance=var,
                           bglevel=bglevel, extra=extra)
    elif mode == 'C3D':
        result = stack_c3d(
            posdata, header, variance=var, extra=extra)
    elif mode == 'CM':
        result = stack_cm(
            posdata, header, variance=var, extra=extra)
    elif mode == 'STARE':
        result = stack_stare(posdata, header, variance=var)
    else:
        msg = "Stack failed (invalid instrument mode)"
        addhist(header, msg)
        log.error(msg)
        return

    if result is None:
        msg = "Aborting stack - stacking failed"
        addhist(header, msg)
        log.error(msg)
        return
    else:
        stacked, var = result

    # add NaNs back in where result is identically zero
    stacked[stacked == 0] = np.nan
    if var is not None:
        var[var == 0] = np.nan

    if mode != 'STARE':
        choptsa = getpar(header, 'CHOPTSAC', default=None, dtype=int,
                         comment="Chop convention. Should be -1 after OC1B")
        if choptsa is None:
            msg = "Value of chop tsa convention was " \
                  "not found. Using default: -1"
        elif abs(choptsa) == 1:
            msg = "Value of chop tsa convention is %i" % choptsa
            stacked = -1 * choptsa * stacked
        else:
            msg = "Value of chop tsa convention (%i) " % choptsa
            msg += "is different from -1, 1. "
            msg += "Using default value: -1"
        addhist(header, msg)
        log.info(msg)

    # Clean jailbars if JBCLEAN='MEDIAN'.  If method = is 'FFT', cleaning
    # is done in sofia_redux.instruments.forcast.clean instead.
    jbmethod = getpar(header, 'JBCLEAN', dtype=str, default='x',
                      comment="Jail bar cleaning algorithm")
    if jbmethod.strip().lower() == 'median':
        result = jbclean(stacked, header, variance=var)
        if result is None:
            msg = "Jailbars not removed (jbclean MEDIAN method failed)"
            addhist(header, msg)
            log.warning(msg)
        else:
            log.info('Jailbars cleaned with MEDIAN method')
            stacked, var = result
    else:
        log.info('Jailbars not removed')

    # Convert to mega-electrons per second
    result = convert_to_electrons(stacked, header, variance=var)
    if result is None:
        # Not a fatal error
        msg = "Could not convert to Mega-electrons per second"
        addhist(header, msg)
        log.warning(msg)
    else:
        stacked, var = result

    # Remove any residual background remaining
    bgsub = subtract_background(stacked, header=header, mask=mask, stat=stat)
    if isinstance(bgsub, np.ndarray):
        addhist(header, "Residual background subtracted from final stack")
        stacked = bgsub

    # update the product type
    hdinsert(header, 'PRODTYPE', 'STACKED', refkey=kref)
    return stacked, var
