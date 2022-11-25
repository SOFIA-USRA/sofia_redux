# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

from astropy import log
import bottleneck as bn
import numpy as np

from sofia_redux.instruments.exes.lincor import lincor
from sofia_redux.instruments.exes.get_badpix import get_badpix
from sofia_redux.instruments.exes.utils import get_reset_dark

__all__ = ['readraw']


def readraw(data, header, do_lincor=False, algorithm=None,
            toss_nint=0, copy_int=True):
    """
    Correct for nonlinearity combine individual readouts.

    First corrects nonlinearity for each readout frame (`exes.lincor`),
    then determines the readout pattern from the OTPAT keyword.

    For readout coadding methods, this step currently has support for Fowler
    mode, simple destructive read, and equally spaced sample-up-the-ramp
    patterns.  Frames are combined and variance is calculated based on the
    readout pattern, using algorithms from the following paper:

        Nonlinearity Corrections and Statistical Uncertainties
        Associated with Near-Infrared Arrays, William D. Vacca,
        Michael C. Cushing and John T. Rayner (2004, PASP 116, 352).

    The readout algorithm may also be directly specified by the `algorithm`
    parameter, which takes the following possible integer values:

        - 0 : Use the last destructive frame only
        - 1 : Simple destructive mode
        - 2 : Use the first and last frames only
        - 3 : Use the second and penultimate frames only
        - 4 : Fowler mode
        - 5 : Sample-up-the-ramp mode

    After readout coadd, multiple frames taken at the same nod position (as
    indicated by the NINT keyword) are averaged.  Extra frames that
    do not match the OTPAT or NINT pattern are dropped.

    Parameters
    ----------
    data : numpy.ndarray
        [nframe, nspec, nspat] array of float values.
    header : fits.Header
        FITS header produced by `exes.readhdr`.  Note that the header will
        be updated in-place.
    do_lincor : bool, optional
        If True, do the nonlinear correction.  If False, no check is
        performed on data quality: the output mask will be all False
        (no bad pixels).
    algorithm : int, optional
        Used to select processing mode.  If None, defaults to that determined
        by `check_readout_pattern`.
    toss_nint : int, optional
        If provided and greater than 0 but less than the number of
        integrations present in the data, `toss_nint` integrations will be
        discarded from the beginning of the data array.  If the number
        equals the number of integrations, the first nod is a B nod,
        and there is another B nod in the data array, the first integrations
        are instead replaced with a copy of the second B nod.
    copy_int : bool, optional
        If `toss_nint` is greater than 0 and `copy_int` is True, replacement
        integrations are always copied from the next B nod, regardless of
        the number of integrations available.

    Returns
    -------
    coadd_data, variance, mask : 3-tuple of numpy.ndarray
       The coadded data, variance, and good data mask.  The coadd_data and
       variance have shape (nframes, ny, nx).  The mask has shape (ny, nx)
       and Boolean data type, where True indicates a good pixel; False
       indicates a bad pixel.
    """
    _check_header(header)
    data = _check_data(data)
    data, mask = _get_data_subarray(data, header)
    data, readmode = _check_readout_pattern(data, header)

    if isinstance(algorithm, int):
        touse_lookup = {
            0: 'last destructive',
            1: 'destructive',
            2: 'first/last nd',
            3: 'second/penultimate nd',
            4: 'fowler',
            5: 'sample-up-the-ramp'}
        touse = touse_lookup.get(algorithm)
        if touse is None:
            raise ValueError(f'Invalid algorithm: {touse}')
    else:
        touse = readmode['mode']

    if do_lincor:
        log.info("Applying linear correction")
        lindata, linmask = lincor(data, header)
        # "And" the output mask to get all nonlinear detector pixels
        mask &= np.all(linmask, axis=0)
    else:
        log.info("Linear correction not applied: do_lincor=False")
        lindata = data.copy()

    log.info('')
    log.info(f'Using read mode algorithm: {touse}')
    log.info('')
    if touse == 'destructive' or touse == 'last destructive':
        coadd_data, variance = _process_destructive(data, header, readmode)
    elif touse == 'first/last nd':
        coadd_data, variance = _process_nondestructive1(
            lindata, header, readmode)
    elif touse == 'second/penultimate nd':
        coadd_data, variance = _process_nondestructive2(
            lindata, header, readmode)
    elif touse == 'fowler':
        coadd_data, variance = _process_fowler(lindata, header, readmode)
    elif touse == 'sample-up-the-ramp':
        coadd_data, variance = _process_sample_up_the_ramp(
            data, header, readmode)
    else:  # pragma: no cover
        # shouldn't be reachable
        raise ValueError(f'Unrecognized readmode: {readmode["mode"]}')

    coadd_data, variance = _combine_nods(coadd_data, variance, header,
                                         readmode, toss_nint, copy_int)

    return coadd_data, variance, mask


def _get_data_subarray(data, header):
    ectpat = str(header.get('ECTPAT', 'UNKNOWN')).split()
    if len(ectpat) == 6:
        ectpat = np.asarray(ectpat, dtype=int)
        ystart = ectpat[2] * 2
        ystop = ystart + (ectpat[3] * 2)
        xstart = ectpat[4]
        xstop = xstart + (ectpat[5])
    else:
        xstart = 0
        xstop = data.shape[2]
        ystart = 0
        ystop = data.shape[1]

    nframes, ny, nx = data.shape

    bpm = get_badpix(header)
    if bpm is not None:
        if bpm.shape[0] < ny or bpm.shape[1] < nx:
            raise ValueError(
                "Bad pixel mask is too small %s; data shape is %s" %
                (repr(bpm.shape), repr(data.shape[1:])))

        # Reference columns are marked as 2 in the badpix mask
        refidx = bpm == 2
        if refidx.any():
            xrange = np.where(~refidx)[1]
            data = data[:, :, xrange.min():xrange.max() + 1]
            bpm = bpm[:, xrange.min():xrange.max() + 1]
            nx = data.shape[2]
            xstart = 0
            xstop = nx

        # Take subarray if needed
        bpm = bpm[ystart:ystop, xstart:xstop]

    mask = np.full((ny, nx), True)
    if bpm is not None:
        mask[bpm != 1] = False

    # Store data size
    header['DETSEC'] = '[%i:%i,%i:%i]' % (
        xstart + 1, xstop, ystart + 1, ystop)
    header['NSPAT'] = nx
    header['NSPEC'] = ny

    # Correct frametime for subarray size as needed
    header['FRAMETIM'] *= ny / 1024.

    return data, mask


def _check_header(header):
    """Ensure header values are of the correct type"""
    header['OTPAT'] = str(header['OTPAT']).upper().strip()
    header['NINT'] = int(header['NINT'])
    header['FRAMETIM'] = float(header['FRAMETIM'])
    header['READNOIS'] = float(header['READNOIS'])
    header['DARKVAL'] = float(header['DARKVAL'])

    try:
        header['PAGAIN'] = float(header['PAGAIN'])
    except (KeyError, ValueError):
        header['PAGAIN'] = 1.0

    try:
        header['EPERADU'] = float(header['EPERADU'])
    except (KeyError, ValueError):
        header['EPERADU'] = 1.0


def _check_data(data):
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError("Data must be a 3-D cube (nframe, nspec, nspat)")
    return data


def _check_readout_pattern(data, header):
    regex = re.compile('[STNDC][0-9]+')
    if regex.match(header['OTPAT']) is None:
        raise ValueError("Unreadable OT pattern. OTPAT=%s" % header['OTPAT'])
    patterns = regex.findall(header['OTPAT'])

    spin = trash = nondest = dest = coadd = nread = npass = 0
    for pattern in patterns:
        optype, n_op = pattern[0], int(pattern[1:]) + 1
        if optype == 'S':
            spin += n_op
            npass += n_op
        elif optype == 'T':
            trash += n_op
        elif optype == 'D':
            dest += n_op
            npass += n_op
        elif optype == 'C':
            coadd += n_op
            dest += n_op
            npass += n_op
        elif optype == 'N':
            nread += 1
            nondest += n_op
            npass += n_op

    nframes = coadd if coadd > 0 else (nondest + dest)
    npattern = data.shape[0] // nframes
    if npattern == 0:
        raise ValueError(f"Data does not match OTPAT={header['OTPAT']}."
                         " A full pattern is not present; aborting.")
    elif (npattern * nframes) != data.shape[0]:
        log.warning(f"Data does not match OTPAT={header['OTPAT']}."
                    " Dropping extra frames.")
        data = data[:npattern * nframes]

    if nread == 2 or (nread == 1 and nondest == 1):
        mode = 'fowler'
    elif nondest == 0 and dest == 1 and nframes == 1:
        mode = 'destructive'
    elif nondest != 0 and dest == 1:
        mode = 'sample-up-the-ramp'
    else:
        raise ValueError("Unrecognized readmode")

    log.info('')
    log.info(f"Readout pattern: {header['OTPAT']}")
    log.info(f"Frame(s) per pattern: {nframes}")
    log.info(f"Total frames: {data.shape[0]}")
    log.info(f"Recommended readout mode: {mode}")

    readmode = {'spin': spin,
                'trash': trash,
                'nondest': nondest,
                'dest': dest,
                'coadd': coadd,
                'nread': nread,
                'npass': npass,
                'nframes': nframes,
                'npattern': npattern,
                'mode': mode}

    return data, readmode


def _process_destructive(data, header, readmode):
    # Using D only
    log.info('Last destructive read')
    frametime = float(header['FRAMETIM'])
    interval = (readmode['dest'] + readmode['nondest']
                + readmode['spin']) * frametime
    log.info(f'Interval = {interval}')

    read_noise = float(header['READNOIS'])
    gain = float(header['PAGAIN'])
    eperadu = float(header['EPERADU'])
    zeroval = float(header['DARKVAL']) / (frametime * gain)
    gain_factor = interval * gain
    v_gain_factor = interval * eperadu

    shape = readmode['npattern'], data.shape[1], data.shape[2]
    coadd_data = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)

    dark1s = get_reset_dark(header)
    for i in range(readmode['npattern']):
        if readmode['coadd'] == 1:  # pragma: no cover
            # The Fowler coadd/subtraction should have been done in hardware
            # (this OTPAT was never used for EXES)
            coadd_data[i] = zeroval - data[i] / gain_factor
        else:
            pattern_start = i * readmode['nframes']
            pattern_end = pattern_start + readmode['nframes']
            signal = data[pattern_end - 1]
            coadd_data[i] = zeroval - (signal - dark1s[None]) / gain_factor

        variance[i] = (np.abs(coadd_data[i]) / v_gain_factor
                       + (read_noise / v_gain_factor) ** 2)

    header['NFRAME'] = 1
    header['BEAMTIME'] = interval
    return coadd_data, variance


def _process_nondestructive1(lindata, header, readmode):
    if readmode['nondest'] + readmode['dest'] < 2:
        raise ValueError('OTPAT is not suitable for First/Last ND mode')

    log.info("First/Last ND mode")
    frametime = float(header['FRAMETIM'])
    nread = (readmode['nondest'] + readmode['dest']) // 2

    interval = (readmode['dest']
                + readmode['nondest']
                + readmode['spin'] - 1) * frametime
    log.info(f'Interval = {interval}')

    gain = float(header['PAGAIN'])
    gainfac = gain * interval
    vgainfac = float(header['EPERADU']) * interval
    readnoise = float(header['READNOIS'])
    darkval = float(header['DARKVAL'])
    zeroval = darkval / (frametime * gain)
    nframes = readmode['nframes']

    shape = readmode['npattern'], lindata.shape[1], lindata.shape[2]
    coadd_data = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)

    for i in range(shape[0]):
        if readmode['coadd'] == 1:  # pragma: no cover
            # (this OTPAT was never used for EXES)
            coadd_data[i] = zeroval - lindata[i] / gainfac
        else:
            pattern_start = i * nframes  # 1st frame
            pattern_end = pattern_start + nframes  # last frame
            pattern_data = lindata[pattern_start:pattern_end]
            pedestal = pattern_data[0]
            signal = pattern_data[pattern_end - pattern_start - 1]
            coadd_data[i] = zeroval - ((signal - pedestal) / gainfac)

        variance[i] = ((np.abs(coadd_data[i]) / vgainfac)
                       * (1 - (frametime * (nread ** 2 - 1))
                          / (3 * interval * nread))
                       + 2 * (readnoise ** 2) / ((vgainfac ** 2) * nread))

    header['NFRAME'] = nread
    header['BEAMTIME'] = interval
    return coadd_data, variance


def _process_nondestructive2(lindata, header, readmode):
    if readmode['nondest'] + readmode['dest'] < 4:
        raise ValueError('OTPAT is not suitable for '
                         'Second/Penultimate ND mode')

    log.info("Second/Penultimate ND mode")
    frametime = float(header['FRAMETIM'])
    nread = (readmode['nondest'] - 1) // 2

    interval = (readmode['nondest'] + readmode['spin'] - 2) * frametime
    log.info(f'Interval = {interval}')

    gain = float(header['PAGAIN'])
    gainfac = gain * interval
    vgainfac = float(header['EPERADU']) * interval
    readnoise = float(header['READNOIS'])
    darkval = float(header['DARKVAL'])
    zeroval = darkval / (frametime * gain)
    nframes = readmode['nframes']

    shape = readmode['npattern'], lindata.shape[1], lindata.shape[2]
    coadd_data = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)

    for i in range(shape[0]):
        if readmode['coadd'] == 1:  # pragma: no cover
            # (this OTPAT was never used for EXES)
            coadd_data[i] = zeroval - lindata[i] / gainfac
        else:
            pattern_start = i * nframes + 1  # 2nd frame
            pattern_end = pattern_start + nframes - 2  # penultimate frame
            pattern_data = lindata[pattern_start:pattern_end]
            pedestal = pattern_data[0]

            signal = pattern_data[pattern_end - pattern_start - 1]
            coadd_data[i] = zeroval - ((signal - pedestal) / gainfac)

            variance[i] = ((np.abs(coadd_data[i]) / vgainfac)
                           * (1 - (frametime * (nread ** 2 - 1))
                              / (3 * interval * nread))
                           + 2 * (readnoise ** 2) / ((vgainfac ** 2) * nread))

    header['NFRAME'] = nread
    header['BEAMTIME'] = interval
    return coadd_data, variance


def _process_fowler(lindata, header, readmode):
    log.info("Fowler mode")
    frametime = float(header['FRAMETIM'])
    gain = float(header['PAGAIN'])
    eperadu = float(header['EPERADU'])
    readnoise = float(header['READNOIS'])
    darkval = float(header['DARKVAL'])
    zeroval = darkval / (frametime * gain)

    nread = (readmode['nondest'] + readmode['dest']) // 2
    interval = (readmode['npass'] - nread) * frametime
    gainfac = interval * gain
    vgainfac = interval * eperadu
    log.info(f'Interval = {interval}')

    shape = readmode['npattern'], lindata.shape[1], lindata.shape[2]
    coadd_data = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)

    for i in range(shape[0]):
        if readmode['coadd'] == 1:  # pragma: no cover
            # The Fowler coadd/subtraction should have been done in hardware
            # (this OTPAT was never used for EXES)
            coadd_data[i] = zeroval - lindata[i] / gainfac
        else:
            pattern_start = i * readmode['nframes']
            pattern_end = pattern_start + readmode['nframes']
            pattern_data = lindata[pattern_start:pattern_end]

            if nread > 1:
                # Add initial reads for pedestal level
                pedestal = bn.nansum(pattern_data[:nread], axis=0)
                # Add final reads for signal level
                signal = bn.nansum(
                    pattern_data[nread:readmode['nframes']], axis=0)
            else:
                pedestal = pattern_data[0]
                signal = pattern_data[1]
            coadd_data[i] = zeroval - ((signal - pedestal) / (nread * gainfac))

        variance[i] = ((np.abs(coadd_data[i]) / vgainfac)
                       * (1 - (frametime * (nread ** 2 - 1))
                          / (3 * interval * nread))
                       + 2 * (readnoise ** 2) / ((vgainfac ** 2) * nread))

    header['NFRAME'] = nread
    header['BEAMTIME'] = interval
    return coadd_data, variance


def _process_sample_up_the_ramp(data, header, readmode):
    if readmode['nondest'] + readmode['dest'] < 3:
        raise ValueError('OTPAT is not suitable for '
                         'Second/Penultimate ND mode')

    log.info("Sample-up-the-ramp mode")
    frametime = float(header['FRAMETIM'])
    gain = float(header['PAGAIN'])
    eperadu = float(header['EPERADU'])
    darkval = float(header['DARKVAL'])
    zeroval = darkval / (frametime * gain)
    readnoise = float(header['READNOIS'])

    nread = readmode['nframes']
    interval = (readmode['npass'] - 1) * frametime
    alpha = nread * (nread + 1) // 12
    gainfac = interval * gain
    vgainfac = interval * eperadu
    log.info(f'Interval = {interval}')

    shape = readmode['npattern'], data.shape[1], data.shape[2]
    coadd_data = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)

    for i in range(readmode['npattern']):
        if readmode['coadd'] == 1:  # pragma: no cover
            # The SUTR fit should have been done in hardware
            # (this OTPAT was never used for EXES)
            coadd_data[i] = zeroval - (data[i] / gainfac)
        else:
            nframes = readmode['nframes']
            pattern_start = i * nframes
            pattern_end = pattern_start + nframes
            pattern_data = data[pattern_start:pattern_end]

            fac = ((np.arange(nread) + 1) - ((nread + 1) / 2)) / alpha
            s = bn.nansum(pattern_data[:nread] * fac[:, None, None], axis=0)
            coadd_data[i] = zeroval - (s / gainfac)

        variance[i] = (6 * np.abs(coadd_data[i]) * ((nread ** 2) + 1))
        variance[i] /= (5 * vgainfac * nread * (nread + 1))
        variance[i] += ((12 * (readnoise ** 2) * (nread - 1))
                        / ((vgainfac ** 2) * nread * (nread + 1)))

    header['NFRAME'] = nread
    header['BEAMTIME'] = interval
    return coadd_data, variance


def _combine_nods(coadd_data, variance, header, readmode, toss_nint, copy_int):
    # Check for multiple frames at the same nod position
    nint = int(header['NINT'])
    if nint <= 1:
        return coadd_data, variance

    npatt = readmode['npattern']
    nout = npatt // nint

    itot = header['BEAMTIME'] * nint * nout
    header['INTTIME'] = itot
    if 'OFF_SLIT' in str(header.get('INSTMODE', 'UNKNOWN')).upper():
        header['EXPTIME'] = itot / 2.
    else:
        header['EXPTIME'] = itot
    header['NEXP'] = nint * nout

    log.info(f"Combining {nint} frames per nod position")
    if npatt < nint:
        raise ValueError(f"Data does not match NINT={nint}")
    elif npatt % nint != 0:
        log.warning(f"Data does not match NINT={nint}; "
                    f"ignoring extra frames.")

    shape = nout, coadd_data.shape[1], coadd_data.shape[2]
    mcoadd = np.empty(shape, dtype=float)
    mvar = np.empty(shape, dtype=float)
    nodn = header.get('NODN', 1)
    for i in range(nout):
        if i == 0:
            if toss_nint > 0:
                if toss_nint < nint and (not copy_int or nodn < 2):
                    log.info(f'Dropping the first {toss_nint} frames')
                    start = toss_nint
                else:
                    if toss_nint <= nint and nodn >= 2:
                        # allow copying the sky data from the next nod
                        # if available
                        log.warning('Copying the first '
                                    'integration(s) from the next B nod')
                        cstart = 2 * nint
                        cstop = 2 * nint + nint
                        coadd_data[0:nint] = coadd_data[cstart:cstop]
                        variance[0:nint] = variance[cstart:cstop]
                    else:
                        # otherwise, just ignore the toss for this file
                        log.warning('TOSS_NINT is set higher than '
                                    'NINT; ignoring.')

                    # in either case, use all frames
                    start = 0
            else:
                start = 0
        else:
            start = i * nint
        stop = i * nint + nint
        ncomb = stop - start
        mcoadd[i] = bn.nansum(coadd_data[start:stop], axis=0) / ncomb
        mvar[i] = bn.nansum(variance[start:stop], axis=0) / (ncomb ** 2)

    return mcoadd, mvar
