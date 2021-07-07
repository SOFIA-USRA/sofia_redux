# Licensed under a 3-clause BSD style license - see LICENSE.rst

from datetime import datetime
import os

from astropy import log
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import numpy as np

from sofia_redux.instruments.fifi_ls.make_header import make_header
from sofia_redux.toolkit.utilities \
    import (hdinsert, write_hdul, gethdul, multitask, valid_num)


__all__ = ['get_channel', 'get_split_params', 'trim_data',
           'name_output_files', 'separate_chops',
           'split_grating_and_chop', 'wrap_split_grating_and_chop']


def get_channel(hdul, channel_index=3):
    """
    Returns RED or BLUE channel extracted from header and data.

    If the channel cannot be read from the prime header, it is read
    from the data header.

    Parameters
    ----------
    hdul : fits.HDUList
    channel_index : int, optional
        data header location determining channel

    Returns
    -------
    str
        RED, BLUE, or UNKNOWN
    """
    channel = str(hdul[0].header.get('CHANNEL', 'UNKNOWN'))
    result = 'UNKNOWN'
    channel = channel.strip().upper()
    if valid_num(channel):
        if channel == '1':
            result = 'BLUE'
        elif channel == '0':
            result = 'RED'
    elif channel != 'UNKNOWN':
        if channel in ['BLUE', 'RED']:
            result = channel
    else:
        blue = (hdul[1].data['HEADER'][0][channel_index] & 2) // 2 == 1
        result = 'BLUE' if blue else 'RED'
    return result


def get_split_params(hdul, channel_index=3, sample_index=4,
                     ramp_index=5):
    """
    Check if a file can be split by basic header checks.  If so,
    return the necessary parameters for splitting the file.

    Parameters
    ----------
    hdul : fits.HDUList
    channel_index : int, optional
        data header index indicating channel
    sample_index : int, optional
        data header index indicating number of samples in data
    ramp_index : int, optional
        data header index indicating number of ramps in data

    Returns
    -------
    dict
        G_STRT (int) -> grating/inductosyn start
        G_PSUP (int) -> number of grating positions going up
        G_SZUP (int) -> step size on the way up
        G_PSDN (int) -> number of grating positions coming down
        G_SZDN (int) -> step size on the way down
        G_CYC (int) -> number of grating sweeps (up-down)
        C_CYC (int) -> number of chop cycles per grating position
        P_STRT (int) -> primary array start
        P_PSUP (int) -> primary array number of positions going up
        P_SZUP (int) -> primary array step size on the way up
        RAMPLN (int) -> ramp length
        CHOPLN (int) -> chop length
        C_AMP (float) -> chop amplitude
        CHANNEL (str) -> channel 'RED' or 'BLUE'
        success (bool) -> True indicates all parameters are valid
    """
    x = {'success': True}
    header = hdul[0].header
    chop_scheme = str(header.get('C_SCHEME', 'UNKNOWN')).strip()
    fname = header.get('FILENAME', 'UNKNOWN')
    if chop_scheme != '2POINT':
        log.error("Invalid chopper scheme for file %s" % fname)
        x['success'] = False
    try:
        x['C_SCHEME'] = int(chop_scheme[0])
    except ValueError:
        x['success'] = False
        x['C_SCHEME'] = None

    x['channel_index'] = channel_index
    x['sample_index'] = sample_index
    x['ramp_index'] = ramp_index

    x['CHANNEL'] = get_channel(hdul, channel_index=x['channel_index'])
    if x['CHANNEL'] not in ['RED', 'BLUE']:
        log.error("Invalid spectral channel for file %s" % fname)
        x['success'] = False
    hdinsert(header, 'CHANNEL', x['CHANNEL'], comment='Detector channel')
    x['CHOPLN'] = header.get('C_CHOPLN')
    x['C_AMP'] = header.get('C_AMP', 0)
    for key in ['G_STRT', 'G_PSUP', 'G_SZUP', 'G_PSDN',
                'G_SZDN', 'G_CYC', 'C_CYC', 'RAMPLN']:
        color_key = key + '_%s' % ('R' if x['CHANNEL'] == 'RED' else 'B')
        x[key] = header.get(color_key)

    x['PRIMARAY'] = header.get('PRIMARAY', 'UNKNOWN').upper().strip()
    suffix = 'R' if x['PRIMARAY'] == 'RED' else 'B'
    for key in ['STRT', 'PSUP', 'SZUP']:
        x['P_' + key] = header.get('G_%s_%s' % (key, suffix))

    for key, value in x.items():
        if key in ['CHANNEL', 'PRIMARAY', 'success']:
            continue
        if not valid_num(value) or float(value) == -9999:
            log.error("Invalid %s for file %s" % (key, fname))
            x['success'] = False
        else:
            dtype = float if key == 'C_AMP' else int
            x[key] = dtype(value)

    rl, cl = x['RAMPLN'], x['CHOPLN']
    # ramp length and chop length must be integer multiples of each other
    try:
        lengths = sorted([rl, cl])
    except TypeError:
        log.error("Bad choplength or ramplength")
        x['success'] = False
    else:
        if int(lengths[1] / lengths[0]) * lengths[0] != lengths[1]:
            log.error(
                "Choplength is not an integer multiple/quotient of ramplength")
            x['success'] = False

        if rl == cl:
            log.error("Unexpected 2POINT chop, case 1: "
                      "ramplength = choplength for file %s" % fname)
            x['success'] = False

        if (rl > cl) and (rl // cl) % 2 == 0:
            log.error("Unexpected 2POINT chop, case 2: "
                      "ramplength = 2*n*choplength where "
                      "n = 1, 2, 3... for file %s" % fname)
            x['success'] = False

    # if c_amp is zero, the chop length is irrelevant - set to rampln
    if x['C_AMP'] == 0:
        x['CHOPLN'] = x['RAMPLN']

    return x


def trim_data(hdul, params):
    """
    Remove partial ramps and "unpaired" chop plateaus.

    Note that 4POINT chop is untested

    Assumptions:
      1. file starts with chop0
      2. chop0 and a following chop1 (2POINT) / chop2 (4POINT)
         makes a pair (chop1 and chop3 makes another pair for
         4POINT chop scheme).
      3. no missing ramps except at beginning or end of file

    Parameters
    ----------
    hdul : fits.HDUList
    params : dict

    Returns
    -------
    fits.HDUList
    """
    # get scan position data if present
    if 'SCANPOS' in hdul:
        posdata = hdul['SCANPOS'].data
    else:
        posdata = None

    # remove partial ramps
    data = hdul[1].data
    head = np.argwhere(data['header'][:, params['sample_index']] == 0)
    head = 0 if len(head) == 0 else head[0, 0]
    last_frame = data['header'][-1][params['sample_index']]
    tail = len(data)
    if last_frame != (params['RAMPLN'] - 1):
        last_ramp = data['header'][-1][params['ramp_index']]
        tail = np.argwhere(data['header'][:, params['ramp_index']]
                           >= last_ramp)
        tail = tail[tail > head]
        tail = tail[0] if len(tail) != 0 else len(data)
    data = data[head:tail]
    if posdata is not None:
        posdata = posdata[head:tail]

    # Remove unpaired chops
    # number of ramps per chop phase or number of ramps to co-add in fit_ramps
    n = params['CHOPLN'] // params['RAMPLN']
    if n > 0:
        index = data['header'][:, params['ramp_index']] // n
        chop_phase = index % params['C_SCHEME']
        head = np.argwhere(chop_phase == 0)
        head = head[0, 0] if len(head) != 0 else 0
        tail = np.argwhere(chop_phase == np.nanmax(chop_phase))
        tail = (tail[-1, 0] + 1) if len(tail) != 0 else len(chop_phase)
        data = data[head:tail]
        if posdata is not None:
            posdata = posdata[head:tail]

    hdu1 = fits.BinTableHDU(data, header=hdul[1].header)
    hdul[1] = hdu1

    if posdata is not None:
        hdul['SCANPOS'] = fits.BinTableHDU(posdata, name='SCANPOS')

    return hdul


def name_output_files(hdul):
    """
    Names the split files.

    Parameters
    ----------
    hdul : fits.HDUList

    Returns
    -------
    2-tuple
        str : chop0 filename
        str : chop1 filename
    """
    filenum = hdul[0].header.get('FILENUM', 'UNKNOWN')
    aorid = hdul[0].header.get('AOR_ID', 'UNKNOWN').strip().replace('_', '')
    if aorid == '':
        aorid = 'UNKNOWN'
    flight = hdul[0].header.get('MISSN-ID', 'UNKNOWN').split('_')[-1][1:]
    if valid_num(flight):
        flight = 'F%04i' % int(flight)
    else:
        flight = 'UNKNOWN'
    channel = get_channel(hdul)
    if channel in ['RED', 'BLUE']:
        channel = channel[:3]
    else:
        channel = 'UN'
    result = tuple('%s_FI_IFS_%s_%s_CP%i_%s.fits' %
                   (flight, aorid, channel, i, filenum) for i in range(2))
    return result


def separate_chops(hdul, params):
    """
    Separate data into different files, by chop index.

    Parameters
    ----------
    hdul : fits.HDUList
    params : dict

    Returns
    -------
    list of fits.HDUList
    """
    if params['RAMPLN'] > params['CHOPLN']:
        log.error("Case where ramplength > choplength not accounted for")
        return
    elif params['RAMPLN'] == params['CHOPLN'] and params['C_AMP'] != 0:
        log.error("Case where ramplength >= choplength not accounted for")
        return

    # Make arrays of inductosyn positions
    pos = params['G_STRT'] + np.arange(params['G_PSUP']) * params['G_SZUP']

    # For reference in spatial offset calculations, later
    prime = params['P_STRT'] + np.arange(params['P_PSUP']) * params['P_SZUP']

    # This shouldn't happen, but just in case: if the number of primary
    # positions is somehow less than the number of secondary positions,
    # extend the array and copy the last primary position into all
    # remaining elements
    if len(prime) < len(pos):
        prime = np.concatenate(
            (prime, np.full(len(pos) - len(prime), prime[-1])))
    n = params['CHOPLN'] // params['RAMPLN']  # ramps per chop phase

    data = hdul[1].data
    readouts = len(data)
    log.debug("number of readouts = %i" % readouts)
    log.debug("ramplength = %i" % params['RAMPLN'])
    log.debug("chop amplitude = %f" % params['C_AMP'])
    binsize = readouts // (n * (params['G_PSUP'] + params['G_PSDN']))
    log.debug("grating and chop binsize = %i" % binsize)

    # get scan position data if present
    if 'SCANPOS' in hdul:
        posdata = hdul['SCANPOS'].data
    else:
        posdata = None

    outfiles = name_output_files(hdul)
    if params['C_AMP'] != 0:
        data_header = hdul[1].data['HEADER']
        chop = ((data_header[:, params['ramp_index']] // n).astype(int)) % 2
    else:
        chop = np.zeros(len(data), dtype=int)

    nodpos = hdul[0].header.get('NODPOS', 0)
    fname = hdul[0].header.get('FILENAME', 'UNKNOWN')
    result = []
    for idx, outfile in enumerate(outfiles):
        if not (chop == idx).any():
            continue
        hdu0 = hdul[0].copy()
        primehead = hdu0.header
        chop_idx = np.argwhere(chop == idx).ravel()
        primehead['HISTORY'] = '----------------------'
        primehead['HISTORY'] = 'Data reduction history'
        primehead['HISTORY'] = '----------------------'
        primehead['HISTORY'] = 'Chops split into separate filenames'
        primehead['HISTORY'] = 'Grating scans split into separate extensions'
        hdinsert(primehead, 'NGRATING', params['G_PSUP'],
                 'Number of grating positions')
        hdinsert(primehead, 'CHOPNUM', idx, 'Chop number')
        hdinsert(primehead, 'PRODTYPE', 'grating_chop_split')
        hdinsert(primehead, 'FILENAME', outfile)

        newhdul = fits.HDUList([hdu0])

        for i in range(params['G_PSUP']):
            try:
                ext = data[chop_idx[(i * binsize): (i + 1) * binsize]]
            except (IndexError, TypeError):
                log.error("Array data does not match header "
                          "data for file %s" % fname)
                return

            image_data = ext['DATA']
            image_hdr = fits.ImageHDU(image_data).header
            t = Time(datetime.utcnow(), format='datetime').isot.split('.')[0]
            image_hdr['DATE'] = t, 'Creation UTC data of FITS header'
            image_hdr['INDPOS'] = pos[i], 'Inductosyn position '
            image_hdr['INDPOS_P'] = (
                prime[i], 'Prime array inductosyn position')
            image_hdr['CHOPNUM'] = idx
            image_hdr['NODPOS'] = nodpos
            image_hdr['BUNIT'] = ('adu', 'Data units')
            newext = fits.ImageHDU(
                image_data, name=f'FLUX_G{i}',
                header=image_hdr)
            newhdul.append(newext)

            if posdata is not None:
                posd = posdata[chop_idx[(i * binsize): (i + 1) * binsize]]
                newhdul.append(
                    fits.BinTableHDU(posd, name=f'SCANPOS_G{i}'))

        result.append(newhdul)
    return result


def _derive_positions(hdul, params):
    """
    Derive position data for OTF scans.

    Parameters
    ----------
    hdul : fits.HDUList
        Input OTF A nod data.
    params : dict
        Split parameters, as returned from get_split_params.

    Returns
    -------
    fits.HDUList
        Updated HDUList with 'SCANPOS' binary table attached.
    """
    header = hdul[0].header

    # sample size
    data = hdul[1].data['DATA']
    nreadout = data.shape[0]

    # time keys
    # start of frame 1
    unixstart = header['UNIXSTRT']
    # start of scan motion
    otfstart = header['OTFSTART']
    # data rate (sec / frame)
    alpha = header['ALPHA']
    # scan duration
    duration = header['TRK_DRTN']

    # first valid frame/ramp after motion starts
    frame1 = int(np.ceil((otfstart - unixstart) / alpha))
    ramp_start = frame1 - frame1 % params['RAMPLN']

    # check start and end values.
    # If bad, warn but allow it
    if ramp_start < 0 or frame1 < 0:
        log.warning(f'Bad OTF keywords: calculated '
                    f'scan start {ramp_start} < 0.')
        log.warning('Check UNIXSTRT, OTFSTART.')
        log.warning('Setting scan start to start of readouts.')
        ramp_start = 0
        frame1 = 0

    # last valid frame/ramp after motion ends
    frame2 = int(np.floor(frame1 + duration / alpha))
    ramp_end = frame2 + params['RAMPLN'] - frame2 % params['RAMPLN'] - 1
    if ramp_end >= nreadout:
        log.warning(f'Bad OTF keywords: calculated scan end '
                    f'{ramp_end} > {nreadout} readouts')
        log.error('Check UNIXSTRT, OTFSTART, TRK_DRTN.')
        log.warning('Setting scan end to end of readouts.')
        ramp_end = nreadout - 1

    # flag useful range
    log.debug(f'Useful OTF range: readout {ramp_start} to {ramp_end} '
              f'out of {nreadout}')
    flag = np.full(nreadout, True)
    flag[:ramp_start] = False
    flag[ramp_end + 1:] = False

    # compute UNIX time for center of ramp
    ftime = np.full(nreadout, unixstart, dtype=float)
    ftime += np.arange(nreadout, dtype=float) * alpha + alpha / 2.0

    # update exptime in header, subtracting time for unusable data
    header['EXPTIME'] -= (~flag).sum() * alpha

    # scan speed in RA/Dec directions, in arcsec/sec
    obslamv = header['OBSLAMV']
    obsbetv = header['OBSBETV']

    # check for zero speed in either direction
    if np.allclose(obslamv, 0):
        obslamv = 0.0
    if np.allclose(obsbetv, 0):
        obsbetv = 0.0

    # base position
    dlam_base = header['DLAM_MAP']
    dbet_base = header['DBET_MAP']

    # update positions for motion: allow extrapolation
    # within the first and last valid ramps
    motion_index = np.arange(nreadout) - frame1
    dlam = dlam_base + obslamv * alpha * motion_index
    dbet = dbet_base + obsbetv * alpha * motion_index

    # set to assumed position beyond ramp ends
    dlam[:ramp_start] = dlam[ramp_start]
    dbet[:ramp_start] = dbet[ramp_start]
    dlam[ramp_end + 1:] = dlam[ramp_end]
    dbet[ramp_end + 1:] = dbet[ramp_end]

    positions = Table()
    positions['DLAM_MAP'] = dlam
    positions['DBET_MAP'] = dbet
    positions['FLAG'] = flag
    positions['UNIXTIME'] = ftime

    hdul.append(fits.BinTableHDU(positions, name='SCANPOS'))

    return hdul


def split_grating_and_chop(filename, write=False, outdir=None):
    """
    Split FIFI-LS raw data file into separate FITS files.

    Files are split based on the chop cycle, with each of the grating
    positions in a separate FITS extension in each file.

    The procedure is:

        1. Read a raw data FITS file
        2. (optional) Check header for compliance with SOFIA requirements.
           Abort if failure.
        3. Reorganize data according to chopper phase and inductosyn
           position.
        4. (optional) Write output files to disk

    FITS files written to disk will each contain data from a single
    chopper phase.  The typical case is a 2-point chop, for which two
    FITS files are written.  No-chop mode is also supported, in which
    case one file is written to disk.  The zeroth extension in the output
    file contains the prime header only.  There are n_scan image
    extensions, containing a header (includes the keyword INDPOS,
    indicating the inductosyn positions of the scan) and a data array,
    with dimensions [26, 18] x (readout frames).

    For A nods taken in OTF (scanning) mode, an additional binary table is
    attached to the file, containing the on-sky position for each readout
    sample.  The table has columns DLAM_MAP, DBET_MAP, FLAG, and UNIXTIME,
    where DLAM_MAP and DBET_MAP indicate the RA and Dec offset from the
    base position, respectively, and UNIXTIME indicates the time in UNIX
    seconds for the readout.  The FLAG column holds a Boolean value
    indicating scanning status: if True, the telescope was in scanning
    motion for the readout; if False, scanning motion had either not
    begun, or had stopped prior to the readout.  Readouts for which the
    FLAG is False may not have accurate position data.

    Only 2-point chopper schemes are allowed (C_SCHEME='2POINT').  If the
    chop amplitude (C_AMP) is zero, it will be treated as a no-chop
    ("total power") observation.

    The output file name is generated from the flight number, AOR-ID,
    detector channel, and raw file number, as::

        chop 0 = <FLIGHT>_FI_IFS_<AOR-ID>_<DETCHAN>_CP0_<FILENUM>.fits
        chop 1 = <FLIGHT>_FI_IFS_<AOR-ID>_<DETCHAN>_CP1_<FILENUM>.fits

    The prime header is also modified from the original, to conform to
    SOFIA standards (via fifi_ls.make_header).

    Parameters
    ----------
    filename : str
        File path to the raw FITS data file.
    write : bool, optional
        If True, write generated HDU Lists to file
    outdir : str, optional
        Name of the output path.  By default new files will be saved
        to the same directory as `filename`.

    Returns
    -------
    tuple of fits.HDUList or tuple of str
        Contains HDULists is write is False, otherwise paths to output
        files
    """
    if isinstance(outdir, str):
        if not os.path.isdir(outdir):
            log.error("Output directory %s does not exist" % outdir)
            return

    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return
    if isinstance(filename, str):
        # input file is string: standardize header
        new_header = make_header(hdul[0].header)
        if new_header is None:
            log.error('Problem updating header')
    else:
        # assume header is already standardized;
        # get filename from header
        filename = hdul[0].header['FILENAME']

    if not isinstance(outdir, str):
        outdir = os.path.dirname(filename)

    if len(hdul) < 2:
        log.error("HDUList missing extension 1 for file %s" % filename)
        return

    if hdul[1].header.get('XTENSION').strip().upper() != 'BINTABLE':
        log.error("Expected BINTABLE extension in extension 1 for file %s" %
                  filename)
        return

    extname = hdul[1].header.get('EXTNAME', 'UNKNOWN').strip().upper()
    if extname != 'FIFILS_RAWDATA':
        log.error("Can only split FIFILS_rawdata: "
                  "extension 1 EXTNAME = %s for %s" % (extname, filename))
        return

    colnames = hdul[1].data.columns.names
    colnames = [x.strip().upper() for x in colnames]
    if 'HEADER' not in colnames or 'DATA' not in colnames:
        log.error('Missing expected DATA and HEADER columns in BINTABLE %s' %
                  filename)
        return

    params = get_split_params(hdul)
    if not params['success']:
        log.error('Problem getting split parameters')
        return

    # attach a position offset table for OTF mode scans
    instmode = str(hdul[0].header.get('INSTMODE', 'UNKNOWN')).upper()
    nodbeam = str(hdul[0].header.get('NODBEAM', 'B')).upper().strip()
    if 'OTF' in instmode and nodbeam == 'A':
        try:
            hdul = _derive_positions(hdul, params)
        except KeyError:
            log.error("Missing required keywords for OTF position data")
            hdul = None
        if hdul is None:
            log.error('Problem reading OTF data data')
            return

    hdul = trim_data(hdul, params)
    if hdul is None:
        log.error('Problem trimming data')
        return

    hduls = separate_chops(hdul, params)
    if hduls is None:
        log.error('Problem separating chops')
        return

    if not write:
        return hduls
    else:
        result = []
        for hdul in hduls:
            result.append(write_hdul(hdul, outdir=outdir, overwrite=True))
        return result


def split_wrap_helper(_, kwargs, filename):
    return split_grating_and_chop(filename, **kwargs)


def wrap_split_grating_and_chop(files, outdir=None,
                                allow_errors=False, write=False,
                                jobs=None):
    """
    Wrapper for split_grating_and_chop over multiple files.

    Parameters
    ----------
    files : array_like of str
        list of filepaths to FIFI-LS FITS files
    outdir : str, optional
        name of the output path.  If None, will be saved to the same
        directory as the input file used to generate the split files.
    allow_errors : bool, optional
        If True, return all created files on error.  Otherwise, return None
    write : bool, optional
        If True, write the output to disk and return the filename instead
        of the HDU.
    jobs : int, optional
        Specifies the maximum number of concurrently running jobs.
        Values of 0 or 1 will result in serial processing.  A negative
        value sets jobs to `n_cpus + 1 + jobs` such that -1 would use
        all cpus, and -2 would use all but one cpu.

    Returns
    -------
    tuple of str
        output filenames written to disk
    """
    if isinstance(files, str):
        files = [files]
    if not hasattr(files, '__len__'):
        log.error("Invalid input files type (%s)" % repr(files))
        return

    kwargs = {'outdir': outdir, 'write': write}

    output = multitask(split_wrap_helper, files, None, kwargs,
                       jobs=jobs)

    failure = False
    result = []
    for x in output:
        if x is None:
            failure = True
        else:
            result.extend(x)
    if failure:
        if len(result) > 0:
            if not allow_errors:
                log.error("Errors were encountered but the following "
                          "files were created:\n%s" % '\n'.join(result))
                return

    return tuple(result)
