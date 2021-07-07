# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities \
    import (gethdul, goodfile, hdinsert, write_hdul, multitask)


__all__ = ['clear_spatial_cache', 'get_spatial_from_cache',
           'store_spatial_in_cache', 'offset_xy',
           'get_deltavec_coeffs', 'calculate_offsets',
           'spatial_calibrate', 'wrap_spatial_calibrate']

__spatial_cache = {}


def clear_spatial_cache():
    """
    Clear all data from the spatial cache.
    """
    global __spatial_cache
    __spatial_cache = {}


def get_spatial_from_cache(spatialfile, obsdate):
    """
    Retrieves spatial data from the spatial cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    spatialfile : str
        File path to the spatial file
    obsdate : int
        Observation date

    Returns
    -------
    arg1, arg2
        arg1 : str or dict
            If reading an offsets file, `arg1` will be the filename.
            If reading a coefficients file, `arg1` will be a dictionary
            of coefficients.
        arg2 : numpy.ndarray or dict
            If reading an offsets file, `arg2` will be an array of offsets.
            If reading a coefficients file, `arg2` will be a dictionary of
            secondary coefficients.
    """
    global __spatial_cache

    key = spatialfile, int(obsdate)

    if key not in __spatial_cache:
        return

    if not goodfile(spatialfile, verbose=True):
        try:
            del __spatial_cache[key]
        except KeyError:   # pragma: no cover
            pass
        return

    modtime = str(os.path.getmtime(spatialfile))
    if modtime not in __spatial_cache.get(key, {}):
        return

    log.debug("Retrieving spatial data from cache (%s, date %s)" % key)
    return __spatial_cache.get(key, {}).get(modtime)


def store_spatial_in_cache(spatialfile, obsdate, arg1, arg2):
    """
    Store spatial data in the spatial cache.

    Parameters
    ----------
    spatialfile : str
        File path to the spatial file
    obsdate : int
        Observation date
    arg1 : str or dict
        If reading an offsets file, `arg1` will be the filename.
        If reading a coefficients file, `arg1` will be a dictionary
        of coefficients.
    arg2 : numpy.ndarray or dict
        If reading an offsets file, `arg2` will be an array of offsets.
        If reading a coefficients file, `arg2` will be a dictionary of
        secondary coefficients.
    """
    global __spatial_cache
    key = spatialfile, int(obsdate)
    log.debug("Storing spatial data in cache (%s, date %s)" % key)
    __spatial_cache[key] = {}
    modtime = str(os.path.getmtime(spatialfile))
    __spatial_cache[key][modtime] = arg1, arg2


def offset_xy(date, blue=False):
    """
    Calculate X and Y offsets for each spaxel.

    The procedure is:

        1. Identify calibration file by date and channel
        2. Read offsets from file and reform into array
        3. Return array

    Parameters
    ----------
    date : array_like of int
        Date of observation as a [YYYY, M, D] vector (e.g. [2009,3,23]
    blue : bool, optional
        If True, BLUE channel is assumed, otherwise, RED

    Returns
    -------
    str, numpy.ndarray
        calibration file used, (25, 2) X and Y offsets of each spaxel
        (in mm)
    """
    fifi_datapath = os.path.join(os.path.dirname(fifi_ls.__file__), 'data')
    calfile_default = os.path.join(
        fifi_datapath, 'spatial_cal', 'poscal_default.txt')
    if not goodfile(calfile_default, verbose=True):
        return

    if not hasattr(date, '__len__') or len(date) != 3 or date[0] < 1e3:
        log.error("DATE must be a 3-element array [yyyy, m, d]")
        return

    obsdate = int(''.join([str(x).zfill(2) for x in date]))
    df = pandas.read_csv(
        calfile_default, comment='#', names=['date', 'chan', 'file'],
        delim_whitespace=True, dtype={'date': int})
    df = df.sort_values('date')
    df['chan'].apply(str.lower)
    channel = 'b' if blue else 'r'
    df = df[(df['chan'] == channel) & (df['date'] >= obsdate)]

    if len(df) == 0:
        log.error("No spatial calibration file found for "
                  "channel %s" % channel)
        return

    calfile = os.path.join(fifi_datapath, df['file'].values[0])

    offset_data = get_spatial_from_cache(calfile, obsdate)
    if offset_data is not None:
        return offset_data

    if not goodfile(calfile, verbose=True):
        return
    offsets = pandas.read_csv(
        calfile, names=['x', 'y'], dtype=float, delim_whitespace=True)
    offsets = np.array([offsets['x'], offsets['y']]).T

    store_spatial_in_cache(calfile, obsdate, calfile, offsets)

    return calfile, offsets


def get_deltavec_coeffs(header, obsdate, telsim2det=0.842):
    """
    Read the correct delta vector spatial coefficients.

    Parameters
    ----------
    header : fits.Header
    obsdate : array_like of int
        Date of observation as a [YYYY, M, D] vector (e.g. [2009,3,23]
    telsim2det : float, optional
        Conversion factor from telsim to detector units

    Returns
    -------
    dict, dict
    """
    prime_array = header.get('PRIMARAY')
    channel = header.get('CHANNEL')
    dichroic = header.get('DICHROIC')
    coeff_file = os.path.join(
        os.path.dirname(fifi_ls.__file__), 'data', 'spatial_cal',
        'FIFI_LS_DeltaVector_Coeffs.txt')

    longdate = int(''.join([str(x).zfill(2) for x in obsdate]))
    coefficients = get_spatial_from_cache(coeff_file, longdate)
    if coefficients is not None:
        return coefficients

    if not goodfile(coeff_file, verbose=True):
        return
    df = pandas.read_csv(
        coeff_file, comment='#', delim_whitespace=True,
        names=['dt', 'ch', 'dch', 'bx', 'ax', 'rx', 'by', 'ay', 'ry'])

    # The offsets from instrument boresight to telescope boresight
    try:
        rows = df[(df['dt'] >= longdate)
                  & (df['ch'] == prime_array[0].lower())
                  & (df['dch'] == dichroic)].sort_values('dt')
    except TypeError:
        rows = []
    c1 = {'ax': 0.0, 'bx': 0.0, 'rx': 0.0,
          'ay': 0.0, 'by': 0.0, 'ry': 0.0}
    c2 = c1.copy()
    c2['required'] = False

    if len(rows) == 0:
        log.error("No boresight offsets found for %s" % longdate)
        return None
    else:
        c1 = rows.iloc[0].to_dict()
        if (channel == 'RED' and prime_array == 'BLUE') or \
                (channel == 'BLUE' and prime_array == 'RED'):

            # The offsets between primary and secondary arrays
            rows2 = df[(df['dt'] >= longdate)
                       & (df['ch'] == channel[0].lower())
                       & (df['dch']
                          == dichroic)].sort_values('dt').reset_index()
            c2 = rows2.loc[0].to_dict()
            c2['required'] = True
        for k in c1.keys():
            if k[0] in ['a', 'b']:
                c1[k] *= telsim2det
                c2[k] *= telsim2det

    store_spatial_in_cache(coeff_file, longdate, c1, c2)

    return c1, c2


def calculate_offsets(hdul, obsdate=None, flipsign=None, rotate=False):
    """
    Calculate X and Y spatial offsets.

    Parameters
    ----------
    hdul : fits.HDUList
    obsdate : array_like of int, optional
    flipsign : bool, optional
    rotate : bool, optional
        If True, rotate by detector angle to set N up

    Returns
    -------
    fits.HDUList
        HDUList updated with 'XS' and 'YS' spatial offsets
    """

    result = fits.HDUList(fits.PrimaryHDU(header=hdul[0].header))

    header = result[0].header
    outname = os.path.basename(header.get('FILENAME', 'UNKNOWN'))
    outname = outname.replace('WAV', 'XYC')
    header['HISTORY'] = 'XY offsets added'
    hdinsert(header, 'PRODTYPE', 'spatial_calibrated')
    hdinsert(header, 'FILENAME', outname)
    channel = header.get('CHANNEL')
    angle = np.radians(header.get('DET_ANGL', 0))

    if obsdate is None:
        obsdate = header.get('DATE-OBS')
        if not isinstance(obsdate, str) or '-' not in obsdate:
            log.error("DATE-OBS not found in file header")
            return
        try:
            obsdate = [int(x) for x in obsdate[:10].split('-')]
        except (ValueError, IndexError, TypeError):
            log.error("Bad DATE-OBS found in file header")
            return

    caloffsets = offset_xy(obsdate, blue=channel == 'BLUE')
    if caloffsets is None:
        return
    calfile, xy = caloffsets
    if len(xy) != 25:
        log.error("Invalid number of XY offsets")
        return

    hdinsert(header, 'SPATFILE', os.path.basename(calfile),
             comment='Spatial calibration file')

    telsim = 'TELSIM' in [header.get('OBJ_NAME').strip().upper(),
                          header.get('OBJECT').strip().upper()]
    rotation = 0.0 if (telsim or rotate) else angle
    hdinsert(header, 'SKY_ANGL', np.rad2deg(rotation),
             comment='Sky angle after calibration (deg)')

    # plate scale in arcsec/mm
    plate_scale = header.get('PLATSCAL', 0)
    # Map offset in arcsec
    dlam_map = header.get('DLAM_MAP', 0)
    dbet_map = header.get('DBET_MAP', 0)
    if telsim:
        # x-y stage, mm
        dx = header.get('DLAM_MAP')
        dy = header.get('DBET_MAP')
        if None in [dx, dy]:
            dx = header.get('OBSDEC', 0) / 10  # OBSRA
            dy = header.get('OBSRA', 0) / 10  # OBSDEC
        xs = dx - xy[:, 1]
        ys = dy - xy[:, 0]
        delta_coeffs = None, None
    else:
        # Real target, not TELSIM
        # flip sign on DBET and DLAM if needed (mostly just older data)
        if flipsign is None:
            longdate = int(''.join([str(x).zfill(2) for x in obsdate]))
            flipsign = longdate < 20150501  # May, 2015
        log.debug("DLam/DBet sign convention: %s" % ('-' if flipsign else '+'))
        if flipsign:
            dbet_map *= -1
            dlam_map *= -1
        delta_coeffs = get_deltavec_coeffs(header, obsdate)
        if delta_coeffs is None:
            log.error('Problem in deltavec coefficients')
            return
        xs, ys = None, None

    ngrating = hdul[0].header.get('NGRATING', 1)
    for idx in range(ngrating):
        name = f'FLUX_G{idx}'

        # if scanpos present, then use array of dlam/dbet instead
        # of header keyword
        pname = name.replace('FLUX', 'SCANPOS')
        if pname in hdul:
            posdata = hdul[pname].data
            # broadcast to nramp x nspaxel
            nramp = len(posdata)
            dlam_map = np.broadcast_to(posdata['DLAM_MAP'], (25, nramp)).T
            dbet_map = np.broadcast_to(posdata['DBET_MAP'], (25, nramp)).T
            if flipsign:
                dlam_map = dlam_map * -1.0
                dbet_map = dbet_map * -1.0
        else:
            nramp = 1

        exthdr = hdul[name].header
        if not telsim:
            c1, c2 = delta_coeffs
            indpos_p = exthdr['INDPOS_P']
            # Calculate offset from instrument boresight to telescope boresight
            dx = c1['bx'] + c1['ax'] * indpos_p - c1['rx']
            dy = -c1['by'] - c1['ay'] * indpos_p - c1['ry']
            if c2['required']:
                # Calculate spatial offset between primary and secondary array
                indpos = exthdr['INDPOS']
                dx += (c2['bx'] - c1['bx']) \
                    + (c2['ax'] * indpos - c1['ax'] * indpos_p)
                dy -= (c2['by'] - c1['by']) \
                    + (c2['ay'] * indpos - c1['ay'] * indpos_p)

            # Calculate offsets with detangl
            xs = (plate_scale * (dx + xy[:, 0])
                  + np.cos(angle) * dlam_map - np.sin(angle) * dbet_map)
            ys = -(plate_scale * (dy - xy[:, 1])
                   + np.sin(angle) * dlam_map + np.cos(angle) * dbet_map)

            # Rotate by 180 for changed convention in KOSMA translator
            if not flipsign:
                xsr = xs * np.cos(-np.pi) - ys * np.sin(-np.pi)
                ysr = xs * np.sin(-np.pi) + ys * np.cos(-np.pi)
                xs = xsr
                ys = ysr

            # Rotate coords by detangl to set N up if desired
            if rotate:
                xsr = xs * np.cos(angle) - ys * np.sin(angle)
                ysr = xs * np.sin(angle) + ys * np.cos(angle)
                xs = xsr
                ys = ysr

        result.append(hdul[name].copy())
        result.append(hdul[name.replace('FLUX', 'STDDEV')].copy())
        result.append(hdul[name.replace('FLUX', 'LAMBDA')].copy())

        # reshape xs and ys data if necessary
        exthdr['BUNIT'] = 'arcsec'
        if xs.ndim > 1:
            result.append(fits.ImageHDU(xs.reshape((nramp, 1, 25)),
                                        header=exthdr,
                                        name=name.replace('FLUX', 'XS')))
            result.append(fits.ImageHDU(ys.reshape((nramp, 1, 25)),
                                        header=exthdr,
                                        name=name.replace('FLUX', 'YS')))
        else:
            result.append(fits.ImageHDU(xs, header=exthdr,
                                        name=name.replace('FLUX', 'XS')))
            result.append(fits.ImageHDU(ys, header=exthdr,
                                        name=name.replace('FLUX', 'YS')))

    return result


def spatial_calibrate(filename, obsdate=None, flipsign=None,
                      rotate=False, outdir=None, write=False):
    """
    Apply spatial calibration (x and y offsets).

    Each data extension will be updated with xs and ys fields, containing
    spatial coordinates for each pixel in the data array.  The input
    file should have been created by fifi_ls.lambda_calibrate.
    This procedure handles only ERF coordinates;  SIRF mapping is neither
    detected nor handled properly.

    The FIFI-LS optics are not perfectly aligned so the pixels do not
    fall on an even grid.  This routine calculates the x, y locations
    for each pixel and shifts them according to dither positions.  The
    procedure is, for each grating scan extension:

        1. Calculate spatial positions in mm (from fifi_ls.offset_xy)
        2. Convert positions to arcseconds, using PLATSCAL, dither
           offsets (DLAM_MAP and DBET_MAP, possibly with a sign-flip
           on each), and offsets of the secondary array from the
           primary (from data/secondary_offset.txt).
        3. Create FITS file and (optionally) write results to disk.

    For OTF mode data, each input ramp sample has a separate
    DLAM_MAP and DBET_MAP value, as recorded in the SCANPOS table in the
    input FITS file.  These values are used to calculate the X and Y
    offsets for each sample.  The output XS and YS image extensions
    have dimension 25 x 1 x n_ramp.  The SCANPOS table is not propagated
    to the output.

    For all other data, the X and Y offsets are calculated from the
    DLAM_MAP and DBET_MAP keywords in the header, and are
    attached to the output as a 25-element array in the XS and YS
    image extensions.

    Parameters
    ----------
    filename : str
        File to be spatial-calibrated
    obsdate : array_like of int, optional
        Date of observation.  Intended for files that do not have
        the DATE-OBS keyword (and value) in the FITS primary header
        (early files do not).  Format is [YYYY,MM,DD].
    flipsign : bool, optional
        If True, DLAM_MAP and DBET_MAP will be multiplied by -1.  If
        False, DLAM_MAP and DBET_MAP will be used as is.  If None
        (default), the observation date will be used to determine
        what the sign convention should be.
    rotate : bool, optional
        If True, rotate by the detector angle to set N up.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUList. The output filename is
        created from the input filename, with the suffix 'WAV' replaced
        with 'XYC'.

    Returns
    -------
    fits.HDUList or str
        Either the HDUList (if write is False) or the filename of
        the output file (if write is True).
    """
    if isinstance(outdir, str):
        if not os.path.isdir(outdir):
            log.error("Output directory %s does not exist" % outdir)
            return

    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return
    if not isinstance(filename, str):
        filename = hdul[0].header['FILENAME']
    if not isinstance(outdir, str):
        outdir = os.path.dirname(filename)

    result = calculate_offsets(hdul, obsdate=obsdate, flipsign=flipsign,
                               rotate=rotate)
    if result is None:
        log.error('Offset calculation failed')
        return

    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def spatial_calibrate_wrap_helper(_, kwargs, filename):
    return spatial_calibrate(filename, **kwargs)


def wrap_spatial_calibrate(files, outdir=None, obsdate=None, flipsign=None,
                           rotate=True, allow_errors=False, write=False,
                           jobs=None):
    """
    Wrapper for spatial_calibrate over multiple files.

    See `spatial_calibrate` for full description of reduction
    on a single file.

    Parameters
    ----------
    files : array_like of str
        paths to files to be spatially calibrated
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    obsdate : array_like of int, optional
        Date of observation.  Intended for files that do not have
        the DATE-OBS keyword (and value) in the FITS primary header
        (early files do not).  Format is [YYYY,MM,DD].
    flipsign : bool, optional
        If True, DLAM_MAP and DBET_MAP will be multiplied by -1.  If
        False, DLAM_MAP and DBET_MAP will be used as is.  If None
        (default), the observation date will be used to determine
        what the sign convention should be.
    rotate : bool, optional
        If True, rotate by the detector angle to set N up.
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

    clear_spatial_cache()

    kwargs = {'outdir': outdir, 'obsdate': obsdate, 'write': write,
              'flipsign': flipsign, 'rotate': rotate}

    output = multitask(spatial_calibrate_wrap_helper, files, None, kwargs,
                       jobs=jobs)

    failure = False
    result = []
    for x in output:
        if x is None:
            failure = True
        else:
            result.append(x)
    if failure:
        if len(result) > 0:
            if not allow_errors:
                log.error("Errors were encountered but the following "
                          "files were created:\n%s" % '\n'.join(result))
                return

    clear_spatial_cache()

    return tuple(result)
