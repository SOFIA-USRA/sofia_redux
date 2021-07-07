# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import astropy.constants as const
from astropy.io import fits
from astropy import log
from astropy import units
import bottleneck as bn
import numpy as np
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities \
    import (gethdul, write_hdul, hdinsert, goodfile,
            multitask)

__all__ = ['clear_wavecal_cache', 'get_wavecal_from_cache',
           'store_wavecal_in_cache', 'read_wavecal', 'wave',
           'lambda_calibrate', 'wrap_lambda_calibrate']

__wavecal_cache = {}


def clear_wavecal_cache():
    """
    Clear all data from the wavecal cache.
    """
    global __wavecal_cache
    __wavecal_cache = {}


def get_wavecal_from_cache(wavecalfile):
    """
    Retrieves wavelength calibration data from the wavecal cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    wavecalfile : str
        File path to the wavecal file

    Returns
    -------
    wavecal : pandas.DataFrame
        Wavelength calibration table
    """
    global __wavecal_cache

    if wavecalfile not in __wavecal_cache:
        return

    if not goodfile(wavecalfile):
        try:
            del __wavecal_cache[wavecalfile]
        except KeyError:   # pragma: no cover
            pass
        return

    modtime = str(os.path.getmtime(wavecalfile))
    if modtime not in __wavecal_cache.get(wavecalfile, {}):
        return

    log.debug("Retrieving data from wavecal file (%s) in cache" % wavecalfile)
    return __wavecal_cache.get(wavecalfile, {}).get(modtime)


def store_wavecal_in_cache(wavecalfile, wavecal):
    """
    Store wavecal data in the wavecal cache.

    Parameters
    ----------
    wavecalfile : str
        File path to the wavecal file
    wavecal : pandas.DataFrame
        Wave calibration table
    """
    global __wavecal_cache
    log.debug("Storing wavecal data (%s) in cache" % wavecalfile)
    __wavecal_cache[wavecalfile] = {}
    modtime = str(os.path.getmtime(wavecalfile))
    __wavecal_cache[wavecalfile][modtime] = wavecal


def column_level1_mapper(name):
    s = name.lower()
    if 'red' in s:
        if '105' in s:
            return 'r_105'
        elif '130' in s:
            return 'r_130'
    elif 'blue' in s:
        if '1st' in s:
            return 'b1'
        elif '2nd' in s:
            return 'b2'

    raise ValueError("Bad column name in dataframe: %s" % name)


def read_wavecal(calfile=None):
    """
    Read and return the data from CalibrationResults.csv.

    Parameters
    ----------
    calfile : str, optional
       Path to the calibration results CSV file.  If not specified,
       it will be read from fifi_ls/data/wave_cal/CalibrationResults.csv.

    Returns
    -------
    pandas.DataFrame
    """
    if calfile is None:
        calfile = os.path.join(
            os.path.dirname(fifi_ls.__file__),
            'data', 'wave_cal', 'CalibrationResults.csv')

    wavecal = get_wavecal_from_cache(calfile)
    if wavecal is not None:
        return wavecal

    if not goodfile(calfile, verbose=True):
        raise ValueError(
            "Cannot read wavelength calibration file: %s" % calfile)
    try:
        df = pandas.read_csv(calfile, header=[0, 1], index_col=[0, 1])
        col0 = list(df.columns.get_level_values(0))
        newcol0 = [np.nan if x.startswith('Unnamed') else float(x)
                   for x in col0]
        newcol0 = bn.push(newcol0).astype(int)

        df.rename(level=0, inplace=True, columns=dict(zip(col0, newcol0)))
        df.rename(level=1, inplace=True, columns=column_level1_mapper)
        df.rename(lambda x: 'isoff' if not isinstance(x, str) else x.lower(),
                  inplace=True, level=0)
        df.rename(lambda x: int(x) if not np.isnan(x) else -1,
                  inplace=True, level=1)
    except Exception as err:
        raise ValueError("Cannot parse %s to dataframe: %s" %
                         (calfile, str(err)))
    df.calfile = calfile

    log.debug("Loaded wavelength calibration file: %s" % calfile)

    store_wavecal_in_cache(calfile, df)

    return df


def wave(ind, date, dichroic, blue=None, wavecal=None):
    """
    Calculate wavelengths for each spectral pixel.

    Requires CalibrationResults.csv file, in fifi_ls/data/wave_cal/,
    as provided by FIFI-LS team.

    The procedure is:

        1. Read wavelength calibration file.  This file is stored in CSV
           format, as generated from an Excel spreadsheet provided by the
           FIFI-LS instrument team.  The FIFI-LS team also provides the
           wavelength equation.
        2. Loop over spaxels, calculating the wavelength and spectral
           width for each spexel, from the parameters specified in the
           calibration file.
        3. Return an array of wavelengths.

    Parameters
    ----------
    ind : int
        Inductosyn position
    date : array_like of int
        [Y, M, D] vector
    dichroic : int
        Dichroic value (105 or 130)
    blue : str, optional
        Either 'B1' to indicate first BLUE order of 'B2' to indicate
        second BLUE order.  If not provided, data is assumed to be
        RED channel
    wavecal : pandas.DataFrame, optional
        DataFrame containing wave calibration data.  May be supplied in
        order to remove the overhead of reading CalibrationResults.csv
        every iteration of this function.

    Returns
    -------
    dict
        Contains:
            - 'wavelength' (numpy.ndarray): a 2-D array (16, 25) containing
              the wavelength of each pixel for each module in microns.
            - 'width' (numpy.ndarray): a 2-D array (16, 25) containing the
              spectral width of each pixel for each spatial module in microns
            - 'calfile' (str): the name of the wavelength calibration file used

    """
    if blue is None:
        channel = 'R'
    elif str(blue).upper() in ['B1', 'B2']:
        channel = str(blue).upper()
    else:
        log.error("BLUE must be B1 or B2")
        return
    try:
        ind = int(ind)
    except (TypeError, ValueError):
        log.error("Invalid IND")
        return
    if ind < 0:
        log.error("IND must be positive")
        return
    if len(date) != 3:
        log.error("DATE must be an array of length 3")
        return

    if not isinstance(wavecal, pandas.DataFrame) or len(wavecal) == 0:
        wavecal = read_wavecal()

    # Get date in YYMM format
    obsdate = int(''.join([str(x).zfill(4)[-2:] for x in date[:2]]))
    wavecal_dates = np.array(wavecal.columns.levels[0])
    wavecal_dates = wavecal_dates[wavecal_dates <= obsdate]
    if len(wavecal_dates) == 0:
        log.error("No calibration data found for date %s" % obsdate)
        return
    last_date = wavecal_dates.max()
    if 'b' in channel.lower():
        # no dichroic difference for blue
        config = ('%s' % channel).lower()
    else:
        config = ('%s_%s' % (channel, dichroic)).lower()
    column = last_date, config
    if column not in wavecal.columns:
        log.error('%s column not in wavecal data' % repr(column))
        return

    cal = wavecal[column]

    # axis 0
    pix = np.arange(1, 17)
    sign = 2 * (((pix - cal['qoff', -1]) > 0) - 0.5)
    delta = ((pix - 8.5) * cal['ps', -1])
    delta += sign * cal['qs', -1] * (pix - cal['qoff', -1]) ** 2

    # axis 1
    isoff = cal['isoff'].values
    modules = cal['isoff'].index.values - 1
    slitpos = 25 - (6 * (modules // 5)) + (modules % 5)
    g = cal['g0', -1]
    g *= np.cos(np.arctan((slitpos - cal['np', -1]) / cal['a', -1]))
    phi = 2 * np.pi * cal['isf', -1] * (ind + isoff) / (2 ** 24)

    # cross terms for wavelength
    w = np.sin(np.add.outer(delta + cal['gamma', -1], phi))
    w += np.sin(phi - cal['gamma', -1])[None]
    w *= 1000 * g[None]

    # cross terms for pixel width
    p = np.cos(np.add.outer(delta + cal['gamma', -1], phi))
    p *= (cal['ps', -1] + 2 * sign
          * cal['qs', -1] * (pix - cal['qoff', -1]))[..., None]
    p *= 1000 * g[None]

    if channel.lower() == 'b2':
        w /= 2
        p /= 2

    wavefile = wavecal.calfile if hasattr(wavecal, 'calfile') else None
    return {'wavelength': w, 'width': p, 'wavefile': wavefile}


def lambda_calibrate(filename, obsdate=None, outdir=None, write=False,
                     wavecal=None):
    """
    Apply spectral calibration.

    The procedure is:

        1. Read input file.
        2. Call fifi_ls.wave to calculate wavelength values and spectral
           widths in microns for each spexel.
        3. Create FITS file and (optionally) write results to disk.

    The generated FITS file contains n_scan * 3 image extensions:
    DATA, STDDEV, and LAMBDA for each scan, named with a '_G{i}' suffix
    for each grating scan index i.  The LAMBDA arrays are always 25 x 16.
    The DATA and STDDEV arrays are 25 x 16 x n_ramp for OTF mode A nods,
    and 25 x 16 otherwise. If a SCANPOS_G{i} table is present in the input,
    it will also be attached unmodified to the output.

    Parameters
    ----------
    filename : str
        File to be wavelength-calibrated.  Should have been generated
        by fifi_ls.combine_nods.
    obsdate : array_like of int, optional
        Date of observation.  Intended for files that do not have
        the DATE-OBS keyword (and value) in the FITS primary header
        (early files do not).  Format is [YYYY,MM,DD].
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUList.  The output directory
        is the same as the directory of the input file.
    wavecal : pandas.DataFrame, optional
        DataFrame containing wave calibration data.  May be supplied in
        order to remove the overhead of reading CalibrationResults.csv
        every iteration of this function.

    Returns
    -------
    fits.HDUList or str
        Either the HDUList (if write is False) or the filename of the
        output file (if write is True).
    """
    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return
    if not isinstance(filename, str):
        filename = hdul[0].header['FILENAME']

    if isinstance(outdir, str):
        if not os.path.isdir(outdir):
            log.error("Output directory %s does not exist" % outdir)
            return
    if not isinstance(outdir, str):
        outdir = os.path.dirname(filename)

    if not isinstance(wavecal, pandas.DataFrame) or len(wavecal) == 0:
        wavecal = read_wavecal()
        if wavecal is None:
            log.warning("Unable to read wave calibration data")
            return

    dichroic = hdul[0].header.get('DICHROIC')
    channel = hdul[0].header.get('CHANNEL')
    b_order = int(hdul[0].header.get('G_ORD_B', -1))
    if channel == 'BLUE':
        if b_order in [1, 2]:
            blue = 'B%i' % b_order
        else:
            log.error("Invalid Blue grating order in file %s" %
                      filename)
            return
    else:
        blue = None

    if obsdate is None:
        obsdate = hdul[0].header.get('DATE-OBS')
        try:
            obsdate = [int(x) for x in obsdate[:10].split('-')]
        except (ValueError, TypeError, IndexError):
            log.error('Invalid DATE-OBS in file %s' % filename)
            return

    outname = os.path.basename(filename)
    for replace in ['NCM', 'CSB', 'RP0', 'RP1']:
        outname = outname.replace(replace, 'WAV')

    result = fits.HDUList(fits.PrimaryHDU(header=hdul[0].header))
    wavefile = None
    speed_of_light = const.c.to(units.um / units.s).value

    ngrating = hdul[0].header.get('NGRATING', 1)
    for idx in range(ngrating):
        name = f'FLUX_G{idx}'
        stdname = f'STDDEV_G{idx}'

        exthdr = hdul[name].header
        data = hdul[name].data
        stddev = hdul[stdname].data
        ind = exthdr.get('INDPOS')

        calibration = wave(ind, obsdate, dichroic, blue=blue, wavecal=wavecal)
        if calibration is None:
            log.error("Wavelength calibration failed")
            return

        if calibration['wavefile'] is not None:
            wavefile = calibration['wavefile']

        wavelength = calibration['wavelength']
        pixelwidth = calibration['width']

        # transform the fluxes into flux densities by dividing the
        # flux by the frequency interval
        dnu = speed_of_light * pixelwidth / (wavelength ** 2)
        data /= dnu
        stddev /= dnu

        log.debug("Pixel width (dlambda) at 2,2: %s" %
                  ', '.join([str(x) for x in pixelwidth[:, 12]]))
        log.debug("Pixel width (dnu) at 2,2: %s" %
                  ', '.join([str(x) for x in dnu[:, 12]]))

        exthdr['BUNIT'] = ('adu/(Hz s)', 'Data units')
        result.append(fits.ImageHDU(data, header=exthdr,
                                    name=name))
        result.append(fits.ImageHDU(stddev, header=exthdr,
                                    name=stdname))
        exthdr['BUNIT'] = 'um'
        result.append(fits.ImageHDU(wavelength, header=exthdr,
                                    name=name.replace('FLUX', 'LAMBDA')))
        # propagate scan positions forward if present
        pname = f'SCANPOS_G{idx}'
        if pname in hdul:
            result.append(hdul[pname].copy())

    # update header
    result[0].header['HISTORY'] = 'Wavelength calibrated'
    result[0].header['PRODTYPE'] = 'wavelength_calibrated'
    result[0].header['FILENAME'] = outname
    if wavefile is not None:
        hdinsert(result[0].header, 'WAVEFILE', os.path.basename(wavefile),
                 comment='Wavelength calibration file')

    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def lambda_calibrate_wrap_helper(_, kwargs, filename):
    return lambda_calibrate(filename, **kwargs)


def wrap_lambda_calibrate(files, outdir=None, obsdate=None,
                          allow_errors=False, write=False, jobs=None):
    """
    Wrapper for lambda_calibrate over multiple files.

    See `lambda_calibrate` for full description of reduction
    on a single file.

    Parameters
    ----------
    files : array_like of str
        paths to files to be wavelength calibrated
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    obsdate : array_like of int, optional
        Date of observation.  Intended for files that do not have
        the DATE-OBS keyword (and value) in the FITS primary header
        (early files do not).  Format is [YYYY,MM,DD].
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

    clear_wavecal_cache()

    kwargs = {'outdir': outdir, 'obsdate': obsdate, 'write': write}

    output = multitask(lambda_calibrate_wrap_helper, files, None, kwargs,
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

    clear_wavecal_cache()

    return tuple(result)
