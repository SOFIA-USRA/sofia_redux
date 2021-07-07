# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities \
    import (goodfile, gethdul, hdinsert, write_hdul,
            multitask)

__all__ = ['get_chop_pair', 'subtract_extensions', 'subtract_chops',
           'wrap_subtract_chops']


def get_chop_pair(chop0_file):
    """
    Given the chop 0 filename, return the chop 0 and chop 1 HDU lists.

    Parameters
    ----------
    chop0_file : str
        File path to the chop 0 file

    Returns
    -------
    fits.HDUList, fits.HDUList
    """
    if not goodfile(chop0_file, verbose=True):
        return
    elif 'RP0' not in chop0_file:
        log.error("Chop0_file must contain 'RP0' in name")
        return

    c0_hdul = gethdul(chop0_file)
    if c0_hdul is None:
        log.error('Invalid RP0 file')
        return
    c0_hdul[0].header['FILENAME'] = os.path.basename(chop0_file)

    c_amp = c0_hdul[0].header.get('C_AMP', 0)
    if c_amp == 0:
        hduls = [c0_hdul]
        return hduls

    chop1_file = str.replace(chop0_file, 'RP0', 'RP1')
    if not goodfile(chop1_file, verbose=True):
        log.error('No RP1 file found')
        return

    c1_hdul = gethdul(chop1_file)
    if c1_hdul is None:
        log.error('Invalid RP1 file')
        return

    hduls = [c0_hdul, c1_hdul]
    hduls[1][0].header['FILENAME'] = os.path.basename(chop1_file)

    valid = False
    for idx, hdul in enumerate(hduls):
        ngrating = hdul[0].header.get('NGRATING', 1)
        for idx in range(ngrating):
            name = f'FLUX_G{idx}'
            stdname = f'STDDEV_G{idx}'
            if name not in hdul:
                log.error(f"Missing extension: {name}")
                break
            if stdname not in hdul:
                log.error(f"Missing extension: {stdname}")
                break
        else:
            continue
        break
    else:
        if len(hduls[0]) == len(hduls[1]):
            valid = True
        else:
            log.error("Differing number of inductosyn positions in chops")

    if not valid:
        for hdul in hduls:
            if isinstance(hdul, fits.HDUList):
                hdul.close()
        return

    return hduls


def subtract_extensions(hdul0, hdul1, add_only=False):
    """
    Subtract extensions in the correct order.

    Parameters
    ----------
    hdul0 : fits.HDUList
        HDU list containing chop 0 extensions
    hdul1 : fits.HDUList
        HDU list containing chop 1 extensions
    add_only : bool, optional
        If True, chop files will be added rather than subtracted.
        This is intended to be used for flat files only.

    Returns
    -------
    fits.HDUList
    """
    hdul = fits.HDUList(fits.PrimaryHDU(header=hdul0[0].header))
    primeheader = hdul[0].header
    hdinsert(primeheader, 'PRODTYPE', 'chop_subtracted')
    outfile = os.path.basename(
        primeheader.get('FILENAME', 'UNKNOWN').replace('RP0', 'CSB'))
    hdinsert(primeheader, 'FILENAME', outfile)
    primeheader['HISTORY'] = 'Chops subtracted'

    nodpos0 = primeheader.get('NODBEAM', 'UNKNOWN').strip().upper()
    nodstyle = primeheader.get('NODSTYLE', 'UNKNOWN').strip().upper()
    multiplier = 1
    if nodstyle in ['SYMMETRIC', 'NMC'] and nodpos0 != 'A' and not add_only:
        multiplier = -1

    ngrating = primeheader.get('NGRATING', 1)
    for idx in range(ngrating):
        name = f'FLUX_G{idx}'
        stdname = f'STDDEV_G{idx}'
        ext0 = hdul0[name]
        ext1 = hdul1[name]
        std0 = hdul0[stdname]
        std1 = hdul1[stdname]

        h0 = ext0.header
        h1 = ext1.header
        if h0['INDPOS'] != h1['INDPOS']:
            log.error(f"Inductosyn positions do not line up for "
                      f"grating extension {idx}")
            return None

        if add_only:
            data = ext0.data + ext1.data
        elif h0.get('CHOPNUM', 0) < 1:
            data = multiplier * (ext0.data - ext1.data)
        else:
            data = multiplier * (ext1.data - ext0.data)

        stddev = np.hypot(std0.data, std1.data)
        bglevels = [h0.get('BGLEVL_A', np.nan), h1.get('BGLEVL_A', np.nan)]
        bglevel = 0.0 if np.isnan(bglevels).all() else np.nanmean(bglevels)
        exthdr = h0
        hdinsert(exthdr, 'BGLEVL_A', bglevel)

        # add flux and stddev extensions to output file
        # note that OTF scanpos table is not expected or propagated
        hdu1 = fits.ImageHDU(data, header=exthdr, name=name)
        hdu2 = fits.ImageHDU(stddev, header=exthdr, name=stdname)
        hdul.append(hdu1)
        hdul.append(hdu2)

    return hdul


def subtract_chops(chop0_file, outdir=None, add_only=False, write=False):
    """
    Subtract chops of ramp-fitted data.

    One HDUL/file is created for a chop pair, containing n_scan
    binary table extensions, each containing DATA and STDDEV data
    cubes of shape (18, 5, 5).  The output filename is created
    from the input filename with the suffix 'RP0' replaced with
    'CSB'.

    The procedure is:

        1. Check chop amplitude: if zero, assume observation was taken
           in no-chop mode and return without subtracting chops
        2. Identify chop 1 file corresponding to input chop 0 file and
           read both from disk.
        3. For each extension, subtract OFF data from ON data.

             a. symmetric mode: For A nods, chop 1 is subtracted from
                chop 0.  For B nods, chop 0 is subtracted from chop 1.
                This should result in positive source flux in the
                resulting output, whether nod A or B, so that files can
                be simply added in the nod-combination algorithm.
             b. asymmetric mode: chop 0 is subtracted from chop 1,
                regardless of nod position.  Sky files are subtracted
                from sources files in the nod-combination algorithm.

        4. Propagate the error as the square root of the sum of the
           input variances.
        5. (optional) Create FITS file and write to disk.

    Parameters
    ----------
    chop0_file : str
        File path to the chop 0 file (*RP0*.fits).  The chop 1
        file is assumed to have the same root and directory
        location, with the designator RP1 in place of RP0.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    add_only : bool, optional
        If True, chop files will be added rather than subtracted.
        This is intended to be used for flat files only.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUL.

    Returns
    -------
    fits.HDUList or str
        Either the HDU (if write is False) or the filename of the output file
        (if write is True)
    """
    if isinstance(outdir, str):
        if not os.path.isdir(outdir):
            log.error("Output directory %s does not exist" % outdir)
            return

    # check for input HDULists
    if isinstance(chop0_file, fits.HDUList):
        hduls = [gethdul(chop0_file)]
    elif ((isinstance(chop0_file, tuple)
           or isinstance(chop0_file, list))
          and len(chop0_file) == 2):
        hduls = [gethdul(chop0_file[0]), gethdul(chop0_file[1])]
    elif ((isinstance(chop0_file, tuple)
           or isinstance(chop0_file, list))
          and len(chop0_file) == 1):
        hduls = [gethdul(chop0_file[0])]
    else:
        if not goodfile(chop0_file, verbose=True):
            return

        if not isinstance(outdir, str):
            outdir = os.path.dirname(chop0_file)

        hduls = get_chop_pair(chop0_file)
        if hduls is None:
            log.error('Problem retrieving chop pair')
            return

    primeheader = hduls[0][0].header.copy()
    if primeheader.get('C_AMP', 0) == 0:  # not an error
        log.info("No chop subtraction performed for NOCHOP mode "
                 "(chop amp = 0)")
        if write:
            # not really much point in writing - just return name
            return tuple((os.path.join(outdir, x[0].header['FILENAME'])
                          for x in hduls))
        else:
            return hduls

    hdul = subtract_extensions(hduls[0], hduls[1], add_only=add_only)
    if hdul is None:
        log.error('Problem subtracting extensions')

    if write:
        return write_hdul(hdul, outdir=outdir, overwrite=True)
    else:
        return hdul


def subtract_chops_wrap_helper(_, kwargs, filename):
    return subtract_chops(filename, **kwargs)


def wrap_subtract_chops(files, outdir=None, add_only=False, write=False,
                        allow_errors=False, jobs=None):
    """
    Wrapper for subtract_chops over multiple files.

    See `subtract_chops` for full description of reduction on a single file.

    Parameters
    ----------
    files : array_like of str
        paths to chop 0 *RP0* files
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    add_only : bool, optional
        If True, chop files will be added rather than subtracted.
        This is intended to be used for flat files only.
    write : bool, optional
        If True, write the output to disk and return the filename instead
        of the HDU.
    allow_errors : bool, optional
        If True, return all created files on error.  Otherwise, return None
        jobs : int, optional
        Specifies the maximum number of concurrently running jobs.
        Values of 0 or 1 will result in serial processing.  A negative
        value sets jobs to `n_cpus + 1 + jobs` such that -1 would use
        all cpus, and -2 would use all but one cpu.
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

    kwargs = {'outdir': outdir, 'add_only': add_only, 'write': write}

    output = multitask(subtract_chops_wrap_helper, files, None, kwargs,
                       jobs=jobs)

    failure = False
    result = []
    for x in output:
        if x is None:
            failure = True
        elif isinstance(x, list) and not isinstance(x, fits.HDUList):
            result.extend(x)
        else:
            result.append(x)
    if failure:
        if len(result) > 0:
            if not allow_errors:
                log.error("Errors were encountered but the following "
                          "files were created:\n%s" % '\n'.join(result))
                return

    return tuple(result)
