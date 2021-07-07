# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numba as nb
import numpy as np

from sofia_redux.instruments.fifi_ls.get_atran \
    import get_atran, clear_atran_cache
from sofia_redux.instruments.fifi_ls.get_resolution \
    import get_resolution, clear_resolution_cache
from sofia_redux.toolkit.utilities \
    import (gethdul, write_hdul, hdinsert, multitask)

__all__ = ['apply_atran', 'telluric_correct', 'wrap_telluric_correct']


@nb.njit(fastmath={'nsz', 'ninf'}, cache=True, nogil=True)
def apply_atran_correction(wave, data, var, atran, cutoff):  # pragma: no cover
    """
    Apply an atmospheric transmission correction to the flux data.

    Parameters
    ----------
    wave : array of float
        nwave elements
    data : array of float
        ndata x nwave
    var : array of float
        ndata x nwave
    atran : array of float
        ATRAN wavelengths and transmission values. 2 x natran
    cutoff : float
        Below this transmission value, set the corrected data to NaN.

    Returns
    -------
    tel_corr, var_corr, atran_store : array, array, array
        The corrected flux, corrected variance, and correction
        values applied.
    """

    aw = atran[0]
    at = atran[1]
    na = atran.shape[1]
    shape = data.shape
    ndata = shape[0]
    nwave = shape[1]

    tel_corr = np.empty(shape, dtype=nb.float64)
    var_corr = np.empty(shape, dtype=nb.float64)
    atran_store = np.empty(shape, dtype=nb.float64)

    for n in range(ndata):
        for i in range(25):
            x = wave[:, i]
            y = data[n, :, i]
            v = var[n, :, i]
            buffer = 0.1 * (np.nanmax(x) - np.nanmin(x))
            lower_lim = x[0] - buffer
            upper_lim = x[-1] + buffer
            a0 = 0
            a1 = na
            found = False
            for ai in range(na):
                w = aw[ai]
                if found and w > upper_lim:
                    a1 = ai
                    break
                if not found and w >= lower_lim:
                    a0 = ai
                    found = True

            itrans = np.interp(x, aw[a0:a1], at[a0:a1])
            for k in range(nwave):
                transmission = itrans[k]

                atran_store[n, k, i] = transmission
                if transmission >= cutoff:
                    tel_corr[n, k, i] = y[k] / transmission
                    var_corr[n, k, i] = v[k] / transmission / transmission
                else:
                    tel_corr[n, k, i] = np.nan
                    var_corr[n, k, i] = np.nan

    return tel_corr, var_corr, atran_store


def apply_atran(hdul, atran, cutoff=0.6, skip_corr=False, unsmoothed=None):
    """
    Apply transmission data to data in an HDUList.

    Parameters
    ----------
    hdul : fits.HDUList
    atran : numpy.ndarray
        (2, nwave) where [0, :] = wavelength and [1, :] = transmission
    cutoff : float, optional
        Used as the fractional transmission below which data will
        be set to NaN. Set to 0 to keep all data.
    skip_corr : bool, optional
        If set, telluric correction will not be applied, but ATRAN
        spectra will still be attached to the output file.
    unsmoothed : numpy.ndarray, optional
        Unsmoothed transmission to attach to output file.
        (2, nwave) where [0, :] = wavelength and [1, :] = transmission

    Returns
    -------
    fits.HDUList
    """
    wave = np.asarray(hdul['LAMBDA'].data, dtype=float)
    var = np.asarray(hdul['STDDEV'].data, dtype=float) ** 2
    data = np.asarray(hdul['FLUX'].data, dtype=float)

    if data.ndim < 3:
        data = data.reshape((1, *data.shape))
        var = var.reshape((1, *var.shape))
        do_reshape = True
    else:
        do_reshape = False

    tel_corr, var_corr, atran_store = apply_atran_correction(
        wave, data, var, atran, cutoff)

    primehead = hdul[0].header.copy()
    outname = os.path.basename(
        primehead.get('FILENAME', 'UNKNOWN').replace('SCM', 'TEL'))
    hdinsert(primehead, 'FILENAME', outname)
    hdinsert(primehead, 'PRODTYPE', 'telluric_corrected')

    result = fits.HDUList(fits.PrimaryHDU(header=primehead))
    exthdr = hdul['FLUX'].header
    if skip_corr:
        result[0].header['HISTORY'] = 'Telluric spectrum attached'
        result.append(hdul['FLUX'].copy())
        result.append(hdul['STDDEV'].copy())
    else:
        result[0].header['HISTORY'] = 'Telluric corrected'
        if do_reshape:
            # standard: 2D data, take first plane
            result.append(fits.ImageHDU(tel_corr[0], header=exthdr,
                                        name='FLUX'))
            result.append(fits.ImageHDU(np.sqrt(var_corr[0]), header=exthdr,
                                        name='STDDEV'))
        else:
            # OTF: 3D data, use all planes
            result.append(fits.ImageHDU(tel_corr, header=exthdr, name='FLUX'))
            result.append(fits.ImageHDU(np.sqrt(var_corr), header=exthdr,
                                        name='STDDEV'))
        result.append(fits.ImageHDU(hdul['FLUX'].data.copy(), header=exthdr,
                                    name='UNCORRECTED_FLUX'))
        result.append(fits.ImageHDU(hdul['STDDEV'].data.copy(), header=exthdr,
                                    name='UNCORRECTED_STDDEV'))
    result.append(hdul['LAMBDA'].copy())
    result.append(hdul['XS'].copy())
    result.append(hdul['YS'].copy())
    exthdr['BUNIT'] = ''
    result.append(fits.ImageHDU(atran_store[0], header=exthdr,
                                name='ATRAN'))
    if unsmoothed is not None:
        # trim unsmoothed data for unused wavelengths, by channel
        channel = primehead.get('CHANNEL', 'UNKNOWN')
        if channel == 'BLUE':
            keep = (unsmoothed[0] < 130.)
            unsmoothed = unsmoothed[:, keep]
        elif channel == 'RED':
            keep = (unsmoothed[0] > 90.)
            unsmoothed = unsmoothed[:, keep]
        result.append(fits.ImageHDU(unsmoothed, header=exthdr,
                                    name='UNSMOOTHED_ATRAN'))

    return result


def telluric_correct(filename, atran_dir=None, cutoff=0.6, use_wv=False,
                     skip_corr=False, write=False, outdir=None):
    """
    Correct spectra for atmospheric absorption features.

    The procedure is:

        1. Identify ATRAN file to use.  Smooth it to the spectral
           resolution of the input file.
        2. Interpolate the atmospheric transmission data onto the wavelength
           value of each spexel. Divide the data at each point by the
           transmission value.
        3. Create FITS file and (optionally) write results to disk.

    The output FITS file contains FLUX, STDDEV, LAMBDA, XS, and YS
    arrays in the same dimensions as the input. Additionally,
    UNCORRECTED_FLUX and UNCORRECTED_STDDEV image extensions are appended,
    containing a copy of the input FLUX and STDDEV arrays. An interpolated
    ATRAN extension is also appended, matching the dimensions of LAMBDA.
    The full unsmoothed transmission data is also appended as an image
    extension, in a n_atran x 2 array, for reference.

    Parameters
    ----------
    filename : str
        FITS file to be telluric corrected.  Should have been
        generated by fifi_ls.combine_grating_scans.
    atran_dir : str, optional
        Path to a directory containing ATRAN reference FITS files.
        If not provided, the default set of files packaged with the
        pipeline will be used.
    cutoff : float, optional
        Used as the fractional transmission below which data will
        be set to NaN. Set to 0 to keep all data.
    use_wv : bool, optional
        If set, water vapor values from the header will be used
        to select the correct ATRAN file.
    skip_corr : bool, optional
        If set, telluric correction will not be applied, but ATRAN
        spectra will still be attached to the output file.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUL.  The output filename is
        created from the input filename, with the suffix 'SCM'
        replaced with 'TEL'.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.

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

    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return
    if not isinstance(filename, str):
        filename = hdul[0].header['FILENAME']
    if not isinstance(outdir, str):
        outdir = os.path.dirname(filename)

    log.info('')
    log.info(filename)

    # Get spectral resolution (RESFILE keyword is added to primehead)
    resolution = get_resolution(hdul[0].header)
    if resolution is None:
        log.error("Unable to determine spectral resolution")
        return

    # Get ATRAN data from unput file or default on disk, smoothed to
    # current resolution
    atran_data = get_atran(hdul[0].header, resolution=resolution,
                           atran_dir=atran_dir, use_wv=use_wv,
                           get_unsmoothed=True)
    if atran_data is None or atran_data[0] is None:
        log.error("Unable to get ATRAN data")
        return
    atran, unsmoothed = atran_data

    result = apply_atran(hdul, atran, cutoff=cutoff, skip_corr=skip_corr,
                         unsmoothed=unsmoothed)
    if result is None:
        log.error('Unable to apply ATRAN correction')
        return

    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def telluric_correct_wrap_helper(_, kwargs, filename):
    return telluric_correct(filename, **kwargs)


def wrap_telluric_correct(files, outdir=None, allow_errors=False,
                          atran_dir=None, cutoff=0.6, use_wv=False,
                          skip_corr=False, write=False,
                          jobs=None):
    """
    Wrapper for telluric_correct over multiple files.

    See `telluric_correct` for full description of reduction
    on a single file.

    Parameters
    ----------
    files : array_like of str
        paths to files to be telluric corrected
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    allow_errors : bool, optional
        If True, return all created files on error.  Otherwise, return None
    atran_dir : str, optional
        Path to a directory containing ATRAN reference FITS files.
        If not provided, the default set of files packaged with the
        pipeline will be used.
    cutoff : float, optional
        Used as the fractional transmission below which data will
        be set to NaN. Set to 0 to keep all data.
    use_wv : bool, optional
        If set, water vapor values from the header will be used
        to select the correct ATRAN file.
    skip_corr : bool, optional
        If set, telluric correction will not be applied, but ATRAN
        spectra will still be attached to the output file.
    write : bool, optional
        If True, write output files to disk.
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

    clear_resolution_cache()
    clear_atran_cache()

    kwargs = {'outdir': outdir, 'write': write, 'atran_dir': atran_dir,
              'cutoff': cutoff, 'use_wv': use_wv, 'skip_corr': skip_corr}

    output = multitask(telluric_correct_wrap_helper, files, None, kwargs,
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

    clear_resolution_cache()
    clear_atran_cache()

    return tuple(result)
