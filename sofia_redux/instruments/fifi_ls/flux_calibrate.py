# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.fifi_ls.get_response \
    import get_response, clear_response_cache
from sofia_redux.toolkit.utilities \
    import (gethdul, hdinsert, write_hdul, multitask)

__all__ = ['apply_response', 'flux_calibrate', 'wrap_flux_calibrate']


def apply_response(hdul, response):
    """
    Apply response data to data in an HDUList.

    Parameters
    ----------
    hdul : fits.HDUList
    response : numpy.ndarray
        (3, nwave) where [0, :] = wavelength and [1, :] = response data,
        and [2, :] = response error.

    Returns
    -------
    fits.HDUList
    """
    wave = np.asarray(hdul['LAMBDA'].data, dtype=float)
    data = np.asarray(hdul['FLUX'].data, dtype=float)
    var = np.asarray(hdul['STDDEV'].data, dtype=float) ** 2
    udata, uvar = None, None
    if 'UNCORRECTED_FLUX' in hdul:
        udata = np.asarray(hdul['UNCORRECTED_FLUX'].data, dtype=float)
        uvar = np.asarray(
            hdul['UNCORRECTED_STDDEV'].data, dtype=float) ** 2

    shape = data.shape
    resp_corr = np.full(shape, np.nan, dtype=float)
    var_corr = np.full(shape, np.nan, dtype=float)
    resp_store = np.full_like(wave, np.nan, dtype=float)

    if udata is not None:
        var_uncorr = np.full(shape, np.nan, dtype=float)
        resp_uncorr = np.full(shape, np.nan, dtype=float)
    else:
        var_uncorr = resp_uncorr = None

    for i in range(25):
        iresp = np.interp(wave[:, i], response[0], response[1],
                          left=np.nan, right=np.nan)
        if np.isnan(iresp).all():
            continue
        resp_corr[..., i] = data[..., i] / iresp
        var_corr[..., i] = var[..., i] / (iresp ** 2)
        resp_store[:, i] = iresp

        # propagate uncorrected data if present
        if udata is not None:
            resp_uncorr[..., i] = udata[..., i] / iresp
            var_uncorr[..., i] = uvar[..., i] / (iresp ** 2)

    if np.isnan(resp_store).all():
        log.error("No valid response data found for input wavelengths")
        return

    result = fits.HDUList(fits.PrimaryHDU(header=hdul[0].header))
    exthdr = hdul['FLUX'].header

    primehead = result[0].header
    outname = os.path.basename(primehead.get('FILENAME', 'UNKNOWN'))
    outname = outname.replace('SCM', 'CAL').replace('TEL', 'CAL')
    primehead['HISTORY'] = 'Flux calibrated'
    hdinsert(primehead, 'PRODTYPE', 'flux_calibrated')
    hdinsert(primehead, 'PROCSTAT', 'LEVEL_3')
    hdinsert(primehead, 'BUNIT', 'Jy/pixel', comment='Data units')
    hdinsert(primehead, 'RAWUNITS', 'adu/(Hz s)',
             comment='Raw data units before calibration')
    hdinsert(primehead, 'FILENAME', outname)

    # Add mean calibration error to header
    hdinsert(primehead, 'CALERR', np.nanmean(response[2] / response[1]),
             comment='Overall fractional flux cal error')

    # Also calibrate the background level in the header at the mean
    # wavelength for the observation
    mean_response = np.interp(np.nanmean(wave), response[0], response[1],
                              left=np.nan, right=np.nan)
    if not np.isnan(mean_response):
        for x in ['A', 'B']:
            hdinsert(primehead, 'BGLEVL_%s' % x,
                     primehead.get('BGLEVL_%s' % x, 0) / mean_response,
                     comment='Background level %s nod (Jy/pixel)' % x)

    exthdr['BUNIT'] = ('Jy/pixel', 'Data units')
    result.append(fits.ImageHDU(resp_corr, header=exthdr, name='FLUX'))
    result.append(fits.ImageHDU(np.sqrt(var_corr), header=exthdr,
                                name='STDDEV'))
    if udata is not None:
        result.append(fits.ImageHDU(resp_uncorr, header=exthdr,
                                    name='UNCORRECTED_FLUX'))
        result.append(fits.ImageHDU(np.sqrt(var_uncorr), header=exthdr,
                                    name='UNCORRECTED_STDDEV'))
    result.append(hdul['LAMBDA'].copy())
    result.append(hdul['XS'].copy())
    result.append(hdul['YS'].copy())
    if 'ATRAN' in hdul:
        result.append(hdul['ATRAN'].copy())
    exthdr['BUNIT'] = ('adu/(Hz s Jy)', 'Data units')
    result.append(fits.ImageHDU(resp_store, header=exthdr,
                                name='RESPONSE'))
    if 'UNSMOOTHED_ATRAN' in hdul:
        result.append(hdul['UNSMOOTHED_ATRAN'].copy())
    return result


def flux_calibrate(filename, response_file=None, write=False, outdir=None):
    """
    Convert spectra to physical flux units.

    The procedure is:

        1. Identify response file to use.  Smooth it to the spectral resolution
           of the input file.
        2. Read the spectral data from the input file.
        3. Loop over the spaxels.
        4. For each spaxel, interpolate the response data onto the wavelengths
           of the spexels and multiply the data at each point by the response
           value and its associated wavelength.
        5. Create a FITS file and (optionally) write results to disk.

    The output FITS file matches the extensions and dimensions of the
    input FITS file, with an additional RESPONSE extension attached,
    containing the interpolated response correction.  The dimensions of
    the RESPONSE array match the dimensions of the LAMBDA array.

    Parameters
    ----------
    filename : str
        FITS file to be flux calibrated.  Should have been created by
        fifi_ls.combine_grating_scans or fifi_ls.telluric_correct.
    response_file : str, optional
        Response file to be used.  If not provided, a default file will
        be used.  If provided, should be a FITS image file containing
        wavelength, response data, and response error in an array
        of three rows in the primary extension.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUList.  The output filename is
        created from the input filename, with the suffix 'SCM' or 'TEL'
        replaced with 'CAL'.
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

    response = get_response(hdul[0].header, filename=response_file)
    if response is None:
        log.error('Failed to get response')
        return

    result = apply_response(hdul, response)
    if result is None:
        log.error('Failed to apply response')
        return
    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def flux_calibrate_wrap_helper(_, kwargs, filename):
    return flux_calibrate(filename, **kwargs)


def wrap_flux_calibrate(files, outdir=None, allow_errors=False,
                        response_file=None, write=False,
                        jobs=None):
    """
    Wrapper for flux_calibrate over multiple files.

    See `flux_calibrate` for full description of reduction on a single file.

    Parameters
    ----------
    files : array_like of str
        paths to files to be flux calibrated
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    allow_errors : bool, optional
        If True, return all created files on error.  Otherwise, return None
    response_file : str, optional
        Response file to be used.  If not provided, a default file will
        be used.  If provided, should be a FITS image file containing
        wavelength, response data, and response error in an array
        of three rows in the primary extension.
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

    clear_response_cache()

    kwargs = {'outdir': outdir, 'write': write,
              'response_file': response_file}

    output = multitask(flux_calibrate_wrap_helper, files, None, kwargs,
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

    return tuple(result)
