# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits

from sofia_redux.spectroscopy.radvel import radvel
from sofia_redux.toolkit.utilities \
    import (gethdul, hdinsert, write_hdul, multitask)

__all__ = ['correct_lambda', 'correct_wave_shift', 'wrap_correct_wave_shift']


def correct_lambda(hdul):
    """
    Correct LAMBDA extension in HDU list.

    Parameters
    ----------
    hdul : fits.HDUList

    Returns
    -------
    fits.HDUList
        HDUList with extension LAMBDA corrected and UNCORRECTED_LAMBDA
        appended.
    """

    result = fits.HDUList(fits.PrimaryHDU(header=hdul[0].header))
    primehead = result[0].header

    history = ''.join(primehead['HISTORY']).lower()
    if 'telluric corrected' not in history:
        log.error("No telluric correction performed, "
                  "not shifting wavelengths")
        return

    primehead['HISTORY'] = 'Barycentric wavelength shift removed'
    hdinsert(primehead, 'PRODTYPE', 'wavelength_shifted')
    outname = os.path.basename(primehead.get('FILENAME', 'UNKNOWN'))
    for repl in ['SCM', 'TEL', 'CAL']:
        outname = outname.replace(repl, 'WSH')
    hdinsert(primehead, 'FILENAME', outname)

    dw_bary, dw_lsr = radvel(primehead)
    hdinsert(primehead, 'BARYSHFT', dw_bary,
             comment='Barycentric motion dl/l shift (applied)')
    hdinsert(primehead, 'LSRSHFT', dw_lsr,
             comment='Additional dl/l shift to LSR (unapplied)')
    w = hdul['LAMBDA'].data
    w = w + (dw_bary * w)

    for hdu in hdul[1:]:
        name = hdu.header.get('EXTNAME')
        if name == 'LAMBDA':
            result.append(fits.ImageHDU(w, header=hdu.header.copy(),
                                        name='LAMBDA'))
            result.append(fits.ImageHDU(hdu.data.copy(),
                                        header=hdu.header.copy(),
                                        name='UNCORRECTED_LAMBDA'))
        else:
            result.append(hdu.copy())

    return result


def correct_wave_shift(filename, write=False, outdir=None):
    """
    Correct wavelength shift due to motion of the Earth.

    The procedure is as follows.

        1. Read the FITS header and the wavelengths from the input file.
        2. Calculate the wavelength shift due to barycentric motion.
        3. Apply the (reverse) shift to wavelengths in the LAMBDA extension.
        4. Create a FITS file and (optionally) write results to disk.

    The output FITS file matches the extensions and dimensions of the
    input FITS file, with an additional UNCORRECTED_LAMBDA extension
    attached, containing a copy of the input, uncorrected wavelengths.

    Parameters
    ----------
    filename : str
        FITS file to be corrected.  Should have been created with
        fifi_ls.combine_grating_scans or fifi_ls.telluric_correct
        or fifi_ls.flux_calibrate.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUList.  The output filename is
        created from the input filename, with the suffix 'SCM', 'TEL',
        or 'CAL' replaced with 'WSH'.
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

    result = correct_lambda(hdul)
    if result is None:
        return
    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def correct_wave_wrap_helper(_, kwargs, filename):
    return correct_wave_shift(filename, **kwargs)


def wrap_correct_wave_shift(files, outdir=None, allow_errors=False,
                            write=False, jobs=None):
    """
    Wrapper for correct_wave_shift over multiple files.

    See `correct_wave_shift` for full description of reduction
    on a single file.

    Parameters
    ----------
    files : array_like of str
        Paths to files to be corrected
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
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

    output = multitask(correct_wave_wrap_helper, files, None, kwargs,
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
