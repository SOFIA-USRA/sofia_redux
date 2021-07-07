# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities \
    import (gethdul, write_hdul, hdinsert, multitask)

__all__ = ['get_lambda_overlap', 'combine_extensions',
           'combine_grating_scans', 'wrap_combine_grating_scans']


def get_lambda_overlap(hdul):
    """
    Get overlapping wavelengths for all extensions.

    Parameters
    ----------
    hdul : fits.HDUList

    Returns
    -------
    2-tuple
        overlap minimum, overlap maximum
    """
    overlap_min, overlap_max = [], []
    ng = hdul[0].header.get('NGRATING', 1)

    for idx in range(ng):
        name = f'LAMBDA_G{idx}'
        w = hdul[name].data
        overlap_min.append(w.min())
        overlap_max.append(w.max())
    overlap_min = max(overlap_min)
    overlap_max = min(overlap_max)
    return overlap_min, overlap_max


def combine_extensions(hdul, correct_bias=False):
    """
    Combine all extensions into a single extension.

    Parameters
    ----------
    hdul : fits.HDUList
    correct_bias : bool, optional
        If True, additive offset between separate grating scans will
        be removed, using overlapping wavelength regions to determine
        the value of the bias.

    Returns
    -------
    fits.HDUList
    """
    # get number of grating scans
    ng = hdul[0].header.get('NGRATING', 1)

    # check data dimensions for first flux extension
    nd = hdul['FLUX_G0'].data.ndim
    if nd > 2 and ng > 1:
        msg = 'Grating scans are not supported in OTF mode.'
        log.error(msg)
        return

    # for single grating scan, just grab data from input
    if ng == 1:
        fluxes = hdul['FLUX_G0'].data
        stddevs = hdul['STDDEV_G0'].data
        lambdas = hdul['LAMBDA_G0'].data

        # expand x and y values to match data shape
        xes = np.zeros_like(fluxes)
        yes = np.zeros_like(fluxes)
        xes[..., :] = hdul['XS_G0'].data[..., :]
        yes[..., :] = hdul['YS_G0'].data[..., :]

        exthdr = hdul['FLUX_G0'].header
        mean_bg = [exthdr.get('BGLEVL_A', 0),
                   exthdr.get('BGLEVL_B', 0)]
    else:
        lambdas = np.zeros((16 * ng, 25))
        fluxes = np.zeros_like(lambdas)
        stddevs = np.zeros_like(lambdas)
        xes = np.zeros_like(lambdas)
        yes = np.zeros_like(lambdas)
        mval = np.full((ng, 25), np.nan)
        bgval = np.zeros((ng, 2))
        overlap_min, overlap_max = 0, 0
        if correct_bias:
            overlap_min, overlap_max = get_lambda_overlap(hdul)
            if overlap_min >= overlap_max:
                log.info("No overlapping wavelengths; "
                         "not correcting bias offset")
                correct_bias = False

        for idx in range(ng):
            name = f'FLUX_G{idx}'
            widx = np.arange(16 * idx, 16 * (idx + 1))
            data = hdul[name].data
            stddev = hdul[name.replace('FLUX', 'STDDEV')].data
            w = hdul[name.replace('FLUX', 'LAMBDA')].data

            # Read the raw background levels from the extension headers
            # for averaging
            exthdr = hdul[name].header
            bgval[idx, :] = np.array(
                [exthdr.get('BGLEVL_A', 0),
                 exthdr.get('BGLEVL_B', 0)])
            lambdas[widx, :] = w
            fluxes[widx, :] = data
            stddevs[widx, :] = stddev
            xes[widx, :] = hdul[name.replace('FLUX', 'XS')].data
            yes[widx, :] = hdul[name.replace('FLUX', 'YS')].data

            # If correcting bias, get the mean value of all overlapping
            # data for this extension
            if correct_bias:
                midx = (w[:, 0] >= overlap_min) & (w[:, 0] <= overlap_max)
                if midx.sum() > 3:
                    mval[idx, :] = np.nanmean(data[midx, :], axis=0)
                else:
                    mval[idx, :] = np.nan

        if correct_bias:
            # Fix bias offset for each grating position at each spaxel
            mlevel = np.resize(np.nanmean(mval, axis=0), (ng, 25))
            diff = mlevel - mval
            diff[np.isnan(diff)] = 0

            for idx in range(ng):
                widx = np.arange(idx * 16, (idx + 1) * 16)
                fluxes[widx, :] = fluxes[widx, :] + diff[idx, :]

        # sort by wavelength
        sortidx = lambdas.argsort(axis=0)
        static_inds = np.indices(lambdas.shape)
        lambdas = lambdas[sortidx, static_inds[1]]
        fluxes = fluxes[sortidx, static_inds[1]]
        stddevs = stddevs[sortidx, static_inds[1]]
        xes = xes[sortidx, static_inds[1]]
        yes = yes[sortidx, static_inds[1]]

        mean_bg = np.nanmean(bgval, axis=0)

    # Update header with combined values

    result = fits.HDUList(fits.PrimaryHDU(header=hdul[0].header))
    primehead = result[0].header

    hdinsert(primehead, 'BGLEVL_A', mean_bg[0],
             comment='Background level in A nod (ADU/s/spaxel)')
    hdinsert(primehead, 'BGLEVL_B', mean_bg[1],
             comment='Background level in B nod (ADU/s/spaxel)')
    hdinsert(primehead, 'PRODTYPE', 'scan_combined')
    outname = os.path.basename(hdul[0].header.get('FILENAME', 'UNKNOWN'))
    outname = outname.replace('FLF', 'SCM').replace('XYC', 'SCM')
    hdinsert(primehead, 'FILENAME', outname)
    primehead['HISTORY'] = "Grating scans combined into single extension"

    exthdr = fits.Header()
    exthdr['BUNIT'] = ('adu/(Hz s)', 'Data units')
    result.append(fits.ImageHDU(fluxes, header=exthdr, name='FLUX'))
    result.append(fits.ImageHDU(stddevs, header=exthdr, name='STDDEV'))
    exthdr['BUNIT'] = 'um'
    result.append(fits.ImageHDU(lambdas, header=exthdr, name='LAMBDA'))
    exthdr['BUNIT'] = 'arcsec'
    result.append(fits.ImageHDU(xes, header=exthdr, name='XS'))
    result.append(fits.ImageHDU(yes, header=exthdr, name='YS'))
    return result


def combine_grating_scans(filename, correct_bias=False,
                          write=False, outdir=None):
    """
    Combine separate grating positions in FIFI-LS data.

    Combines separate grating positions in FIFI-LS data into a
    single extension.  Will produce an unevenly spaced wavelength
    grid if multiple grating (inductosyn) extensions exist in the
    input file.

    The procedure is:

    1. Read the wavelength values for all grating scans to determine
       overlapping wavelength regions.
    2. Loop over the grating scan extensions, reading the flux from
       each. Calculate the mean value of any overlapping wavelength
       regions. Store all data in an unsorted cube.
    3. Loop over the spaxels in the cube, correcting the flux to the
       mean value of all overlap regions, then sorting the data by
       its associated wavelength value.
    4. Create FITS file and (optionally) write results to disk.

    The output FITS file contains FLUX, STDDEV, LAMBDA, XS, and
    YS data arrays.  For most data, all arrays have dimension
    25 x nw, where nw is the combined number of wavelength samples
    across all input grating scans.

    For OTF data, only one grating scan is expected or allowed, so nw
    is always 16. Output cubes for FLUX, STDDEV, XS, and YS are
    25 x 16 x n_ramp.  The LAMBDA array is 25 x 16.

    Parameters
    ----------
    filename : str
        Path to the file to be combined
    correct_bias : bool, optional
        If True, additive offset between separate grating scans will
        be removed, using overlapping wavelength regions to determine
        the value of the bias.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUL.  The output filename is
        created from the input filename, with the suffix 'XYC' or
        'FLF' replaced with 'SCM'.

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

    result = combine_extensions(hdul, correct_bias=correct_bias)
    if result is None:
        log.error('Combination failed')
        return

    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def combine_scans_wrap_helper(_, kwargs, filename):
    return combine_grating_scans(filename, **kwargs)


def wrap_combine_grating_scans(files, outdir=None, correct_bias=True,
                               allow_errors=False, write=False,
                               jobs=None):
    """
    Wrapper for combine_grating_scans over multiple files.

    See `combine_grating_scans` for full description of reduction on a
    single file.

    Parameters
    ----------
    files : array_like of str
        paths to files to be scan-combined
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    correct_bias : bool, optional
        If True, additive offset between separate grating scans will
        be removed, using overlapping wavelength regions to determine
        the value of the bias.
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

    kwargs = {'outdir': outdir, 'write': write, 'correct_bias': correct_bias}

    output = multitask(combine_scans_wrap_helper, files, None, kwargs,
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
