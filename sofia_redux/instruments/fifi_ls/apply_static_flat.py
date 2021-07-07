# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numba as nb
import numpy as np
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities \
    import (gethdul, goodfile, hdinsert, write_hdul,
            multitask)
from sofia_redux.toolkit.interpolate \
    import interp_1d_point_with_error as interp


__all__ = ['clear_flat_cache', 'get_flat_from_cache', 'store_flat_in_cache',
           'get_flat', 'apply_flat_to_hdul', 'apply_static_flat',
           'wrap_apply_static_flat']

__flat_cache = {}


def clear_flat_cache():
    """
    Clear all data from the flat cache.
    """
    global __flat_cache
    __flat_cache = {}


def get_flat_from_cache(specfile, spatfile, obsdate):
    """
    Retrieves flat data from the flat cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    specfile : str
        File path to the spectral flat file
    spatfile : str:
        File path to the spatial flat file

    Returns
    -------
    filename, spat_flat, spec_flat, spec_wave, spec_err
        filename : str
            Used to update FLATFILE in FITS headers
        spat_flat : numpy.ndarray
            Spatial flat (nwave, nspaxel, nspexel)
        spec_flat : numpy.ndarray
            Spectral flat (nwave, nspaxel, nspexel)
        spec_wave : numpy.ndarray
            Spectral wavelengths (nwave)
        spec_err : numpy.ndarray
            Error at each wavelength (nwave)
    """
    global __flat_cache

    key = specfile, spatfile, obsdate

    if key not in __flat_cache:
        return

    for f in [specfile, spatfile]:
        if not goodfile(f):
            try:
                del __flat_cache[key]
            except KeyError:   # pragma: no cover
                # this could potentially fail in a race condition
                pass
            return

    modkey = ','.join([str(os.path.getmtime(f))
                       for f in [specfile, spatfile]])
    if modkey not in __flat_cache.get(key, {}):
        return

    log.debug("Retrieving flat data from cache (%s, %s, %s)" % key)
    return __flat_cache.get(key, {}).get(modkey)


def store_flat_in_cache(specfile, spatfile, obsdate,
                        filename, spat_flat, spec_flat, spec_wave, spec_err):
    """
    Store flat data in the flat cache.

    Parameters
    ----------
    specfile : str
        File path to the spectral flat file
    spatfile : str:
        File path to the spatial flat file
    filename : str
        Used to update FLATFILE in FITS headers
    spat_flat : numpy.ndarray
        Spatial flat (nwave, nspaxel, nspexel)
    spec_flat : numpy.ndarray
        Spectral flat (nwave, nspaxel, nspexel)
    spec_wave : numpy.ndarray
        Spectral wavelengths (nwave)
    spec_err : numpy.ndarray
        Error at each wavelength (nwave)

    Returns
    -------
    None
    """
    global __flat_cache
    key = specfile, spatfile, obsdate
    log.debug("Storing flat data in cache (%s, %s, %s)" % key)
    __flat_cache[key] = {}
    modkey = ','.join([str(os.path.getmtime(f))
                       for f in [specfile, spatfile]])
    __flat_cache[key][modkey] = (
        filename, spat_flat, spec_flat, spec_wave, spec_err)


def get_flat(header):
    """
    Return flat data table

    Parameters
    ----------
    header : fits.Header
        Information from the header will be used to
        determine the correct default.

    Returns
    -------
    filename, spat_flat, spec_flat, spec_wave, spec_err
        filename : str
            Used to update FLATFILE header keyword
        spat_flat : numpy.ndarray
            Spatial flat (nwave, nspaxel, nspexel)
        spec_flat : numpy.ndarray
            Spectral flat (nwave, nspaxel, nspexel)
        spec_wave : numpy.ndarray
            Spectral wavelengths (nwave)
        spec_err : numpy.ndarray
            Error at each wavelength (nwave)
    """
    if not isinstance(header, fits.Header) or 'DATE-OBS' not in header:
        log.error("Cannot determine DATE-OBS from header")
        return

    for required_key in ['DATE-OBS', 'CHANNEL', 'DICHROIC', 'G_ORD_B']:
        if required_key not in header:
            log.error("Header is missing %s keyword" % required_key)
            return

    # get the date
    dateobs = str(header.get('DATE-OBS'))
    try:
        obsdate = int(''.join([x.zfill(2) for x in dateobs[:10].split('-')]))
    except ValueError:
        obsdate = -9999
    if obsdate < 20000000:
        log.error("Invalid DATE-OBS %s" % dateobs)
        return

    # get the channel, dichroic, and order
    channel = str(header.get('CHANNEL', 'UNKNOWN')).strip().upper()
    dichroic = int(header.get('DICHROIC'))
    b_order = str(header.get('G_ORD_B')).strip().upper()
    dstr = 'D130' if dichroic == 130 else 'D105'

    # also check for the order filter for blue
    # This keyword is only present from 10/2019 on.
    # If not present, or it's a bad value, assume the value
    # matches the order
    if 'G_FLT_B' in header:
        b_filter = str(header['G_FLT_B']).upper().strip()
        if b_filter not in ['1', '2']:
            b_filter = b_order
    else:
        b_filter = b_order

    if channel == 'RED':
        fstr = 'R'
        cstr = 'R1'
    else:
        fstr = f'B{b_order}'
        if b_order == b_filter:
            cstr = f'B{b_order}'
        else:
            cstr = f'B{b_order}{b_filter}'

    # expected spatial and spectral file names
    spatfile = 'spatialFlat{}.txt'.format(fstr)
    specfile = 'spectralFlats{}{}.fits'.format(cstr, dstr)

    # paths to files
    flat_path = os.path.join(os.path.dirname(fifi_ls.__file__),
                             'data', 'flat_files')
    spatfile = os.path.join(flat_path, spatfile)
    specfile = os.path.join(flat_path, specfile)

    if not goodfile(spatfile, verbose=True):
        log.error("Cannot locate spatial flat {}".format(spatfile))
        return
    if not goodfile(specfile, verbose=True):
        log.error("Cannot locate spectral flat {}".format(specfile))
        return

    # read the spatial flat file
    df = pandas.read_csv(spatfile, comment='#', delim_whitespace=True)
    df.sort_values('date', inplace=True)
    df = df[df['date'] >= obsdate]
    if len(df) == 0:
        log.error("No spatial flat found for {}".format(dateobs))
        return
    spat_row = df.reset_index().loc[0]

    log.debug('For: {} {} {}'.format(cstr, dstr, obsdate))
    log.debug('Found spatial flat: '
              '{}[{:.0f}]'.format(spatfile, spat_row['date']))
    log.debug('Found spectral flat: {}'.format(specfile))
    fname = 'flat_files/{}[{:.0f}],flat_files/{}'.format(
        os.path.basename(spatfile), spat_row['date'],
        os.path.basename(specfile))

    flatdata = get_flat_from_cache(specfile, spatfile, obsdate)
    if flatdata is not None:
        return flatdata

    # spatial flat is an array of 25 numbers
    spat = np.asarray(
        spat_row[['p{}'.format(i) for i in range(25)]], dtype=float)

    # read the spectral flat file: wavelengths, flat values, errors
    spec_hdul = fits.open(specfile)
    spec_wave = np.asarray(spec_hdul[1].data, dtype=float)
    spec_flat = np.asarray(spec_hdul[2].data, dtype=float)
    spec_err = np.asarray(spec_hdul[3].data, dtype=float)
    spec_hdul.close()

    store_flat_in_cache(specfile, spatfile, obsdate,
                        fname, spat, spec_flat, spec_wave, spec_err)

    return fname, spat, spec_flat, spec_wave, spec_err


@nb.njit(fastmath={'nsz', 'ninf'}, cache=True, nogil=True)
def stripnans(x, y, e):   # pragma: no cover
    n = x.size
    xout = np.empty(n, dtype=nb.float64)
    yout = np.empty(n, dtype=nb.float64)
    eout = np.empty(n, dtype=nb.float64)
    found = 0
    for i in range(n):
        yi = y[i]
        if not np.isnan(yi):
            xout[found] = x[i]
            eout[found] = e[i]
            yout[found] = yi
            found += 1
    return xout[:found], yout[:found], eout[:found]


@nb.njit(fastmath={'nsz', 'ninf'}, cache=True)
def calculate_flat(wave, data, var, spatdata, specdata,
                   specwave, specerr, skiperr):   # pragma: no cover
    """Workhorse for math stuff - fast"""

    shape = data.shape
    ndata, nw, ns = shape
    ndns = ndata * ns
    nwave, nspaxel, nspexel = specdata.shape

    flat_corr = np.empty(shape, dtype=nb.float64)
    var_corr = np.empty(shape, dtype=nb.float64)
    flat_store = np.empty(shape, dtype=nb.float64)
    flat_err_store = np.empty(shape, dtype=nb.float64)
    dostrip = np.any(np.isnan(specdata))

    # loop over data samples and spaxels
    for ij in range(ndns):
        i = ij // ns
        j = ij % ns

        x = wave[:, j]
        y = data[i, :, j]
        v = var[i, :, j]
        sfac = spatdata[j]

        # spexel loop: each one has its own spectral response
        for k in nb.prange(nspexel):

            if dostrip:
                sw, sd, se = stripnans(
                    specwave, specdata[:, j, k], specerr[:, j, k])
            else:
                sw, sd, se = specwave, specdata[:, j, k], specerr[:, j, k]

            xk = x[k]

            # Interpolate spectral response onto wavelength for spexel
            flatval, eout = interp(sw, sd, se, xk)

            if flatval == 0 or np.isnan(flatval):
                flat_store[i, k, j] = np.nan
                flat_corr[i, k, j] = np.nan
                flat_err_store[i, k, j] = np.nan
                var_corr[i, k, j] = np.nan
                continue

            # Apply spatial response
            flatval *= sfac
            eout *= sfac

            # Apply flat to data and error
            yk = y[k]
            flat_store[i, k, j] = flatval
            flat_corr[i, k, j] = yk / flatval
            flat_err_store[i, k, j] = eout

            flatval *= flatval  # flat squared in-place for speed
            var_corr[i, k, j] = v[k] / flatval
            if not skiperr:
                # Propagate the flat variance
                # (+= flat_variance * flux^2 / (flat^4)
                ey = eout * yk
                ey *= ey  # this is flat_variance * (flux^2)
                flatval *= flatval  # this is flat^4
                var_corr[i, k, j] += ey / flatval

    return flat_corr, var_corr, flat_store, flat_err_store


def apply_flat_to_hdul(hdul, flatdata, skip_err=True):
    """
    Divide extension data by static flat.  Update header.

    Parameters
    ----------
    hdul : fits.HDUList
    flatdata : 4-tuple of numpy.ndarray
    skip_err : bool, optional

    Returns
    -------
    fits.HDUList
    """
    # flat data from get_flat:
    flatfile, spatdata, specdata, specwave, specerr = flatdata

    # update the header for the output file;
    # add the flat file name to it
    primehead = hdul[0].header

    # loop over extensions
    result = fits.HDUList(fits.PrimaryHDU(header=primehead))
    primehead = result[0].header
    hdinsert(primehead, 'FLATFILE', flatfile,
             comment='Flat filename')
    primehead['HISTORY'] = 'Flat-field corrected'
    hdinsert(primehead, 'PRODTYPE', 'flat_fielded')
    hdinsert(primehead, 'FILENAME', hdul[0].header['FILENAME'].replace(
        'XYC', 'FLF'))

    ngrating = hdul[0].header.get('NGRATING', 1)
    for idx in range(ngrating):
        name = f'FLUX_G{idx}'

        data = np.asarray(hdul[name].data, dtype=float)
        var = np.asarray(hdul[name.replace('FLUX', 'STDDEV')].data,
                         dtype=float) ** 2
        wave = np.asarray(hdul[name.replace('FLUX', 'LAMBDA')].data,
                          dtype=float)
        if data.ndim < 3:
            data = data.reshape((1, *data.shape))
            var = var.reshape((1, *var.shape))
            do_reshape = True
        else:
            do_reshape = False

        hdu_result = calculate_flat(wave, data, var, spatdata, specdata,
                                    specwave, specerr, skip_err)

        # store data in output FITS:
        # FLUX, STDDEV, LAMBDA, XS, YS
        exthdr = hdul[name].header
        if do_reshape:
            # standard data: is 2D, so take first plane only
            result.append(fits.ImageHDU(hdu_result[0][0],
                                        header=exthdr, name=name))
            result.append(fits.ImageHDU(np.sqrt(hdu_result[1][0]),
                                        header=exthdr,
                                        name=name.replace('FLUX', 'STDDEV')))
        else:
            # OTF data: is 3D, propagate all planes
            result.append(fits.ImageHDU(hdu_result[0],
                                        header=exthdr, name=name))
            result.append(fits.ImageHDU(np.sqrt(hdu_result[1]), header=exthdr,
                                        name=name.replace('FLUX', 'STDDEV')))
        result.append(hdul[name.replace('FLUX', 'LAMBDA')].copy())
        result.append(hdul[name.replace('FLUX', 'XS')].copy())
        result.append(hdul[name.replace('FLUX', 'YS')].copy())

        # also store flat and flat error, for diagnostic purposes
        # this is always 2D
        exthdr['BUNIT'] = ''
        result.append(fits.ImageHDU(hdu_result[2][0], header=exthdr,
                                    name=name.replace('FLUX', 'FLAT')))
        result.append(fits.ImageHDU(hdu_result[3][0], header=exthdr,
                                    name=name.replace('FLUX', 'FLATERR')))

    return result


def apply_static_flat(filename, outdir=None, write=None, skip_err=True):
    """
    Apply pre-reduced and normalized flat to FIFI-LS data.

    The procedure is:

        1. Identify a spatial and spectral flat field to use.  The code
           will look for flat files that match the current configuration
           (channel, dichroic, and order), and the observation date
           (spatial flats only).
        2. Loop through each grating scan extension in the input file,
           composing a flat cube from the spatial and spectral flat
           inputs.
        3. Divide the data cube by the flat cube and optionally
           propagate the flat error to the error cube.
        4. Create FITS file and (optionally) write results to disk.

    The output FITS file contains n_scan * 7 image extensions:
    FLUX, STDDEV, LAMBDA, XS, YS, FLAT, and FLATERR data cubes,
    with a '_G{i}' suffix for each grating extension i.
    For most data, FLUX and STDDEV are 25 x 16 arrays; for OTF
    data they are 25 x 16 x n_ramp.  The XS and YS arrays are usually
    25-element 1D arrays; for OTF data they are 25 x 1 x n_ramp.
    The LAMBDA, FLAT, and FLATERR arrays are always 25 x 16.

    Parameters
    ----------
    filename : str
        File to be flat field corrected.  Should have been generated
        by fifi_ls.spatial_calibrate.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUL. The output filename is created
        from the input filename, with the suffix 'XYC' replaced with 'FLF'.
    skip_err : bool, optional
        If True, flat errors will not be propagated.

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

    flatdata = get_flat(header=hdul[0].header)
    if flatdata is None:
        log.error('No flat found.')
        return
    result = apply_flat_to_hdul(hdul, flatdata, skip_err=skip_err)
    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def apply_static_flat_wrap_helper(_, kwargs, filename):
    return apply_static_flat(filename, **kwargs)


def wrap_apply_static_flat(files, outdir=None, allow_errors=False,
                           write=False, skip_err=True, jobs=None):
    """
    Wrapper for apply_static_flat over multiple files.

    See `apply_static_flat` for full description of reduction
    on a single file.

    Parameters
    ----------
    files : array_like of str
        paths to files to be flat corrected
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    allow_errors : bool, optional
        If True, return all created files on error.  Otherwise, return None
    write : bool, optional
        If True, output files will be written to disk.
    skip_err : bool, optional
        If True, flat errors will not be propagated to the flux errors.
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

    clear_flat_cache()

    kwargs = {'write': write, 'outdir': outdir, 'skip_err': skip_err}
    output = multitask(apply_static_flat_wrap_helper, files, None, kwargs,
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

    clear_flat_cache()

    return tuple(result)
