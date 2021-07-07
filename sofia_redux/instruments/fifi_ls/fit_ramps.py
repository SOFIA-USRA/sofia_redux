# Licensed under a 3-clause BSD style license - see LICENSE.rst

from datetime import datetime
import os
import warnings

from astropy import log
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
import bottleneck as bn
import numba
import numpy as np

from sofia_redux.instruments.fifi_ls.get_badpix \
    import get_badpix, clear_badpix_cache
from sofia_redux.toolkit.stats import meancomb
from sofia_redux.toolkit.utilities \
    import (hdinsert, gethdul, write_hdul, multitask)

__all__ = ['get_readout_range', 'resize_data', 'fit_data',
           'process_extension', 'fit_ramps', 'wrap_fit_ramps']

DEBUG = False


def get_readout_range(header):
    """
    Returns the readout range as extracted from header.

    If the observations occurred between January 2014 and March 2015,
    we ignore the first 3 readouts, otherwise ignore the first 2.
    The last readout is also ignored.

    Parameters
    ----------
    header : fits.Header

    Returns
    -------
    2-tuple:
        int : first readout
        int : ramplength
    """
    channel = header.get('CHANNEL', 'UNKNOWN').strip().upper()
    ramplength = header.get(
        'RAMPLN_%s' % ('R' if channel == 'RED' else 'B'))
    dateobs = header.get('DATE-OBS',
                         Time(datetime.utcnow(), format='datetime').isot)
    dateobs = Time(dateobs, format='isot')
    drange = [Time('2014-01-01T00:00:00', format='isot'),
              Time('2015-03-01T00:00:00', format='isot')]
    read_start = 3 if (drange[0] < dateobs < drange[1]) else 2
    return read_start, ramplength


def resize_data(data, readout_range, indpos, remove_first_ramps=True,
                subtract_bias=True, indpos_sigma=3.0):
    """
    Trim and reshape data to separate ramps.

    Parameters
    ----------
    data : numpy.ndarray
        (1 [optional], nramps, nwave (18), nspaxel (26))
    readout_range : array_like of int
        (start readout, ramp length)
    indpos : int
        Expected inductosyn position for the grating
    remove_first_ramps : bool, optional
        Remove the first two ramps.
    subtract_bias : bool, optional
        Subtract the empty pixel value.
    indpos_sigma : float, optional
        If >0, will be used to flag ramps with mean grating
        position that deviates from the indpos value by this
        many sigma, or has a standard deviation greater
        than this value.  Flags are returned in the bad_ramps
        array.

    Returns
    -------
    data, bad_ramps : numpy.ndarray, numpy.ndarray
        Resized data is a floating point array with dimensions
        (ramps/spaxel, ramplength, nwave (16), nspaxel (25)).
        The secondary array is a Boolean array with dimensions
        (ramps/spaxel), containing bad ramp flags (True = bad)
        for later propagation in ramp fits.
    """
    if not isinstance(data, np.ndarray):
        log.error("Invalid data")
        return
    elif (data.ndim != 3
          or data.shape[-1] < 25
          or data.shape[-2] != 18):
        log.error("Invalid data shape")
        return
    elif not hasattr(readout_range, '__len__') or len(readout_range) != 2:
        log.error("Invalid readout_range")
        return

    grating_values = None
    if indpos_sigma > 0:
        # use the values in the last spaxel to check for
        # bad grating position
        mask = 0b0000000000011111
        grating_values = (data[:, 2::4, 25] & 0xffff) + 2 ** 16 \
            * ((data[:, 3::4, 25] & 0xffff) & mask)

        # average the 4 samples
        grating_values = np.mean(grating_values, axis=1)

        # check for appropriate values
        gmean, gmed, gstd = sigma_clipped_stats(grating_values)
        log.debug(f'Expected INDPOS: {indpos}; reported: {gmean} +/- {gstd}')
        if np.allclose(gstd, 0) or np.allclose(gmean, 0) or \
                np.abs(gmean - indpos) > 5 * gstd:
            # mean is well off from expected value -- this likely
            # means the data is older, and does not contain grating
            # value measurements
            log.debug('Grating measurements do not match expected '
                      'value. Turning off grating instability data '
                      'rejection.')
            grating_values = None

    if subtract_bias:
        # subtract the first spexel (empty pixel) value,
        # accounting for invalid int16 values
        fdata = np.float32(data)
        empty_pix = fdata[:, 0, :] + 2**15
        all_pix = fdata + 5000.0
        all_pix -= np.tile(np.expand_dims(empty_pix, 1), (1, 18, 1))
        all_pix[all_pix < -32768] = -32768
        all_pix[all_pix > 32767] = 32767
        data = np.int16(all_pix)

    # remove the last spaxel and the first and last spexel
    data = data[:, 1:17, :25]
    readout_start, ramplength = readout_range
    nreadouts = data.size
    nramps = nreadouts // ramplength
    ramps_per_spaxel = nramps // (25 * 16)
    if nreadouts != (25 * 16 * ramplength * ramps_per_spaxel):
        log.error(f"Number of readouts in data shape {data.shape} "
                  "does not match header")
        return

    # Reshape the data into separate ramp dimensions
    # (nramps, 16, 25) -> (ramps/spaxel, ramplength, 16, 25)
    # i.e. (n_ramp, n_readout, n_wave, n_spaxel)
    newshape = ramps_per_spaxel, ramplength, 16, 25
    data = data.reshape(newshape)
    log.debug(f'# ramps, # readouts/ramp: {ramps_per_spaxel}, {ramplength}')
    if grating_values is not None:
        grating_values = grating_values.reshape(ramps_per_spaxel, ramplength)

    # remove the first ramps from each spaxel if required
    if remove_first_ramps and ramps_per_spaxel > 2:
        log.debug('Removing first 2 ramps')
        data = data[2:]
        if grating_values is not None:
            grating_values = grating_values[2:]

    # remove the first 2 or 3, and last readout if possible
    if ramplength > (readout_start + 1):
        data = data[:, readout_start:-1, :, :]

    # convert data to float
    data = np.asarray(data, dtype=float)

    # make a bad sample mask from the grating values
    bad_ramps = np.full(data.shape[0], False)
    if grating_values is not None:
        # average value over ramp
        ramp_value = np.mean(grating_values, axis=1)
        ramp_std = np.std(grating_values, axis=1)
        # use full clipped standard deviation for flagging
        gmean, gmed, gstd = sigma_clipped_stats(grating_values)

        bad_mean = np.abs(ramp_value - indpos) > indpos_sigma * gstd
        bad_std = np.abs(ramp_std / gstd) > indpos_sigma
        bad_ramps = bad_mean | bad_std
        log.debug(f'Found {np.sum(bad_ramps)} bad ramps '
                  f'(out of {np.size(bad_ramps)}).')

        bad_ramp_idx = np.unique(np.where(bad_ramps)[0])
        if len(bad_ramp_idx) > 0:
            log.debug(f'Bad ramp index: {bad_ramp_idx}')
        if DEBUG:  # pragma: no cover
            log.info(f'Bad ramp index: {bad_ramp_idx}')
            from matplotlib import pyplot as plt
            plt.plot(grating_values)
            plt.plot(ramp_value, color='black', linestyle="-",
                     label='ramp-averaged value')
            plt.axhline(indpos, color='lightgray', linestyle="--",
                        alpha=0.5, label='expected value')
            plt.axhline(indpos - gstd, color='darkgray', linestyle="-.",
                        alpha=0.5)
            plt.axhline(indpos + gstd, color='darkgray', linestyle="-.",
                        alpha=0.5, label='standard deviation')
            plt.axhline(indpos - indpos_sigma * gstd, color='gray',
                        linestyle=":", alpha=0.5)
            plt.axhline(indpos + indpos_sigma * gstd, color='gray',
                        linestyle=":", alpha=0.5,
                        label=f'{indpos_sigma} sigma limit')
            for idx in bad_ramp_idx:
                plt.axvline(idx, alpha=0.2, color='black')
            plt.title(f'Rejected ramps: {bad_ramp_idx}')
            plt.xlabel('Ramp number')
            plt.ylabel('Inductosyn value')
            plt.legend()
            plt.tight_layout()
            plt.show()

    return data, bad_ramps


@numba.njit(cache=True, fastmath={'nsz', 'nnan', 'ninf'},
            nogil=True, parallel=False)
def calculate_fit(data, maxidx):   # pragma: no cover

    ramps_per_spaxel, nramp, nwave, nspaxel = data.shape
    mat = np.empty((nramp, 2), dtype=numba.int64)
    ata = np.empty((nramp, 2, 2), dtype=numba.float64)
    vfac = np.empty(nramp, dtype=numba.float64)
    sum1 = 0
    sum2 = 0

    for i in range(nramp):
        sum1 += i
        sum2 += i * i
        mat[i, 0] = 1
        mat[i, 1] = i
        ata[i, 0, 0] = i + 1
        ata[i, 0, 1] = sum1
        ata[i, 1, 0] = sum1
        ata[i, 1, 1] = sum2
        if i > 0:
            vfac[i] = 1.0 / np.sum((mat[:i + 1, 1] - (i / 2)) ** 2)
        else:
            vfac[i] = 0.0

    slopes = np.empty((ramps_per_spaxel, nwave, nspaxel), dtype=numba.float64)
    var = np.empty((ramps_per_spaxel, nwave, nspaxel), dtype=numba.float64)
    for i in numba.prange(ramps_per_spaxel):
        for j in range(nwave):
            for k in range(nspaxel):
                idx = maxidx[i, j, k]
                if idx == 0 or idx == (nramp - 1):
                    idx = nramp
                elif idx <= 2:
                    slopes[i, j, k] = np.nan
                    var[i, j, k] = np.nan
                    continue

                line = data[i, :idx, j, k]
                a = ata[idx - 1]
                b = np.empty(2, dtype=numba.float64)
                b[0] = 0.0
                b[1] = 0.0
                for ln in range(idx):
                    b[0] += line[ln]
                    b[1] += line[ln] * ln

                c = np.linalg.solve(a, b)
                slopes[i, j, k] = c[1]
                sdevd = 0.0
                for ln in range(idx):
                    diff = c[0] + (c[1] * ln) - line[ln]
                    sdevd += diff * diff
                var[i, j, k] = vfac[idx - 1] * sdevd / (idx - 2)

    return slopes, var


def fit_data(data, s2n=10, threshold=5, allow_zero_variance=True,
             average_ramps=True, bad_ramps=None):
    """
    Applies linear fit (y = ax + b) over the second dimension of a 4D array.

    Highly optimized for this particular problem.  The flux is
    determined by the slope of the second dimension (ramps).
    After collapsing the ramp dimension, multiple ramps are averaged for
    each spaxel and spexel.

    Parameters
    ----------
    data : numpy.ndarray
        (ramps/spaxel, ramplength, nwave, nspaxel)
    s2n : float, optional
        Signal-to-noise below which data will be considered questionable
        and will be ignored.  Set <= 0 to turn off signal-to-noise filtering.
    threshold : float, optional
        Robust rejection threshold (in sigma) for combining slopes of
        individual ramps.
    allow_zero_variance : bool, optional
        If True, does not set data points with zero variance to NaN.  This
        option is here to replicate the behaviour of the previous IDL
        version.
    average_ramps : bool, optional
        If True, all ramps in the extension are averaged together.  This
        is desirable for all modes except OTF scans.
    bad_ramps : numpy.ndarray, optional
        If provided, should be an array of bool, matching the number
        of ramps/spaxel (data.shape[0]) where True indicates a bad ramp
        (e.g. due to grating position instability).

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        flux and standard deviation arrays of size (spexel, spaxel)
        or (16, 25) for FIFI-LS.
    """

    maxidx = bn.nanargmax(data, axis=1)
    if bad_ramps is not None:
        maxidx[bad_ramps] = -1
    slopes, var = calculate_fit(data, maxidx)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if isinstance(s2n, (float, int)) and s2n > 0:
            setnan = np.full(slopes.shape, False)
            nzi = var > 0
            setnan[nzi] = (slopes[nzi] / np.sqrt(var[nzi])) < s2n
            if allow_zero_variance:
                setnan[var == 0] = False
            slopes[setnan] = np.nan

    info = {}
    flux, mvar = meancomb(slopes, robust=threshold, variance=var,
                          axis=0, returned=True, info=info)

    if average_ramps:
        return flux, np.sqrt(mvar)
    else:
        # still flag outliers
        slopes[~info['mask']] = np.nan
        var[~info['mask']] = np.nan
        return slopes, np.sqrt(var)


def process_extension(hdu, readout_range, threshold=5,
                      s2n=10, remove_first=None,
                      subtract_bias=True, badmask=None,
                      average_ramps=True, posdata=None,
                      indpos_sigma=3.0):
    """
    Wrapper to process a single HDU extension.

    Parameters
    ----------
    hdu : ImageHDU
        input HDU extension
    readout_range : array_like of int
        (start readout, ramplength)
    threshold : float, optional
        Robust rejection threshold (in sigma) for combining slopes of
        individual ramps.
    s2n : float, optional
        Signal-to-noise below which data will be considered questionable
        and will be ignored.  Set <= 0 to turn off signal-to-noise filtering.
    remove_first : bool, optional
        if True, remove the first two ramps prior to fitting
    subtract_bias : bool, optional
        If True, subtract the value of the empty zeroth pixel for each
        spaxel prior to fitting.
    badmask : numpy.ndarray, optional
        (npoints, (spexel, spaxel)) array specifying indices of bad pixels
    average_ramps : bool, optional
        If set, data is averaged over ramps (default).  If not, all ramps
        are returned, as appropriate for OTF mode data, for which each ramp
        is at a different sky position.
    posdata : fits.Table, optional
        OTF scan position data table. If provided and `average_ramps` is
        set to False, then the input DLAM_MAP and DBET_MAP for each
        readout are averaged over each ramp; the FLAG key is and-ed over
        the ramp. The UNIXTIME for the first and last retained ramps
        are averaged and added to the header as RAMPSTRT and RAMPEND
        keywords, respectively.
    indpos_sigma : float, optional
        If >0, will be used to discard samples with grating
        position that deviates from the expected INDPOS value by this
        many sigma.

    Returns
    -------
    flux, stddev : ImageHDU, ImageHDU
        Optionally, a third element may be returned: a BinTableHDU containing
        OTF position data, propagated from the `posdata` argument.
    """
    if hdu.data is None:
        log.error("No data in HDU")
        return
    data = hdu.data.copy()
    image_hdr = hdu.header.copy()
    indpos = image_hdr['INDPOS']
    result = resize_data(data, readout_range, indpos,
                         remove_first_ramps=remove_first,
                         subtract_bias=subtract_bias,
                         indpos_sigma=indpos_sigma)
    if result is None:
        return

    flux, bad_ramps = result
    nramp = flux.shape[0]
    nread = readout_range[1]

    # fit slopes to ramps, average if desired
    flux, stddev = fit_data(flux, s2n=s2n, threshold=threshold,
                            average_ramps=average_ramps,
                            bad_ramps=bad_ramps)

    # apply bad mask to spaxels/spexels
    if isinstance(badmask, np.ndarray):
        if average_ramps:
            flux[badmask[:, 0], badmask[:, 1]] = np.nan
            stddev[badmask[:, 0], badmask[:, 1]] = np.nan
        else:
            flux[:, badmask[:, 0], badmask[:, 1]] = np.nan
            stddev[:, badmask[:, 0], badmask[:, 1]] = np.nan

    # average position data for each ramp if available
    # (but don't average over ramps)
    if posdata is not None:
        if average_ramps:
            log.warning('Incompatible arguments: posdata cannot '
                        'be propagated with average_ramps=True.')
            log.warning('Dropping scan position data from further '
                        'reduction.')
            posdata = None
        else:
            # reshape: nreadout -> n_ramp, n_readout
            if len(posdata) != (nramp * nread):
                log.error("Number of readouts does not match header")
                return
            newshape = (nramp, nread)

            # logical-and of flags over each ramp
            reshaped = posdata['FLAG'].reshape(newshape)
            flag = np.all(reshaped, axis=1)

            # get first and last averaged ramp times
            reshaped = posdata['UNIXTIME'].reshape(newshape)
            ftime = np.mean(reshaped, axis=1)[flag]
            hdinsert(image_hdr, 'RAMPSTRT', ftime[0],
                     'UNIX time of first ramp [s]')
            hdinsert(image_hdr, 'RAMPEND', ftime[-1],
                     'UNIX time of last ramp [s]')

            # new table with only good ramp-averaged position data
            # todo: consider averaging spatially local ramps as well
            pdata = Table()
            for name in ['DLAM_MAP', 'DBET_MAP']:
                reshaped = posdata[name].reshape(newshape)
                pdata[name] = np.mean(reshaped, axis=1)[flag]
            posdata = pdata

            # trim flux ramps to useful range as well
            if not flag.all():
                log.debug(f'Trimming {(~flag).sum()} '
                          f'ramps outside scan motion')
                flux = flux[flag, :, :]
                stddev = stddev[flag, :, :]

    # add a bg level header keyword
    mflux = 0.0 if np.isnan(flux).all() else np.nanmean(flux)
    hdinsert(image_hdr, 'BGLEVL_A', mflux,
             comment='BG level nod A (ADU/s)')

    name = image_hdr['EXTNAME']
    image_hdr['BUNIT'] = 'adu/s'
    hdu1 = fits.ImageHDU(flux, header=image_hdr, name=name)
    hdu2 = fits.ImageHDU(stddev, header=image_hdr,
                         name=name.replace('FLUX', 'STDDEV'))
    if posdata is not None:
        hdu3 = fits.BinTableHDU(posdata, name=name.replace('FLUX', 'SCANPOS'))
        return hdu1, hdu2, hdu3
    else:
        return hdu1, hdu2


def fit_ramps(filename, s2n=10, threshold=5, badpix_file=None,
              write=False, outdir=None, remove_first=True,
              subtract_bias=True, indpos_sigma=3.0):
    """
    Fit straight lines to raw voltage ramps to calculate corresponding flux.

    If writing to disk, the output filename is created from the input
    filename, with the suffix 'CP0' or 'CP1' replaced with 'RP0' or
    'RP1'.  The resulting HDU contains n_scan * 2 image extensions, for
    FLUX and STDDEV data for each grating scan, named with a grating scan
    index (e.g. FLUX_G0, STDDEV_G0, FLUX_G1, STDDDEV_G1, etc.).

    For all data except OTF mode A nods, the output image arrays have
    dimensions 25 x 16.  For OTF A nods, the ramps are not averaged to
    produce a single 2D array but rather propagated as a 3D cube, along
    with separate scanning position data.  Dimensions for these data
    are 25 x 16 x n_ramp, and an additional table called SCANPOS_G0
    is attached to the output file, containing the scan position data,
    averaged over each ramp.  It is assumed that OTF data files contain
    only one grating scan.

    Input files for this step should have been generated by
    fifi_ls.split_grating_and_chop.

    The procedure is:
    Loop through the extensions, fitting ramps in each:

        1. Remove the 26th spaxel (chopper values) and the first and
           last spexel.
        2. Remove the first ramp from each spaxel and the first and last
           readout in each ramp.
        3. Loop over spaxels and spexels, fitting each ramp with a line.
           Record the slope and the error on the slope.  Combine the
           slopes from all ramps with a robust weighted mean.  Record
           the error on the mean as the error on the flux.
        4. (optional) Create FITS file and write results to disk.

    Parameters
    ----------
    filename : str
        Name of the chop-split, grating-split file (including the path
        if not in the current working directory)
    s2n : float, optional
        Signal-to-noise below which data will be considered questionable
        and will be ignored.  Set <= 0 to turn off signal-to-noise filtering.
    threshold : float, optional
        Robust rejection threshold (in sigma) for combining slopes of
        individual ramps.
    badpix_file : str, optional
        badpix file to be used.  If not provided, a default file will be
        retrieved from the data/badpix_files directory, matching the date
        and channel of the input header.  If an override file is provided,
        it should be a text file containing two columns, spaxel and
        spexel coordinates.  Spaxels are numbered from 1 to 25, spexels
        from 1 to 16.
    write : bool, optional
        If True, write the output to disk and return the filename instead
        of the HDU.
    outdir : str, optional
        If writing to disk, use to set the output directory.  By default the
        output directory will be the same as the input filename location.
    remove_first : bool, optional
        If True, remove the first two ramps prior to fitting.
    subtract_bias : bool, optional
        If True, subtract the value of the empty zeroth pixel prior to
        fitting.
    indpos_sigma : float, optional
        If >0, will be used to discard samples with grating
        position that deviates from the expected INDPOS value by this
        many sigma.

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

    readout_range = get_readout_range(hdul[0].header)
    if (not hasattr(readout_range, '__len__')
            or len(readout_range) != 2
            or None in readout_range):
        log.error("Invalid readout range")
        return

    if not isinstance(filename, str):
        filename = hdul[0].header['FILENAME']
    if not isinstance(outdir, str):
        outdir = os.path.dirname(filename)

    if DEBUG:  # pragma: no cover
        log.info(f'Working on: {filename}')
    else:
        log.debug(f'Working on: {filename}')

    outfile = os.path.basename(filename).replace('CP', 'RP')
    primehead = hdul[0].header.copy()
    hdinsert(primehead, 'FILENAME', outfile)
    hdinsert(primehead, 'PRODTYPE', 'ramps_fit')

    badmask = get_badpix(primehead, filename=badpix_file)
    hdul_new = fits.HDUList([fits.PrimaryHDU(header=primehead)])
    ngrating = primehead.get('NGRATING', 1)
    remove_first_passed = remove_first
    for idx in range(ngrating):
        hdu = hdul[f'FLUX_G{idx}']

        # check for OTF mode data
        if f'SCANPOS_G{idx}' in hdul:
            average_ramps = False
            remove_first = False
            posdata = hdul[f'SCANPOS_G{idx}'].data
        else:
            average_ramps = True
            remove_first = remove_first_passed
            posdata = None

        ext = process_extension(
            hdu, readout_range, threshold=threshold,
            s2n=s2n, remove_first=remove_first,
            subtract_bias=subtract_bias, badmask=badmask,
            average_ramps=average_ramps, posdata=posdata,
            indpos_sigma=indpos_sigma)
        if ext is None:
            log.error("Failed to process extension %i: %s" %
                      (idx + 1, filename))
            return

        # append flux and stddev extensions
        hdul_new.append(ext[0])
        hdul_new.append(ext[1])

        # append posdata too if present
        if len(ext) > 2:
            hdul_new.append(ext[2])

    hdul_new[0].header['HISTORY'] = "Ramps fit; bad pixels flagged with NaN"
    if remove_first:
        hdul_new[0].header['HISTORY'] = "Ramps fit; removed first 2 ramps"
    hdul_new[0].header['HISTORY'] = \
        "Ramps fit; fitting on readouts [%i: %i]" % (
        readout_range[0] + 1, readout_range[1] - 1)

    if write:
        return write_hdul(hdul_new, outdir=outdir, overwrite=True)
    else:
        return hdul_new


def fit_ramps_wrap_helper(_, kwargs, filename):
    return fit_ramps(filename, **kwargs)


def wrap_fit_ramps(files, s2n=30, threshold=5, badpix_file=None,
                   outdir=None, remove_first=True, subtract_bias=True,
                   indpos_sigma=3.0, allow_errors=False,
                   write=False, jobs=None):
    """
    Wrapper for fit_ramps over multiple files.

    See `fit_ramps` for full description of reduction on a single file.

    Parameters
    ----------
    files : array_like of str
    s2n : float, optional
    threshold : float, optional
    badpix_file : str, optional
    outdir : str, optional
    remove_first : bool, optional
    subtract_bias : bool, optional
    indpos_sigma : float, optional
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

    clear_badpix_cache()

    kwargs = {
        's2n': s2n, 'threshold': threshold, 'badpix_file': badpix_file,
        'outdir': outdir, 'remove_first': remove_first,
        'subtract_bias': subtract_bias, 'indpos_sigma': indpos_sigma,
        'write': write}

    if DEBUG:  # pragma: no cover
        jobs = 1

    output = multitask(fit_ramps_wrap_helper, files, None, kwargs,
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

    clear_badpix_cache()

    return tuple(result)
