# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
from astropy.time import Time
import numba as nb
import numpy as np
from pandas import DataFrame

from sofia_redux.instruments.fifi_ls.make_header import make_header
from sofia_redux.toolkit.interpolate \
    import interp_1d_point_with_error as interp
from sofia_redux.toolkit.utilities \
    import (hdinsert, gethdul, write_hdul)


__all__ = ['classify_files', 'combine_extensions', 'combine_nods']


def _mjd(dateobs):
    """Get the MJD from a DATE-OBS."""
    try:
        mean_time = Time(dateobs).mjd
    except (ValueError, AttributeError):
        mean_time = 0
    return mean_time


def _read_exthdrs(hdul, key, default=0):
    """Read FIFI-LS extension headers."""
    result = []
    if len(hdul) <= 1:
        return result
    ngrating = hdul[0].header.get('NGRATING', 1)
    for idx in range(ngrating):
        name = f'FLUX_G{idx}'
        header = hdul[name].header
        result.append(header.get(key, default))
    return np.array(result)


def _from_hdul(hdul, key):
    """Read a header keyword from the PHU."""
    return hdul[0].header[key.upper().strip()]


def classify_files(filenames, offbeam=False):
    """
    Extract various properties of all files for subsequent combination.

    Parameters
    ----------
    filenames : array_like of str
        File paths to FITS files
    offbeam : bool, optional
        If True, swap 'A' nods with 'B' nods and the following
        associated keywords: DLAM_MAP <-> DLAM_OFF,
        DBET_MAP <->  DBET_OFF.

    Returns
    -------
    pandas.DataFrame
    """
    hduls = []
    fname_list = []
    for fname in filenames:
        hdul = gethdul(fname)
        if hdul is None:
            log.error("Invalid HDUList: %s" % fname)
            continue
        hduls.append(hdul)
        if not isinstance(fname, str):
            fname_list.append(_from_hdul(hdul, 'FILENAME'))
        else:
            fname_list.append(fname)
    filenames = fname_list
    n = len(filenames)
    if n == 0:
        log.error("No good files found.")
        return None

    keywords = ['nodstyle', 'detchan', 'channel', 'nodbeam', 'dlam_map',
                'dbet_map', 'dlam_off', 'dbet_off', 'date-obs']

    init = dict((key, [_from_hdul(hdul, key) for hdul in hduls])
                for key in keywords)
    init['mjd'] = [_mjd(dateobs) for dateobs in init['date-obs']]

    init['indpos'] = [_read_exthdrs(hdul, 'indpos', default=0)
                      for hdul in hduls]
    init['bglevl'] = [_read_exthdrs(hdul, 'bglevl_a', default=0)
                      for hdul in hduls]
    init['asymmetric'] = [x in ['ASYMMETRIC', 'C2NC2']
                          for x in init['nodstyle']]
    init['tsort'] = [0.0] * n
    init['sky'] = [False] * n  # calculate later
    init['hdul'] = hduls
    init['chdul'] = [None] * n
    init['combined'] = [np.full(len(x), False) for x in init['indpos']]
    init['outfile'] = [''] * n

    df = DataFrame(init, index=filenames)

    # If any files are asymmetric, treat them all as asymmetric
    if df['asymmetric'].any() and not df['asymmetric'].all():
        log.warning("Mismatched NODSTYLE. Will attempt to combine anyway.")

    # Drop any bad dates
    bad_dates = df[df['mjd'] == 0]
    if len(bad_dates) > 0:
        for name, row in bad_dates.iterrows():
            log.error('DATE-OBS in header is %s for %s' %
                      (row['date-obs'], name))
        df = df.drop(bad_dates.index)

    # If there's a good detchan value, use it in place of channel,
    # then set channel to either 1 (BLUE) or 0 (RED)
    valid_detchan = (df['detchan'] != 0) & (df['detchan'] != '0')
    df['channel'] = np.where(valid_detchan, df['detchan'], df['channel'])
    df['channel'] = df['channel'].apply(lambda x: 1 if x == 'BLUE' else 0)

    # update headers if offbeam is True
    if offbeam:
        # Switch A and B beams
        df['nodbeam'] = np.where(df['nodbeam'] != 'A', 'A', 'B')

        df = df.rename(index=str, columns={
            'dlam_map': 'dlam_off',
            'dbet_map': 'dbet_off',
            'dlam_off': 'dlam_map',
            'dbet_off': 'dbet_map'})

        for key in ['dlam_map', 'dlam_off', 'dbet_map', 'dbet_off', 'nodbeam']:
            df.apply(lambda x: hdinsert(
                x.hdul[0].header, key.upper(), x[key]), axis=1)

    # set on-source exptime to 0 for asym B beams and track as 'sky' files
    df['sky'] = df['asymmetric'] & (df['nodbeam'] != 'A')
    for hdul in df[df['sky']]['hdul'].values:
        if isinstance(hdul, fits.HDUList):
            hdul[0].header['EXPTIME'] = 0.0
            hdul[0].header['NEXP'] = 0

    return df


@nb.njit(fastmath={'nsz', 'ninf'}, cache=True)
def interp_b_nods(atime, btime, bdata, berr):   # pragma: no cover
    """
    Interpolate two B nods to the A time.

    Parameters
    ----------
    atime : array-like of float
        The UNIX time for each A nod sample.
    btime : array-like float
        Before and after time for the B nods.  Expected to have two
        elements; all `atime` values should fall between the first
        and second values.
    bdata : array-like of float
        2 x nw x ns B nod data to interpolate.
    berr : array-like of float
        2 x nw x ns B nod errors to interpolate.

    Returns
    -------
    bflux, bvar : array-like of float
        nw x ns interpolated B nod flux and variance.
    """
    nt = atime.size
    nn, nw, ns = bdata.shape

    bflux = np.empty((nt, nw, ns), dtype=nb.float64)
    bvar = np.empty((nt, nw, ns), dtype=nb.float64)

    for t in range(nt):
        for i in range(nw):
            for j in range(ns):
                # flux and error at this pixel
                bf = bdata[:, i, j]
                be = berr[:, i, j]

                # Interpolate B flux and error onto A time
                if np.any(np.isnan(bf)) or np.any(np.isnan(be)):
                    bflux[t, i, j] = np.nan
                    bvar[t, i, j] = np.nan
                else:
                    f, e = interp(btime, bf, be, atime[t])
                    bflux[t, i, j] = f
                    bvar[t, i, j] = e * e

    return bflux, bvar


def combine_extensions(df, b_nod_method='nearest'):
    """
    Find a B nod for each A nod.

    For asymmetric data, DLAM and DBET do not need to match,
    B data can be used more than once, and the B needs to be
    subtracted, rather than added (symmetric B nods are
    multiplied by -1 in chop_subtract)

    For the 'interpolate' option for B nod combination for most data, the
    time of interpolation is taken to be the middle of the observation,
    as determined by the FIFISTRT and EXPTIME keywords in the primary
    header.  For OTF data, the time is interpolated between RAMPSTRT
    and RAMPEND times in the extension header, for each ramp.

    Parameters
    ----------
    df : pandas.DataFrame
    b_nod_method : {'nearest', 'average', 'interpolate'}, optional
        Determines the method of combining the two nearest before
        and after B nods.

    Returns
    -------
    list of fits.HDUList
    """
    # check B method parameter
    if b_nod_method not in ['nearest', 'average', 'interpolate']:
        raise ValueError("Bad b_nod_method: should be 'nearest', "
                         "'average', or 'interpolate'.")
    get_two = b_nod_method != 'nearest'

    df.sort_values('mjd', inplace=True)
    blist = df[df['nodbeam'] == 'B']
    alist = df[df['nodbeam'] == 'A']

    # skip if no pairs available
    if len(blist) == 0:
        log.warning('No B nods found')
        return df
    elif len(alist) == 0:
        log.error('No A nods found')
        return df

    for afile, arow in alist.iterrows():

        asymmetric = arow['asymmetric']
        bselect = blist[(blist['channel'] == arow['channel'])
                        & (blist['asymmetric'] == asymmetric)]

        if not asymmetric:
            bselect = bselect[(bselect['dlam_map'] == arow['dlam_map'])
                              & (bselect['dbet_map'] == arow['dbet_map'])]

        # find closest matching B image in time
        if get_two and asymmetric:
            bselect['tsort'] = bselect['mjd'] - arow['mjd']
            after = (bselect[bselect['tsort'] > 0]).sort_values('tsort')
            bselect = (bselect[bselect['tsort'] <= 0]).sort_values(
                'tsort', ascending=False)
        else:
            bselect['tsort'] = abs(bselect['mjd'] - arow['mjd'])
            bselect = bselect.sort_values('tsort')
            after = None

        primehead, combined_hdul = None, None
        for aidx, apos in enumerate(arow['indpos']):
            bidx, bidx2 = [], []
            bfile, bfile2 = None, None
            brow, brow2 = None, None
            for bfile, brow in bselect.iterrows():
                bidx = brow['indpos'] == apos
                if not asymmetric:
                    bidx &= ~brow['combined']
                if np.any(bidx):
                    break

            if after is not None:
                for bfile2, brow2 in after.iterrows():
                    # always asymmetric
                    bidx2 = brow2['indpos'] == apos
                    if np.any(bidx2):
                        break
                if not np.any(bidx) and np.any(bidx2):
                    bidx = bidx2
                    brow = brow2
                    bfile = bfile2
                    bidx2 = []

            describe_a = f"A {os.path.basename(arow.name)} at ext{aidx + 1} " \
                         f"channel {arow['channel']} indpos {apos} " \
                         f"dlam {arow['dlam_map']} dbet {arow['dbet_map']}"
            if np.any(bidx):
                arow['combined'][aidx] = True
                a_fname = f'FLUX_G{aidx}'
                a_sname = f'STDDEV_G{aidx}'
                a_hdr = arow['hdul'][0].header

                bgidx = np.nonzero(bidx)[0][0]
                brow['combined'][bgidx] = True
                b_background = brow['bglevl'][bgidx]
                b_fname = f'FLUX_G{bgidx}'
                b_sname = f'STDDEV_G{bgidx}'
                b_flux = brow['hdul'][b_fname].data
                b_var = brow['hdul'][b_sname].data ** 2
                b_hdr = brow['hdul'][0].header

                # check for offbeam with OTF mode: B nods
                # can't have an extra dimension
                if b_flux.ndim > 2:
                    msg = 'Offbeam option is not available for OTF mode'
                    log.error(msg)
                    raise ValueError(msg)

                combine_headers = [a_hdr, b_hdr]

                # check for a second B nod: if not found, will do
                # 'nearest' for this A file
                if np.any(bidx2):
                    # add in header for combination
                    b2_hdr = brow2['hdul'][0].header
                    combine_headers.append(b2_hdr)

                    # get A and B times
                    try:
                        # unix time at middle of exposure
                        atime = a_hdr['START'] \
                            + a_hdr['FIFISTRT'] * a_hdr['ALPHA'] \
                            + a_hdr['EXPTIME'] / 2.0
                        btime1 = b_hdr['START'] \
                            + b_hdr['FIFISTRT'] * b_hdr['ALPHA'] \
                            + b_hdr['EXPTIME'] / 2.0
                        btime2 = b2_hdr['START'] \
                            + b2_hdr['FIFISTRT'] * b2_hdr['ALPHA'] \
                            + b2_hdr['EXPTIME'] / 2.0
                    except KeyError:
                        raise ValueError('Missing START, FIFISTRT, ALPHA, '
                                         'or EXPTIME keys in headers.')

                    # get index for second B row
                    bgidx2 = np.nonzero(bidx2)[0][0]
                    brow2['combined'][bgidx2] = True

                    if b_nod_method == 'interpolate':
                        # debug message
                        msg = f'Interpolating B {bfile} at {btime1} ' \
                              f'and {bfile2} at {btime2} ' \
                              f'to A time {atime} and subbing from '

                        # interpolate background to header atime
                        b_background = \
                            np.interp(atime, [btime1, btime2],
                                      [b_background, brow2['bglevl'][bgidx2]])

                        # UNIX time is a range of values for OTF data:
                        # retrieve from RAMPSTRT and RAMPEND keys
                        a_hdu_hdr = arow['hdul'][a_fname].header
                        a_shape = arow['hdul'][a_fname].data.shape
                        if len(a_shape) == 3 \
                                and 'RAMPSTRT' in a_hdu_hdr \
                                and 'RAMPEND' in a_hdu_hdr:
                            rampstart = a_hdu_hdr['RAMPSTRT']
                            rampend = a_hdu_hdr['RAMPEND']
                            nramp = a_shape[0]
                            ramp_incr = (rampend - rampstart) / (nramp - 1)
                            atime = np.full(nramp, rampstart)
                            atime += np.arange(nramp, dtype=float) * ramp_incr
                        else:
                            atime = np.array([atime])
                        btime = np.array([btime1, btime2])

                        # interpolate B flux to A time
                        b_fname = f'FLUX_G{bgidx2}'
                        b_sname = f'STDDEV_G{bgidx2}'
                        bdata = np.array([b_flux, brow2['hdul'][b_fname].data])
                        berr = np.array([np.sqrt(b_var),
                                         brow2['hdul'][b_sname].data])
                        b_flux, b_var = \
                            interp_b_nods(atime, btime, bdata, berr)

                        # reshape if there was only one atime
                        if atime.size == 1:
                            b_flux = b_flux[0]
                            b_var = b_var[0]
                    else:
                        # debug message
                        msg = f'Averaging B {bfile} and {bfile2} ' \
                              f'and subbing from '

                        # average flux
                        b_flux += brow2['hdul'][b_fname].data
                        b_flux /= 2.

                        # propagate variance
                        b_var += brow2['hdul'][b_sname].data ** 2
                        b_var /= 4.

                        # average background
                        b_background += brow2['bglevl'][bgidx2]
                        b_background /= 2.

                else:
                    if asymmetric:
                        msg = f'Subbing B {os.path.basename(brow.name)} from '
                    else:
                        msg = f'Adding B {os.path.basename(brow.name)} to '

                log.debug(msg + describe_a)

                # Note: in the OTF case, A data is a 3D cube with
                # ramps x spexels x spaxels, and B data is a
                # 2D array of spexels x spaxels.  The B data is
                # subtracted at each ramp.
                # For other modes, A and B are both spexels x spaxels.

                flux = arow['hdul'][a_fname].data
                stddev = arow['hdul'][a_sname].data ** 2 + b_var
                if asymmetric:
                    flux -= b_flux
                else:
                    flux += b_flux
                    # divide by two for doubled source
                    flux /= 2
                    stddev /= 4
                stddev = np.sqrt(stddev)

                if combined_hdul is None:
                    primehead = make_header(combine_headers)
                    primehead['HISTORY'] = 'Nods combined'
                    hdinsert(primehead, 'PRODTYPE', 'nod_combined')
                    outfile, _ = os.path.splitext(os.path.basename(afile))
                    outfile = '_'.join(outfile.split('_')[:-2])
                    outfile += '_NCM_%s.fits' % primehead.get('FILENUM')
                    df.loc[afile, 'outfile'] = outfile
                    hdinsert(primehead, 'FILENAME', outfile)
                    combined_hdul = fits.HDUList(
                        fits.PrimaryHDU(header=primehead))

                exthead = arow['hdul'][a_fname].header
                hdinsert(exthead, 'BGLEVL_B', b_background,
                         comment='BG level nod B (ADU/s)')
                combined_hdul.append(fits.ImageHDU(flux, header=exthead,
                                                   name=a_fname))
                combined_hdul.append(fits.ImageHDU(stddev, header=exthead,
                                                   name=a_sname))

                # add in scanpos table from A nod if present
                a_pname = f'SCANPOS_G{aidx}'
                if a_pname in arow['hdul']:
                    combined_hdul.append(arow['hdul'][a_pname].copy())
            else:
                msg = "No matching B found for "
                log.debug(msg + describe_a)

        if combined_hdul is not None:
            df.at[afile, 'chdul'] = combined_hdul

    return df


def combine_nods(filenames, offbeam=False, b_nod_method='nearest',
                 outdir=None, write=False):
    """
    Combine nods of ramp-fitted, chop-subtracted data.

    Writes a single FITS file to disk for each A nod found.  Each
    HDU list contains n_san binary table extensions, each containing
    DATA and STDDEV data cubes, each 5x5x18.  The output filename is
    created from the input filename, with the suffix 'CSB', 'RP0' or
    'RP1' replaced with 'NCM', and with input file numbers numbers
    concatenated.  Unless specified, the output directory is the same
    as the input files.

    Input files should have been generated by `subtract_chops`, or
    `fit_ramps` (for total power mode, which has no chops).

    The procedure is:

        1. Read header information from each extension in each of the
           input files, making lists of A data and B data, with relevant
           metadata (dither) position, date/time observed (DATE-OBS),
           inductosyn position, channel, nod style).

        2. Loop though all A data to find matching B data

            a. asymmetric nod style: find closest B nod in time with the
            same channel and inductosyn position.  Dither position does
            not have to match, B data can be used more than once, and
            data must be subtracted rather than added.

            b. symmetric nod style: find closest B nod in time with the
            same channel, inductosyn position, and dither position. Each
            B nod can only be used once, since it contains a source
            observation, and data must be added rather than subtracted.

        3. After addition or subtraction, create a FITS file and write
        results to disk.


    Parameters
    ----------
    filenames : array_like of str
        File paths to the data to be combined
    offbeam : bool, optional
        If True, swap 'A' nods with 'B' nods and the following
        associated keywords: DLAM_MAP <-> DLAM_OFF,
        DBET_MAP <->  DBET_OFF. This option cannot be used with
        OTF-mode A nods.
    b_nod_method : {'nearest', 'average', 'interpolate'}, optional
        For asymmetric, data this option controls how the nearest B nods
        are combined. The 'nearest' option takes only the nearest B nod
        in time.  The 'average' option averages the nearest before and
        after B nods.  The 'interpolate' option linearly interpolates the
        nearest before and after B nods to the time of the A data.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  If False, return the HDUL.

    Returns
    -------
    pandas.DataFrame
        The output pandas dataframe contains a huge variety of
        information indexed by original filename.  The combined
        A-B FITS data are located in the 'chdul' column.  Note that
        only A nod files contain combined data in this 'chdul'
        column.  For example, in order to extract combined FITS
        data, one could issue the command::

            df = combine_nods(filenames)
            combined_hduls = df[df['nodbeam'] == 'A']['chdul']

        In order to extract rows from the dataframe that were not
        combined issue the command::

            not_combined = df[(df['nodbeam'] == 'A') & (df['chdul'] == None)]

        files are considered 'combined' if at least one A extension was
        combined for an A-nod file.  A true signifier of whether an
        extension was combined (both A and B nod files) can be found in the
        'combined' column as a list of bools, one for each extension.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    if not hasattr(filenames, '__len__'):
        log.error("Invalid input files type (%s)" % repr(filenames))
        return

    if isinstance(outdir, str):
        if not os.path.isdir(outdir):
            log.error("Output directory %s does not exist" % outdir)
            return
    df = classify_files(filenames, offbeam=offbeam)
    if df is None:
        log.error("Problem in file classification")
        return

    df = combine_extensions(df, b_nod_method=b_nod_method)

    for filename, row in df[df['nodbeam'] == 'A'].iterrows():

        if outdir is not None:
            outdir = str(outdir)
        else:
            outdir = os.path.dirname(filename)

        if write and row['chdul'] is not None:
            write_hdul(row['chdul'], outdir=outdir, overwrite=True)
        if row['outfile'] is not None:
            df.at[filename, 'outfile'] = os.path.join(
                outdir, os.path.basename(row['outfile']))

    return df
