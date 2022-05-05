# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np
import psutil
from scipy.interpolate import Rbf

from sofia_redux.instruments.fifi_ls.get_resolution \
    import get_resolution, clear_resolution_cache
from sofia_redux.instruments.fifi_ls.get_response \
    import get_response, clear_response_cache
from sofia_redux.instruments.fifi_ls.make_header \
    import make_header
from sofia_redux.toolkit.utilities \
    import gethdul, hdinsert, write_hdul, stack
from sofia_redux.toolkit.resampling.resample import Resample
from sofia_redux.spectroscopy.smoothres import smoothres


__all__ = ['combine_files', 'get_grid_info', 'generate_exposure_map',
           'rbf_mean_combine', 'local_surface_fit', 'make_hdul',
           'resample']


def combine_files(filenames):
    """
    Combine all files into a single dataset.

    For OTF mode, the input data for each file contains
    multiple samples, each with their own X and Y coordinates.
    Each sample is handled separately, as if it came from a
    different input file.

    Parameters
    ----------
    filenames : array_like of str
        File paths to FITS data to be resampled.

    Returns
    -------
    dict
    """
    # process the first filename
    nfiles = len(filenames)
    obslam, obsbet, angle = np.zeros((3, nfiles))
    interpolate = True

    header_list, combined, fnames = [], {}, []
    waves, xs, ys, fluxs, errors = [], [], [], [], []
    ufluxs, uerrs, ulams = [], [], []
    atran = None

    log.info(f'Reading {nfiles} files')
    otf_mode = False
    for i, filename in enumerate(filenames):
        hdul = gethdul(filename, verbose=True)
        if hdul is None:
            msg = 'Could not read file: {filename}'
            log.error(msg)
            raise ValueError(msg)
        header = hdul[0].header
        header_list.append(header)
        obslam[i] = header.get('OBSLAM', 0)
        obsbet[i] = header.get('OBSBET', 0)
        angle[i] = header.get('SKY_ANGL', 0)
        if not isinstance(filename, str):
            fnames.append(header.get('FILENAME', 'UNKNOWN'))
        else:
            fnames.append(filename)

        if interpolate:
            # Interpolate if all dither values are zero, otherwise
            # use the resampling algorithm
            interpolate &= header.get('DBET_MAP', 0) == 0
            interpolate &= header.get('DLAM_MAP', 0) == 0

        # if otf data, will treat each plane as a separate file
        dshape = hdul['FLUX'].data.shape
        otf_mode = len(dshape) > 2
        if otf_mode:
            # don't interpolate by default for OTF_MODE
            interpolate = False

            for plane in range(dshape[0]):
                waves.append(hdul['LAMBDA'].data)
                xs.append(hdul['XS'].data[plane])
                ys.append(hdul['YS'].data[plane])
                fluxs.append(hdul['FLUX'].data[plane])
                errors.append(hdul['STDDEV'].data[plane])
                if 'UNCORRECTED_FLUX' in hdul:
                    ufluxs.append(hdul['UNCORRECTED_FLUX'].data[plane])
                    uerrs.append(hdul['UNCORRECTED_STDDEV'].data[plane])
                if 'UNCORRECTED_LAMBDA' in hdul:
                    ulams.append(hdul['UNCORRECTED_LAMBDA'].data)
        else:
            waves.append(hdul['LAMBDA'].data)
            xs.append(hdul['XS'].data)
            ys.append(hdul['YS'].data)
            fluxs.append(hdul['FLUX'].data)
            errors.append(hdul['STDDEV'].data)
            if 'UNCORRECTED_FLUX' in hdul:
                ufluxs.append(hdul['UNCORRECTED_FLUX'].data)
                uerrs.append(hdul['UNCORRECTED_STDDEV'].data)
            if 'UNCORRECTED_LAMBDA' in hdul:
                ulams.append(hdul['UNCORRECTED_LAMBDA'].data)

        if atran is None and 'UNSMOOTHED_ATRAN' in hdul:
            atran = hdul['UNSMOOTHED_ATRAN'].data

        hdul.close()

    if len(filenames) > 1 or otf_mode:
        log.info(f'Combining and rotating relative to the first file '
                 f'({os.path.basename(fnames[0])}).')
    else:
        interpolate = True

    a = -1.0 * np.radians(angle)
    cosbet = np.cos(np.radians(obsbet[0]))
    da = np.radians(angle[0] - angle)
    betoff = 3600.0 * (obsbet - obsbet[0])
    lamoff = 3600.0 * cosbet * (obslam[0] - obslam)

    # update headers
    for update in np.where(betoff != 0)[0]:
        hdinsert(header_list[update], 'DBET_MAP',
                 header_list[update].get('DBET_MAP', 0) + betoff[update])

    for update in np.where(lamoff != 0)[0]:
        hdinsert(header_list[update], 'DLAM_MAP',
                 header_list[update].get('DLAM_MAP', 0) - lamoff[update])

    idx = abs(a) > 1e-6
    dx = lamoff[idx] * np.cos(a[idx]) - betoff[idx] * np.sin(a[idx])
    dy = lamoff[idx] * np.sin(a[idx]) + betoff[idx] * np.cos(a[idx])
    lamoff[idx] = dx
    betoff[idx] = dy

    for i, (x, y, dx, dy) in enumerate(zip(xs, ys, lamoff, betoff)):
        if i == 0:
            continue
        x += dx
        y += dy
        ra = da[i]
        if abs(ra) <= 1e-6:
            continue

        cda, sda = np.cos(ra), np.sin(ra)
        xr = x * cda - y * sda
        ry = x * sda + y * cda
        x[:] = xr
        y[:] = ry

    # update to the angle of the first header
    for update in np.where(idx)[0]:
        hdinsert(header_list[update], 'SKY_ANGL', angle[0])

    combined['PRIMEHEAD'] = make_header(header_list)
    combined['method'] = 'interpolate' if interpolate else 'resample'
    combined['X'] = np.array(xs)
    combined['Y'] = np.array(ys)
    combined['FLUX'] = np.array(fluxs)
    combined['ERROR'] = np.array(errors)
    combined['WAVELENGTH'] = np.array(waves)
    if len(fluxs) == len(ufluxs):
        combined['UNCORRECTED_FLUX'] = np.array(ufluxs)
        combined['UNCORRECTED_ERROR'] = np.array(uerrs)
    if len(waves) == len(ulams):
        combined['UNCORRECTED_LAMBDA'] = np.array(ulams)
    if atran is not None:
        combined['UNSMOOTHED_TRANSMISSION'] = atran
    return combined


def get_grid_info(combined, xrange=None, yrange=None, wrange=None,
                  oversample=None, spatial_size=None, spectral_size=None):
    """
    Get output coordinate system and useful parameters.

    Parameters
    ----------
    combined : dict
        Dictionary containing combined data
    xrange : array_like of float, optional
        (Min, max) x-value of new grid.  If not provided, minimum and
        maximum input values will be used.
    yrange : array_like of float, optional
        (Min, max) y-value of new grid.  If not provided, minimum and
        maximum input values will be used.
    wrange : array_like of float, optional
        (Min, max) wavelength value of new grid.  If not provided, minimum
        and maximum input values will be used.
    oversample : array_like of int or float, optional
        Number of pixels to sample mean FWHM with, in the (spatial, spectral)
        dimensions. Default is (5.0, 8.0).
    spatial_size : float, optional
        Output pixel size, in the spatial dimensions.
        Units are arcsec.  If specified, the corresponding oversample
        parameter will be ignored.
    spectral_size : float, optional
        Output pixel size, in the spectral dimension.
        Units are um.  If specified, the corresponding oversample
        parameter will be ignored.

    Returns
    -------
    dict
    """
    primehead = combined['PRIMEHEAD']

    # get full set of wavelengths for range testing
    lam = np.hstack([w.ravel() for w in combined['WAVELENGTH']])
    if 'UNCORRECTED_LAMBDA' in combined:
        ulam = np.hstack([w.ravel() for w in combined['UNCORRECTED_LAMBDA']])
        waves = [lam, ulam]
    else:
        ulam = None
        waves = lam
    xs = np.hstack([x.ravel() for x in combined['X']])
    ys = np.hstack([y.ravel() for y in combined['Y']])

    # get grid ranges
    xmin = np.nanmin(xs) if xrange is None else xrange[0]
    xmax = np.nanmax(xs) if xrange is None else xrange[1]
    ymin = np.nanmin(ys) if yrange is None else yrange[0]
    ymax = np.nanmax(ys) if yrange is None else yrange[1]
    wmin = np.nanmin(waves) if wrange is None else wrange[0]
    wmax = np.nanmax(waves) if wrange is None else wrange[1]

    xrange, yrange, wrange = xmax - xmin, ymax - ymin, wmax - wmin

    # get oversample parameter
    if oversample is None:
        xy_oversample, w_oversample = 5.0, 8.0
    else:
        xy_oversample, w_oversample = oversample

    log.info(f'Overall wave min/max (um): {wmin:.5f} {wmax:.5f}')
    log.info(f'Overall X min/max (arcsec): {xmin:.2f} {xmax:.2f}')
    log.info(f'Overall Y min/max (arcsec): {ymin:.2f} {ymax:.2f}')

    # Begin with spectral scalings
    wmid = (wmin + wmax) / 2
    resolution = get_resolution(primehead, wmean=wmid)
    wave_fwhm = wmid / resolution
    if spectral_size is not None:
        delta_wave = spectral_size
        w_oversample = wave_fwhm / delta_wave
    else:
        delta_wave = wave_fwhm / w_oversample
    nw = int(np.round((wmax - wmin) / delta_wave) + 1)
    log.info(f'Average spectral FWHM: {wave_fwhm:.5f} um')
    log.info(f'Output spectral pixel scale: {delta_wave:.5f} um')
    log.info(f'Spectral oversample: {w_oversample:.2f} pixels')

    # Spatial scalings
    xy_fwhm = get_resolution(
        primehead, spatial=True,
        wmean=float(np.nanmean(lam)))

    # pixel size
    if str(primehead['CHANNEL']).upper() == 'RED':
        pix_size = 3.0 * primehead['PLATSCAL']
    else:
        pix_size = 1.5 * primehead['PLATSCAL']
    if spatial_size is not None:
        delta_xy = spatial_size
        xy_oversample = xy_fwhm / delta_xy
    else:
        delta_xy = xy_fwhm / xy_oversample
    nx = int(np.round((xrange / delta_xy) + 1))
    ny = int(np.round((yrange / delta_xy) + 1))

    log.info(f'Pixel size for channel: {pix_size:.2f} arcsec')
    log.info(f'Average spatial FWHM for channel: {xy_fwhm:.2f} arcsec')
    log.info(f'Output spatial pixel scale: {delta_xy:.2f} arcsec/pix')
    log.info(f'Spatial oversample: {xy_oversample:.2f} pixels')

    # Rotation angle before calibration.  If not rotated at calibration,
    # detector rotation is zero.
    det_angle = 0.0 if primehead.get('SKY_ANGL', 0) != 0 else \
        -np.radians(primehead.get('DET_ANGL', 0.0))

    ni = xs.size
    wout = np.arange(nw, dtype=float) * delta_wave + wmin
    xout = np.arange(nx, dtype=float) * delta_xy + xmin
    yout = np.arange(ny, dtype=float) * delta_xy + ymin

    log.info('')
    log.info(f'Output grid size (nw, ny, nx): {nw} x {ny} x {nx}')
    if nx > 2048 or ny > 2048:
        log.error('Spatial range too large.')
        return

    coordinates = stack(xs, ys, lam)
    if ulam is None:
        uncor_coords = None
    else:
        uncor_coords = stack(xs, ys, ulam)
    grid = xout, yout, wout

    return {
        'wmin': wmin, 'wmax': wmax,
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'shape': (ni, nw, ny, nx),
        'wout': wout, 'xout': xout, 'yout': yout,
        'wrange': wrange, 'xrange': xrange, 'yrange': yrange,
        'delta': (delta_wave, delta_xy, delta_xy),
        'oversample': (w_oversample, xy_oversample, xy_oversample),
        'wave_fwhm': wave_fwhm, 'xy_fwhm': xy_fwhm,
        'resolution': resolution,
        'pix_size': pix_size, 'det_angle': det_angle,
        'coordinates': coordinates,
        'uncorrected_coordinates': uncor_coords,
        'grid': grid}


def generate_exposure_map(combined, grid_info, get_good=False):
    """
    Create the exposure map from combined files.

    Parameters
    ----------
    combined : dict
        Dictionary containing combined data
    grid_info : dict
        Dictionary containing output grid coordinates and other necessary
        information.
    get_good : bool, optional
        If set, a list of good pixel arrays will be returned
        instead of an exposure map

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        3D exposure map array, by default.  If get_good is set,
        a list of boolean arrays is returned instead, representing
        the good pixel mask for each input file.
    """
    # get x and y and correct by the det angle to get
    # min and max range in detector CS
    a = grid_info['det_angle']
    if not np.allclose(a, 0):
        cosa, sina = np.cos(a), np.sin(a)
        detx = (combined['X'] * cosa) - (combined['Y'] * sina)
        dety = (combined['X'] * sina) + (combined['Y'] * cosa)
    else:
        cosa, sina = 1.0, 0.0
        detx, dety = combined['X'], combined['Y']
    # wavelengths are okay as is
    detw = combined['WAVELENGTH']

    # get grid info to index into
    dw, dy, dx = grid_info['delta']
    wavesize = dw / 2
    pixsize = grid_info['pix_size'] / 2
    wmin = grid_info['wmin']
    ymin = grid_info['ymin']
    xmin = grid_info['xmin']
    nw, ny, nx = grid_info['shape'][1:]

    # loop over files to get exposure for each one
    if get_good:
        exposure = []
    else:
        exposure = np.zeros((nw, ny, nx), dtype=int)
    for i in range(len(detx)):
        detx_min = detx[i].min() - pixsize
        detx_max = detx[i].max() + pixsize
        dety_min = dety[i].min() - pixsize
        dety_max = dety[i].max() + pixsize
        detw_min = detw[i].min() - wavesize
        detw_max = detw[i].max() + wavesize

        # use wavelengths directly
        wl = int(np.floor((detw_min - wmin) / dw))
        wh = int(np.ceil((detw_max - wmin) / dw)) + 1
        wl = 0 if wl < 0 else wl
        wh = nw if wh > nw else wh

        # unrotate square corners between min and max to get grid locations
        # in clockwise order
        cx, cy = [], []
        cx.append(detx_min * cosa + dety_min * sina)
        cy.append(-detx_min * sina + dety_min * cosa)
        cx.append(detx_min * cosa + dety_max * sina)
        cy.append(-detx_min * sina + dety_max * cosa)
        cx.append(detx_max * cosa + dety_max * sina)
        cy.append(-detx_max * sina + dety_max * cosa)
        cx.append(detx_max * cosa + dety_min * sina)
        cy.append(-detx_max * sina + dety_min * cosa)

        # convert to pixel indices
        idx = (np.array(cx) - xmin) / dx
        idy = (np.array(cy) - ymin) / dy

        # make a square large enough to contain the FOV
        xl = int(np.floor(idx.min()))
        xh = int(np.ceil(idx.max())) + 1
        xl = 0 if xl < 0 else xl
        xh = nx if xh > nx else xh

        yl = int(np.floor(idy.min()))
        yh = int(np.ceil(idy.max())) + 1
        yl = 0 if yl < 0 else yl
        yh = ny if yh > ny else yh

        square = np.zeros((yh - yl, xh - xl), dtype=int)

        # set values for FOV between corner vertices
        fov = np.full(square.shape, True)
        fov_y, fov_x = np.indices(square.shape)
        for j in range(len(idx)):
            p1x = idx[j - 1] - xl
            p1y = idy[j - 1] - yl
            p2x = idx[j] - xl
            p2y = idy[j] - yl

            # check for vertical line
            if p2x == p1x:
                # mark data to the correct side of the line
                max_x = p1x
                sign = np.sign(p2y - p1y)
                fov &= (fov_x * sign) >= (max_x * sign)
            else:
                # otherwise: mark area under the line between
                # previous vertex and current
                # (or above, as appropriate)
                m = (p2y - p1y) / (p2x - p1x)
                max_y = m * (fov_x - p1x) + p1y
                sign = np.sign(p2x - p1x)

                fov &= (fov_y * sign) <= (max_y * sign)

        square[fov] = 1

        if get_good:
            exposure.append((square.astype(bool), (xl, xh), (yl, yh)))
        else:
            exposure[wl:wh, yl:yh, xl:xh] += square[None, :, :]

    # For accounting purposes, multiply the exposure map by 2 for NMC
    if not get_good:
        nodstyle = combined['PRIMEHEAD'].get('NODSTYLE', 'UNK').upper().strip()
        if nodstyle in ['SYMMETRIC', 'NMC']:
            exposure *= 2

    return exposure


def rbf_mean_combine(combined, grid_info, window=None,
                     error_weighting=True, smoothing=None,
                     order=0, robust=None, neg_threshold=None,
                     fit_threshold=None, edge_threshold=None,
                     skip_uncorrected=False, **kwargs):
    """
    Combines multiple datasets using radial basis functions and mean combine.

    The combined data is stored in the combined dictionary, in the
    GRID_FLUX, GRID_ERROR, and GRID_COUNTS keys.

    Parameters
    ----------
    combined : dict
        Dictionary containing combined data
    grid_info : dict
        Dictionary containing output grid coordinates and other necessary
        information.
    window : float or array_like of float, optional
        Region to consider for local polynomial fits, given as a factor
        of the mean FWHM, in the wavelength dimension. If three elements
        are passed, the third will be used.  Default is
        0.5.
    error_weighting : bool, optional
        If True (default), weight polynomial fitting by the `error` values
        of each sample.
    smoothing : float or array_like of float, optional
        Radius over which to smooth the input data, given as a factor of
        the mean FWHM, in the wavelength dimension. If three elements are
        passed, the third will be used. Default is 0.25.
    order : int or array_like of int, optional
        Maximum order of local polynomial fits.
    robust : float, optional
        Rejection threshold for input data to local fits, given as a
        factor of the standard deviation.
    neg_threshold : float, optional
        First-pass rejection threshold for negative input data, given
        as a factor of the standard deviation; if None or <= 0,
        first-pass rejection will not be performed.
    fit_threshold : float, optional
        Rejection threshold for output fit values, given as a factor
        of the standard deviation in the input data.  If exceeded,
        weighted mean value is used in place of fit.
    edge_threshold : float or array_like or float
        If set to a value > 0 and < 1, edges of the fit will be masked out
        according to `edge_algorithm`. Values close to zero will result in
        a high degree of edge clipping, while values close to 1 ckip edges
        to a lesser extent. The clipping threshold is a fraction of window.
        An array may be used to specify values for each dimension.
    skip_uncorrected: bool, optional
        If set, the uncorrected flux cube will not be computed, even
        if present in the input data.  This option is primarily intended
        for testing or quicklook, when the full data product is not needed.
    kwargs : dict, optional
        Optional keyword arguments to pass into `scipy.interpolate.Rbf`.
        Please see the options here.  By default, the weighting function
        is a multi-quadratic sqrt(r/epsilon)**2 + 1) rather than the
        previous version inverse distance weighting scheme.
    """
    log.info('')
    log.info('Resampling wavelengths with polynomial fits.')
    log.info('Interpolating spatial coordinates with radial basis functions.')

    shape = grid_info['shape'][1:]
    flux = np.zeros(shape)
    std = np.zeros(shape)
    counts = np.zeros(shape)

    do_uncor = 'UNCORRECTED_FLUX' in combined and not skip_uncorrected

    if do_uncor:
        log.debug('Resampling uncorrected cube alongside corrected cube')
        uflux = np.zeros(shape)
        ustd = np.zeros(shape)
        ucounts = np.zeros(shape)
    else:
        uflux = ustd = ucounts = None

    # check parameters -- may be passed with all three coordinates.
    # If so, assume wavelength is the last one
    if hasattr(window, '__len__') and len(window) == 3:
        window = window[2]
    if hasattr(smoothing, '__len__') and len(smoothing) == 3:
        smoothing = smoothing[2]
    if hasattr(order, '__len__') and len(order) == 3:
        order = order[2]

    if order != 0:
        log.warning('Setting wavelength order to 0 for stability.')
        order = 0

    if window is None:
        window = 0.5
    if smoothing is None:
        smoothing = 0.25
    if edge_threshold is None:
        edge_threshold = 0.0

    fit_wdw = window * grid_info['wave_fwhm']
    smoothing_wdw = smoothing * grid_info['wave_fwhm']
    log.info(f'Fit window: {fit_wdw:.5f} um')
    log.info(f'Gaussian width of smoothing function: {smoothing_wdw:.5f} um')

    # minimum points in a wavelength slice to attempt to interpolate
    minpoints = 10

    # output grid
    xg, yg, wout = grid_info['grid']
    nx = xg.size
    ny = yg.size
    nw = wout.size
    xgrid = np.resize(xg, (ny, nx))
    ygrid = np.resize(yg, (nx, ny)).T

    # exposure map for grid counts
    good_grid = generate_exposure_map(combined, grid_info, get_good=True)

    # loop over files
    for filei, grididx in enumerate(good_grid):
        square, xrange, yrange = grididx

        if not square.any():
            log.debug('No good values in file {}'.format(filei))
            continue
        xout = xgrid[yrange[0]:yrange[1], xrange[0]:xrange[1]][square]
        yout = ygrid[yrange[0]:yrange[1], xrange[0]:xrange[1]][square]

        # resample to grid wavelengths
        ws = combined['WAVELENGTH'][filei]
        if do_uncor and 'UNCORRECTED_LAMBDA' in combined:
            uws = combined['UNCORRECTED_LAMBDA'][filei]
        else:
            uws = ws

        shape = (nw,) + ws.shape[1:]
        iflux, istd = np.empty(shape), np.empty(shape)
        if do_uncor:
            iuflux, iustd = np.empty(shape), np.empty(shape)
        else:
            iuflux, iustd = None, None

        # loop over spaxels to resample spexels
        for i in range(shape[1]):
            # all wavelengths, y=i, x=j
            s = slice(None), i
            try:
                resampler = Resample(
                    ws[s], combined['FLUX'][filei][s],
                    error=combined['ERROR'][filei][s],
                    window=fit_wdw, order=order,
                    robust=robust, negthresh=neg_threshold)

                iflux[:, i], istd[:, i] = resampler(
                    wout, smoothing=smoothing_wdw,
                    fit_threshold=fit_threshold,
                    edge_threshold=edge_threshold,
                    edge_algorithm='distribution',
                    get_error=True, error_weighting=error_weighting)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                log.debug('Math error in resampler at '
                          'spaxel {} for file {}'.format(i, filei))
                iflux[:, i], istd[:, i] = np.nan, np.nan

            if do_uncor:
                try:
                    resampler = Resample(
                        uws[s], combined['UNCORRECTED_FLUX'][filei][s],
                        error=combined['UNCORRECTED_ERROR'][filei][s],
                        window=fit_wdw, order=order, robust=robust,
                        negthresh=neg_threshold)
                    iuflux[:, i], iustd[:, i] = resampler(
                        wout, smoothing=smoothing_wdw,
                        fit_threshold=fit_threshold,
                        edge_threshold=edge_threshold,
                        edge_algorithm='distribution',
                        get_error=True, error_weighting=error_weighting)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    log.debug('Math error in resampler at '
                              'spaxel {} for file {}'.format(i, filei))
                    iuflux[:, i], iustd[:, i] = np.nan, np.nan

        # x and y coordinates for resampled fluxes --
        # take from first spexel
        x = combined['X'][filei][0, :]
        y = combined['Y'][filei][0, :]

        # check for useful data
        fidx = np.isfinite(iflux) & np.isfinite(istd)
        waveok = np.sum(fidx, axis=1) > minpoints
        if do_uncor:
            ufidx = np.isfinite(iuflux) & np.isfinite(iustd)
            uwaveok = np.sum(ufidx, axis=1) > minpoints
        else:
            ufidx = None
            uwaveok = np.full_like(waveok, False)

        for wavei in range(nw):
            if waveok[wavei]:
                idx = fidx[wavei]

                rbf = Rbf(x[idx], y[idx], iflux[wavei][idx], **kwargs)
                new_flux = np.zeros(square.shape)
                new_flux[square] = rbf(xout, yout)
                flux[wavei, yrange[0]:yrange[1],
                     xrange[0]:xrange[1]] += new_flux

                rbf = Rbf(x[idx], y[idx], istd[wavei][idx], **kwargs)
                new_std = np.zeros(square.shape)
                new_std[square] = rbf(xout, yout) ** 2
                std[wavei, yrange[0]:yrange[1],
                    xrange[0]:xrange[1]] += new_std

                counts[wavei, yrange[0]:yrange[1],
                       xrange[0]:xrange[1]] += square

            if do_uncor:
                if uwaveok[wavei]:
                    idx = ufidx[wavei]

                    rbf = Rbf(x[idx], y[idx], iuflux[wavei][idx], **kwargs)
                    new_flux = np.zeros(square.shape)
                    new_flux[square] = rbf(xout, yout)
                    uflux[wavei, yrange[0]:yrange[1],
                          xrange[0]:xrange[1]] += new_flux

                    rbf = Rbf(x[idx], y[idx], iustd[wavei][idx], **kwargs)
                    new_std = np.zeros(square.shape)
                    new_std[square] = rbf(xout, yout) ** 2
                    ustd[wavei, yrange[0]:yrange[1],
                         xrange[0]:xrange[1]] += new_std

                    ucounts[wavei, yrange[0]:yrange[1],
                            xrange[0]:xrange[1]] += square

    # average, set zero counts to nan
    log.info('Mean-combining all resampled cubes')
    exposure = ucounts.copy() if do_uncor else counts.copy()

    # For accounting purposes, multiply the exposure map by 2 for NMC
    nodstyle = combined['PRIMEHEAD'].get('NODSTYLE', 'UNK').upper().strip()
    if nodstyle in ['SYMMETRIC', 'NMC']:
        exposure *= 2

    nzi = counts > 0
    flux[nzi] /= counts[nzi]
    std[nzi] = np.sqrt(std[nzi]) / counts[nzi]
    flux[~nzi] = np.nan
    std[~nzi] = np.nan

    # correct flux for pixel size change
    factor = (grid_info['delta'][1] / grid_info['pix_size']) ** 2
    log.info(f'Flux correction factor: {factor:.5f}')
    correction = factor

    combined['GRID_FLUX'] = flux * correction
    combined['GRID_ERROR'] = std * correction
    combined['GRID_COUNTS'] = exposure

    # warn if all NaN
    if np.all(np.isnan(flux)):
        log.warning('Primary flux cube contains only NaN values.')

    if do_uncor:
        nzi = ucounts > 0
        uflux[nzi] /= ucounts[nzi]
        ustd[nzi] = np.sqrt(ustd[nzi]) / ucounts[nzi]
        uflux[~nzi] = np.nan
        ustd[~nzi] = np.nan
        combined['GRID_UNCORRECTED_FLUX'] = uflux * correction
        combined['GRID_UNCORRECTED_ERROR'] = ustd * correction
        if np.all(np.isnan(uflux)):
            log.warning('Uncorrected flux cube contains only NaN values.')

    # Update header
    hdinsert(combined['PRIMEHEAD'], 'WVFITWDW', str(fit_wdw),
             comment='Wave resample fit window (um)')
    hdinsert(combined['PRIMEHEAD'], 'WVFITORD', str(order),
             comment='Wave resample fit order')
    hdinsert(combined['PRIMEHEAD'], 'WVFITSMR', str(smoothing_wdw),
             comment='Wave resample smooth radius (um)')
    hdinsert(combined['PRIMEHEAD'], 'XYRSMPAL',
             'radial basis function interpolation',
             comment='XY resampling algorithm')
    return


def local_surface_fit(combined, grid_info, window=None,
                      adaptive_threshold=None,
                      adaptive_algorithm='scaled',
                      error_weighting=True, smoothing=None,
                      order=2, robust=None, neg_threshold=None,
                      fit_threshold=None, edge_threshold=None,
                      skip_uncorrected=False, jobs=None,
                      check_memory=True):
    """
    Resamples combined data on regular grid using local polynomial fitting.

    Parameters
    ----------
    combined : dict
        Dictionary containing combined data. Returned from `combine_files`.
    grid_info : dict
        Dictionary containing output grid coordinates and other necessary
        information.  Returned from `get_grid_info`.
    window : array_like of float, optional
        Region to consider for local polynomial fits, given as a factor
        of the mean FWHM, in the (x, y, w) dimensions. Default is
        (3.0, 3.0, 0.5).
    adaptive_threshold : array_like of float, optional
        If > 0, determines how the adaptive smoothing algorithm
        will attempt to fit data.  The optimal value is 1.  Will
        automatically enable both distance and error weighting.  For
        dimensions that have adaptive smoothing enabled, `smoothing`
        should be set to the Gaussian width of the data in units of `window`.
        For other dimensions not using adaptive smoothing, `smoothing`
        has the usual definition. Adaptive smoothing is disabled by
        default: (0.0, 0.0, 0.0).
    adaptive_algorithm : {'scaled', 'shaped'}, optional
        Determines the type of variation allowed for the adaptive kernel.
        If 'scaled', only the kernel size is allowed to vary.  If 'shaped',
        kernel shape may also vary.
    error_weighting : bool, optional
        If True, errors will be used to weight the flux fits.
    smoothing : array_like of float, optional
        Distance over which to smooth the data, given as a factor of the
        mean FWHM, in the (x, y, w) dimensions.  If `adaptive_threshold` is
        set for a certain dimension, smoothing should be set to 1.0 for that
        dimension. Default is (1.75, 1.75, 0.25).
    order : int or array of int, optional
        Maximum order of local polynomial fits, in the (x, y, w)
        dimensions.
    robust : float, optional
        Rejection threshold for input data to local fits, given as a
        factor of the standard deviation.
    neg_threshold : float, optional
        First-pass rejection threshold for negative input data, given
        as a factor of the standard deviation; if None or <= 0,
        first-pass rejection will not be performed.
    fit_threshold : float, optional
        Rejection threshold for output fit values, given as a factor
        of the standard deviation in the input data.  If exceeded,
        weighted mean value is used in place of fit.
    edge_threshold : array_like of float, optional
        Threshold for edge marking for (x, y, w) dimensions. Values
        should be between 0 and 1; higher values mean more edge pixels
        marked. Default is (0.7, 0.7, 0.5).
    skip_uncorrected: bool, optional
        If set, the uncorrected flux cube will not be computed, even
        if present in the input data.  This option is primarily intended
        for testing or quicklook, when the full data product is not needed.
    jobs : int, optional
        Specifies the maximum number of concurrently running jobs.
        Values of 0 or 1 will result in serial processing.  A negative
        value sets jobs to `n_cpus + 1 + jobs` such that -1 would use
        all cpus, and -2 would use all but one cpu.
    check_memory : bool, optional
        If set, expected memory use will be checked and used to limit
        the number of jobs if necessary.
    """
    log.info('')
    log.info('Resampling using local polynomial fits')

    # Fit window for FWHM
    if window is None:
        window = (3.0, 3.0, 0.5)
    if smoothing is None:
        smoothing = (2.0, 2.0, 0.25)
    if edge_threshold is None:
        edge_threshold = (0.7, 0.7, 0.5)

    fit_wdw = \
        (window[0] * grid_info['xy_fwhm'],
         window[1] * grid_info['xy_fwhm'],
         window[2] * grid_info['wave_fwhm'])

    smoothing_wdw = \
        (smoothing[0] * grid_info['xy_fwhm'],
         smoothing[1] * grid_info['xy_fwhm'],
         smoothing[2] * grid_info['wave_fwhm'])

    log.info(f'Fit window (x, y, w): {fit_wdw[0]:.2f} arcsec, '
             f'{fit_wdw[1]:.2f} arcsec, {fit_wdw[2]:.5f} um')

    log.info(f'Gaussian width of smoothing function (x, y, w): '
             f'{smoothing_wdw[0]:.2f} arcsec, {smoothing_wdw[1]:.2f} arcsec, '
             f'{smoothing_wdw[2]:.5f} um')

    if adaptive_threshold is not None:
        log.info(f'Adaptive algorithm: {adaptive_algorithm}')
        log.info(f'Adaptive smoothing threshold (x, y, w): '
                 f'{adaptive_threshold[0]:.2f}, {adaptive_threshold[1]:.2f}, '
                 f'{adaptive_threshold[2]:.2f}')
    else:
        adaptive_threshold = 0
        adaptive_algorithm = None

    flxvals = np.hstack([f.ravel() for f in combined['FLUX']])
    errvals = np.hstack([e.ravel() for e in combined['ERROR']])

    # check whether data fits in memory
    log.info('')
    max_bytes = Resample.estimate_max_bytes(grid_info['coordinates'],
                                            fit_wdw, order=order)
    max_avail = psutil.virtual_memory().total

    max_size = max_bytes
    for unit in ['B', 'kB', 'MB', 'GB', 'TB', 'PB']:
        if max_size < 1024 or unit == 'PB':
            break
        max_size /= 1024.0
    log.debug(f'Maximum expected memory needed: {max_size:.2f} {unit}')

    if check_memory:
        # let the resampler handle it - it has more sophisticated checks
        large_data = None
    else:
        # with check_memory false, set large data if the reduction is
        # likely to take up a significant percentage of memory, to give
        # it a chance to succeed
        if max_bytes >= max_avail / 10:
            log.debug('Splitting data tree into blocks.')
            large_data = True
        else:
            large_data = False

    if np.all(np.isnan(flxvals)):
        log.warning('Primary flux cube contains only NaN values.')
        flux = np.full(grid_info['shape'][1:], np.nan)
        std = np.full(grid_info['shape'][1:], np.nan)
        weights = np.full(grid_info['shape'][1:], 0.0)
    else:
        resampler = Resample(
            grid_info['coordinates'].copy(), flxvals, error=errvals,
            window=fit_wdw, order=order, robust=robust,
            negthresh=neg_threshold, large_data=large_data,
            check_memory=check_memory)

        flux, std, weights = resampler(
            *grid_info['grid'], smoothing=smoothing_wdw,
            adaptive_algorithm=adaptive_algorithm,
            adaptive_threshold=adaptive_threshold,
            fit_threshold=fit_threshold, edge_threshold=edge_threshold,
            edge_algorithm='distribution', get_error=True,
            get_distance_weights=True,
            error_weighting=error_weighting, jobs=jobs)

    do_uncor = 'UNCORRECTED_FLUX' in combined and not skip_uncorrected

    if do_uncor:
        log.info('')
        log.info('Now resampling uncorrected cube.')

        if grid_info['uncorrected_coordinates'] is None:
            coord = grid_info['coordinates'].copy()
        else:
            coord = grid_info['uncorrected_coordinates'].copy()
        flxvals = np.hstack([f.ravel() for f in combined['UNCORRECTED_FLUX']])
        errvals = np.hstack([e.ravel() for e in combined['UNCORRECTED_ERROR']])

        if np.all(np.isnan(flxvals)):
            log.warning('Uncorrected flux cube contains only NaN values.')
            uflux = np.full(grid_info['shape'][1:], np.nan)
            ustd = np.full(grid_info['shape'][1:], np.nan)
            uweights = np.full(grid_info['shape'][1:], 0.0)
        else:
            resampler = Resample(
                coord, flxvals, error=errvals,
                window=fit_wdw, order=order, robust=robust,
                negthresh=neg_threshold, large_data=large_data,
                check_memory=check_memory)
            uflux, ustd, uweights = resampler(
                *grid_info['grid'], smoothing=smoothing_wdw,
                adaptive_algorithm=adaptive_algorithm,
                adaptive_threshold=adaptive_threshold,
                fit_threshold=fit_threshold, edge_threshold=edge_threshold,
                edge_algorithm='distribution', get_error=True,
                get_distance_weights=True,
                error_weighting=error_weighting, jobs=jobs)
    else:
        uflux, ustd, uweights = None, None, None

    # make exposure map
    log.info('')
    log.info('Making the exposure map.')
    exposure = generate_exposure_map(combined, grid_info)
    combined['GRID_COUNTS'] = exposure

    # store distance weights
    combined['GRID_WEIGHTS'] = weights

    # correct flux for spatial pixel size change and extrapolation
    factor = (grid_info['delta'][1] / grid_info['pix_size']) ** 2
    log.debug('Flux correction factor: {}'.format(factor))

    correction = np.full(flux.shape, factor)
    correction[exposure == 0] = np.nan
    combined['GRID_FLUX'] = flux * correction
    combined['GRID_ERROR'] = std * correction
    if do_uncor:
        combined['GRID_UNCORRECTED_FLUX'] = uflux * correction
        combined['GRID_UNCORRECTED_ERROR'] = ustd * correction
        combined['GRID_UNCORRECTED_WEIGHTS'] = uweights

    # Update header
    hdinsert(combined['PRIMEHEAD'], 'XYFITWD', str(fit_wdw),
             comment='WXY Resample fit window (arcsec,arcsec,um)')
    hdinsert(combined['PRIMEHEAD'], 'XYFITORD', str(order),
             comment='WXY Resample fit order')
    hdinsert(combined['PRIMEHEAD'], 'XYFITWTS', 'error and distance',
             comment='WXY Resample weights')
    hdinsert(combined['PRIMEHEAD'], 'XYFITSMR', str(smoothing_wdw),
             comment='WXY Resample smooth radius (arcsec,arcsec,um)')
    hdinsert(combined['PRIMEHEAD'], 'XYRSMPAL',
             'local polynomial surface fits',
             comment='WXY resampling algorithm')
    return


def make_hdul(combined, grid_info, append_weights=False):
    """
    Create final HDU List from combined data and gridding info.

    Parameters
    ----------
    combined : dict
    grid_info : dict
    append_weights : bool, optional
        If set, distance weights will be appended as an additional
        extension.

    Returns
    -------
    fits.HDUList
    """
    primehead = combined['PRIMEHEAD']
    outname = os.path.basename(primehead.get('FILENAME', 'UNKNOWN'))
    outname, _ = os.path.splitext(outname)
    for repl in ['SCM', 'TEL', 'CAL', 'WSH']:
        outname = outname.replace(repl, 'WXY')
    outname = f"{'_'.join(outname.split('_')[:-1])}_" \
              f"{primehead.get('FILENUM', 'UNK')}.fits"
    hdinsert(primehead, 'FILENAME', outname)
    hdinsert(primehead, 'NAXIS', 0)
    hdinsert(primehead, 'PRODTYPE', 'resampled')
    primehead['HISTORY'] = 'Resampled to regular grid'
    hdinsert(primehead, 'PIXSCAL', grid_info['delta'][1])
    hdinsert(primehead, 'XYOVRSMP', str(grid_info['oversample']),
             comment='WXY Oversampling (pix per mean FWHM)')

    obsbet = primehead['OBSDEC'] - (primehead['DBET_MAP'] / 3600)
    obslam = (primehead['OBSRA'] * 15)
    obslam -= primehead['DLAM_MAP'] / (3600 * np.cos(np.radians(obsbet)))
    rot = float(np.radians(primehead.get('SKY_ANGL', 0)))
    xref, yref = grid_info['xout'][0], grid_info['yout'][0]
    if rot != 0:
        xrot = (xref * np.cos(rot)) - (yref * np.sin(rot))
        yrot = (xref * np.sin(rot)) + (yref * np.cos(rot))
        xref, yref = xrot, yrot
    crval2 = obsbet + (yref / 3600)
    crval1 = obslam - (xref / (3600 * np.cos(np.radians(crval2))))

    # Save reference value in TELRA/TELDEC, since those are archive-
    # searchable values
    hdinsert(primehead, 'TELRA', obslam / 15)
    hdinsert(primehead, 'TELDEC', obsbet)
    procstat = str(primehead.get('PROCSTAT')).upper()
    imagehdu = fits.ImageHDU(combined['GRID_FLUX'])

    exthdr = imagehdu.header.copy()
    exthdr_1d = imagehdu.header.copy()
    hdinsert(exthdr, 'DATE-OBS', primehead['DATE-OBS'],
             comment='Observation date')
    hdinsert(exthdr_1d, 'DATE-OBS', primehead['DATE-OBS'],
             comment='Observation date')

    if procstat == 'LEVEL_3':
        hdinsert(exthdr, 'BUNIT', 'Jy/pixel', comment='Data units')
        # New convention:: always set calibrated WXY to LEVEL_4
        # even if it contains data from one mission only
        hdinsert(primehead, 'PROCSTAT', 'LEVEL_4')
    else:
        hdinsert(exthdr, 'BUNIT', 'ADU/(s Hz)',
                 comment='Data units')

    # Add WCS keywords to primary header and ext header
    for h in [primehead, exthdr]:
        hdinsert(h, 'EQUINOX', 2000.0, comment='Coordinate equinox')
        hdinsert(h, 'CTYPE1', 'RA---TAN',
                 comment='Axis 1 type and projection')
        hdinsert(h, 'CTYPE2', 'DEC--TAN',
                 comment='Axis 2 type and projection')
        hdinsert(h, 'CTYPE3', 'WAVE', comment='Axis 3 type and projection')
        hdinsert(h, 'CUNIT1', 'deg', comment='Axis 1 units')
        hdinsert(h, 'CUNIT2', 'deg', comment='Axis 2 units')
        hdinsert(h, 'CUNIT3', 'um', comment='Axis 3 units')
        hdinsert(h, 'CRPIX1', 1.0, comment='Reference pixel (x)')
        hdinsert(h, 'CRPIX2', 1.0, comment='Reference pixel (y)')
        hdinsert(h, 'CRPIX3', 1.0, comment='Reference pixel (z)')
        hdinsert(h, 'CRVAL1', crval1, comment='RA (deg) at CRPIX1,2')
        hdinsert(h, 'CRVAL2', crval2, comment='Dec (deg) at CRPIX1,2')
        hdinsert(h, 'CRVAL3', grid_info['wout'][0],
                 comment='Wavelength (um) at CRPIX3')
        hdinsert(h, 'CDELT1', -grid_info['delta'][1] / 3600,
                 comment='RA pixel scale (deg/pix)')
        hdinsert(h, 'CDELT2', grid_info['delta'][2] / 3600,
                 comment='Dec pixel scale (deg/pix)')
        hdinsert(h, 'CDELT3', grid_info['delta'][0],
                 comment='Wavelength pixel scale (um/pix)')
        hdinsert(h, 'CROTA2', -primehead.get('SKY_ANGL', 0.0),
                 comment='Rotation angle (deg)')
        hdinsert(h, 'SPECSYS', 'BARYCENT',
                 comment='Spectral reference frame')
        # add beam keywords
        hdinsert(h, 'BMAJ', grid_info['xy_fwhm'] / 3600,
                 comment='Beam major axis (deg)')
        hdinsert(h, 'BMIN', grid_info['xy_fwhm'] / 3600,
                 comment='Beam minor axis (deg)')
        hdinsert(h, 'BPA', 0.0, comment='Beam position angle (deg)')

    # interpolate smoothed ATRAN and response data onto new grid for
    # reference
    resolution = grid_info['resolution']
    dw = grid_info['delta'][0] / 2
    wout = grid_info['wout']
    wmin = wout.min()
    wmax = wout.max()
    if 'UNSMOOTHED_TRANSMISSION' in combined:

        unsmoothed_atran = combined['UNSMOOTHED_TRANSMISSION']
        try:
            smoothed = smoothres(unsmoothed_atran[0], unsmoothed_atran[1],
                                 resolution)

            # Interpolate transmission to new wavelengths
            w = unsmoothed_atran[0]
            atran = np.interp(wout, w, smoothed, left=np.nan, right=np.nan)

            # Keep unsmoothed data as is, but cut to wavelength range
            keep = (w >= (wmin - dw)) & (w <= (wmax + dw))
            unsmoothed_atran = unsmoothed_atran[:, keep]
        except (ValueError, TypeError, IndexError):
            log.error('Problem in interpolation.  '
                      'Setting TRANSMISSION to 1.0.')
            atran = np.full(wout.shape, 1.0)
    else:
        atran = np.full(wout.shape, 1.0)
        unsmoothed_atran = None

    response = get_response(primehead)
    try:
        resp = np.interp(wout, response[0], response[1],
                         left=np.nan, right=np.nan)
    except (ValueError, TypeError, IndexError):
        log.error('Problem in interpolation.  '
                  'Setting RESPONSE to 1.0.')
        resp = np.full(wout.shape, 1.0)

    # Add the spectral keys to primehead
    hdinsert(primehead, 'RESOLUN', resolution)
    hdinsert(primehead, 'SPEXLWID', grid_info['delta'][0])

    # make HDUList
    hdul = fits.HDUList(fits.PrimaryHDU(header=primehead))
    hdul.append(fits.ImageHDU(data=combined['GRID_FLUX'],
                              name='FLUX', header=exthdr))
    hdul.append(fits.ImageHDU(data=combined['GRID_ERROR'],
                              name='ERROR', header=exthdr))

    if 'GRID_UNCORRECTED_FLUX' in combined:
        hdul.append(fits.ImageHDU(data=combined['GRID_UNCORRECTED_FLUX'],
                                  name='UNCORRECTED_FLUX', header=exthdr))
        hdul.append(fits.ImageHDU(data=combined['GRID_UNCORRECTED_ERROR'],
                                  name='UNCORRECTED_ERROR', header=exthdr))

    hdinsert(exthdr_1d, 'BUNIT', 'um', comment='Data units')
    hdul.append(fits.ImageHDU(data=grid_info['wout'], name='WAVELENGTH',
                              header=exthdr_1d))
    exthdr_1d['BUNIT'] = 'arcsec'
    hdul.append(fits.ImageHDU(data=grid_info['xout'], name='X',
                              header=exthdr_1d))
    hdul.append(fits.ImageHDU(data=grid_info['yout'], name='Y',
                              header=exthdr_1d))

    exthdr_1d['BUNIT'] = ''
    hdul.append(fits.ImageHDU(data=atran, name='TRANSMISSION',
                              header=exthdr_1d))
    exthdr_1d['BUNIT'] = 'adu/(s Hz Jy)'
    hdul.append(fits.ImageHDU(data=resp, name='RESPONSE',
                              header=exthdr_1d))

    exthdr['BUNIT'] = ''
    hdul.append(fits.ImageHDU(data=combined['GRID_COUNTS'],
                              name='EXPOSURE_MAP', header=exthdr))
    exthdr_1d['BUNIT'] = ''
    hdul.append(fits.ImageHDU(data=unsmoothed_atran,
                              name='UNSMOOTHED_TRANSMISSION',
                              header=exthdr_1d))
    if append_weights and 'GRID_WEIGHTS' in combined:
        hdul.append(fits.ImageHDU(data=combined['GRID_WEIGHTS'],
                                  name='IMAGE_WEIGHTS', header=exthdr))
        if 'GRID_UNCORRECTED_WEIGHTS' in combined:
            hdul.append(
                fits.ImageHDU(data=combined['GRID_UNCORRECTED_WEIGHTS'],
                              name='UNCORRECTED_IMAGE_WEIGHTS',
                              header=exthdr))

    return hdul


def resample(filenames, xrange=None, yrange=None, wrange=None,
             interp=False, oversample=None, spatial_size=None,
             spectral_size=None, window=None,
             adaptive_threshold=None, adaptive_algorithm=None,
             error_weighting=True, smoothing=None, order=2,
             robust=None, neg_threshold=None, fit_threshold=None,
             edge_threshold=None, append_weights=False,
             skip_uncorrected=False, write=False, outdir=None,
             jobs=None, check_memory=True):
    """
    Resample unevenly spaced FIFI-LS pixels to regular grid.

    Spatial and spectral pixels from all dither positions are
    resampled onto a regular grid.

    The procedure is:

        1. Read input files
        2. Define output grid based on input parameter values or values from
           file data.
        3. Resample data: perform local polynomial fits at each
           output grid point.
        4. Correct flux for change to pixel size.  Factor is new pixel area
           (dx^2) / input pixel size (12" red, 6" blue).
        5. Update header for new WCS, from OBSRA/OBSDEC and offsets, Update
           PROCSTAT to LEVEL_4 if input data comes from multiple missions.
        6. Create FITS file and write results to disk.

    Parameters
    ----------
    filenames : array_like of str
        File paths to FITS data to be resampled.
    xrange : array_like of float, optional
        (Min, max) x-value of new grid.  If not provided, minimum and
        maximum input values will be used.
    yrange : array_like of float, optional
        (Min, max) y-value of new grid.  If not provided, minimum and
        maximum input values will be used.
    wrange : array_like of float, optional
        (Min, max) wavelength value of new grid.  If not provided, minimum
        and maximum input values will be used.
    interp : bool, optional
        If True, alternate algorithm will be used for spatial resampling
        (interpolation / mean combine, rather than local polynomial fits).
    oversample : array_like of int or float, optional
        Number of pixels to sample mean FWHM with, in the
        (spatial, spectral) dimensions.
        Default is (5.0, 8.0).
    spatial_size : float, optional
        Output pixel size, in the spatial dimensions.
        Units are arcsec.  If specified, the corresponding oversample
        parameter will be ignored.
    spectral_size : float, optional
        Output pixel size, in the spectral dimension.
        Units are um.  If specified, the corresponding oversample
        parameter will be ignored.
    window : array_like of float, optional
        Region to consider for local polynomial fits, given as a factor
        of the mean FWHM, in the (x, y, w) dimensions. Default is
        (3.0, 3.0, 0.5).
    adaptive_threshold : array_like of float, optional
        If > 0, determines how the adaptive smoothing algorithm
        will attempt to fit data.  The optimal value is 1.  Will
        automatically enable both distance and error weighting.  For
        dimensions that have adaptive smoothing enabled, `smoothing`
        should be set to the Gaussian width of the data in units of `window`.
        For other dimensions not using adaptive smoothing, `smoothing`
        has the usual definition. Adaptive smoothing is disabled by
        default: (0.0, 0.0, 0.0).
    adaptive_algorithm : {'scaled', 'shaped'}, optional
        Determines the type of variation allowed for the adaptive kernel.
        If 'scaled', only the kernel size is allowed to vary.  If 'shaped',
        kernel shape may also vary.
    error_weighting : bool, optional
        If True, errors will be used to weight the flux fits.
    smoothing : array_like of float, optional
        Radius over which to smooth the input data, specified as a
        factor of the mean FWHM, if distance weights are being used.
        Default is (2.0, 2.0, 0.25).
    order : array_like or int, optional
        (nfeatures,) array of single integer value specifying the
        polynomial fit order for each dimension (x, y, w).
    robust : float, optional
        Rejection threshold for input data to local fits, given as a
        factor of the standard deviation.
    neg_threshold : float, optional
        First-pass rejection threshold for negative input data, given
        as a factor of the standard deviation; if None or <= 0,
        first-pass rejection will not be performed.
    fit_threshold : float, optional
        Rejection threshold for output fit values, given as a factor
        of the standard deviation in the input data.  If exceeded,
        weighted mean value is used in place of fit.
    edge_threshold : float or array_like or float
        If set to a value > 0 and < 1, edges of the fit will be masked out
        according to `edge_algorithm`. Values close to zero will result in
        a high degree of edge clipping, while values close to 1 ckip edges
        to a lesser extent. The clipping threshold is a fraction of window.
        An array may be used to specify values for each dimension.
    append_weights: bool, optional
        If set, distance weights will be appended as an additional
        extension.
    skip_uncorrected: bool, optional
        If set, the uncorrected flux cube will not be computed, even
        if present in the input data.  This option is primarily intended
        for testing or quicklook, when the full data product is not needed.
    write : bool, optional
        If True, write to disk and return the path to the output
        file.  The output filename is created from the input filename,
        with the product type suffix replaced with 'WXY'.
    outdir : str, optional
        Directory path to write output.  If None, output files
        will be written to the same directory as the input files.
    jobs : int, optional
        Specifies the maximum number of concurrently running jobs.
        Values of 0 or 1 will result in serial processing.  A negative
        value sets jobs to `n_cpus + 1 + jobs` such that -1 would use
        all cpus, and -2 would use all but one cpu.
    check_memory : bool, optional
        If set, expected memory use will be checked and used to limit
        the number of jobs if necessary.

    Returns
    -------
    fits.HDUList or str
        Either the HDU (if write is False) or the filename of the output
        file (if write is True).  The output contains the following
        extensions: FLUX, ERROR, WAVELENGTH, X, Y, TRANSMISSION,
        RESPONSE, EXPOSURE_MAP.  The following extensions will be appended
        if possible: UNCORRECTED_FLUX, UNCORRECTED_ERROR,
        UNSMOOTHED_TRANSMISSION.
    """
    clear_resolution_cache()
    clear_response_cache()

    if isinstance(filenames, str):
        filenames = [filenames]
    if not hasattr(filenames, '__len__'):
        log.error(f'Invalid input files type ({repr(filenames)})')
        return

    if isinstance(outdir, str):
        if not os.path.isdir(outdir):
            log.error(f'Output directory {outdir} does not exist')
            return
    else:
        if isinstance(filenames[0], str):
            outdir = os.path.dirname(filenames[0])

    combined = combine_files(filenames)
    grid_info = get_grid_info(combined, xrange=xrange, yrange=yrange,
                              wrange=wrange, oversample=oversample,
                              spatial_size=spatial_size,
                              spectral_size=spectral_size)
    if grid_info is None:
        log.error('Problem in grid calculation')
        return

    interp |= combined['method'] == 'interpolate'
    if interp:
        try:
            rbf_mean_combine(combined, grid_info, window=window,
                             error_weighting=error_weighting,
                             smoothing=smoothing, order=order,
                             robust=robust, neg_threshold=neg_threshold,
                             fit_threshold=fit_threshold,
                             skip_uncorrected=skip_uncorrected)
        except Exception as err:
            log.error(err, exc_info=True)
            return None
    else:
        try:
            local_surface_fit(combined, grid_info, window=window,
                              adaptive_threshold=adaptive_threshold,
                              adaptive_algorithm=adaptive_algorithm,
                              error_weighting=error_weighting,
                              smoothing=smoothing, order=order,
                              robust=robust, neg_threshold=neg_threshold,
                              fit_threshold=fit_threshold,
                              edge_threshold=edge_threshold,
                              skip_uncorrected=skip_uncorrected,
                              jobs=jobs, check_memory=check_memory)
        except Exception as err:
            log.error(err, exc_info=True)
            return None

    result = make_hdul(combined, grid_info, append_weights=append_weights)
    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)
