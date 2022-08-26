# Licensed under a 3-clause BSD style license - see LICENSE.rst

import gc
import os

from astropy import log, units
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import cloudpickle
import numpy as np
import psutil
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
import shutil
import tempfile

from sofia_redux.instruments.fifi_ls.get_resolution \
    import get_resolution, clear_resolution_cache
from sofia_redux.instruments.fifi_ls.get_response \
    import get_response, clear_response_cache
from sofia_redux.instruments.fifi_ls.make_header \
    import make_header
from sofia_redux.toolkit.utilities \
    import gethdul, hdinsert, write_hdul
from sofia_redux.toolkit.resampling.resample import Resample
from sofia_redux.spectroscopy.smoothres import smoothres


__all__ = ['combine_files', 'get_grid_info', 'generate_exposure_map',
           'rbf_mean_combine', 'local_surface_fit', 'make_hdul',
           'resample', 'perform_scan_reduction', 'cleanup_scan_reduction']


def extract_info_from_file(filename, get_atran=True):
    """
    Extract all necessary resampling data from a single file.

    Parameters
    ----------
    filename : str or fits.HDUList
        The file or HDUList to examine.
    get_atran : bool, optional
        If `True`, attempt to extract the ATRAN data when present and
        include in the result.

    Returns
    -------
    file_info : dict
    """
    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        msg = f'Could not read file: {filename}'
        log.error(msg)
        raise ValueError(msg)

    header = hdul[0].header.copy()
    obsra = header.get('OBSRA', 0) * 15  # hourangle to degree
    obsdec = header.get('OBSDEC', 0)
    sky_angle = header.get('SKY_ANGL', 0)
    det_angle = header.get('DET_ANGL', 0)
    dbet_map = header.get('DBET_MAP', 0) / 3600  # arcsec to degree
    dlam_map = header.get('DLAM_MAP', 0) / 3600  # arcsec to degree
    obs_lam = header.get('OBSLAM', 0)
    obs_bet = header.get('OBSBET', 0)

    flux = hdul['FLUX'].data.copy()
    otf_mode = flux.ndim > 2
    ra = hdul['RA'].data.copy()
    dec = hdul['DEC'].data.copy()
    xs = hdul['XS'].data.copy()
    ys = hdul['YS'].data.copy()
    if otf_mode:
        wave = np.empty(flux.shape, dtype=float)
        wave[:] = hdul['LAMBDA'].data
    else:
        wave = hdul['LAMBDA'].data.copy()
    error = hdul['STDDEV'].data.copy()

    if 'UNCORRECTED_FLUX' in hdul:
        u_flux = hdul['UNCORRECTED_FLUX'].data.copy()
        u_error = hdul['UNCORRECTED_STDDEV'].data.copy()
    else:
        u_flux = None
        u_error = None

    if 'UNCORRECTED_LAMBDA' in hdul:
        if otf_mode:
            u_wave = np.empty(flux.shape, dtype=float)
            u_wave[:] = hdul['UNCORRECTED_LAMBDA'].data
        else:
            u_wave = hdul['UNCORRECTED_LAMBDA'].data.copy()
    else:
        u_wave = None

    if get_atran and 'UNSMOOTHED_ATRAN' in hdul:
        atran = hdul['UNSMOOTHED_ATRAN'].data
    else:
        atran = None

    hdul.close()

    # Now convert XS/YS to RA/DEC
    if 'DATE-OBS' in header:
        flip_sign = Time(header['DATE-OBS']) < Time('2015-05-01')
    else:
        flip_sign = False

    result = {
        'header': header,
        'obsra': obsra,
        'obsdec': obsdec,
        'obs_lam': obs_lam,
        'obs_bet': obs_bet,
        'dlam_map': dlam_map,
        'dbet_map': dbet_map,
        'flux': flux,
        'u_flux': u_flux,
        'error': error,
        'u_error': u_error,
        'wave': wave,
        'u_wave': u_wave,
        'ra': ra,
        'dec': dec,
        'xs': xs,
        'ys': ys,
        'sky_angle': sky_angle,
        'det_angle': det_angle,
        'flip_sign': flip_sign,
        'atran': atran}
    return result


def analyze_input_files(filenames, naif_id_key='NAIF_ID'):
    """
    Extract necessary reduction information on the input files.

    Parameters
    ----------
    filenames : str or list (str)
    naif_id_key : str, optional
        The name of the NAIF ID keyword.  If present in the header, should
        indicate the associated file contains a nonsidereal observation.

    Returns
    -------
    file_info : dict
    """
    atran = None
    otf_mode = False
    nonsidereal_values = True
    definite_nonsidereal = False
    interpolate = True
    uncorrected = False
    if isinstance(filenames, str):
        filenames = [x.strip() for x in filenames.split(',')]
    elif isinstance(filenames, fits.HDUList) or not isinstance(
            filenames, list):
        filenames = [filenames]  # Don't want to iterate on HDUs

    for filename in filenames:
        hdul = gethdul(filename, verbose=True)
        if hdul is None:
            msg = f'Could not read file: {filename}'
            log.error(msg)
            raise ValueError(msg)

        if not otf_mode:
            if 'FLUX' in hdul and hdul['FLUX'].data.ndim > 2:
                otf_mode = True

        if not uncorrected and 'UNCORRECTED_FLUX' in hdul:
            uncorrected = True

        h = hdul[0].header
        if not definite_nonsidereal and naif_id_key in h:
            definite_nonsidereal = True

        if nonsidereal_values:
            obs_lam, obs_bet = h.get('OBSLAM', 0), h.get('OBSBET', 0)
            if obs_lam != 0 or obs_bet != 0:
                nonsidereal_values = False

        if interpolate:
            d_lam, d_bet = h.get('DLAM_MAP', 0), h.get('DBET_MAP', 0)
            if d_lam != 0 or d_bet != 0:
                interpolate = False

        if atran is None and 'UNSMOOTHED_ATRAN' in hdul:
            atran = hdul['UNSMOOTHED_ATRAN'].data

        if not isinstance(filename, fits.HDUList):
            hdul.close()

        if (otf_mode and atran is not None and uncorrected
                and not (interpolate
                         or nonsidereal_values)):  # pragma: no cover
            # don't need to continue
            break

    if not(len(filenames) > 1 or otf_mode):
        interpolate = True

    file_info = {'otf': otf_mode,
                 'nonsidereal_values': nonsidereal_values,
                 'definite_nonsidereal': definite_nonsidereal,
                 'can_interpolate': interpolate,
                 'uncorrected': uncorrected,
                 'atran': atran}
    return file_info


def normalize_spherical_coordinates(info):
    """
    Normalize all detector coordinates/header to be relative to first file.

    Will insert the normalized detector coordinates into the information as
    longitude/latitude (lon/lat) key values.

    Parameters
    ----------
    info : dict

    Returns
    -------
    None
    """
    a = -np.radians([d['sky_angle'] for d in info.values()])
    beta = np.asarray([d['obs_bet'] for d in info.values()])
    lam = np.asarray([d['obs_lam'] for d in info.values()])
    da = a - a[0]
    cos_beta = np.cos(beta[0])
    beta_off = 3600.0 * (beta - beta[0])
    lam_off = 3600.0 * cos_beta * (lam[0] - lam)
    header_list = [d['header'] for d in info.values()]  # referenced

    # update headers
    for update in np.where(beta_off != 0)[0]:
        hdinsert(header_list[update], 'DBET_MAP',
                 header_list[update].get('DBET_MAP', 0) + beta_off[update])

    for update in np.where(lam_off != 0)[0]:
        hdinsert(header_list[update], 'DLAM_MAP',
                 header_list[update].get('DLAM_MAP', 0) - lam_off[update])

    idx = abs(a) > 1e-6
    dx = lam_off[idx] * np.cos(a[idx]) - beta_off[idx] * np.sin(a[idx])
    dy = lam_off[idx] * np.sin(a[idx]) + beta_off[idx] * np.cos(a[idx])
    lam_off[idx] = dx
    beta_off[idx] = dy

    for i, d in enumerate(info.values()):
        d['lon'], d['lat'] = d['xs'].copy(), d['ys'].copy()
        if i == 0:
            continue
        lon, lat = d['lon'], d['lat']
        dx, dy, delta_a = lam_off[i], beta_off[i], da[i]
        lon += dx
        lat += dy
        if abs(delta_a) <= 1e-6:
            continue

        cda, sda = np.cos(delta_a), np.sin(delta_a)
        xr = lon * cda - lat * sda
        ry = lon * sda + lat * cda
        lon[:] = xr
        lat[:] = ry

    # update to the angle of the first header
    first_angle = info[list(info.keys())[0]]['sky_angle']
    for update in np.where(idx)[0]:
        hdinsert(header_list[update], 'SKY_ANGL', first_angle)


def combine_files(filenames, naif_id_key='NAIF_ID',
                  scan_reduction=False, save_scan=False, scan_kwargs=None,
                  skip_uncorrected=False, insert_source=True):
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
    naif_id_key : str, optional
        The header key which if present, indicates that an observation is
        nonsidereal.
    scan_reduction: bool, optional
        If `True`, indicates the user wished to perform a scan reduction on
        the files.  This will only be possible if the files contain OTF data.
    save_scan : bool, optional
        If `True`, the output from the scan algorithm, prior to resampling,
        will be saved to disk.
    scan_kwargs : dict, optional
        Optional parameters for a scan reduction if performed.
    skip_uncorrected : bool, optional
        If `True`, skip reduction of the uncorrected flux values.
    insert_source : bool, optional
        If `True`, will perform a full scan reduction (if applicable) and
        reinsert the source after.  Otherwise, the reduction is used to
        calculate gains, offsets, and correlations which will then be
        applied to the original data. If `True`, note that timestream
        filtering will not be applied to the correction and should therefore
        be excluded from the scan reduction runtime parameters in order to
        reduce processing pressure.

    Returns
    -------
    dict
    """
    log.info(f'Reading {len(filenames)} files')
    info = analyze_input_files(filenames, naif_id_key=naif_id_key)
    combined = {'OTF': info['otf'],
                'definite_nonsidereal': info['definite_nonsidereal'],
                'nonsidereal_values': info['nonsidereal_values']}
    do_uncorrected = info['uncorrected'] and not skip_uncorrected
    atran = info.get('atran')
    if atran is not None:
        combined['UNSMOOTHED_TRANSMISSION'] = atran

    scan_reduction &= info['otf']  # Scan reduction must use OTF data
    if scan_reduction:  # pragma: no cover
        combined['scan_reduction'] = True
        combined.update(perform_scan_reduction(
            filenames, save_scan=save_scan, scan_kwargs=scan_kwargs,
            reduce_uncorrected=do_uncorrected, insert_source=insert_source))
    else:
        combined['scan_reduction'] = False
        files_info = {}
        for filename in filenames:
            file_info = extract_info_from_file(filename, get_atran=False)
            if not isinstance(filename, str):
                fname = file_info['header'].get('FILENAME', 'UNKNOWN')
            else:
                fname = filename
            files_info[fname] = file_info

        normalize_spherical_coordinates(files_info)
        header_list = [d['header'] for d in files_info.values()]
        combined['PRIMEHEAD'] = make_header(header_list)
        if info['can_interpolate']:
            combined['method'] = 'interpolate'
        else:
            combined['method'] = 'resample'

        ra, dec, xs, ys, wave, flux, error, samples = (
            [], [], [], [], [], [], [], [])

        if info['otf']:
            for d in files_info.values():
                x_ra, x_dec, x_xs, x_ys = d['ra'], d['dec'], d['lon'], d['lat']
                x_wave, x_flux, x_error = d['wave'], d['flux'], d['error']
                samples.append(x_flux.size)
                frames = x_flux.shape[0]

                for frame in range(frames):
                    flux.append(x_flux[frame])
                    error.append(x_error[frame])
                    wave.append(x_wave[frame] if x_wave.ndim == 3 else x_wave)
                    ra.append(x_ra[frame] if x_ra.ndim == 3 else x_ra)
                    dec.append(x_dec[frame] if x_dec.ndim == 3 else x_dec)
                    xs.append(x_xs[frame] if x_xs.ndim == 3 else x_xs)
                    ys.append(x_ys[frame] if x_ys.ndim == 3 else x_ys)

        else:
            for d in files_info.values():
                ra.append(d['ra'])
                dec.append(d['dec'])
                xs.append(d['lon'])
                ys.append(d['lat'])
                wave.append(d['wave'])
                flux.append(d['flux'])
                error.append(d['error'])
                samples.append(d['flux'].size)

        combined.update({
            'RA': ra, 'DEC': dec, 'XS': xs, 'YS': ys, 'WAVE': wave,
            'FLUX': flux, 'ERROR': error, 'SAMPLES': samples})

        if not do_uncorrected:
            return combined

        u_flux, u_error, u_wave = [], [], []
        for d in files_info.values():
            uf = d.get('u_flux')
            ue = d.get('u_error')

            if uf is None or ue is None:
                do_uncorrected = False
                break

            uw = d.get('u_wave')
            if uw is None:
                uw = d.get('wave')

            if info['otf']:
                for frame in range(uf.shape[0]):
                    u_flux.append(uf[frame])
                    u_error.append(ue[frame])
                    u_wave.append(uw[frame])
            else:
                u_flux.append(uf)
                u_error.append(ue)
                u_wave.append(uw)
        else:
            do_uncorrected = True

        if do_uncorrected:
            combined['UNCORRECTED_FLUX'] = u_flux
            combined['UNCORRECTED_ERROR'] = u_error
            combined['UNCORRECTED_WAVE'] = u_wave

    return combined


def combine_scan_reductions(reduction, uncorrected_reduction=None
                            ):  # pragma: no cover
    """
    Extract necessary resampling information from scan reductions.

    Parameters
    ----------
    reduction : str or Reduction
    uncorrected_reduction : str or Reduction, optional

    Returns
    -------
    combined : dict
    """
    if isinstance(reduction, str):
        delete = True
        with open(reduction, 'rb') as f:
            reduction = cloudpickle.load(f)
    else:
        delete = False

    info = reduction.info.combine_reduction_scans_for_resampler(reduction)
    combined = {
        'PRIMEHEAD': make_header(info['headers']),
        'method': 'resample',
        'RA': info['coordinates'][0],
        'DEC': info['coordinates'][1],
        'WAVE': info['coordinates'][2],
        'XS': info['xy_coordinates'][0],
        'YS': info['xy_coordinates'][1],
        'FLUX': info['flux'],
        'ERROR': info['error'],
        'SAMPLES': info['samples'],
        'CORNERS': info['corners'],
        'XY_CORNERS': info['xy_corners'],
        'scan_reduction': True}

    if delete:
        del reduction
    gc.collect()

    if uncorrected_reduction is None:
        return combined

    if isinstance(uncorrected_reduction, str):
        delete = True
        with open(uncorrected_reduction, 'rb') as f:
            uncorrected_reduction = cloudpickle.load(f)
    else:
        delete = False

    info = uncorrected_reduction.info.combine_reduction_scans_for_resampler(
        uncorrected_reduction)
    combined['UNCORRECTED_RA'] = info['coordinates'][0]
    combined['UNCORRECTED_DEC'] = info['coordinates'][1]
    combined['UNCORRECTED_WAVE'] = info['coordinates'][2]
    combined['UNCORRECTED_XS'] = info['xy_coordinates'][0]
    combined['UNCORRECTED_YS'] = info['xy_coordinates'][1]
    combined['UNCORRECTED_FLUX'] = info['flux']
    combined['UNCORRECTED_ERROR'] = info['error']
    combined['UNCORRECTED_SAMPLES'] = info['samples']
    combined['UNCORRECTED_CORNERS'] = info['corners']
    combined['UNCORRECTED_XY_CORNERS'] = info['xy_corners']

    if delete:
        del uncorrected_reduction
    gc.collect()
    return combined


def get_grid_info(combined, oversample=None,
                  spatial_size=None, spectral_size=None,
                  target_x=None, target_y=None, target_wave=None,
                  ctype1='RA---TAN', ctype2='DEC--TAN', ctype3='WAVE',
                  detector_coordinates=False):
    """
    Get output coordinate system and useful parameters.

    Parameters
    ----------
    combined : dict
        Dictionary containing combined data
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
    target_x : float, optional
        The target right ascension (hourangle) or map center along the x-axis
        (arcsec).  The default is the mid-point of all values in the
        combined data.
    target_y : float, optional
        The target declination (degree) or map center along the y-axis
        (arcsec).  The default is the mid-point of all values in the
        combined data.
    target_wave : float, optional
        The center wavelength (um).  The default is the mid-point of all
    ctype1 : str, optional
        The coordinate frame for the x spatial axis using FITS standards.
    ctype2 : str, optional
        The coordinate frame for the y spatial axis using FITS standards.
    ctype3 : str, optional
        The coordinate frame for the w spectral axis using FITS standards.
    detector_coordinates : bool, optional
        If `True`, reduce using detector native coordinates, otherwise
        project using the CTYPE* keys on RA/DEC.

    Returns
    -------
    dict
    """
    prime_header = combined['PRIMEHEAD']

    east_to_west = not detector_coordinates

    if detector_coordinates:
        wcs_dict = {
            'CUNIT1': 'arcsec',
            'CUNIT2': 'arcsec',
            'CUNIT3': 'um'
        }
        x_key, y_key = 'XS', 'YS'
    else:
        wcs_dict = {
            'CTYPE1': ctype1.upper(), 'CUNIT1': 'deg',
            'CTYPE2': ctype2.upper(), 'CUNIT2': 'deg',
            'CTYPE3': ctype3.upper(), 'CUNIT3': 'um',
        }
        x_key, y_key = 'RA', 'DEC'

    scan_reduction = combined.get('scan_reduction', False)

    if scan_reduction:  # pragma: no cover
        ux_key, uy_key = f'UNCORRECTED_{x_key}', f'UNCORRECTED_{y_key}'
        wave = combined['WAVE'].copy()
        x = combined[x_key].copy()
        y = combined[y_key].copy()
    else:
        ux_key, uy_key = x_key, y_key
        wave = np.hstack([x.ravel() for x in combined['WAVE']])
        x = np.hstack([x.ravel() for x in combined[x_key]])
        y = np.hstack([x.ravel() for x in combined[y_key]])

    if not detector_coordinates:
        x *= 15  # hourangle to degrees

    u_wave = combined.get('UNCORRECTED_WAVE')

    if u_wave is not None:
        for uw in u_wave:
            if uw is None:
                do_uncorrected = False
                break
        else:
            do_uncorrected = True
    else:
        do_uncorrected = False

    if do_uncorrected:
        if scan_reduction:  # pragma: no cover
            ux = combined[ux_key].copy()
            uy = combined[uy_key].copy()
            u_wave = u_wave.copy()
            if not detector_coordinates:
                ux *= 15  # hourangle to degrees
        else:
            ux, uy = x, y
            u_wave = np.hstack([w.ravel() for w in u_wave])
    else:
        ux, uy, u_wave = x, y, wave

    min_x, max_x = np.nanmin(x), np.nanmax(x)
    min_y, max_y = np.nanmin(y), np.nanmax(y)
    min_w, max_w = np.nanmin(wave), np.nanmax(wave)

    if do_uncorrected:
        min_w = min(min_w, np.nanmin(u_wave))
        max_w = max(max_w, np.nanmax(u_wave))
        if x is not ux:  # pragma: no cover
            min_x = min(min_x, np.nanmin(ux))
            max_x = max(max_x, np.nanmax(ux))
            min_y = min(min_y, np.nanmin(uy))
            max_y = max(max_y, np.nanmax(uy))

    x_range = [min_x, max_x]
    y_range = [min_y, max_y]
    wave_range = [min_w, max_w]

    if target_x is None:
        target_x = 0.0 if detector_coordinates else sum(x_range) / 2
    elif not detector_coordinates:
        target_x *= 15

    if target_y is None:
        target_y = 0.0 if detector_coordinates else sum(y_range) / 2

    mid_wave = sum(wave_range) / 2
    if target_wave is None:
        target_wave = mid_wave

    # get oversample parameter
    if oversample is None:
        xy_oversample, w_oversample = 5.0, 8.0
    else:
        xy_oversample, w_oversample = oversample

    log.info(f'Overall w range: '
             f'{wave_range[0]:.5f} -> {wave_range[1]:.5f} (um)')

    if detector_coordinates:
        x_str = f'{x_range[0]:.5f} -> {x_range[1]:.5f} (arcsec)'
        y_str = f'{y_range[0]:.5f} -> {y_range[1]:.5f} (arcsec)'
    else:
        x_str = ' -> '.join(
            Angle(x_range * units.Unit('degree')).to('hourangle').to_string(
                sep=':'))
        x_str += ' (hourangle)'
        y_str = ' -> '.join(
            Angle(y_range * units.Unit('degree')).to_string(sep=':'))
        y_str += ' (degree)'

    log.info(f'Overall x range: {x_str}')
    log.info(f'Overall y range: {y_str}')

    # Begin with spectral scalings
    resolution = get_resolution(prime_header, wmean=mid_wave)
    wave_fwhm = mid_wave / resolution
    if spectral_size is not None:
        delta_wave = spectral_size
        w_oversample = wave_fwhm / delta_wave
    else:
        delta_wave = wave_fwhm / w_oversample

    log.info(f'Average spectral FWHM: {wave_fwhm:.5f} um')
    log.info(f'Output spectral pixel scale: {delta_wave:.5f} um')
    log.info(f'Spectral oversample: {w_oversample:.2f} pixels')

    # Spatial scalings
    xy_fwhm = get_resolution(
        prime_header, spatial=True,
        wmean=float(np.nanmean(wave)))  # in arcseconds
    if not detector_coordinates:
        xy_fwhm /= 3600  # to degrees

    # pixel size
    if str(prime_header['CHANNEL']).upper() == 'RED':
        pix_size = 3.0 * prime_header['PLATSCAL']
    else:
        pix_size = 1.5 * prime_header['PLATSCAL']

    if spatial_size is not None:
        if not detector_coordinates:
            delta_xy = spatial_size / 3600  # to degrees
        else:
            delta_xy = spatial_size  # arcsec
        xy_oversample = xy_fwhm / delta_xy
    else:
        delta_xy = xy_fwhm / xy_oversample

    log.info(f'Pixel size for channel: {pix_size:.2f} arcsec')

    fac = 1 if detector_coordinates else 3600
    log.info(f'Average spatial FWHM for channel: {xy_fwhm * fac:.2f} arcsec')
    log.info(f'Output spatial pixel scale: {delta_xy * fac:.2f} arcsec/pix')
    log.info(f'Spatial oversample: {xy_oversample:.2f} pixels')

    # Figure out the map dimensions in RA/DEC
    wcs_dict['CRPIX1'] = 0
    wcs_dict['CRPIX2'] = 0
    wcs_dict['CRPIX3'] = 0
    wcs_dict['CRVAL1'] = target_x
    wcs_dict['CRVAL2'] = target_y
    wcs_dict['CRVAL3'] = target_wave
    if east_to_west:
        wcs_dict['CDELT1'] = -delta_xy
    else:
        wcs_dict['CDELT1'] = delta_xy
    wcs_dict['CDELT2'] = delta_xy
    wcs_dict['CDELT3'] = delta_wave

    wcs = WCS(wcs_dict)

    # Convert coordinates to the correct units (degrees, meters)
    if not detector_coordinates:
        wave *= 1e-6  # um to m
        if do_uncorrected:
            u_wave *= 1e-6

    # Calculate the input coordinates as pixels about origin 0
    pix_xyw = np.asarray(wcs.wcs_world2pix(x, y, wave, 0))
    n_xyw = np.round(np.ptp(pix_xyw, axis=1)).astype(int) + 1
    n_xyw += (n_xyw % 2) == 0  # center on pixel
    min_pixel = np.floor(np.nanmin(pix_xyw, axis=1)).astype(int)
    wcs_dict['CRPIX1'] -= min_pixel[0]
    wcs_dict['CRPIX2'] -= min_pixel[1]
    wcs_dict['CRPIX3'] -= min_pixel[2]
    wcs = WCS(wcs_dict)

    # This centers the reference pixel on the target coordinates
    pix_xyw = np.asarray(wcs.wcs_world2pix(x, y, wave, 0))
    n_pix = np.round(np.nanmax(pix_xyw, axis=1)).astype(int) + 1
    ni = x.size

    # Rotation angle before calibration.  If not rotated at calibration,
    # detector rotation is zero.
    if prime_header.get('SKY_ANGL', 0) != 0:
        det_angle = 0.0
    else:
        det_angle = -prime_header.get('DET_ANGL', 0.0)

    log.info('')
    log.info(f'Output grid size (nw, ny, nx): '
             f'{n_pix[2]} x {n_pix[1]} x {n_pix[0]}')
    if (n_pix[:2] > 2048).any():
        log.error('Spatial range too large.')
        return

    if do_uncorrected:
        u_pix_xyw = np.asarray(
            wcs.wcs_world2pix(ux, uy, u_wave, 0))
    else:
        u_pix_xyw = None

    x_out = np.arange(n_pix[0], dtype=float)
    y_out = np.arange(n_pix[1], dtype=float)
    w_out = np.arange(n_pix[2], dtype=float)
    grid = x_out, y_out, w_out

    x_max, x_min = np.nanmax(pix_xyw[0]), np.nanmin(pix_xyw[0])
    y_max, y_min = np.nanmax(pix_xyw[1]), np.nanmin(pix_xyw[1])
    w_max, w_min = np.nanmax(pix_xyw[2]), np.nanmin(pix_xyw[2])
    x_range, y_range, w_range = x_max - x_min, y_max - y_min, w_max - w_min

    um = units.Unit('um')
    arcsec = units.Unit('arcsec')
    degree = units.Unit('degree')
    xy_unit = arcsec if detector_coordinates else degree

    if east_to_west:
        delta = (delta_wave * um, delta_xy * xy_unit, -delta_xy * xy_unit)
    else:
        delta = (delta_wave * um, delta_xy * xy_unit, delta_xy * xy_unit)

    return {
        'wcs': wcs,
        'shape': (ni, n_pix[2], n_pix[1], n_pix[0]),
        'w_out': w_out, 'x_out': x_out, 'y_out': y_out,
        'x_min': x_min, 'y_min': y_min, 'w_min': w_min,
        'x_max': x_min, 'y_max': y_min, 'w_max': w_min,
        'x_range': x_range, 'y_range': y_range, 'w_range': w_range,
        'delta': delta,
        'oversample': (w_oversample, xy_oversample, xy_oversample),
        'wave_fwhm': wave_fwhm * um, 'xy_fwhm': xy_fwhm * 3600 * arcsec,
        'resolution': resolution,
        'pix_size': pix_size * arcsec,
        'det_angle': det_angle * degree,
        'coordinates': pix_xyw,
        'uncorrected_coordinates': u_pix_xyw,
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
    nw, ny, nx = grid_info['shape'][1:]

    # loop over files to get exposure for each one
    if get_good:
        exposure = []
    else:
        exposure = np.zeros((nw, ny, nx), dtype=int)

    xy_pix_size = (grid_info['pix_size'] / grid_info['delta'][1]
                   ).decompose().value / 2
    dr = np.sqrt(2) * xy_pix_size

    if not combined.get('scan_reduction', False):
        x, y, w, = [], [], []
        start = 0
        c = grid_info['coordinates']
        for i in range(len(combined['FLUX'])):
            flux = combined['FLUX'][i]
            end = start + flux.size
            x.append(c[0, start:end].reshape(flux.shape))
            y.append(c[1, start:end].reshape(flux.shape))
            w.append(c[2, start:end].reshape(flux.shape))
            start = end

    else:  # pragma: no cover
        wcs = grid_info['wcs']
        if 'CTYPE1' in wcs.to_header():
            corners = combined['CORNERS']
            detector_coordinates = False
        else:
            corners = combined['XY_CORNERS']
            detector_coordinates = True

        n_scans = len(corners)
        n_frames = np.asarray([len(corners[i][0]) for i in range(n_scans)])
        total_frames = n_frames.sum()
        min_wave = np.zeros(total_frames)
        max_wave = np.zeros(total_frames)
        xc = np.concatenate([corners[i][0] for i in range(n_scans)], axis=0)
        yc = np.concatenate([corners[i][1] for i in range(n_scans)], axis=0)
        if not detector_coordinates:
            xc *= 15  # hourangle to degrees for RA/DEC coordinates

        start_frame = 0
        i0 = 0
        for (frame_count, samples) in zip(n_frames, combined['SAMPLES']):
            end_frame = start_frame + frame_count
            i1 = i0 + samples
            wave = combined['WAVE'][i0:i1]
            min_wave[start_frame:end_frame] = wave.min()
            max_wave[start_frame:end_frame] = wave.max()
            start_frame, i0 = end_frame, i1

        wave = np.stack([min_wave, min_wave, max_wave, max_wave], axis=1)

        if detector_coordinates:
            # degrees to arcseconds
            x, y, w = wcs.wcs_world2pix(xc, yc, wave, 0)
        else:
            # um to m (it's strange, I know)
            x, y, w = wcs.wcs_world2pix(xc, yc, wave * 1e-6, 0)

    for i in range(len(x)):  # loop through files or frames
        # Find the min and max coordinates at each point
        wi, yi, xi = w[i], y[i], x[i]
        min_w, max_w = wi.min(), wi.max()
        max_x = xi.max()

        # Since the coordinates are already rotated, we need to determine
        # the corners of the detector footprint, then store then in clockwise
        # order
        points = np.stack([xi.ravel(), yi.ravel()], axis=1)
        hull = np.array([points[index] for index in
                         ConvexHull(points).vertices])
        angles = np.arctan2(hull[:, 1] - yi.mean(),
                            hull[:, 0] - xi.mean())
        hull = hull[np.argsort(angles)[::-1]]
        # calculate the relative angles between each vertex and add 45 deg
        # to calculate diagonal offset
        #  n-1->0, 0->1, 1->2, 2->3 ... n-2->n-1

        angles = np.asarray(
            [np.arctan2(hull[v, 1] - hull[v - 1, 1],
                        hull[v, 0] - hull[v - 1, 0])
             for v in range(hull.shape[0])]) + np.deg2rad(45)
        for v, angle in enumerate(angles):
            hull[v, 0] += np.cos(angle) * dr
            hull[v, 1] += np.sin(angle) * dr

        xl, yl = np.clip((np.min(hull, axis=0)).astype(int), 0, None)
        xh, yh = np.clip((np.ceil(np.max(hull, axis=0))).astype(int) + 1,
                         None, [nx, ny])
        wl = np.clip(int(min_w - 0.5), 0, None)
        wh = np.clip(int(np.ceil(max_w + 0.5)) + 1, None, nw)

        # Make a square large enough to contain the FOV
        square = np.zeros((yh - yl, xh - xl), dtype=int)

        # set values for FOV between corner vertices
        fov = np.full(square.shape, True)
        fov_y, fov_x = np.indices(square.shape)

        for j, (p2x, p2y) in enumerate(hull):
            p1x, p1y = hull[j - 1]
            x1, y1, x2, y2 = p1x - xl, p1y - yl, p2x - xl, p2y - yl

            # Check for vertical line
            if x1 == x2:
                # Mark data to the correct side of line
                sign = np.sign(y2 - y1)
                fov &= (fov_x * sign) >= (max_x * sign)
            else:
                # otherwise: mark area under the line between previous
                # vertex and current (or above, as appropriate)
                m = (y2 - y1) / (x2 - x1)
                max_y = m * (fov_x - x1) + y1
                sign = np.sign(x2 - x1)
                fov &= (fov_y * sign) <= (max_y * sign)

        square[fov] = 1

        if get_good:
            exposure.append((square.astype(bool), (xl, xh), (yl, yh)))
        else:
            exposure[wl:wh, yl:yh, xl:xh] += square[None, :, :]

    # For accounting purposes, multiply the exposure map by 2 for NMC
    if not get_good:
        nodstyle = combined['PRIMEHEAD'].get(
            'NODSTYLE', 'UNK').upper().strip()
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

    log.info(f'Fit window: {grid_info["wave_fwhm"] * window:.5f}')
    log.info(f'Gaussian width of smoothing function: '
             f'{smoothing * grid_info["wave_fwhm"]:.5f}')

    fit_wdw = window * grid_info['oversample'][0]
    smoothing_wdw = smoothing * grid_info['oversample'][0]

    # minimum points in a wavelength slice to attempt to interpolate
    min_points = 10

    # output grid
    xg, yg, w_out = grid_info['grid']
    nx = xg.size
    ny = yg.size
    nw = w_out.size
    x_grid = np.resize(xg, (ny, nx))
    y_grid = np.resize(yg, (nx, ny)).T

    # exposure map for grid counts
    good_grid = generate_exposure_map(combined, grid_info, get_good=True)
    n_spax = combined['FLUX'][0].shape[-1]

    f, e = combined['FLUX'], combined['ERROR']
    x, y, w = grid_info['coordinates']

    if do_uncor:
        uf = combined['UNCORRECTED_FLUX']
        ue = combined['UNCORRECTED_ERROR']
        uw = grid_info.get('uncorrected_coordinates')
        uw = w if uw is None else uw[2]  # .reshape(f.shape)
    else:
        uf, ue, uw = f, e, w

    # Create some work arrays
    temp_shape = (nw, n_spax)
    iflux, istd = np.empty(temp_shape), np.empty(temp_shape)
    if do_uncor:
        iuflux, iustd = np.empty(temp_shape), np.empty(temp_shape)
    else:
        iuflux, iustd = None, None

    start = 0
    for file_idx, (square, xr, yr) in enumerate(good_grid):

        file_flux = f[file_idx]
        end = start + file_flux.size

        if not square.any():
            log.debug(f'No good values in file {file_idx}')
            start = end
            continue

        file_wave = w[start:end].reshape(file_flux.shape)
        file_x = x[start:end].reshape(file_flux.shape)
        file_y = y[start:end].reshape(file_flux.shape)
        file_error = e[file_idx]
        if do_uncor:
            file_u_flux = uf[file_idx]
            file_u_wave = uw[start:end].reshape(file_u_flux.shape)
            file_u_error = ue[file_idx]
        else:
            file_u_flux = file_flux
            file_u_wave = file_wave
            file_u_error = file_error

        x_out = x_grid[yr[0]:yr[1], xr[0]:xr[1]][square]
        y_out = y_grid[yr[0]:yr[1], xr[0]:xr[1]][square]

        # loop over spaxels to resample spexels
        for spaxel in range(n_spax):
            # all wavelengths, y=i, x=j
            s = slice(None), spaxel

            try:
                resampler = Resample(
                    file_wave[s], file_flux[s], error=file_error[s],
                    window=fit_wdw, order=order,
                    robust=robust, negthresh=neg_threshold)

                iflux[:, spaxel], istd[:, spaxel] = resampler(
                    w_out, smoothing=smoothing_wdw,
                    fit_threshold=fit_threshold,
                    edge_threshold=edge_threshold,
                    edge_algorithm='distribution',
                    get_error=True, error_weighting=error_weighting)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                log.debug(f'Math error in resampler at '
                          f'spaxel {spaxel} for file {file_idx}')
                iflux[:, spaxel], istd[:, spaxel] = np.nan, np.nan

            if do_uncor:
                try:
                    resampler = Resample(
                        file_u_wave[s], file_u_flux[s], error=file_u_error[s],
                        window=fit_wdw, order=order, robust=robust,
                        negthresh=neg_threshold)
                    iuflux[:, spaxel], iustd[:, spaxel] = resampler(
                        w_out, smoothing=smoothing_wdw,
                        fit_threshold=fit_threshold,
                        edge_threshold=edge_threshold,
                        edge_algorithm='distribution',
                        get_error=True, error_weighting=error_weighting)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    log.debug(f'Math error in resampler at '
                              f'spaxel {spaxel} for file {file_idx}')
                    iuflux[:, spaxel], iustd[:, spaxel] = np.nan, np.nan

        # x and y coordinates for resampled fluxes -- take from first spexel
        xi, yi = file_x[0], file_y[0]

        # check for useful data
        valid = np.isfinite(iflux) & np.isfinite(istd)
        wave_ok = np.sum(valid, axis=1) > min_points
        if do_uncor:
            u_valid = np.isfinite(iuflux) & np.isfinite(iustd)
            u_wave_ok = np.sum(u_valid, axis=1) > min_points
        else:
            u_valid = None
            u_wave_ok = np.full_like(wave_ok, False)

        for wave_i in range(nw):
            if wave_ok[wave_i]:
                idx = valid[wave_i]
                rbf = Rbf(xi[idx], yi[idx], iflux[wave_i][idx], **kwargs)
                new_flux = np.zeros(square.shape)
                new_flux[square] = rbf(x_out, y_out)
                flux[wave_i, yr[0]:yr[1], xr[0]:xr[1]] += new_flux

                rbf = Rbf(xi[idx], yi[idx], istd[wave_i][idx], **kwargs)
                new_std = np.zeros(square.shape)
                new_std[square] = rbf(x_out, y_out) ** 2
                std[wave_i, yr[0]:yr[1], xr[0]:xr[1]] += new_std

                counts[wave_i, yr[0]:yr[1], xr[0]:xr[1]] += square

            if do_uncor:
                if u_wave_ok[wave_i]:
                    idx = u_valid[wave_i]

                    rbf = Rbf(xi[idx], yi[idx], iuflux[wave_i][idx], **kwargs)
                    new_flux = np.zeros(square.shape)
                    new_flux[square] = rbf(x_out, y_out)
                    uflux[wave_i, yr[0]:yr[1], xr[0]:xr[1]] += new_flux

                    rbf = Rbf(xi[idx], yi[idx], iustd[wave_i][idx], **kwargs)
                    new_std = np.zeros(square.shape)
                    new_std[square] = rbf(x_out, y_out) ** 2
                    ustd[wave_i, yr[0]:yr[1], xr[0]:xr[1]] += new_std

                    ucounts[wave_i, yr[0]:yr[1], xr[0]:xr[1]] += square

        start = end

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
    factor = (grid_info['delta'][1] / grid_info['pix_size']
              ).decompose().value ** 2
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
    unit_fit_wdw = (grid_info["wave_fwhm"] * window).to('um').value
    unit_smooth_wdw = (grid_info["wave_fwhm"] * smoothing).to('um').value

    hdinsert(combined['PRIMEHEAD'], 'WVFITWDW', str(unit_fit_wdw),
             comment='Wave resample fit window (um)')
    hdinsert(combined['PRIMEHEAD'], 'WVFITORD', str(order),
             comment='Wave resample fit order')
    hdinsert(combined['PRIMEHEAD'], 'WVFITSMR', str(unit_smooth_wdw),
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

    # In pixel units
    xy_fwhm = grid_info['oversample'][1]
    w_fwhm = grid_info['oversample'][0]

    fit_wdw = (window[0] * xy_fwhm,
               window[1] * xy_fwhm,
               window[2] * w_fwhm)

    smth_wdw = (smoothing[0] * xy_fwhm,
                smoothing[1] * xy_fwhm,
                smoothing[2] * w_fwhm)

    log.info(f'Fit window (x, y, w): '
             f'{(fit_wdw[0] * abs(grid_info["delta"][2]).to("arcsec")):.2f} '
             f'{(fit_wdw[1] * abs(grid_info["delta"][1]).to("arcsec")):.2f} '
             f'{(fit_wdw[2] * abs(grid_info["delta"][0]).to("um")):.5f}')

    log.info(f'Gaussian width of smoothing function (x, y, w): '
             f'{(smth_wdw[0] * abs(grid_info["delta"][2]).to("arcsec")):.2f} '
             f'{(smth_wdw[1] * abs(grid_info["delta"][1]).to("arcsec")):.2f} '
             f'{(smth_wdw[2] * abs(grid_info["delta"][0]).to("um")):.5f}')

    if adaptive_threshold is not None:
        log.info(f'Adaptive algorithm: {adaptive_algorithm}')
        log.info(f'Adaptive smoothing threshold (x, y, w): '
                 f'{adaptive_threshold[0]:.2f}, {adaptive_threshold[1]:.2f}, '
                 f'{adaptive_threshold[2]:.2f}')
    else:
        adaptive_threshold = 0
        adaptive_algorithm = None

    scan_reduction = combined.get('scan_reduction', False)
    if scan_reduction:  # pragma: no cover
        flxvals = combined['FLUX']
        errvals = combined['ERROR']
    else:
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
            *grid_info['grid'], smoothing=smth_wdw,
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

        if grid_info['uncorrected_coordinates'] is None:  # pragma: no cover
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
                *grid_info['grid'], smoothing=smth_wdw,
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
    factor = (grid_info['delta'][1] / grid_info['pix_size']
              ).decompose().value ** 2
    log.info(f'Flux correction factor: {factor}')

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
    hdinsert(combined['PRIMEHEAD'], 'XYFITSMR', str(smth_wdw),
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
    hdinsert(primehead, 'PIXSCAL', grid_info['delta'][1].to('arcsec').value)
    hdinsert(primehead, 'XYOVRSMP', str(grid_info['oversample']),
             comment='WXY Oversampling (pix per mean FWHM)')

    obsbet = primehead['OBSDEC'] - (primehead['DBET_MAP'] / 3600)
    obslam = (primehead['OBSRA'] * 15)
    obslam -= primehead['DLAM_MAP'] / (3600 * np.cos(np.radians(obsbet)))

    wcs = grid_info['wcs']
    wcs_info = wcs.to_header()

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
        if 'CTYPE1' in wcs_info:
            detector_coordinates = False
            hdinsert(h, 'EQUINOX', 2000.0, comment='Coordinate equinox')
            hdinsert(h, 'CTYPE1', wcs_info['CTYPE1'],
                     comment='Axis 1 type and projection')
            hdinsert(h, 'CTYPE2', wcs_info['CTYPE2'],
                     comment='Axis 2 type and projection')
            hdinsert(h, 'CTYPE3', wcs_info['CTYPE3'],
                     comment='Axis 3 type and projection')
            hdinsert(h, 'CUNIT1', 'deg', comment='Axis 1 units')
            hdinsert(h, 'CUNIT2', 'deg', comment='Axis 2 units')
            hdinsert(h, 'CUNIT3', 'um', comment='Axis 3 units')
            hdinsert(h, 'CRVAL1', wcs_info['CRVAL1'],
                     comment='RA (deg) at CRPIX1,2')
            hdinsert(h, 'CRVAL2', wcs_info['CRVAL2'],
                     comment='Dec (deg) at CRPIX1,2')
            hdinsert(h, 'CRVAL3', wcs_info['CRVAL3'] * 1e6,
                     comment='Wavelength (um) at CRPIX3')
            hdinsert(h, 'CDELT1', wcs_info['CDELT1'],
                     comment='RA pixel scale (deg/pix)')
            hdinsert(h, 'CDELT2', wcs_info['CDELT2'],
                     comment='Dec pixel scale (deg/pix)')
            hdinsert(h, 'CDELT3', wcs_info['CDELT3'] * 1e6,
                     comment='Wavelength pixel scale (um/pix)')
        else:
            detector_coordinates = True
            hdinsert(h, 'CTYPE1', 'X',
                     comment='Axis 1 type and projection')
            hdinsert(h, 'CTYPE2', 'Y',
                     comment='Axis 2 type and projection')
            hdinsert(h, 'CTYPE3', 'WAVE',
                     comment='Axis 3 type and projection')
            hdinsert(h, 'CUNIT1', 'arcsec', comment='Axis 1 units')
            hdinsert(h, 'CUNIT2', 'arcsec', comment='Axis 2 units')
            hdinsert(h, 'CUNIT3', 'um', comment='Axis 3 units')
            hdinsert(h, 'CRVAL1', wcs_info['CRVAL1'],
                     comment='X (arcsec) at CRPIX1,2')
            hdinsert(h, 'CRVAL2', wcs_info['CRVAL2'],
                     comment='Y (arcsec) at CRPIX1,2')
            hdinsert(h, 'CRVAL3', wcs_info['CRVAL3'],
                     comment='Wavelength (um) at CRPIX3')
            hdinsert(h, 'CDELT1', wcs_info['CDELT1'],
                     comment='RA pixel scale (arcsec/pix)')
            hdinsert(h, 'CDELT2', wcs_info['CDELT2'],
                     comment='Dec pixel scale (arcsec/pix)')
            hdinsert(h, 'CDELT3', wcs_info['CDELT3'],
                     comment='Wavelength pixel scale (um/pix)')

        hdinsert(h, 'CRPIX1', wcs_info['CRPIX1'],
                 comment='Reference pixel (x)')
        hdinsert(h, 'CRPIX2', wcs_info['CRPIX2'],
                 comment='Reference pixel (y)')
        hdinsert(h, 'CRPIX3', wcs_info['CRPIX3'],
                 comment='Reference pixel (z)')

        hdinsert(h, 'CROTA2', -primehead.get('SKY_ANGL', 0.0),
                 comment='Rotation angle (deg)')
        hdinsert(h, 'SPECSYS', 'BARYCENT',
                 comment='Spectral reference frame')
        # add beam keywords
        hdinsert(h, 'BMAJ', grid_info['xy_fwhm'].to('degree').value,
                 comment='Beam major axis (deg)')
        hdinsert(h, 'BMIN', grid_info['xy_fwhm'].to('degree').value,
                 comment='Beam minor axis (deg)')
        hdinsert(h, 'BPA', 0.0, comment='Beam position angle (deg)')

    # interpolate smoothed ATRAN and response data onto new grid for
    # reference
    resolution = grid_info['resolution']
    dw = (grid_info['delta'][0] / 2).to('um').value

    gx, gy, gw = grid_info['grid']

    x_out = wcs.wcs_pix2world(gx, [0], [0], 0)[0]
    y_out = wcs.wcs_pix2world([0], gy, [0], 0)[1]
    w_out = wcs.wcs_pix2world([0], [0], gw, 0)[2]
    if not detector_coordinates:
        w_out *= 1e6

    wmin = w_out.min()
    wmax = w_out.max()
    if 'UNSMOOTHED_TRANSMISSION' in combined:

        unsmoothed_atran = combined['UNSMOOTHED_TRANSMISSION']
        try:
            smoothed = smoothres(unsmoothed_atran[0], unsmoothed_atran[1],
                                 resolution)

            # Interpolate transmission to new wavelengths
            w = unsmoothed_atran[0]
            atran = np.interp(w_out, w, smoothed, left=np.nan, right=np.nan)

            # Keep unsmoothed data as is, but cut to wavelength range
            keep = (w >= (wmin - dw)) & (w <= (wmax + dw))
            unsmoothed_atran = unsmoothed_atran[:, keep]
        except (ValueError, TypeError, IndexError):
            log.error('Problem in interpolation.  '
                      'Setting TRANSMISSION to 1.0.')
            atran = np.full(w_out.shape, 1.0)
    else:
        atran = np.full(w_out.shape, 1.0)
        unsmoothed_atran = None

    response = get_response(primehead)
    try:
        resp = np.interp(w_out, response[0], response[1],
                         left=np.nan, right=np.nan)
    except (ValueError, TypeError, IndexError):
        log.error('Problem in interpolation.  '
                  'Setting RESPONSE to 1.0.')
        resp = np.full(w_out.shape, 1.0)

    # Add the spectral keys to primehead
    hdinsert(primehead, 'RESOLUN', resolution)
    hdinsert(primehead, 'SPEXLWID', grid_info['delta'][0].to('um').value)

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
    hdul.append(fits.ImageHDU(data=w_out, name='WAVELENGTH',
                              header=exthdr_1d))
    if detector_coordinates:
        exthdr_1d['BUNIT'] = 'arcsec'
        hdul.append(fits.ImageHDU(data=x_out, name='XS', header=exthdr_1d))
        hdul.append(fits.ImageHDU(data=y_out, name='YS', header=exthdr_1d))
    else:
        exthdr_1d['BUNIT'] = 'degree'
        hdul.append(fits.ImageHDU(data=x_out, name=wcs_info['CTYPE1'],
                                  header=exthdr_1d))
        hdul.append(fits.ImageHDU(data=y_out, name=wcs_info['CTYPE2'],
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


def perform_scan_reduction(filenames, scan_kwargs=None,
                           reduce_uncorrected=True,
                           save_scan=False,
                           insert_source=True):  # pragma: no cover
    """
    Reduce the files using a scan reduction.

    Parameters
    ----------
    filenames : list (str)
        A list of FIFI-LS WSH FITS files to reduce.
    scan_kwargs : dict, optional
        Keyword arguments to pass into the scan reduction.
    reduce_uncorrected : bool, optional
        If `True`, reduce the uncorrected flux values as well.
    save_scan : bool, optional
        If `True`, files produced by the scan reduction will be saved to disk.
    insert_source : bool, optional
        If `True`, will perform a full scan reduction and reinsert the source
        after.  Otherwise, the reduction is used to calculate gains, offsets,
        and correlations which will then be applied to the original data.
        If `True`, note that timestream filtering will not be applied to the
        correction and should therefore be excluded from the scan reduction
        runtime parameters in order to reduce processing pressure.

    Returns
    -------
    combined : dict
    """
    # lazy import, in case scan is not installed
    from sofia_redux.scan.reduction.reduction import Reduction

    if scan_kwargs is None:
        scan_kwargs = {}

    if 'grid' not in scan_kwargs:
        hdul = gethdul(filenames[0])
        header = hdul[0].header

        w_mid = (hdul['LAMBDA'].data.max() + hdul['LAMBDA'].data.min()) / 2
        if not isinstance(filenames[0], fits.HDUList):
            hdul.close()

        # Begin with spectral scalings
        resolution = get_resolution(header, wmean=w_mid)
        spectral_fwhm = w_mid / resolution
        spatial_fwhm = get_resolution(header, spatial=True, wmean=w_mid)
        scan_kwargs['grid'] = f'{spatial_fwhm / 3},{spectral_fwhm / 3}'

    # save intermediate files if desired
    if save_scan:
        scan_kwargs['write'] = {'source': True}
    else:
        scan_kwargs['write'] = {'source': False}

    scan_kwargs.update({'fifi_ls': {'resample': 'True',
                                    'insert_source': insert_source}})

    reduction = Reduction('fifi_ls')
    reduction.run(filenames, **scan_kwargs)

    temporary_directory = tempfile.mkdtemp('fifi_resample_scan_reductions')
    reduction_file = os.path.join(temporary_directory, 'reduction.p')

    with open(reduction_file, 'wb') as f:
        cloudpickle.dump(reduction, f)

    # Save to disk for now to free up memory
    del reduction
    gc.collect()

    # "fifi_ls.uncorrected" is the toggle to perform a SOFSCAN reduction on
    # the uncorrected data values
    if reduce_uncorrected:
        u_reduction = Reduction('fifi_ls')
        scan_kwargs['fifi_ls']['uncorrected'] = True
        u_reduction.run(filenames, **scan_kwargs)
        u_reduction_file = os.path.join(temporary_directory, 'u_reduction.p')
        with open(u_reduction_file, 'wb') as f:
            cloudpickle.dump(u_reduction, f)
        del u_reduction
        gc.collect()
    else:
        u_reduction_file = None

    combined = combine_scan_reductions(
        reduction_file, uncorrected_reduction=u_reduction_file)

    combined['reduction_file'] = reduction_file
    combined['uncorrected_reduction_file'] = u_reduction_file
    return combined


def resample(filenames, target_x=None, target_y=None, target_wave=None,
             ctype1='RA---TAN', ctype2='DEC--TAN', ctype3='WAVE',
             interp=False, oversample=None, spatial_size=None,
             spectral_size=None, window=None,
             adaptive_threshold=None, adaptive_algorithm=None,
             error_weighting=True, smoothing=None, order=2,
             robust=None, neg_threshold=None, fit_threshold=None,
             edge_threshold=None, append_weights=False,
             skip_uncorrected=False, write=False, outdir=None,
             jobs=None, check_memory=True, scan_reduction=False,
             scan_kwargs=None, save_scan=False, detector_coordinates=None,
             naif_id_key='NAIF_ID', insert_source=True):
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
    target_x : float, optional
        The target right ascension (hourangle).  The default is the mid-point
        of all RA values in the combined data.
    target_y : float, optional
        The target declination (degree).  The default is the mid-point of all
        DEC values in the combined data.
    target_wave : float, optional
        The center wavelength (um).  The default is the mid-point of all
        wavelength values in the combined data.
    ctype1 : str, optional
        The coordinate frame for the x spatial axis using FITS standards.
    ctype2 : str, optional
        The coordinate frame for the y spatial axis using FITS standards.
    ctype3 : str, optional
        The coordinate frame for the w spectral axis using FITS standards.
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
    scan_reduction : bool, optional
        If `True`, run a scan reduction first before performing the resampling
        step.  This may be a very time consuming operation, but may also remove
        many correlated noise signals from the data.
    save_scan : bool, optional
        If `True`, the output from the scan algorithm, prior to resampling,
        will be saved to disk.
    scan_kwargs : dict, optional
        Optional keyword arguments to pass into the scan reduction.
    detector_coordinates : bool, optional
        If `True`, reduce using detector coordinates instead of RA/DEC.  if
        `None`, will attempt to auto-detect based on OBSLAM/OBSDEC header
        values (`True` if all OBSLAM/DEC = 0, `False` otherwise).
    naif_id_key : str, optional
        The name of the NAIF ID keyword.  If present in the header, should
        indicate the associated file contains a nonsidereal observation.
    insert_source : bool, optional
        If `True`, will perform a full scan reduction (if applicable) and
        reinsert the source after.  Otherwise, the reduction is used to
        calculate gains, offsets, and correlations which will then be
        applied to the original data. If `True`, note that timestream
        filtering will not be applied to the correction and should therefore
        be excluded from the scan reduction runtime parameters in order to
        reduce processing pressure.

    Returns
    -------
    fits.HDUList or str
        Either the HDU (if write is False) or the filename of the output
        file (if write is True).  The output contains the following
        extensions: FLUX, ERROR, WAVELENGTH, X, Y, RA, DEC,
        TRANSMISSION, RESPONSE, EXPOSURE_MAP.  The following extensions
        will be appended if possible: UNCORRECTED_FLUX, UNCORRECTED_ERROR,
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

    combined = combine_files(filenames, naif_id_key=naif_id_key,
                             scan_reduction=scan_reduction,
                             save_scan=save_scan,
                             scan_kwargs=scan_kwargs,
                             skip_uncorrected=skip_uncorrected,
                             insert_source=insert_source)

    interp |= combined['method'] == 'interpolate'
    if detector_coordinates is None:
        if combined['definite_nonsidereal']:
            log.info('Resampling using detector coordinates: nonsidereal')
            detector_coordinates = True
        elif combined['nonsidereal_values']:
            log.info('Resampling using detector coordinates: '
                     'possible non-sidereal observation')
            detector_coordinates = True
        else:
            log.info('Resampling using equatorial coordinates')
            detector_coordinates = False

    grid_info = get_grid_info(combined, oversample=oversample,
                              spatial_size=spatial_size,
                              spectral_size=spectral_size,
                              target_x=target_x,
                              target_y=target_y,
                              target_wave=target_wave,
                              ctype1=ctype1,
                              ctype2=ctype2,
                              ctype3=ctype3,
                              detector_coordinates=detector_coordinates)

    if grid_info is None:
        log.error('Problem in grid calculation')
        cleanup_scan_reduction(combined)
        return

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
        finally:
            cleanup_scan_reduction(combined)
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
        finally:
            cleanup_scan_reduction(combined)

    result = make_hdul(combined, grid_info, append_weights=append_weights)

    if not write:
        return result
    else:
        return write_hdul(result, outdir=outdir, overwrite=True)


def cleanup_scan_reduction(combined):
    """
    Remove all temporary files created during a scan reduction.

    Parameters
    ----------
    combined : dict
        The combined data set.  The 'reduction_file' and
        'uncorrected_reduction_file' keys should have string values pointing
        to the pickle file on disk.  Their parent directory will also be
        deleted.

    Returns
    -------
    None
    """
    try:
        delete_dir = None
        for key in ['reduction_file', 'uncorrected_reduction_file']:
            filename = combined.get(key)
            if filename is not None and os.path.isfile(filename):
                os.remove(filename)
                if delete_dir is None:
                    delete_dir = os.path.dirname(filename)
        if delete_dir is not None and os.path.isdir(delete_dir):
            shutil.rmtree(delete_dir)
    except Exception as err:  # pragma: no cover
        log.error(f"Problem cleaning scan reduction temporary files: {err}")
