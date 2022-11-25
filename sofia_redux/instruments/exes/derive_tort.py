# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import bottleneck as bn
import numpy as np
import pandas
from scipy.optimize import curve_fit

from sofia_redux.instruments.exes.get_resolution import get_resolution
from sofia_redux.instruments.exes.tort import tort
from sofia_redux.instruments.exes.utils import parse_central_wavenumber
from sofia_redux.toolkit.utilities.func import goodfile
from sofia_redux.toolkit.stats import meancomb
from sofia_redux.toolkit.convolve.filter import sobel

__all__ = ['derive_tort']


def derive_tort(data, header, maxiter=5, fixed=False,
                edge_method='deriv', custom_wavemap=None,
                top_pixel=None, bottom_pixel=None,
                start_pixel=None, end_pixel=None):
    """
    Derive distortion parameters and identify orders and illuminated regions.

    The input flat is first undistorted (`tort`) using default parameters.
    For cross-dispersed data, the distortion is tested and (optionally)
    optimized by doing the following:

        1. Enhance the edges in the image, using `edge_method`.
        2. Take the 2D FFT of the resulting image.
        3. Use the power maxima in the FFT image to determine the
           order spacing.
        4. Calculate the rotation angle from the FFT.
        5. If the angle is not close to zero, use it to correct the
           `krot` parameter and redo the distortion correction by
           reiterating this algorithm.

    This process must be closely monitored: if the flat frame has
    insufficient signal, or the optical parameters used to calculate the
    distortion correction are insufficiently accurate, the spacing and
    rotation angle may be wrongly calculated at this step.

    Parameters
    ----------
    data : numpy.ndarray
        2D image [nspec, nspat], typically a blackbody flat frame.
    header : fits.Header
        FITS header for the flat file containing distortion parameters.
    maxiter : int, optional
        The maximum number of iterations to use while optimizing distortion
        parameters.
    fixed : bool, optional
        If True, will not attempt to optimize distortion parameters.
    edge_method : {'deriv', 'sqderiv', 'sobel'}, optional
        Sets the edge enhancement method for optimizing the cross-dispersed
        distortion parameters.  May be one derivative ('deriv'),
        squared derivative ('sqderiv'), or Sobel ('sobel').
    custom_wavemap : str or bool, optional
        Filename for a text file containing explicit order edges for a
        cross-dispersed mode, as whitespace-separated integers indicating
        bottom, top, left, and right edges for each order (one per line).
        If set, it is used to directly set the order mask for a
        cross-dispersed mode. If set to a value other than a string or
        None, a 'customWVM.dat' file is expected in the current directory.
    top_pixel : int, optional
        If provided, is used to directly set the top edge of the order.
        Used for single-order modes only (medium, low); ignored for
        cross-dispersed modes.
    bottom_pixel : int, optional
        If provided, is used to directly set the bottom edge of the order.
        Used for single-order modes only (medium, low); ignored for
        cross-dispersed modes.
    start_pixel : int, optional
        If provided, is used to directly set the left edge of all orders.
        May be used for either single-order or cross-dispersed modes.
    end_pixel : int, optional
        If provided, is used to directly set the right edge of all orders.
        May be used for either single-order or cross-dispersed modes.

    Returns
    -------
    tortdata, tortillum : 2-tuple of numpy.ndarray
        Undistorted flat data and illumination array. Both have shape
        [nspec, nspat].  The illumination array contains integer values with
        1 = illuminated, 0 = unilluminated, -1 = pixel that
        does not correspond to any region in the raw frame (after
        distortion correction).

    """
    # tort data, with bilinear interpolation, no echelon slit skewing
    # The illumination (illum) marks data as out of bounds with -1 and
    # in bounds with 1.

    cross_dispersed = str(
        header['INSTCFG']).upper() in ['HIGH_MED', 'HIGH_LOW']
    if cross_dispersed:
        result = _process_cross_dispersed(
            data, header,
            method=edge_method, maxiter=maxiter,
            fixed=fixed, custom_wavemap=custom_wavemap,
            start_pixel=start_pixel, end_pixel=end_pixel)
    else:
        result = _process_single_order(
            data, header,
            top_pixel=top_pixel, bottom_pixel=bottom_pixel,
            start_pixel=start_pixel, end_pixel=end_pixel)

    tortdata, tortillum, b1, t1, s1, e1 = result
    _update_header(header, b1, t1, s1, e1)
    return tortdata, tortillum


def _gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y):
    return (amplitude
            * np.exp(-((xy[1] - y0) ** 2) / (2 * (sigma_y ** 2)))
            * np.exp(-((xy[0] - x0) ** 2) / (2 * (sigma_x ** 2)))).ravel()


def _get_derivative(data, header, illum, method='deriv', axis=1):

    nx = header['NSPAT']
    ny = header['NSPEC']

    # Make column and row index arrays
    y, x = np.mgrid[:ny, :nx]
    derivative = np.zeros((ny, nx), dtype=np.float64)

    if method in ['deriv', 'sqderiv']:
        select = (x >= 4) & (x < (nx - 4))
        select &= (y >= 4) & (y < (ny - 4))
        # 'and' by the illumination mask shifted by +/-4 in each direction
        illum_mask = np.pad(illum > 0, 4, mode='edge')
        select &= illum_mask[4:-4, 4:-4]
        select &= illum_mask[8:, 4:-4] & illum_mask[:-8, 4:-4]
        select &= illum_mask[4:-4, 8:] & illum_mask[4:-4, :-8]

        if select.any():
            derivative[select] = np.gradient(
                data, axis=axis, edge_order=2)[select]
            if method == 'sqderiv':  # square the derivative
                derivative *= derivative
    else:  # use sobel instead
        derivative = sobel(data * 1000 / data.max())

    return derivative


def _describe_orders(tortdata, header, illum, method='deriv'):
    # Make the first derivative array

    hrr = float(header['HRR'])  # R number for echelon grating
    hrdgr = float(header['HRDGR'])  # echelon grating groove spacing
    xdr = float(header['XDR'])  # Cross-dispersed R number
    xdfl = float(header['XDFL'])  # Assumed cross-dispersed focal length
    pixelwd = float(header['PIXELWD'])  # Pixel width
    hdr_spacing = float(header['SPACING'])  # Order separation in pixels
    krot = float(header['KROT'])

    # Planned or updated central wavenumber
    waveno0 = parse_central_wavenumber(header)

    derivative = _get_derivative(tortdata, header, illum, method=method)

    # Use a 2D FFT to determine orientation of orders
    # Array is twice the size of the input array
    ny, nx = derivative.shape
    dfft = np.zeros((ny * 2, nx * 2), dtype=np.float64)
    dfft[:ny, :nx] = derivative
    dfft = np.fft.fft2(dfft, norm='forward')

    dw = 0.5 / (np.sqrt((hrr ** 2) / (1 + (hrr ** 2))) * hrdgr)
    instcfg = header['INSTCFG'].upper()

    if instcfg == 'HIGH_LOW':
        cpredict = 2 * (2 * xdr) * (1.2 * xdfl) * dw / (waveno0 * pixelwd)
    else:
        cpredict = 2 * xdr * xdfl * dw / (waveno0 * pixelwd)

    if hdr_spacing != -9999:
        predict = hdr_spacing
    else:
        predict = cpredict
        header['SPACING'] = cpredict

    predict_idx = np.full(2, 2 * nx / predict)

    # Coordinate arrays for doubled array
    y2, x2 = np.mgrid[:(ny * 2), :(nx * 2)]

    ps = np.abs(dfft) ** 2

    # Check for peak near 2nd and fundamental harmonics
    xlim1 = predict_idx * [1.6, 0.8]
    xlim2 = predict_idx * [2.4, 1.2]
    ylim = predict_idx * [0.4, 0.2]

    npow = 512
    ky2 = 32
    powers = np.zeros((npow, npow), dtype=np.float64)
    powmax = np.zeros(2, dtype=np.float64)
    xloc_max = np.zeros(2, dtype=np.float64)
    yloc_max = np.zeros(2, dtype=np.float64)
    iymax = ixmax = -1

    log.info('')
    for harmonic in range(2):
        ipowmax = 0

        # bottom of image
        idx = (x2 >= xlim1[harmonic]) & (x2 <= xlim2[harmonic])
        idx &= y2 <= ylim[harmonic]
        if np.any(idx):
            maxi = np.argmax(ps[idx])
            ipowmax = ps[idx][maxi]
            ixmax = x2[idx][maxi]
            iymax = y2[idx][maxi]
            powers[y2[idx] + ky2, x2[idx]] = ps[idx]

        # top of image
        idx = (x2 >= xlim1[harmonic]) & (x2 <= xlim2[harmonic])
        idx &= y2 >= ps.shape[0] - ylim[harmonic]
        if np.any(idx):
            maxi = np.argmax(ps[idx])
            ipowmax2 = ps[idx][maxi]
            yidx = ky2 + y2[idx] - ps.shape[0]

            if ipowmax2 > ipowmax:
                ipowmax = ipowmax2
                ixmax = x2[idx][maxi]
                iymax = yidx[maxi]

            powers[yidx, x2[idx]] = ps[idx]

        if (ipowmax <= 0
                or ixmax in [xlim1[harmonic], xlim2[harmonic]]
                or iymax in [ky2 - ylim[harmonic], ky2 + ylim[harmonic]]):
            log.warning("Can't find harmonic max in 2D FFT")
            log.warning(f"ixmax={ixmax}, iymax={iymax}")

        powmax[harmonic] = ipowmax

        # Fit a 2D peak to power image
        xmin = int(np.clip(xlim1[harmonic], 0, npow - 7))
        xmax = int(np.clip(xlim2[harmonic], 6, npow)) + 1
        ymin = int(np.clip(ky2 - ylim[harmonic], 0, npow - 7))
        ymax = int(np.clip(ky2 + ylim[harmonic], 6, npow)) + 1
        fitdata = powers[ymin:ymax, xmin:xmax] / ipowmax

        yp, xp = np.mgrid[:fitdata.shape[0], :fitdata.shape[1]]
        guess_max = np.argmax(fitdata)
        p0 = (
            1.0,  # amplitude
            xp.ravel()[guess_max],  # x0
            yp.ravel()[guess_max],  # y0
            1.0,  # sigma_x
            1.0  # sigma_y
        )
        lower_bounds = [-np.inf, 0.0, 0.0, -np.inf, -np.inf]
        upper_bounds = [np.inf,
                        xlim2[harmonic] - xlim1[harmonic],
                        2 * ylim[harmonic],
                        np.inf,
                        np.inf]

        try:
            popt, _ = curve_fit(_gaussian_2d, (xp, yp), fitdata.ravel(),
                                p0=p0, bounds=(lower_bounds, upper_bounds))
            xloc_max[harmonic] = popt[1] + xmin
            yloc_max[harmonic] = popt[2] + ymin - ky2
        except (ValueError, RuntimeError) as err:
            log.debug(f"Error encountered during FFT peak location: {err}")
            xloc_max[harmonic] = ixmax
            yloc_max[harmonic] = iymax

    log.info(f"KROT={krot}")
    log.info(f"2D FFT 2nd harmonic vector (x, y): "
             f"{xloc_max[0]}, {yloc_max[0]}")
    log.info(f"2D FFT fundamental vector (x, y): "
             f"{xloc_max[1]}, {yloc_max[1]}")

    if powmax[0] > powmax[1]:
        log.info("Using 2nd harmonic for frequency")
        xmax = xloc_max[0] / 2
        ymax = yloc_max[0] / 2
        powmax = powmax[0]
    else:
        xmax = xloc_max[1]
        ymax = yloc_max[1]
        powmax = powmax[1]

    if powmax == 0:
        raise ValueError(
            "Can't determine order spacing: power spectrum maximum = 0")

    # r_spacing = 2 * nx / np.sqrt((xmax ** 2) + (ymax ** 2))
    spacing = 2 * nx / xmax
    angle = np.arctan2(ymax, xmax)

    log.info(f"Order spacing, angle: {spacing}, {angle}")
    log.info(f"Predicted order spacing (header): {predict} ({cpredict})")

    if np.abs(spacing - predict) > (predict / 50):
        log.warning("Order spacing disagrees with prediction")
    if np.abs(angle) > 0.02:
        log.warning("FFT angle > 0.02")

    return spacing, angle


def _get_xd_power(tortdata, header):

    # Sum derivative over y and orders to find orders
    spacing = header['SPACING']
    nx = header['NSPAT']
    threshold_factor = header['THRFAC']
    nxo = spacing + 1
    norder = int((nx / nxo) - 2)

    nx2 = int(2 * nxo)

    # Make a summed spatial profile (x-direction) of the order illumination
    power = np.empty(nx2, dtype=np.float64)
    deriv = np.empty(nx2, dtype=np.float64)

    offsets = (np.round((np.arange(norder) + 1) * spacing)).astype(int)
    for i in range(nx2):
        cols = i + offsets
        power[i] = meancomb(tortdata[:, cols], robust=4.0, returned=False)
        diff = tortdata[:, cols + 1] - tortdata[:, cols - 1]
        deriv[i] = meancomb(diff, robust=4.0, returned=False)

    # Find the beginning of the order by looking for the min/max values
    # in the derivative.
    nx1 = int((nxo / 2 - 1))
    nx2 = int(nx1 + nxo)

    dermin_idx = np.argmin(deriv[nx1:nx2]) + nx1
    dermax_idx = np.argmax(deriv[nx1:nx2]) + nx1

    # Average nearby positions to get better value for min and max derivative
    # (location of rise and fall)
    d = deriv[dermin_idx - 1: dermin_idx + 2]
    x_fall = dermin_idx + 0.5 * (d[0] - d[2]) / (d[0] - (2 * d[1]) + d[2])
    if np.isnan(x_fall):
        x_fall = dermin_idx

    d = deriv[dermax_idx - 1: dermax_idx + 2]
    x_rise = dermax_idx + 0.5 * (d[0] - d[2]) / (d[0] - (2 * d[1]) + d[2])
    if np.isnan(x_rise):
        x_rise = dermax_idx

    if x_fall > x_rise:
        x_fall -= spacing

    x_order1 = (x_fall + x_rise) / 2

    if x_order1 < 0:  # pragma: no cover
        x_order1 += spacing
    elif x_order1 > spacing:
        x_order1 -= spacing

    # Make a new spatial profile, starting at the beginning of the order
    # (x_order1)
    norder = int((nx - x_order1) / spacing)
    nxo = int(spacing) + 1
    power = np.zeros(nxo, dtype=np.float64)
    offsets = (np.round(np.arange(norder) * spacing + x_order1)).astype(int)
    for i in range(nxo):
        cols = i + offsets
        power[i] = meancomb(tortdata[:, cols], robust=4.0, returned=False)

    # Find the illumination threshold based on the power profile
    # and mark unilluminated edges of the order profile
    powmin = np.min(power)
    powmax = np.max(power)
    threshold = powmin + (threshold_factor * (powmax - powmin))
    illum_profile = power >= threshold
    if not illum_profile.any():
        raise ValueError("No illuminated pixels")
    illum_idx = np.nonzero(illum_profile)[0]

    # Illumination profile in x-direction
    illum_x = np.full(nx, -1)
    start_x = (np.round(x_order1 + (np.arange(norder) * spacing))).astype(int)
    nprof = illum_profile.size
    i0 = illum_idx[0]
    for i, start_i in enumerate(start_x):
        nleft = nx - start_i
        if start_i < 0:  # pragma: no cover
            illum_x[(start_i + i0):(start_i + nprof)] = \
                illum_profile[i0:].copy()
        elif nleft < nprof:  # pragma: no cover
            illum_x[start_i: (start_i + nprof)] = illum_profile.copy()
        else:
            maxidx = min(nleft, nprof)
            illum_x[start_i:(start_i + maxidx)] = illum_profile[:maxidx]

    log.info(f"NORDERS, SPACING, XORDER1, NT = "
             f"{norder}, {spacing}, {x_order1}, {int(spacing)}")
    header['NORDERS'] = norder
    header['XORDER1'] = x_order1

    # Check there are not too many unilluminated pixels
    nbelow = (illum_x == 0).sum() / norder
    log.info(f"Power below threshold in {nbelow} pixels per order")
    if (not header['PINHOLE']
            and nbelow > (int(spacing + 1) / 2)):  # pragma: no cover
        log.warning("Too many pixels with power below threshold")
    header['NBELOW'] = nbelow

    return power, illum_x


def _process_cross_dispersed(data, header, method='deriv',
                             maxiter=5, fixed=False, custom_wavemap=None,
                             start_pixel=None, end_pixel=None):

    tortdata, tortillum = tort(data, header, order=1,
                               missing=0.0, skew=False, get_illum=True)
    spacing, angle = _describe_orders(
        tortdata, header, tortillum, method=method)

    angle_limit = 0.001
    for iteration in range(maxiter):

        if not fixed and np.abs(angle) > angle_limit:
            log.info('')
            log.info(f"KROT Iteration {iteration + 1}")
            header['KROT'] -= angle
            log.info(f"Changing KROT to correct angle; krot={header['KROT']}")

            tortdata, tortillum = tort(
                data, header, order=1, missing=0.0,
                skew=False, get_illum=True)

            spacing, angle = _describe_orders(
                tortdata, header, tortillum, method=method)

            if iteration == (maxiter - 1):
                log.warning(
                    f"{maxiter} iterations used, not adjusting KROT further")

            elif np.abs(angle) > angle_limit:
                continue  # attempt to find angle again

        elif fixed and np.abs(angle) > angle_limit:
            log.warning(f"Angle > {angle_limit}.  KROT may need adjusting")

        old_spacing = header['SPACING']
        if np.isclose(spacing, old_spacing, atol=0.001):
            log.warning(f"Order spacing changed from "
                        f"{old_spacing} to {spacing}")
            log.warning("Setting it back to avoid unnecessary modification")
        else:
            header['SPACING'] = spacing
            header['NT'] = int(spacing)

        try:
            power, illum_x = _get_xd_power(tortdata, header)
        except (IndexError, ValueError):
            raise ValueError('Illuminated power could not be found') from None

        tortillum, illum_y = _update_illumination(
            tortillum, tortdata, header, illum_x)

        b1, t1, ss1, ee1 = _get_top_bottom_pixels(
            header, illum_x, illum_y, custom_wavemap=custom_wavemap)

        if b1 is not None and t1 is not None:
            break

    else:
        raise ValueError("Order edges could not be found.  Try increasing "
                         "edge tolerance.")

    s1, e1 = _get_left_right_pixels(header, tortillum, b1, t1,
                                    ss1=ss1, ee1=ee1,
                                    start_pixel=start_pixel,
                                    end_pixel=end_pixel)

    return tortdata, tortillum, b1, t1, s1, e1


def _process_single_order(data, header,
                          bottom_pixel=None, top_pixel=None,
                          start_pixel=None, end_pixel=None):

    tortdata, tortillum = tort(data, header, order=1,
                               missing=0.0, skew=False, get_illum=True)

    nx = header['NSPAT']
    illum_x = np.full(nx, 1)
    header['NORDERS'] = 1
    header['XORDER1'] = 0

    tortillum, illum_y = _update_illumination(
        tortillum, tortdata, header, illum_x)

    b1, t1, ss1, ee1 = _get_top_bottom_pixels(
        header, illum_x, illum_y,
        bottom_pixel=bottom_pixel, top_pixel=top_pixel)

    s1, e1 = _get_left_right_pixels(
        header, tortillum, b1, t1, ss1=ss1, ee1=ee1,
        start_pixel=start_pixel, end_pixel=end_pixel)

    return tortdata, tortillum, b1, t1, s1, e1


def _update_illumination(tortillum, tortdata, header, illum_x):

    ny = header['NSPEC']
    cross_dispersed = str(header['INSTCFG']).upper() in [
        'HIGH_MED', 'HIGH_LOW']

    # Keep the minimum value for each pixel
    tortillum = np.clip(tortillum, None, illum_x[None])

    # Determine illumination in y
    # Take the minimum illuminated pixels from
    # 3/4 * total pixels illuminated in x-direction
    illmin = 3 * int(np.sum(illum_x > 0)) // 4

    # Get power by summing over y and dividing by the number of illuminated
    # pixels.  Ignore unilluminated pixels and rows where the number of
    # pixels is less than the minimum required.
    idx = tortillum != 1
    illdata = tortillum.copy()
    illdata[idx] = 0
    illsum = illdata.sum(axis=1)
    powdata = tortdata.copy()
    powdata[idx] = 0
    sumb = bn.nansum(tortdata, axis=1)
    idx = illsum > illmin
    power = np.zeros(ny, dtype=np.float64)
    power[idx] = sumb[idx] / illsum[idx]

    # Use power threshold to make illumination mask
    thresh = header['THRFAC'] * power.max()
    illum_y = power >= thresh

    # Update illum with illum_y values keeping minumum values.
    # NOTE: There is an inconsistency here in the source code.  illum is
    # updated here for all modes, but it is overwritten after makeflat
    # by setillum, which uses illx only.  illx is updated for the
    # y-illumination only for LOW and MED modes.
    if not cross_dispersed:
        tortillum = np.clip(tortillum, None, illum_y[:, None])

    return tortillum, illum_y


def _get_top_bottom_pixels(header, illum_x, illum_y,
                           bottom_pixel=None, top_pixel=None,
                           custom_wavemap=None):

    nx = header['NSPAT']
    ny = header['NSPEC']

    cross_dispersed = str(header['INSTCFG']).upper() in [
        'HIGH_MED', 'HIGH_LOW']
    y_idx = np.nonzero(illum_y)[0]
    ss1 = None
    ee1 = None

    if not cross_dispersed:
        header['ROTATION'] = 0
        b1 = y_idx[0] if bottom_pixel is None else int(bottom_pixel)
        t1 = y_idx[-1] if top_pixel is None else int(top_pixel)
        b1 = np.array([b1])
        t1 = np.array([t1])

    elif custom_wavemap is not None:

        header['ROTATION'] = 3

        if not isinstance(custom_wavemap, str):
            custom_wavemap_file = 'customWVM.dat'
        else:
            custom_wavemap_file = custom_wavemap

        log.info('')
        if not goodfile(custom_wavemap_file, verbose=True):
            raise ValueError(f"Could not apply modification "
                             f"from {custom_wavemap_file}")
        log.info(f'Using custom wavemap {custom_wavemap_file} to '
                 f'modify order edges.')
        table = pandas.read_csv(custom_wavemap_file, delim_whitespace=True,
                                dtype=int, comment='#',
                                names=['b1', 't1', 'ss1', 'ee1'])

        b1 = table['b1'].values
        t1 = table['t1'].values
        ss1 = table['ss1'].values
        ee1 = table['ee1'].values
        header['NORDERS'] = len(b1)
    else:
        # cross-dispersion, no modification file
        header['ROTATION'] = 3
        x = np.arange(nx)
        b1 = (x < (nx - 1)) & (illum_x <= 0) & (np.roll(illum_x, -1) > 0)
        t1 = (x > 0) & (np.roll(illum_x, 1) > 0) & (illum_x <= 0)
        b1 = np.nonzero(b1)[0] + 1
        t1 = np.nonzero(t1)[0] - 1
        if b1.size < t1.size:
            norder = b1.size
            t1 = t1[1: (norder + 1)]
        elif t1.size < b1.size:
            norder = t1.size
            b1 = b1[0: norder]
        else:
            norder = t1.size
        header['NORDERS'] = norder

        # Check to see if all orders found
        if norder == 0 or (b1.size != norder) or (t1.size != norder):
            log.info(f"Number of orders: {norder}")
            log.info(f"Number of top edges: {t1.size}")
            log.info(f"Number of bottom edges {b1.size}")
            log.warning("Not all orders were found")
            return None, None, ss1, ee1

        # Invert B and T to correspond with rotated data
        tmp = b1.copy()
        b1 = ny - t1 - 1
        t1 = ny - tmp - 1

    return b1, t1, ss1, ee1


def _get_left_right_pixels(header, tortillum, b1, t1,
                           ss1=None, ee1=None,
                           start_pixel=None,
                           end_pixel=None):

    nx = header['NSPAT']
    ny = header['NSPEC']
    norders = header['NORDERS']
    cross_dispersed = str(header['INSTCFG']).upper() in [
        'HIGH_MED', 'HIGH_LOW']
    y, x = np.mgrid[:ny, :nx]

    s1 = np.empty(norders, dtype=int)
    e1 = np.empty(norders, dtype=int)
    for i in range(norders):
        i2ord = str(norders - i).zfill(2)
        onum = f'ODR{i2ord}'

        if cross_dispersed:
            header[onum + '_B1'] = b1[i]
            header[onum + '_T1'] = t1[i]

            xlim1 = ny - t1[i] - 1
            xlim2 = ny - b1[i] - 1
            illum_ord = (x >= xlim1) & (x <= xlim2) & (tortillum == 1)

            if ss1 is None:
                bxr = illum_ord & ((y == 0)
                                   | (np.roll(tortillum, 1, axis=0) < 1))
                txr = illum_ord & ((y == (ny - 1))
                                   | (np.roll(tortillum, -1, axis=0) < 1))
                if start_pixel is not None:
                    ts1 = int(start_pixel)
                else:
                    if not np.any(bxr):  # pragma: no cover
                        ts1 = 2
                    else:
                        ts1 = np.argwhere(bxr)[-1][0] + 2
                if end_pixel is not None:
                    te1 = int(end_pixel)
                else:
                    if not np.any(txr):  # pragma: no cover
                        te1 = ny - 3
                    else:
                        te1 = np.argwhere(txr)[0][0] - 2
            else:
                # where b1,t1,ss1,ee1 were retrieved from file
                ts1 = ss1[i]
                te1 = ee1[i]
        else:
            # single order
            header[onum + '_B1'] = b1[i]
            header[onum + '_T1'] = t1[i]
            illum_ord = (y >= b1[i]) & (y <= t1[i]) & (tortillum == 1)

            bxr = illum_ord & ((x == 0) | (np.roll(tortillum, 1, axis=1) < 1))
            txr = illum_ord & ((x == (nx - 1))
                               | (np.roll(tortillum, -1, axis=1) < 1))

            if start_pixel is not None:
                ts1 = int(start_pixel)
            else:
                if not np.any(bxr):
                    ts1 = 2
                else:
                    ts1 = np.argwhere(bxr)[-1][1] + 2
            if end_pixel is not None:
                te1 = int(end_pixel)
            else:
                if not np.any(txr):
                    te1 = nx - 3
                else:
                    te1 = np.argwhere(txr)[0][1] - 2

        s1[i] = ts1
        e1[i] = te1

        header[onum + '_XR'] = f'{ts1},{te1}'
        header.comments[onum + '_XR'] = f'Extraction range for order {i2ord}'
        header.comments[onum + '_T1'] = \
            f'Edge coefficient for top of order {i2ord}'
        header.comments[onum + '_B1'] = \
            f'Edge coefficient for bottom of order {i2ord}'

    return s1, e1


def _update_header(header, b1, t1, s1, e1):

    norders = header['NORDERS']

    # Set more values in header
    header['ORDR_B'] = ','.join([str(x) for x in b1])
    header['ORDR_T'] = ','.join([str(x) for x in t1])
    header['ORDR_S'] = ','.join([str(x) for x in s1])
    header['ORDR_E'] = ','.join([str(x) for x in e1])
    header['EDGEDEG'] = 0
    header['ORDERS'] = ','.join(reversed([str(x + 1) for x in range(norders)]))
    header['SLTH_PIX'] = float(np.mean(np.abs(t1 - b1)))
    header['SLTH_ARC'] = header['SLTH_PIX'] * header['PLTSCALE']

    # get slit width from assumed value
    header['SLTW_ARC'] = header['SLITWID']
    header['SLTW_PIX'] = header['SLITWID'] / header['PLTSCALE']

    # get RP from slitwid, by central wavelength
    header['RP'] = get_resolution(header)

    # Set comments for new keywords
    header.comments['EDGEDEG'] = 'Degree of the polynomial fit to order edges'
    header.comments['ORDERS'] = 'Orders identified'
    header.comments['SLTH_PIX'] = 'Slit height in pixels'
    header.comments['SLTH_ARC'] = 'Slit height in arcseconds'
    header.comments['SLTW_PIX'] = 'Slit width in pixels'
    header.comments['SLTW_ARC'] = 'Slit width in arcseconds'
    header.comments['RP'] = 'Estimated spectral resolution'
    header.comments['ROTATION'] = 'IDL rotate value'
