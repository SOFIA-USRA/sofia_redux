# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import astropy.constants as const
import astropy.units as u
import numpy as np

from sofia_redux.instruments.exes.clean import clean
from sofia_redux.instruments.exes.derive_tort import derive_tort
from sofia_redux.instruments.exes.tortcoord import tortcoord
from sofia_redux.instruments.exes.utils import parse_central_wavenumber
from sofia_redux.toolkit.stats import meancomb

__all__ = ['makeflat', 'blackbody_pnu', 'bnu', 'bb_cal_factor']


def makeflat(cards, header, variance, robust=4.0, radius=10,
             black_frame=None, dark=None,
             fix_tort=False, edge_method='deriv', custom_wavemap=None,
             start_pixel=None, end_pixel=None,
             top_pixel=None, bottom_pixel=None):
    """
    Generate calibrated flat frame; set distortion parameters.

    The procedure is:

        1. Black, sky, and shiny frames are identified. Typically,
           only black frames are present for EXES flats.
        2. The black frame is used to set and test distortion parameters.
        3. A difference frame is calculated (typically black-dark)
           and normalized by the black-body function at the ambient
           temperature (hdr['BBTEMP']).
        4. The inverse frame (bb / (black - dark)) is calculated.
        5. Unilluminated pixels are set to zero in the inverse frame.

    Calibration is performed by multiplying science data by the output frame.

    Parameters
    ----------
    cards : numpy.ndarray
        3D cube [nframe, nspec, nspat] or 2D image [nspec, nspat] containing
        flat frames (black, sky, or shiny).
    header : fits.Header
        FITS header for the flat file.
    variance : numpy.ndarray
        Variance array, matching `cards` shape.
    robust : float, optional
        Threshold for outlier rejection in robust mean combination, specified
        as a factor times the standard deviation.
    radius : int, optional
        Pixel radius to search for good pixels, used for interpolation over
        bad pixels in the flat frames.
    black_frame : int, optional
        Index of the black frame in the input `cards` (typically 0). If
        not provided, the black frame is set as the card with the highest
        mean value.
    dark : numpy.ndarray, optional
        Slit dark frame image [nspec, nspat]. If provided, and the input
        flat has a black card only, it will be subtracted from the black
        frame to make the difference image.
    fix_tort : bool, optional
        If True, no attempt will be made to optimize distortion parameters
        for cross-dispersed modes.
    edge_method : {'deriv', 'sqderiv', 'sobel'}, optional
        Sets the edge enhancement method for optimizing the cross-dispersed
        distortion parameters.  May be one derivative ('deriv'),
        squared derivative ('sqderiv'), or Sobel ('sobel').    custom_wavemap
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

    """

    params = _check_inputs(cards, header, variance,
                           robust=robust, dark=dark, radius=radius)
    _set_black_frame(params, black_frame)
    _set_shiny_and_sky_frames(params)
    _set_process_type(params)
    _check_saturation(params)
    _process_cards(params)
    _calculate_responsive_quantum_efficiency(params)
    _undistort_flat(params, edge_method=edge_method,
                    fix_tort=fix_tort, custom_wavemap=custom_wavemap,
                    start_pixel=start_pixel,
                    end_pixel=end_pixel, top_pixel=top_pixel,
                    bottom_pixel=bottom_pixel)

    # If no cards/flat desired, set them to 1, now that testtort is done
    if params['cardmode'] in ['NONE', 'UNKNOWN']:
        shape = params['ny'], params['nx']
        params['cards'].fill(1.0)
        params['flat_variance'] = np.zeros(shape, dtype=np.float64)
        params['flat'] = np.ones(shape, dtype=np.float64)
        params['illum'] = np.ones(shape, dtype=int)
        return params

    _create_flat(params, robust=robust)
    return params


def blackbody_pnu(wavenumber, temperature):
    """
    Black-body photon function.

    Accepts arrays for wavenumber and/or temperature, as long
    as their shapes can be broadcast together.

    Parameters
    ----------
    wavenumber : float or array-like of float
        Wavenumber values for computing the black-body function.
    temperature : float or array-like of float
        Temperature values for computing the black-body function.

    Returns
    -------
    pnu : float or array-like of float
        Photons at input values, in Hz/cm.
    """
    t = u.Quantity(temperature, 'Kelvin')
    v = u.Quantity(wavenumber, 'Kayser').to('Hz', equivalencies=u.spectral())

    pnu = 2 * (v ** 2) / const.c
    pnu /= np.exp((const.h * v) / (const.k_B * t)) - 1.0
    return pnu.to(u.Hz / u.cm).value


def bnu(wavenumber, temperature):
    """
    Black-body intensity function.

    Accepts arrays for wavenumber and/or temperature, as long
    as their shapes can be broadcast together.

    Parameters
    ----------
    wavenumber : float or array-like of float
        Wavenumber values for computing the black-body function.
    temperature : float or array-like of float
        Temperature values for computing the black-body function.

    Returns
    -------
    bnu : float or array-like of float
        Blackbody intensity in erg s-1 cm-2 (cm-1)-1.
    """
    t = u.Quantity(temperature, 'Kelvin')
    v = u.Quantity(wavenumber, 'Kayser').to('Hz', equivalencies=u.spectral())

    bnu = 2 * const.h * (v ** 3) / const.c
    bnu /= np.exp((const.h * v) / (const.k_B * t)) - 1.0
    return bnu.value * 1e5


def bb_cal_factor(wavenumber, bb_temp, flat_tamb, flat_emis):
    """
    Calibration factor for EXES blackbody source + flat mirror.

    The EXES blackbody plate is located outside of the dewar,
    with one reflection off of a flat mirror before entering the
    EXES window. This flat field system seems to produce photons
    like the sum of two blackbodies:

        B(BB_TEMP, lambda) * (1-emissivity) + emissivity * B(T_ambient),

    where emissivity is (1-reflectance) of the flat mirror.

    Expected values are emissivity = 0.1. (reflectance = 0.9) and
    T_ambient = 290 K.

    Parameters
    ----------
    wavenumber : float or array-like of float
        Wavenumber values for computing the black-body function.
    bb_temp : float
        Blackbody source temperature value.
    flat_tamb : float
        Ambient temperature for flat mirror.
    flat_emis : float
        Emissivity factor (1 - reflectance) for the flat mirror.

    Returns
    -------

    """
    bnu_t = bnu(wavenumber, bb_temp)
    bnu_t_amb = bnu(wavenumber, flat_tamb)

    cal_factor = bnu_t * (1 - flat_emis) + bnu_t_amb * flat_emis
    return cal_factor


def _check_inputs(cards, header, variance, robust=4.0, dark=None, radius=10):
    """Check input dimensions and modes."""
    cards = np.asarray(cards, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if cards.shape != variance.shape:
        raise ValueError("Card shape does not match variance shape")

    cardmode = str(header['CARDMODE']).strip().upper()
    instcfg = str(header['INSTCFG']).strip().upper()
    nx = int(header['NSPAT'])
    ny = int(header['NSPEC'])
    eperadu = float(header['EPERADU'])
    slitval = float(header['SLITVAL'])
    temp = float(header['BB_TEMP'])
    flat_tamb = float(header['FLATTAMB'])
    flat_emis = float(header['FLATEMIS'])
    pixel_width = float(header['PIXELWD'])
    frametime = float(header['FRAMETIM'])
    gain = float(header['PAGAIN'])
    satval = float(header['SATVAL'])

    # Planned or updated central wavenumber
    waveno0 = parse_central_wavenumber(header)

    if dark is not None:
        dark = np.asarray(dark, dtype=float)
        if dark.shape != (ny, nx):
            raise ValueError("Dark does not match expected shape from header")

    # Check no cards or tort, or if tort is from object
    if cardmode in ['NONE', 'UNKNOWN']:
        raise ValueError("CARDMODE is unspecified")

    # If camera, then use flatmode = shiny or sky
    if instcfg == 'CAMERA':
        if cardmode != 'SKY':
            log.info("Setting flatmode = shiny for camera mode")
            cardmode = 'SHINY'

    # Get parameters for instrument configuration
    if instcfg in ['HIGH_MED', 'HIGH_LOW']:
        focal_length = float(header['HRFL0'])
        r_number = float(header['HRR'])
    else:
        focal_length = float(header['XDFL0'])
        r_number = float(header['XDR'])

    frgain = frametime * gain
    if frgain > 0:
        maxval = satval / frgain
    else:
        maxval = satval

    # Check cards
    if cards.ndim == 2:
        cards = cards[None]
    ncards = cards.shape[0]

    # Check the number of cards against mode
    if ncards == 1:
        cardmode = 'BLK'
    elif ncards < 3 and cardmode in ['SHINY', 'BLKSHINY']:
        log.warning("Shiny frame not read.  Changing cardmode to BLKSKY")
        cardmode = 'BLKSKY'

    # Check the values in each card
    means = np.empty(ncards, dtype=float)
    for i in range(ncards):
        means[i] = meancomb(cards[i].ravel(), robust=robust, returned=False)

    bad_frames = np.isnan(means)
    if bad_frames.any():
        badidx = np.nonzero(bad_frames)[0]
        for frame in badidx:
            msg = f"Bad data in flat frame {frame}"
            if frame < 2:
                raise ValueError(f"Cannot proceed: {msg}")
            elif cardmode in ['SHINY', 'BLKSHINY']:
                log.warning(msg)
                log.info("Changing mode to BLKSKY")
                cardmode = 'BLKSKY'
            else:
                log.info(msg)
                log.info(f"This is allowable for frame: {frame}")

    return {
        'cards': cards,
        'variance': variance,
        'header': header,
        'dark': dark,
        'cardmode': cardmode,
        'instcfg': instcfg,
        'ncards': ncards,
        'nx': nx,
        'ny': ny,
        'eperadu': eperadu,
        'slitval': slitval,
        'temp': temp,
        'flat_tamb': flat_tamb,
        'flat_emis': flat_emis,
        'waveno0': waveno0,
        'focal_length': focal_length,
        'r_number': r_number,
        'maxval': maxval,
        'pixel_width': pixel_width,
        'card_means': means,
        'radius': radius
    }


def _set_black_frame(params, black_frame):
    """Set the black frame index."""
    if black_frame is not None:
        try:
            black_frame = int(black_frame)
        except (TypeError, ValueError):
            black_frame = -1
        if not (0 <= black_frame < params['ncards']):
            raise ValueError(f"Cannot use black_frame={black_frame} "
                             f"for {params['ncards']} cards")

    elif params['cardmode'] == 'SKY':
        black_frame = 0
    else:
        black_frame = np.argmax(params['card_means'])
        log.info(f"Flat frame {black_frame} is brightest")

    params['black_frame'] = black_frame


def _set_shiny_and_sky_frames(params):
    """Set shiny and sky frame indices."""
    # Note: this logic was copied from the original TEXES pipeline,
    # but shiny has never been used for EXES

    black_frame = params['black_frame']
    nc = params['ncards']
    means = params['card_means']

    if black_frame == 0:
        shiny_frame = 2
        sky_frame = 3 if (nc == 4 and (means[3] < means[1])) else 1

    elif black_frame == 1:
        shiny_frame = 3
        sky_frame = 2 if (nc == 4 and (means[2] < means[0])) else 0

    else:
        shiny_frame = black_frame - 2
        sky_frame = black_frame - 1

    if shiny_frame >= nc:
        shiny_frame = black_frame
    if sky_frame >= nc:
        sky_frame = black_frame

    if sky_frame >= 2:
        sky_frame2 = sky_frame - 2
    else:
        sky_frame2 = sky_frame + 2
    if sky_frame2 >= nc:
        sky_frame2 = sky_frame

    params['sky_frame'] = sky_frame
    params['sky_frame2'] = sky_frame2
    params['shiny_frame'] = shiny_frame


def _set_process_type(params):
    """Check cardmode is correct for current parameters."""
    cardmode = params['cardmode']
    black_frame = params['black_frame']
    shiny_frame = params['shiny_frame']

    if cardmode in ['BLK', 'NONE', 'UNKNOWN']:
        process_type = 'BLK'

    elif cardmode in ['SKY']:
        process_type = 'SKY'

    elif cardmode in ['SHINY'] and shiny_frame != black_frame:
        process_type = 'SHINY'

    elif cardmode in ['BLKSKY', 'OBJ', 'BLKOBJ']:
        process_type = 'BLKSKY'

    elif cardmode in ['BLKSHINY'] and shiny_frame != black_frame:
        process_type = 'BLKSHINY'

    else:
        if shiny_frame != black_frame:
            raise ValueError(f"Unrecognizable cardmode: {cardmode}")
        else:
            raise ValueError("Cardmode is unusable without shiny frame")

    params['process_type'] = process_type


def _check_saturation(params, max_saturation=0.04):
    """Generate a saturation mask and check for too many bad pixels."""

    process_type = params['process_type']
    maxval = params['maxval']

    if maxval <= 0:
        mask = np.full((params['ny'], params['nx']), True)

    elif process_type == 'SKY':
        mask = params['cards'][params['sky_frame']] <= maxval

    elif process_type == 'SHINY':
        mask = params['cards'][params['shiny_frame']] <= maxval

    else:
        mask = params['cards'][params['black_frame']] <= maxval

    saturated = np.sum(~mask)
    if saturated > (max_saturation * params['nx']):
        msg = f"{saturated} pixels saturated in black."
        if process_type != 'SHINY':
            msg += "  Try using cardmode='SHINY'."
        raise ValueError(msg)

    params['mask'] = mask


def _process_cards(params):
    """Process flat cards according to mode."""
    process_type = params['process_type']

    if process_type == 'BLK':
        _process_blk(params)
    elif process_type == 'SKY':
        _process_sky(params)
    elif process_type == 'SHINY':
        _process_shiny(params)
    elif process_type == 'BLKSKY':
        _process_blksky(params)
    elif process_type == 'BLKSHINY':
        _process_blkshiny(params)
    else:
        raise ValueError("Unknown process type")

    # at this point, there should be a card1 and card2 in params.
    # Replace the cards with this set
    params['cards'] = np.array([params['card1'], params['card2']])


def _process_blk(params):
    """Make diff and stddev frames for BLK mode."""
    # Accounts for change in the EXES observing pattern of only doing
    # stares at the blackbody + slit darks
    log.info("Processing BLK type cards:")
    shape = params['ny'], params['nx']
    cards = params['cards']
    black_frame = params['black_frame']

    if params['dark'] is None:
        log.info('No slit dark available.')
        # Mark saturated pixels
        black_card = cards[black_frame]
        card1 = black_card.copy()
        diff = card1.copy()
        card2 = np.zeros(shape, dtype=float)
        if params['sky_frame'] != black_frame:
            sky_card = cards[params['sky_frame']]
            nzi = black_card != 0
            card2[nzi] = (black_card[nzi] - sky_card[nzi]) / black_card[nzi]
        params['card_variance'] = params['variance'][black_frame].copy()

    else:
        log.info('Subtracting slit dark.')
        black_card = cards[0]
        card1 = black_card.copy()
        card2 = black_card - params['dark'].copy()
        diff = card2.copy()
        card2[black_card == 0] = 0
        params['card_variance'] = params['variance'][0].copy()

    params['card1'] = card1
    params['card2'] = card2
    params['diff'] = diff
    params['stddev'] = np.sqrt(params['card_variance'])


def _process_sky(params):
    """Make diff and stddev frames for SKY mode."""
    log.info("Processing SKY type cards")
    sky_frame = params['sky_frame']
    sky_card = params['cards'][sky_frame]
    card1 = sky_card.copy()
    card2 = sky_card.copy()
    diff = sky_card.copy()

    params['card1'] = card1
    params['card2'] = card2
    params['diff'] = diff
    params['card_variance'] = params['variance'][sky_frame].copy()
    params['stddev'] = np.sqrt(params['card_variance'])


def _process_shiny(params):
    """Make diff and stddev frames for SHINY mode."""
    log.info("Processing SHINY type cards")
    shiny_frame = params['shiny_frame']
    shiny_card = params['cards'][shiny_frame]
    sky_card = params['cards'][params['sky_frame']]

    card1 = shiny_card.copy()
    diff = shiny_card.copy()
    card2 = np.zeros(shiny_card.shape, dtype=float)
    if shiny_frame != params['sky_frame']:
        nzi = shiny_card != 0
        card2[nzi] = sky_card[nzi] / shiny_card[nzi]

    params['card1'] = card1
    params['card2'] = card2
    params['diff'] = diff
    params['card_variance'] = params['variance'][shiny_frame].copy()
    params['stddev'] = np.sqrt(params['card_variance'])


def _process_blksky(params):
    """Make diff and stddev frames for BLKSKY mode."""
    log.info("Processing BLKSKY type cards")
    black_frame = params['black_frame']
    shiny_frame = params['shiny_frame']

    black_card = params['cards'][black_frame]
    do_dark = params['dark'] is not None

    if black_frame != shiny_frame:
        card1 = black_card - params['cards'][shiny_frame]
    else:
        card1 = black_card.copy()

    if do_dark:
        log.info('Subtracting slit dark instead of sky.')
        diff = black_card - params['dark']
    else:
        diff = black_card - params['cards'][params['sky_frame']]

    card2 = np.zeros(black_card.shape, dtype=float)
    nzi = black_card != 0
    card2[nzi] = diff[nzi]
    if not do_dark:
        card2[nzi] /= black_card[nzi]

    params['card1'] = card1
    params['card2'] = card2
    params['diff'] = diff
    params['card_variance'] = params['variance'][black_frame].copy()
    params['stddev'] = np.sqrt(params['card_variance'])


def _process_blkshiny(params):
    """Make diff and stddev frames for BLKSHINY mode."""
    log.info("Processing BLKSHINY type cards")
    black_frame = params['black_frame']
    shiny_frame = params['shiny_frame']

    black_card = params['cards'][black_frame]
    shiny_card = params['cards'][shiny_frame]
    sky_card = params['cards'][params['sky_frame']]

    card1 = black_card - shiny_card
    diff = card1.copy()
    card2 = np.zeros(card1.shape, dtype=float)
    nzi = black_card != 0
    card2[nzi] = (black_card[nzi] - sky_card[nzi]) / black_card[nzi]

    params['card1'] = card1
    params['card2'] = card2
    params['diff'] = diff
    params['card_variance'] = params['variance'][black_frame].copy()
    params['stddev'] = np.sqrt(params['card_variance'])


def _calculate_responsive_quantum_efficiency(params):
    """Calculate RQE for the black frame."""
    dwno = params['waveno0'] * params['slitval']
    dwno /= 2 * params['focal_length'] * params['r_number']

    pnut = blackbody_pnu(params['waveno0'], params['temp'])
    a_omega = (params['pixel_width'] ** 2) * np.pi / (4 * 36)

    black_mean = params['card_means'][params['black_frame']]
    rqe = black_mean * params['eperadu']
    rqe /= pnut * a_omega * dwno
    log.info(f"Mean RQE over black frame: {rqe}")
    params['rqe'] = rqe


def _undistort_flat(params, edge_method='deriv', custom_wavemap=None,
                    fix_tort=False, start_pixel=None, end_pixel=None,
                    top_pixel=None, bottom_pixel=None):
    """Undistort the black frame and tune distortion parameters."""
    header = params['header'].copy()

    # clean card 0 before using
    card0 = params['cards'][0].copy()
    card0, _ = clean(card0, params['header'], params['stddev'].copy(),
                     mask=params['mask'], radius=params['radius'])

    tortdata, tortillum = derive_tort(
        card0, header, maxiter=5, fixed=fix_tort,
        edge_method=edge_method, custom_wavemap=custom_wavemap,
        top_pixel=top_pixel, bottom_pixel=bottom_pixel,
        start_pixel=start_pixel, end_pixel=end_pixel)

    # tortdata is not used later, but tortillum is
    params['tortdata'] = tortdata
    params['tortillum'] = tortillum
    params['header'] = header


def _create_flat(params, robust=4.0):
    """Create the flat from the diff frame."""

    # Clean the diff frame (normally black-sky or black-dark)
    diff = params['diff']
    header = params['header'].copy()
    ny, nx = shape = params['ny'], params['nx']
    illum = params['tortillum'].copy()

    diff, stddev = clean(diff, header, params['stddev'],
                         mask=params['mask'], radius=params['radius'])
    params['diff'] = diff
    params['stddev'] = stddev

    # Check the diff frame for overall negative value
    mean_diff = meancomb(diff.ravel(), robust=robust, returned=False)
    if mean_diff <= 0:
        log.warning("Mean flat diff <= 0; Setting flat = 1")
        params['flat'] = np.ones(shape, dtype=np.float64)
        params['flat_variance'] = np.zeros(shape, dtype=np.float64)
        params['illum'] = np.ones(shape, dtype=int)
        return

    # Set flat = 0 if diff is small to prevent huge flat values.
    thrfac = float(header['THRFAC'])
    if thrfac < 0.2:
        dmin = 0.05 * mean_diff
    elif thrfac > 1:
        dmin = 0.25 * mean_diff
    else:
        dmin = 0.25 * thrfac * mean_diff

    # Find pixels above threshold
    idx = diff > dmin
    if not idx.any():
        log.warning("No pixels found above threshold; Setting flat = 1")
        params['flat'] = np.ones(shape, dtype=np.float64)
        params['flat_variance'] = np.zeros(shape, dtype=np.float64)
        params['illum'] = np.ones(shape, dtype=int)
        return

    flat = np.zeros(shape, dtype=np.float64)
    flat_variance = np.zeros(shape, dtype=np.float64)

    cal_factor = bb_cal_factor(params['waveno0'], params['temp'],
                               params['flat_tamb'], params['flat_emis'])
    flat[idx] = cal_factor / diff[idx]
    flat_variance[idx] = ((params['stddev'][idx] ** 2)
                          * (cal_factor ** 2)
                          / (diff[idx] ** 4))
    header['BNU_T'] = cal_factor
    header['BUNIT'] = 'erg s-1 cm-2 sr-1 (cm-1)-1 ct-1'

    ux, uy = tortcoord(header, skew=True)
    u1 = ux.astype(int)
    v1 = uy.astype(int)
    u2 = u1 + 1
    v2 = v1 + 1

    # Check for pixels out of bounds and mark with -1
    cond1 = (u1 >= 0) & (u2 < nx) & (v1 >= 0) & (v2 < ny)
    if not cond1.all():
        idx = np.where(~cond1)
        illum[idx] = -1

    # Check the four nearest pixels for bad values and mark with 0
    u1 = np.clip(u1, 0, nx - 1)
    u2 = np.clip(u2, 0, nx - 1)
    v1 = np.clip(v1, 0, ny - 1)
    v2 = np.clip(v2, 0, ny - 1)

    cond2 = diff[v1, u1] <= dmin
    cond2 |= diff[v2, u1] <= dmin
    cond2 |= diff[v1, u2] <= dmin
    cond2 |= diff[v2, u2] <= dmin
    idx = cond1 & cond2
    if idx.any():
        illum[np.where(idx)] = 0

    # Update params
    params['header'] = header
    params['illum'] = illum
    params['flat'] = flat
    params['flat_variance'] = flat_variance
