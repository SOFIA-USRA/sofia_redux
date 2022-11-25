# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes.utils import parse_central_wavenumber

__all__ = ['wavecal']


def wavecal(header, order=None):
    """
    Generate a wavelength calibration map from the grating equation.

    Dispersion parameters are read from the header and used to
    calculate the wavenumber values for each pixel.  The assumed central
    wavenumber for the observation (WAVENO0) is used for the first
    pass calibration; this value may be overridden by setting
    header['WNO0'] to a more accurate value.

    Parameters
    ----------
    header : fits.Header
        Header of EXES FITS file.
    order : int, optional
        Cross-dispersed order to calculate wavelengths for. If
        provided, only the 1D wavelength array is produced.

    Returns
    -------
    wavemap : numpy.ndarray
        Array containing the wavelength values. If `order` is not
        provided, then a 3D data cube is produced, where the first
        plane is an image containing the wavelength value at each pixel,
        the second plane is an image containing the spatial coordinates
        for each pixel, and the third is an image containing the order
        number for each pixel. Keywords describing the wavelength
        calibration are added to `header`. If `order` is specified, only
        the 1D wavelength array is produced and the header is not modified.
    """
    try:
        params = _parse_inputs(header, order)
    except ValueError as msg:
        log.error(msg)
        return np.empty(0)

    _setup_wavemap(params)
    _populate_wavecal(params)
    _update_header(params)

    return params['wavemap']


def _parse_inputs(header, order):
    """Read wavecal parameters from input header."""
    nx = header['NSPAT']
    ny = header['NSPEC']
    pixelwd = header['PIXELWD']
    hrr = header['HRR']
    hrdgr = header['HRDGR']
    hrfl = header['HRFL']
    xdr = header['XDR']
    xdfl = header['XDFL']
    slitoff = header['SLITOFF']
    norders = header['NORDERS']
    orders = header['ORDERS']
    ordr_b = header['ORDR_B']
    ordr_t = header['ORDR_T']
    ordr_s = header['ORDR_S']
    ordr_e = header['ORDR_E']
    instcfg = header['INSTCFG']
    pltscale = header['PLTSCALE']

    crossdisp = instcfg in ['HIGH_MED', 'HIGH_LOW']
    wnoc = parse_central_wavenumber(header)

    if crossdisp:
        ns = nx
        nw = ny
        dlnw = pixelwd / (2 * hrr * (1 - slitoff / 20.) * hrfl)
        dw = 0.5 / (np.sqrt(hrr ** 2 / (1 + hrr ** 2)) * hrdgr)
    else:
        ns = ny
        nw = nx
        dlnw = pixelwd / (2 * xdr * xdfl)
        dw = 1

    ob = _check_order(ordr_b)
    ot = _check_order(ordr_t, ns)
    os = _check_order(ordr_s)
    oe = _check_order(ordr_e, nw)
    order_numbers = _check_order(orders)

    if crossdisp:
        # invert top and bottom for rotation
        tmp = ob.copy()
        ob = ny - ot - 1
        ot = ny - tmp - 1
        if (len(ob) != norders or len(ot) != norders
                or len(order_numbers) != norders):
            message = (f"Can't determine edges for XD orders. "
                       f"Not calculating wavenumbers\n"
                       f"  Number of orders      : {norders}\n"
                       f"  Number of order names : {len(order_numbers)}\n"
                       f"  Number of bottom edges: {len(ob)}\n"
                       f"  Number of top edges   : {len(ot)}")
            raise ValueError(message)

    params = {'header': header, 'crossdisp': crossdisp, 'order': order,
              'nx': nx, 'ny': ny, 'norders': norders,
              'order_numbers': order_numbers,
              'ns': ns, 'nw': nw, 'dlnw': dlnw, 'dw': dw,
              'ob': ob, 'ot': ot, 'os': os, 'oe': oe,
              'wnoc': wnoc, 'pltscale': pltscale, 'instcfg': instcfg
              }
    return params


def _setup_wavemap(params):
    """Initialize the wavecal map."""
    nx = params['nx']
    ny = params['ny']
    norders = params['norders']
    oe = params['oe']
    os = params['os']
    crossdisp = params['crossdisp']
    order = params['order']
    try:
        order = int(order)
    except (ValueError, TypeError):
        valid_order = False
    else:
        valid_order = True
    if valid_order and 0 < order <= norders:
        order_idx = norders - order + 1
        wavemap = np.zeros(oe[order_idx - 1] - os[order_idx - 1])
        wavecal_ = np.empty(0)
        spatcal = np.empty(0)
        order_mask = np.empty(0)
        if crossdisp:
            w = np.arange(ny)
            nw2 = ny / 2
        else:
            w = np.arange(nx)
            nw2 = nx / 2
        s = np.empty(0)
    else:
        order_idx = -1
        wavemap = np.full((3, ny, nx), np.nan)
        wavecal_ = wavemap[0]
        spatcal = wavemap[1]
        order_mask = wavemap[2]

        # Make column and row index arrays
        w = np.zeros((ny, nx), dtype=int)
        s = np.zeros_like(w, dtype=int)
        if crossdisp:
            w.T[:] = np.arange(ny).T
            s[:] = np.arange(nx)
            nw2 = ny / 2
        else:
            w[:] = np.arange(nx)
            s.T[:] = np.arange(ny).T
            nw2 = nx / 2

    params['wavemap'] = wavemap
    params['wavecal'] = wavecal_
    params['spatcal'] = spatcal
    params['order_mask'] = order_mask
    params['w'] = w
    params['s'] = s
    params['nw2'] = nw2
    params['order_idx'] = order_idx


def _check_order(order, default=0):
    """Parse integer orders from a string list."""
    if order != 'UNKNOWN':
        checked = [int(o) for o in order.split(',')]
    else:
        checked = [default]
    return np.array(checked)


def _populate_wavecal(params):
    """Populate the wavecal map with calibrated values."""
    s = params['s']
    w = params['w']
    wave_cal = params['wavecal']
    spat_cal = params['spatcal']
    order_mask = params['order_mask']
    dlnw = params['dlnw']
    nw2 = params['nw2']
    plate_scale = params['pltscale']
    order_numbers = params['order_numbers']

    # start with order mask set to zero
    order_mask[:] = 0

    for i in range(1, params['norders'] + 1):
        if params['order_idx'] != -1:
            if i != params['order']:
                continue
            idx = params['order_idx']
        else:
            idx = i

        wnoi = params['wnoc'] + params['dw'] * (
            idx - (params['norders'] + 1) / 2)
        bottom = params['ob'][idx - 1]
        top = params['ot'][idx - 1]
        start = params['os'][idx - 1]
        stop = params['oe'][idx - 1]

        if params['order_idx'] == -1:
            in_range = ((s >= bottom) & (s <= top)
                        & (w >= start) & (w <= stop))
            if np.sum(in_range) == 0:  # pragma: no cover
                continue

            wave_cal[in_range] = wnoi * np.exp(
                dlnw * (w[in_range] - nw2 + 0.5))
            if params['crossdisp']:
                spat_cal[in_range] = (top - s[in_range]) * plate_scale
            else:
                spat_cal[in_range] = (s[in_range] - bottom) * plate_scale
            order_mask[in_range] = order_numbers[i - 1]
        else:
            in_range = (w >= start) & (w <= stop)
            if np.sum(in_range) == 0:  # pragma: no cover
                continue
            params['wavemap'] = wnoi * np.exp(dlnw * (w[in_range] - nw2 + 0.5))


def _update_header(params):
    """Update the header with wavecal keys."""
    header = params['header']
    if params['order_idx'] == -1:
        params['wavemap'][0] = params['wavecal']
        params['wavemap'][1] = params['spatcal']
        params['wavemap'][2] = params['order_mask']
        header['WCTYPE'] = ('1D', 'Wavecal type (2D or 1D)')
        header['BUNIT1'] = ('cm-1', 'Data units for first plane of image')
        header['BUNIT2'] = ('arcsec', 'Data units for second plane of image')
        header['BUNIT3'] = ('', 'Data units for third plane of image')
