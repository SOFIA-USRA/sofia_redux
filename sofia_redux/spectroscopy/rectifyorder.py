# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np
from scipy.sparse import csr_matrix

from sofia_redux.toolkit.utilities.fits import hdinsert
from sofia_redux.toolkit.fitting.polynomial import polyinterp2d
from sofia_redux.toolkit.interpolate import tabinv, Interpolate
from sofia_redux.toolkit.image.fill import polyfillaa

__all__ = ['get_rect_xy', 'trim_xy', 'rectifyorder', 'update_wcs',
           'rectifyorder']


def get_rect_xy(xarray, yarray, xvals, yvals, dx=None, dy=None, mask=None,
                poly_order=3):
    """
    Given arrays of x and y coordinates, interpolate to defined grids

    Parameters
    ----------
    xarray : array_like of float
        (nrows, ncols)
    yarray : array_like of float
        (nrows, ncols)
    xvals : array_like of float
        (nrows, ncols)
    yvals : array_like of float
        (nrows, ncols)
    dx : float, optional
        Spacing of the output grids in the x-direction.  If not supplied,
        then taken to be range(xarray)/range(xvals)
    dy : float, optional
        Spacing of the output grids in the y-direction.  If not supplied,
        then taken to be range(yarray)/range(yvals)
    mask : numpy.ndarray, optional
    poly_order : int, optional
        Polynomial order used to interpolate to the new coordinates

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        (rectx, recty, gridx, gridy)
        where rectx and recty are the x and y arrays interpolated onto
        a regular grid.  gridx and gridy are the coordinates of the
        regular grid.
    """
    s = xarray.shape
    if yarray.shape != s or xvals.shape != s or yvals.shape != s:
        log.error("Incompatible array dimensions")
        return
    gi = np.isfinite(xarray) & np.isfinite(yarray)
    gi &= np.isfinite(xvals) & np.isfinite(yvals)
    if isinstance(mask, np.ndarray):
        if mask.shape != s:
            log.error("Incompatible mask dimensions")
            return
        gi &= mask
    if not gi.any():
        log.error("All valid x, y values are invalid or masked out")
        return

    xv, yv = xvals[gi].copy(), yvals[gi].copy()
    xa, ya = xarray[gi].copy(), yarray[gi].copy()
    xrange, yrange = np.ptp(xa), np.ptp(ya)
    if dx is None:
        dx = xrange / np.ptp(xv)
    if dy is None:
        dy = yrange / np.ptp(yv)

    nx = int(xrange / dx) + 1
    ny = int(yrange / dy) + 1
    yout, xout = np.mgrid[:ny, :nx]
    xout = xout * dx + xa.min()
    yout = yout * dy + ya.min()
    ix = polyinterp2d(xa, ya, xv, xout, yout, order=poly_order, full=False)
    iy = polyinterp2d(xa, ya, yv, xout, yout, order=poly_order, full=False)
    return ix, iy, xout[0], yout[:, 0]


def trim_xy(xarray, yarray, xgrid, ygrid, ybuffer=None, xbuffer=None,
            xrange=None, yrange=None):
    """
    Trim rows and columns from the edges of the coordinate arrays.

    Parameters
    ----------
    xarray : numpy.ndarray
        (nrows, ncols) array of x-coordinates
    yarray : numpy.ndarray
        (nrows, ncols) array of y-coordinates
    xgrid : numpy.ndarray
        (ncols,) array of x-coordinates along the x-axis of the rectified
        arrays.
    ygrid : numpy.ndarray
        (nrows,) array of y-coordinates along the y-axis of the rectified
        arrays.
    ybuffer : int, optional
        number of pixels to cut from the top and bottom of the arrays
    xbuffer : int, optional
        number of pixels to cut from the left and right of the arrays
    xrange : array_like of float
        (2,) [lower limit, upper limit] defining the range of valid
        x values.
    yrange : array_like of float
        (2,) [lower limit, upper limit] defining the range of valid
        y values.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        (xarray2, yarray2, xgrid2, ygrid2) where array dimensions may
        be smaller than the original input arrays.
    """
    ix, iy = xarray.copy(), yarray.copy()
    xg, yg = xgrid.copy(), ygrid.copy()

    # apply buffers to edges of arrays
    if isinstance(ybuffer, int) and ybuffer > 0:
        ix = ix[ybuffer:-ybuffer, :]
        iy = iy[ybuffer:-ybuffer, :]
        yg = yg[ybuffer:-ybuffer]
    if isinstance(xbuffer, int) and xbuffer > 0:
        ix = ix[:, xbuffer:-xbuffer]
        iy = iy[:, xbuffer:-xbuffer]
        xg = xg[xbuffer:-xbuffer]

    # trim partial NaNs near the left, right, top, and bottom edges
    xf = np.argwhere(np.any(np.isfinite(ix), axis=0)).ravel()
    yf = np.argwhere(np.any(np.isfinite(iy), axis=1)).ravel()
    xf = [xf.min(), xf.max() + 1] if len(xf) >= 2 else [0, ix.shape[1]]
    yf = [yf.min(), yf.max() + 1] if len(yf) >= 2 else [0, iy.shape[0]]
    ix, iy = ix[yf[0]: yf[1], xf[0]: xf[1]], iy[yf[0]: yf[1], xf[0]: xf[1]]
    xg, yg = xg[xf[0]: xf[1]], yg[yf[0]: yf[1]]

    # trim to avoid the edge of the arrays
    if xrange is not None:
        xf = ix.copy()
        xf[np.isnan(xf)] = xrange[0] - 1
        xf = np.any((xf <= xrange[0]) | (xf >= xrange[1]), axis=0)
        xf = np.argwhere(~xf).ravel()
        xf = [xf.min(), xf.max() + 1] if len(xf) >= 2 else [0, ix.shape[1]]
    else:
        xf = [0, ix.shape[1]]

    if yrange is not None:
        yf = iy.copy()
        yf[np.isnan(yf)] = yrange[0] - 1
        yf = np.any((yf <= yrange[0]) | (yf >= yrange[1]), axis=1)
        yf = np.argwhere(~yf).ravel()
        yf = [yf.min(), yf.max() + 1] if len(yf) >= 2 else [0, iy.shape[0]]
    else:
        yf = [0, iy.shape[0]]

    ix, iy = ix[yf[0]: yf[1], xf[0]: xf[1]], iy[yf[0]: yf[1], xf[0]: xf[1]]
    xg, yg = xg[xf[0]: xf[1]], yg[yf[0]: yf[1]]
    return ix, iy, xg, yg


def reconstruct_slit(image, xarray, yarray, xgrid, ygrid, header=None,
                     variance=None, bitmask=None, badpix_mask=None,
                     badfrac=0.1, xrange=None, yrange=None):

    if header is None:
        header = fits.Header()

    ix, iy = xarray.copy(), yarray.copy()
    xg, yg = xgrid.copy(), ygrid.copy()
    nx, ny = xg.size - 1, yg.size - 1
    dx, dy = np.abs(xg[1] - xg[0]), np.abs(yg[1] - yg[0])

    if xrange is None:
        xrange = 0, image.shape[1]
    if yrange is None:
        yrange = 0, image.shape[0]
    xrange = np.clip(xrange, 0, image.shape[1])
    yrange = np.clip(yrange, 0, image.shape[0])

    # slits defines the vertices of the slit
    slits = np.zeros((2, 4, nx * ny))

    # Order of the vertices is bl -> tl -> tr -> br:
    # 2->3
    # |  |
    # 1  4
    # in a clockwise order
    vx, vy = [0, 0, 1, 1], [0, 1, 1, 0]
    for k, (j, i) in enumerate(zip(vy, vx)):
        slits[0, k] = iy[j:ny + j, i:nx + i].ravel()
        slits[1, k] = ix[j:ny + j, i:nx + i].ravel()
    # add 0.5 to index pixels at the centre rather than lower-left
    slits += 0.5

    # remove invalid entries
    keep = np.all(np.all(np.isfinite(slits), axis=1), axis=0)
    slits = slits[:, :, keep]

    # Format the data for polyfillaa (calculates area of slit shape)
    px, py = slits[1].T.ravel(), slits[0].T.ravel()
    # poly_ind gives the start and end indices defining each shape
    poly_ind = np.arange(slits.shape[-1]) * 4

    pixels, areas = polyfillaa(px, py, xrange=xrange, yrange=yrange,
                               start_indices=poly_ind, area=True)[:]

    gs, gw = np.mgrid[:ny, :nx]
    gs, gw = gs.ravel()[keep], gw.ravel()[keep]
    dovar = isinstance(variance, np.ndarray)
    dobits = isinstance(bitmask, np.ndarray)

    # Reorder
    keys = np.array(list(pixels.keys()))
    maxind = keys.max() + 1
    pv = np.full((maxind,), None, dtype=object)
    av = np.full((maxind,), None, dtype=object)
    av[keys] = list(areas.values())
    pv[keys] = list(pixels.values())
    npix = np.array(list(map(lambda val: 0 if val is None else len(val), pv)))
    minpix, maxpix = np.min(npix), np.max(npix) + 1
    nbins = maxpix - minpix
    pixbins = (npix - minpix).astype(int)
    npoly_ind = np.arange(npix.size)
    s = csr_matrix(
        (npoly_ind, [pixbins, np.arange(npix.size)]),
        shape=(nbins, npix.size))

    bpm = badpix_mask if badpix_mask is None else badpix_mask.astype(float)

    result = {'image': np.zeros((ny, nx)),
              'wave': xgrid[:nx] + dx / 2,
              'spatial': ygrid[:ny] + dy / 2,
              'mask': np.zeros((ny, nx)),
              'bitmask': np.zeros((ny, nx), dtype=int),
              'pixsum': np.zeros((ny, nx)),
              'variance': np.zeros((ny, nx)),
              'header': header}

    for i, put in enumerate(np.split(s.data, s.indptr[1:-1])):
        numpix = i + minpix
        if numpix == 0 or len(put) == 0:  # pragma: no cover
            continue
        pix_arr = np.array(list(pv[put]))
        area_arr = np.array(list(av[put]))
        takeidx = np.ravel_multi_index(
            np.reshape(pix_arr, (pix_arr.shape[0] * numpix, 2)).T,
            image.shape)
        val_arr = np.reshape(np.take(image, takeidx), area_arr.shape)
        putidx = tuple(np.array(list(zip(gs[put], gw[put]))).T)

        result['image'][putidx] += np.sum(area_arr * val_arr, axis=1)
        asum = np.sum(area_arr, axis=1)
        result['pixsum'][putidx] += asum
        if bpm is not None:
            bpm_arr = np.reshape(np.take(bpm, takeidx), area_arr.shape)
            result['mask'][putidx] += np.sum(area_arr * bpm_arr, axis=1)
        else:
            result['mask'][putidx] += asum
        if dovar:
            var_arr = np.reshape(np.take(variance, takeidx), area_arr.shape)
            result['variance'][putidx] += np.sum(area_arr * var_arr, axis=1)
        if dobits:
            bit_arr = np.reshape(np.take(bitmask, takeidx), area_arr.shape)
            bit_arr = np.mod(np.product(bit_arr, axis=1), 256)
            result['bitmask'][putidx] = bit_arr

    result['mask'] = np.array(result['mask'] > (1 - badfrac))
    result['mask'] &= ~np.isnan(result['image'])

    return result


def update_wcs(result, spatcal):
    """
    Update a FITS header with spectral WCS information.

    This function assumes that result['header'] is a fits.Header,
    to be updated in place, and that result['wave'] and
    result['spatial'] are appropriately populated.

    Parameters
    ----------
    result : dict
        Rectified result, as produced by `reconstruct_slit`.
    spatcal : numpy.ndarray of float (nrow, ncol)
        Spatial coordinates of each input pixel.
    """
    header = result['header']
    wave = result['wave']
    space = result['spatial']

    # get original keywords from input header
    do_secondary = True
    specsys = header.get('SPECSYS', 'TOPOCENT')
    try:
        # this assumes a simple input WCS convention
        crpix1 = header['CRPIX1']
        crpix2 = header['CRPIX2']
        crval1 = header['CRVAL1']
        crval2 = header['CRVAL2']
        crota2 = np.radians(header.get('CROTA2', 0.0))
        radesys = header.get('RADESYS', 'FK5')
        equinox = header.get('EQUINOX', 2000.0)
    except (KeyError, ValueError, TypeError):
        do_secondary = False
        crpix1, crpix2, crval1, crval2 = None, None, None, None
        crota2, radesys, equinox = None, None, None

    # dw and ds for the new image
    wave_scale = np.mean(wave[1:] - wave[:-1])
    space_scale = np.mean(space[1:] - space[:-1])

    # try to get wave units from input header,
    # assume um if not present
    wave_units = header.get('XUNITS', 'um')

    # assume spatial units are always arcsec/degrees
    space_units = 'arcsec'
    cunit1 = 'deg'
    cunit2 = 'deg'

    # middle pixel
    middle_wave = wave.size // 2
    middle_space = space.size // 2

    # add a simple linear primary WCS, referenced to the middle pixel
    hdinsert(header, 'CTYPE1', 'LINEAR',
             comment='Name of the coordinate axis')
    hdinsert(header, 'CTYPE2', 'LINEAR',
             comment='Name of the coordinate axis')
    hdinsert(header, 'CUNIT1', wave_units,
             comment='Units of the coordinate axis')
    hdinsert(header, 'CUNIT2', space_units,
             comment='Units of the coordinate axis')
    hdinsert(header, 'CRPIX1', middle_wave + 1,
             comment='Coordinate system reference pixel')
    hdinsert(header, 'CRPIX2', middle_space + 1,
             comment='Coordinate system reference pixel')
    hdinsert(header, 'CRVAL1', wave[middle_wave],
             comment='Coordinate system value at reference pixel')
    hdinsert(header, 'CRVAL2', space[middle_space],
             comment='Coordinate system value at reference pixel')
    hdinsert(header, 'CDELT1', wave_scale,
             comment='Coordinate increment at reference point')
    hdinsert(header, 'CDELT2', space_scale,
             comment='Coordinate increment at reference point')
    hdinsert(header, 'CROTA2', 0.0,
             comment='Coordinate system rotation angle')
    hdinsert(header, 'SPECSYS', specsys,
             comment='Spectral reference frame')

    # if the input header is non-trivial, add a secondary WCS
    # with full spatial sky coordinates.
    # This WCS is not primary because DS9 doesn't handle it well
    if do_secondary:
        # get the reference pixel slit position from the spatial cal
        interp2d = Interpolate(spatcal, mode='nearest', method='cubic')
        ref_arcsec = interp2d(crpix1 - 1, crpix2 - 1)

        # invert to get the effective index in the new spatial grid
        ref_pix = tabinv(space, ref_arcsec)

        hdinsert(header, 'CTYPE1A', 'WAVE',
                 comment='Name of the coordinate axis')
        hdinsert(header, 'CTYPE2A', 'DEC--TAN',
                 comment='Name of the coordinate axis')
        hdinsert(header, 'CTYPE3A', 'RA---TAN',
                 comment='Name of the coordinate axis')
        hdinsert(header, 'CUNIT1A', wave_units,
                 comment='Units of the coordinate axis')
        hdinsert(header, 'CUNIT2A', cunit2,
                 comment='Units of the coordinate axis')
        hdinsert(header, 'CUNIT3A', cunit1,
                 comment='Units of the coordinate axis')
        hdinsert(header, 'CRPIX1A', middle_wave + 1,
                 comment='Coordinate system reference pixel')
        hdinsert(header, 'CRPIX2A', ref_pix + 1,
                 comment='Coordinate system reference pixel')
        hdinsert(header, 'CRPIX3A', 1.0,
                 comment='Coordinate system reference pixel')
        hdinsert(header, 'CRVAL1A', wave[middle_wave],
                 comment='Coordinate system value at reference pixel')
        hdinsert(header, 'CRVAL2A', crval2,
                 comment='Coordinate system value at reference pixel')
        hdinsert(header, 'CRVAL3A', crval1,
                 comment='Coordinate system value at reference pixel')
        hdinsert(header, 'CDELT1A', wave_scale,
                 comment='Coordinate increment at reference point')
        hdinsert(header, 'CDELT2A', space_scale / 3600.,
                 comment='Coordinate increment at reference point')
        hdinsert(header, 'CDELT3A', - space_scale / 3600.,
                 comment='Coordinate increment at reference point')
        hdinsert(header, 'PC2_2A', np.cos(crota2),
                 comment='Coordinate transformation matrix element')
        hdinsert(header, 'PC2_3A', -np.sin(crota2),
                 comment='Coordinate transformation matrix element')
        hdinsert(header, 'PC3_2A', np.sin(crota2),
                 comment='Coordinate transformation matrix element')
        hdinsert(header, 'PC3_3A', np.cos(crota2),
                 comment='Coordinate transformation matrix element')
        hdinsert(header, 'RADESYSA', radesys,
                 comment='Equatorial coordinate system')
        hdinsert(header, 'EQUINOXA', equinox,
                 comment='[yr] Equinox of equatorial coordinates')
        hdinsert(header, 'SPECSYSA', specsys,
                 comment='Spectral reference frame')


def rectifyorder(image, ordermask, wavecal, spatcal, order,
                 header=None, variance=None, mask=None, bitmask=None,
                 x=None, y=None, dw=None, ds=None,
                 badfrac=0.1, ybuffer=3, xbuffer=None, poly_order=3):
    """
    Construct average spatial profiles for a single order

    See `sofia_redux.spectroscopy.mkspatprof` and
    `sofia_redux.spectroscopy.extspec` for algorithm description.

    Parameters
    ----------
    image : numpy.ndarray of float (nrow, ncol)
        2-d image
    ordermask : numpy.ndarray of int (nrow, ncol)
        Order number of each pixel
    wavecal : numpy.ndarray of float (nrow, ncol)
        Wavelength of each pixel
    spatcal : numpy.ndarray of float (nrow, ncol)
        Spatial coordinates of each pixel
    order : int
        order to process
    header : fits.Header
        Header to update with spectral WCS.
    variance : numpy.ndarray of float (nrow, ncol), optional
        Variance to rectify parallel to the image.
    mask : numpy.ndarray of bool (nrow, ncol), optional
        Mask indicating good (True) and bad (False) pixels.
    bitmask : numpy.ndarray of int (nrow, ncol), optional
        bit-set flags of each pixel.
    x : numpy.array, optional
        (nrow, ncol) x-coordinates
    y : numpy.array, optional
        (nrow, ncol) y-coordinates
    dw : float, optional
        Delta lambda based on the span of the order in pixels and
        wavelengths.
    ds : float, optional
        The spatial sampling of the resampling slit in arcseconds,
        typically given by slth_arc / slth_pix.
    xbuffer : int, optional
        The number of pixels to ignore near the left and right of the slit.
    ybuffer : int, optional
        The number of pixels to ignore near the top and bottom of the slit.
    badfrac : float, optional
        If defines the maximum area of a pixel to be missing before
        that pixel should be considered bad.  For example, a badfrac of 0.1
        means that output flux of a pixel must be the sum of at least
        0.9 input pixels.
    poly_order : int, optional
        Polynomial order to use when converting wavecal and spatcal to
        rectified values.

    Returns
    -------
    dict
        image -> numpy.ndarray (ns, nw)
        wave -> numpy.ndarray (nw,)
        spatial -> numpy.ndarray (ns,)
        mask -> numpy.ndarray (ns, nw)
        bitmask -> numpy.ndarray (ns, nw)
        pixsum -> numpy.ndarray (ns, nw)
        variance -> numpy.ndarray (ns, nw)
        header -> fits.Header
    """
    nrows, ncols = image.shape[:2]
    if x is None or y is None:
        y, x = np.mgrid[:nrows, :ncols]
    omask = (ordermask == order)

    # Straighten the wavelength and spatial coordinates
    rect_xy = get_rect_xy(wavecal, spatcal, x, y, dy=ds, dx=dw,
                          mask=omask, poly_order=poly_order)
    if rect_xy is None:
        log.error("failed to rectify image")
        return

    # Trim bad values around the edges of new coordinate arrays
    rect_xy = trim_xy(*rect_xy, xbuffer=xbuffer, ybuffer=ybuffer,
                      xrange=[2, ncols - 2], yrange=[2, nrows - 2])

    result = reconstruct_slit(image, *rect_xy,
                              header=header, variance=variance,
                              bitmask=bitmask, badpix_mask=mask,
                              badfrac=badfrac, xrange=[0, ncols],
                              yrange=[0, nrows])

    # update the WCS for the new image
    update_wcs(result, spatcal)

    return result
