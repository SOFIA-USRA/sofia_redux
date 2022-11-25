# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

__all__ = ['tortcoord']


def tortcoord(header, skew=False):
    """
    Calculate undistorted image coordinates.

    Use the array size and tort parameters from the FITS header and
    calculate the x and y coordinates in the undistorted array that
    correspond to the x and y coordinates of the raw array, using
    knowledge of the optics of the instrument.

    Parameters
    ----------
    header : fits.Header
        FITS header containing distortion parameters, as produced by
        `exes.readhdr`.
    skew : bool, optional
        If True, correct for echelon slit skewing.

    Returns
    -------
    xcor, ycor : 2-tuple of numpy.ndarray
        Both arrays are of shape (ny, nx) and of type numpy.float64
        `xcor` is the undistorted x-coordinate array.
        `ycor` is the undistorted y-coordinate array.
    """
    instcfg = str(header.get('INSTCFG', 'NONE')).upper()

    if instcfg in ['HIGH_MED', 'HIGH_LOW']:
        xcor, ycor = _get_cross_dispersed_coordinates(header, skew=skew)
    else:
        xcor, ycor = _get_coordinates(header)

    xcor, ycor = _barrel_distortion(xcor, ycor, header)
    xrot, yrot = _rotate_coordinates(xcor, ycor, header)
    return xrot, yrot


def _get_coordinates(header):
    """
    Calculate pixel coordinates.

    Parameters
    ----------
    header : fits.Header

    Returns
    -------
    xcor, ycor : 2-tuple of numpy.ndarray
        The corrected x and y pixel coordinates
    """
    nx = int(header['NSPAT'])  # number of pixels in the x direction
    ny = int(header['NSPEC'])  # number of pixels in the y direction
    slitrot = float(header['SLITROT'])  # slit rotation angle
    pixelwd = float(header['PIXELWD'])  # pixel width
    xdg = float(header['XDG'])  # tort parameter ?
    xdr = float(header['XDR'])  # tort parameter ?
    xdfl = float(header['XDFL'])  # adjusted XD focal length

    # # qtort/testtort version:
    # xdskew = 2 * xdg * xdr + np.tan(slitrot + krot)

    # tort version
    xdskew = (2 * xdg * xdr) + np.tan(slitrot)
    xdsmile = -xdr * pixelwd / xdfl
    xdnlin = -(xdr + 1 / (2 * xdr)) * (pixelwd / xdfl) / 2

    x, y, xcor, ycor = _create_pixel_array(nx, ny)

    # skewing by cross-dispersion smile
    xcor += (xdskew * ycor) + (xdsmile * (ycor ** 2))

    # non-linearity of xd spectrum
    xmid = ((nx + 1) / 2) - 1
    xcor += xdnlin * ((xcor ** 2) - (xmid ** 2))

    return xcor, ycor


def _get_cross_dispersed_coordinates(header, skew=False):
    """
    Calculate pixel coordinates for cross-dispersed data

    Notes:

        - KROT for EXES is the rotation between the chambers.
        - Echelle dispersion is proportional to tan(beta) - sin(delta)
          Other formulas with XDR may need to have sin(delta)
        - XD smile was over-corrected for TEXES, but seems not to be
          for EXES, so fudge factor is removed.
        - Include echelon smile and slit rotation
        - Should include the variation of HRR along an order

    Parameters
    ----------
    header : fits.Header
    skew : bool, optional
        If True, correct for echelon slit skewing.

    Returns
    -------
    xcor, ycor : 2-tuple of numpy.ndarray
        The corrected x and y pixel coordinates

    """
    nx = int(header['NSPAT'])  # number of pixels in the x direction
    ny = int(header['NSPEC'])  # number of pixels in the y direction
    slitrot = float(header['SLITROT'])  # slit rotation angle
    krot = float(header['KROT'])  # k-mirror rotation angle
    pixelwd = float(header['PIXELWD'])  # pixel width
    hrg = float(header['HRG'])  # gamma (out-of-plan) angle for echelon grating
    hrr = float(header['HRR'])  # R number of echelon grating
    xdg = float(header['XDG'])  # gamma angle for XD echelon grating
    xdr = float(header['XDR'])  # tort parameter ?
    xddelta = float(header['XDDELTA'])  # tort parameter ?
    xdfl = float(header['XDFL'])  # adjusted XD focal length
    hrfl = float(header['HRFL'])  # high resolution focal length
    xorder1 = float(header['XORDER1'])  # first pixel of order 1
    spacing = float(header['SPACING'])  # order separation in pixels

    hrskew = (2.0 * hrg * hrr) + np.tan(slitrot)

    # include XD smile and spectrum rotation by k mirror
    xdskew = (2.0 * xdg * xdr) + np.tan(krot)
    xdsmile = -xdr * pixelwd / xdfl
    xddisp = (xdfl * (xdr - xddelta)) / (hrfl * hrr)

    xdnlin = -(xdr + 1 / (2 * xdr)) * (pixelwd / xdfl) / 2
    hrnlin = -(hrr + 1 / (2 * hrr)) * (pixelwd / hrfl) / 2.

    # xdskew and xddisp depend on x because xdr depends on x
    dxdskew = (xdg + xddisp / (2 * xdr)) * (1 + xdr ** 2) * (pixelwd / xdfl)

    xorder0 = xorder1 - spacing / 2

    x, y, xdist, ydist = _create_pixel_array(nx, ny)
    xmid = ((nx + 1) / 2) - 1
    ymid = ((ny + 1) / 2) - 1

    if skew:
        # distance from order center
        order = (x - xorder0) / spacing
        dorder = order - np.round(order)

        # slit skewing within orders by echelon smile
        dx = spacing * dorder
        ydist += hrskew * dx

    # non-linearity of echelon spectrum
    # (subtract ymid^2 so middle moves and ends stay put)
    ycor = ydist + hrnlin * (ydist**2 - ymid**2)

    # skewing by cross dispersion, k mirror, and cross-dispersion smile
    # note: xd dispersion depends on linear y (wavelength)
    xcor = (xdist + (xddisp * ydist) + (xdskew * ycor)
            + (dxdskew * xdist * ycor) + (xdsmile * ycor**2))

    # non-linearity of xd spectrum
    xcor += xdnlin * (xcor**2 - xmid**2)
    return xcor, ycor


def _create_pixel_array(nx, ny):
    """
    Make index arrays and distance from center.

    Parameters
    ----------
    nx : int
    ny : int

    Returns
    -------
    x, y, xdist, ydist : 4-tuple of numpy.ndarray
        X and y index arrays and distances from center.
    """
    y, x = np.mgrid[:ny, :nx]
    xmid = ((nx + 1) / 2) - 1
    ymid = ((ny + 1) / 2) - 1
    xdist = np.asarray(x, dtype=np.float64) - xmid
    ydist = np.asarray(y, dtype=np.float64) - ymid
    return x, y, xdist, ydist


def _barrel_distortion(x, y, header):
    """
    Correct barrel distortion.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates
    y : numpy.ndarray
        y-coordinates
    header : fits.Header
        FITS header

    Returns
    -------
    xnew, ynew : 2-tuple of numpy.ndarray
        `x` and `y` corrected for barrel distortion.
    """
    nx = int(header['NSPAT'])  # number of pixels in the x direction
    brl = float(header['BRL'])  # Barrel tort parameters...
    x0brl = float(header['X0BRL'])
    y0brl = float(header['Y0BRL'])
    xmid = ((nx + 1) / 2) - 1

    barrel = 1 - brl * ((x - x0brl) ** 2 + (y - y0brl) ** 2) / (xmid ** 3)
    xb = x * barrel
    yb = y * barrel
    return xb, yb


def _rotate_coordinates(x, y, header):
    """
    Rotate coordinates by detector rotation angle.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates relative to center of numpy.ndarray
    y : numpy.ndarray
        y-coordinates relative to center of numpy.ndarray
    header : fits.Header
        FITS header

    Returns
    -------
    xrot, yrot : 2-tuple of numpy.ndarray
        `x` and `y` rotated by detector angle.
    """
    detrot = float(header['DETROT'])  # Detector rotation (tort parameter)
    nx = int(header['NSPAT'])  # number of pixels in the x direction
    ny = int(header['NSPEC'])  # number of pixels in the y direction
    cosrot = -np.cos(detrot)
    sinrot = -np.sin(detrot)
    xmid = ((nx + 1) / 2) - 1
    ymid = ((ny + 1) / 2) - 1

    xrot = (x * cosrot) - (y * sinrot) + xmid
    yrot = (y * cosrot) + (x * sinrot) + ymid
    return xrot, yrot
