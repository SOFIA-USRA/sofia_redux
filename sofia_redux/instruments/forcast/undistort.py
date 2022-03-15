# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re

from astropy import log
from astropy.io import fits
import numpy as np
from scipy.optimize import minimize
from skimage import transform as tf

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert, kref
from sofia_redux.toolkit.image.adjust import frebin
from sofia_redux.toolkit.image.warp import warp_image

from sofia_redux.instruments.forcast.distcorr_model import distcorr_model
from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.peakfind import peakfind
from sofia_redux.instruments.forcast.readmode import readmode

addhist = add_history_wrap('Distortion')

__all__ = ['default_pinpos', 'get_pinpos',
           'find_pixat11', 'update_wcs', 'transform_image',
           'rebin_image', 'frame_image', 'find_source',
           'undistort']


def default_pinpos():
    """
    Default values of the model and instrument points to be warped and
    fitted.  Not tested or recommended for use with real FORCAST data.

    Returns
    -------
    dict
        model -> numpy.ndarray
            N(x, y) model reference positions of shape (N, 2)
        pins -> numpy.ndarray
            N(x, y) pin positions of shape (N, 2)
        nx, ny -> int
            define number of pixels for both pins and image in x and y
        dx, dy -> float
            model x, y spacing
        angle -> float
            clockwise rotation of the model about the center
            in degrees.
        order -> int
            order to be used if using the "polynomial" method in
            sofia_redux.instruments.forcast.undistort based upon this model.
    """
    # The following model defines (6 x 6) pins on a (256 x 256) image
    x_model = [13, 58, 104, 149, 194, 239, 13, 58, 104, 149, 194, 239, 13,
               58, 104, 149, 194, 239, 13, 58, 104, 149, 194, 239, 13, 58,
               104, 149, 194, 239, 13, 58, 104, 149, 194, 239]
    y_model = [242, 242, 242, 242, 242, 242, 196, 196, 196, 196, 196, 196,
               151, 151, 151, 151, 151, 151, 106, 106, 106, 106, 106, 106,
               60, 60, 60, 60, 60, 60, 15, 15, 15, 15, 15, 15]
    x_pos = [6, 54, 102, 150, 198, 247, 5, 54, 102, 150, 198, 247, 5,
             54, 102, 150, 198, 246, 5, 54, 102, 150, 198, 247, 6, 54,
             102, 150, 199, 247, 7, 55, 103, 151, 199, 248]
    y_pos = [242, 243, 243, 243, 242, 241, 197, 199, 199, 199, 198,
             196, 151, 152, 152, 152, 151, 151, 105, 106, 106, 106, 106,
             104, 58, 59, 60, 59, 58, 57, 10, 10, 11, 11, 10, 9]
    x_model = np.asarray(x_model, float)
    y_model = np.asarray(y_model, float)
    x_pos = np.asarray(x_pos, float)
    y_pos = np.asarray(y_pos, float)
    return {
        'model': np.stack((x_model, y_model), axis=1),
        'pins': np.stack((x_pos, y_pos), axis=1),
        'nx': 256, 'ny': 256,
        'dx': 1.0, 'dy': 1.0,  # The ratio dy/dx is used in rebin_image()
        'angle': 0.0,  # not used but will appear in header under PIN_MOD
        'order': 3,  # only used when undistorting using 'polynomial'
    }


def get_pinpos(header, pinpos=None, rotate=False):
    """
    Get the pinhole model and coefficients and update header.

    The majority of this code sanity checks pin positions and
    model positions.  If the user has not supplied the pinpos
    dict, one will be retrieved from the drip configuration or
    from a default set defined in this file (not recommended).
    Note that if the header contains the PIN_MOD keyword it
    will override 'dx', 'dy', 'angle', and 'order' from any
    other source.

    The user may optionally rotate the image by NODANGLE in the
    header.  If the angle is non-zero, the model will be
    rotated around the center of the image and be rescaled by a
    factor of two.  Note that we are not interested in scaling
    or offsets at this point as only the relative warping
    between points is important.  Offsets and scalings will be
    determined during the WCS correction and re-binning steps.

    The PIN_MOD keyword in the header will be updated to contain
    values as a string in the format '[dx,dy,angle,order]' where
    all values are described under the pinpos parameter.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        FITS header that will may be used to determine pinhole model
        coefficients or will be updated with them
    pinpos : dict or str or None
        if pinpos == 'default' then the output pin positions will
        be retried from default_pinpos(). If pinpos is None, the pin
        positions will be retrieved according to the drip
        configuration.  Otherwise, the user should explicitly define
        pinpos as a dict with the following keys and values:

            model -> numpy.ndarray
                N(x, y) model reference positions of shape (N, 2)
            pins -> numpy.ndarray
                N(x, y) pin positions of shape (N, 2)
            nx, ny -> int
                define number of pixels for both pins and image in x and y
            dx, dy -> float
                model x, y spacing
            angle -> float
                clockwise rotation of the model about the center in
                degrees.  This is not required for anything but the
                PIN_MID keyword in the header will be updated with
                this value.
            order -> int
                order to be used if using the "polynomial" method in
                sofia_redux.instruments.forcast.undistort based upon
                this model.

    rotate : bool, optional
        rotate the model by 'NODANGLE' in the header

    Returns
    -------
    dict or None
        pinpos if validated else None.  If rotation occurred then
        pinpos['model'] will be updated.  Keys and values are
        described above under the `pinpos` parameter.
    """

    if pinpos is None:
        pinpos = distcorr_model()
    elif isinstance(pinpos, str):
        if pinpos.strip().lower() == 'default':
            pinpos = default_pinpos()
            log.info("default values have been used for pinholes")
        elif os.path.isfile(pinpos):
            pinpos = distcorr_model(pinhole=pinpos)
    fail, msg = False, ''
    if not isinstance(pinpos, dict):
        fail, msg = True, "pinpos must be %s, None, or 'default'" % dict
    elif 'model' not in pinpos or 'pins' not in pinpos:
        fail, msg = True, "missing model or pin positions"
    elif not isinstance(pinpos['model'], np.ndarray):
        fail, msg = True, "model is not an array"
    elif not isinstance(pinpos['pins'], np.ndarray):
        fail, msg = True, "pins is not an array"
    elif len(pinpos['model'].shape) != 2 or len(pinpos['pins'].shape) != 2:
        fail, msg = True, "model and pins must be of shape (N, 2)"
    elif pinpos['model'].shape != pinpos['pins'].shape:
        fail, msg = True, "model shape does not match pins shape"
    if fail:
        log.error(msg)
        addhist(header, 'correction was not applied (Invalid pinpos)')
        return
    pp = pinpos.copy()

    pin_model_read = getpar(
        header, 'PIN_MOD', comment='pinhole model coeffs')
    if pin_model_read is None:
        dx, dy = pp.get('dx'), pp.get('dy')
        if None in [dx, dy]:
            addhist(header, 'correction was not applied (Invalid coeffs)')
            log.error("could not determine pinhole coefficients")
            return
        order = pp.get('order', -1)
        # angle is not required, but will be put in the header
        angle = pp.get('angle', 0.0)
        hdinsert(header, 'PIN_MOD', '[%f,%f,%f,%i]' % (dx, dy, angle, order),
                 comment='pinhole model coeffs', refkey=kref)
    else:
        # pp will be updated with header PIN_MOD coefficients
        pinmod = re.split(r'[\[\],]', pin_model_read)
        try:
            pinmod = [float(x) for x in pinmod if x != '']
        except (ValueError, TypeError):
            addhist(header, 'correction was not applied (Invalid coeffs)')
            log.error("invalid PIN_MOD values in header")
            return
        if len(pinmod) != 4:
            addhist(header, 'correction was not applied (Invalid coeffs)')
            log.error("invalid PIN_MOD values in header")
            return
        pp['dx'], pp['dy'] = pinmod[0], pinmod[1]
        pp['angle'] = pinmod[2]
        pp['order'] = int(pinmod[3])

    angle = getpar(header, 'NODANGLE', dtype=float, default=0)
    if rotate and angle != 0:
        # Note (x, y) convention, not the standard numpy (y, x)
        img_size = np.array([header['NAXIS1'], header['NAXIS2']])
        center = (img_size - 1) / 2
        shift = tf.SimilarityTransform(translation=-center)
        rot = tf.SimilarityTransform(rotation=np.deg2rad(angle))
        recenter = tf.SimilarityTransform(translation=center)
        pp['model'] = recenter(rot(shift(pp['model'].copy())))
        addhist(header, 'Images rotate by NODANGLE=%f' % angle)

    return pp


def find_pixat11(transform, x0, y0, epsilon=1e-8,
                 xrange=(0, 255), yrange=(0, 255),
                 method=None, maxiter=None, verbose=False,
                 direct=True):
    """
    Calculate the position of x0, y0 after a transformation

    Also calculate the position of x1 (= x0 + 1), y1 (= y0 + 1)
    after the transformation in order to determine pixel scaling
    and offsets.

    If we have a transform that can be directly inverted, then
    the solution is simple and we can return an exact solution.
    In this case the all optional arguments are ignored aside
    from eps which will determine the number of decimal places
    in the output (None will not limit precision).

    Otherwise, we will need to perform some type of minimization.
    The default minimization method, 'TNC' is a truncated Newton
    algorithm used to minimize a function subject to bounds.  It
    is suitable for the purposes of
    sofia_redux.instruments.forcast.undistort.  If you wish
    to solve an unbounded problem, leave method = None to allow
    scipy.optimize.minimize to select an appropriate method or
    set your own.

    Parameters
    ----------
    transform : skimage.transform.PolynomialTransform
        as returned by warp_image with get_transform=True.
    x0 : float
        input x coordinate
    y0 : float
        input y coordinate
    epsilon : float, optional
        If a polynomial transform was used, an iterative method is
        used in place of inversion.  The iteration will be terminated
        after the tolerance is lower than eplison.
    xrange : tuple of float, optional
        (xmin, xmax) range of x values to search
    yrange : tuple of float, optional
        (ymin, ymax) range of y values to search
    method : str, optional
        optimization method.  See scipy.optimize.minimize for
        full list of available options.
    maxiter : int, optional
        terminitate search after this many iterations
    verbose : bool, optional
        if True, print convergence messages
    direct : bool, optional
        Attempts direct inversion if True, otherwise use a
        minimization routine.

    Returns
    -------
    dict
        {x0 -> float, y0 -> float,
         x1 -> float, y1 -> float,
         x1ref -> float, y1ref -> float}
    """
    c0 = np.array([[x0, y0]])
    if epsilon is None:
        decimals = 0
        eps = 1e-8
    else:
        decimals = -int(np.log10(epsilon * 10))
        decimals = 0 if decimals < 0 else decimals
        eps = epsilon

    # Check if we can do direct inversion
    if direct and hasattr(transform, 'inverse') and \
            not isinstance(transform, tf.PolynomialTransform):
        c1 = transform.inverse(c0)
        c1p1 = transform(c1 + 1)
        return {'x0': x0, 'y0': y0,
                'x1ref': np.round(float(c1[0][0]), decimals=decimals),
                'y1ref': np.round(float(c1[0][1]), decimals=decimals),
                'x01': np.round(float(c1p1[0][0]), decimals=decimals),
                'y01': np.round(float(c1p1[0][1]), decimals=decimals)}

    # Otherwise we have to do minimization
    if xrange is None and yrange is None:
        bounds = None
    else:
        bounds = [xrange, yrange]
        if method is None:
            method = 'TNC'

    def transform_distance(params):
        txy = transform(np.array([[params[0], params[1]]]))
        return ((txy - c0) ** 2).sum()

    options = {'disp': verbose}
    if isinstance(maxiter, int):
        options['maxiter'] = maxiter
    p = minimize(transform_distance, c0.copy(), method=method,
                 tol=eps, bounds=bounds, options=options)
    if not p.success:
        try:
            msg = p.message.decode()
        except AttributeError:
            msg = str(p)

        log.error("minimization failed: %s" % msg)
        return
    x1ref, y1ref = p.x

    # Find the pixel in the initial image that is at (x1ref+1, y1ref+1)
    # in the resulting image
    x01, y01 = transform(np.array([[x1ref + 1, y1ref + 1]]))[0]

    return {'x0': x0, 'y0': y0,
            'x1ref': np.round(float(x1ref), decimals=decimals),
            'y1ref': np.round(float(y1ref), decimals=decimals),
            'x01': np.round(float(x01), decimals=decimals),
            'y01': np.round(float(y01), decimals=decimals)}


def update_wcs(header, transform, eps=None):
    """
    Update the WCS in the header according to the transform

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        FITS header to update
    transform
        A function that transforms input coordinates to output
        coordinates.  numpy.ndarray (N, 2) -> [N], [N]
    eps : float, optional
        precision

    Returns
    -------
    dict
        contains output from find_pixat11 with an additional
        key, 'update_cdelt'.  update_cdelt will
        be True if CDELT1 and CDELT2 were successfully
        updated and False otherwise.
    """
    if not isinstance(header, fits.header.Header):
        return

    if 'CRPIX1' not in header or 'CRPIX2' not in header:
        addhist(header, 'CRPIX1 or CRPIX2 are not in header. '
                        'Skipping WCS update')
        return

    crpix1 = header.get('CRPIX1', 1.0)
    crpix2 = header.get('CRPIX2', 1.0)
    addhist(header, 'CRPIX from stack= [%s,%s]' % (crpix1, crpix2))

    log.info("updating WCS")
    dxy = find_pixat11(transform, crpix1 - 1, crpix2 - 1, epsilon=eps)

    addhist(header, 'CRPIX = [%s,%s]' % (
        dxy['x0'] + 1, dxy['y0'] + 1))
    addhist(header, 'CRPIX after transform = [%f,%f]' % (
        dxy['x1ref'] + 1, dxy['y1ref'] + 1))
    hdinsert(header, 'CRPIX1', dxy['x1ref'] + 1, refkey=kref)
    hdinsert(header, 'CRPIX2', dxy['y1ref'] + 1, refkey=kref)

    # Update CDELT
    addhist(header, 'CDELT from stack= [%s,%s]' % (
        header.get('CDELT1', -1), header.get('CDELT2', -1)))
    addhist(header,
            'CROT from stack= [ - ,%s]' % header.get('CROTA2', -1))
    for key in ['CROTA2', 'CDELT1', 'CDELT2']:
        if key not in header:
            dxy['update_cdelt'] = False
            addhist(header, 'CROTA2, CDELT1 or CDELT2 are not in '
                            'header. Using default Platescale')
            break
    else:
        dxy['update_cdelt'] = True
        addhist(header, 'Ref CRPIX+1 = [%f,%f]' % (
            dxy['x01'] + 1, dxy['y01'] + 1))
        header['CDELT1'] *= (dxy['x01'] - dxy['x0'])
        header['CDELT2'] *= (dxy['y01'] - dxy['y0'])

    return dxy


def transform_image(data, xin, yin, xout, yout, header=None, variance=None,
                    transform_type='polynomial', order=4, get_dxy=False,
                    extrapolate=False):
    """
    Transform an image and update header using coordinte point mapping

    Transforms an image such that points in (yin, xin) are warped to
    (yout, xout) in the output image.  If a header is supplied, any
    WCS information will be updated accordingly.  Note that order is
    only important if you are doing a polynomial warp.

    Parameters
    ----------
    data : numpy.ndarray
        input image (nrow, ncol)
    xin : array-like
        warping input x-coordinates
    yin : array-like
        warping input y-coordinates
    xout : array-like
        warping output x-coordinates
    yout : array-like
        warping output y-coordinates
    variance : numpy.ndarray, optional
        variance array to update in parallel with the data array
    header : astropy.io.fits.header.Header, optional
        FITS header to update WCS
    transform_type : str, optional
        see scikit.image.transform for a list of available transform
        types.
    order : int, optional
        Order to use if transform_type is 'polynomial'
    get_dxy : bool, optional
        If True

    Returns
    -------
    2-tuple or 3-tuple
        - warped output image (nrow, ncol)
        - warped variance (nrow, ncol)
        - dxy, optional output from update_wcs
    """
    image, transform = warp_image(
        data.copy(), xin, yin, xout, yout, transform=transform_type,
        order=order, mode='constant', cval=np.nan, get_transform=True,
        extrapolate=extrapolate)
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    var = variance.copy() if dovar else None
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance at transform_image)"
        addhist(header, msg)
        log.error(msg)
    if dovar:
        var = warp_image(var, xin, yin, xout, yout, transform=transform_type,
                         order=order, mode='constant', cval=np.nan,
                         extrapolate=extrapolate)
    addhist(header, 'correction model uses order %s' % order)
    log.info("distortion solution order: %s" % order)
    dxy = update_wcs(header, transform)

    if get_dxy:
        return image, var, dxy
    else:
        return image, var


def rebin_image(image, factor, header=None, variance=None, platescale=None):
    r"""
    Rebin the image to square pixels

    Here factor represents dy/dx (pixel height/width).  If the pixels
    were square, then 11\*dx == 11\*dy, or nx\*dy == ny\*dy.  The smaller
    the pixels, the larger the number of them across the array for the
    same angular distance.  We can make the pixels square by re-binning
    such that the above equation holds, but must assume one dimension
    is the reference with 256 pixels.

    We can adopt the principle that the re-binned array should be as
    close to 256 x 256 as possible.  In that case nx'\*ny' == 256\*256
    and nx' == ny' \* (dy/dx).

    Parameters
    ----------
    image : numpy.ndarray
        input image with rectangular pixels (nrow, ncol)
    factor : float
        the ration dy/dx (pixel height/width)
    header : astropy.io.fits.header.Header
        FITS header containing WCS.  Values will be updated and
        HISTORY messages will be written.
    variance : numpy.ndarray
        variance array to update in parallel
    platescale : float
        if set, CDELT will be set to this value in header WCS.
        Should be provided as arc seconds.

    Returns
    -------
    2-tuple
        numpy.ndarray : the rebinned image (nrow, ncol)
        numpy.ndarray : the rebinned variance (nrow, ncol)
    """
    ny = int(np.round(np.sqrt(np.product(image.shape) / factor)))
    nx = int(np.round(ny * factor))
    rebinned = frebin(image.copy(), (ny, nx), total=True)

    dovar = isinstance(variance, np.ndarray) and variance.shape == image.shape
    if not dovar and variance is not None:
        addhist(header, 'variance not propagated (invalid variance at rebin)')
        log.error("variance shape does not match data shape")
    var = variance.copy() if dovar else None
    if dovar:
        # Rebin variance as standard deviation so that the
        # flux-conservation factor gets applied appropriately
        var = frebin(np.sqrt(var), (ny, nx), total=True) ** 2

    if header is None:
        return rebinned, var
    # The reference when rebinning an immage between the initial and
    # the final image is the corner of the image, not the first pixel
    # position.  The corner is 0.5 pixels from the pixel number 0.
    # NOTE: CRPIX values have +1 added to pixel positions, so what
    # we did was the same as x = crpix - 1 + 0.5, and then the same
    # in reverse.
    addhist(header, "new image size: %s x %s" % (nx, ny))
    log.info("new image size: %s x %s" % (nx, ny))
    x = header['CRPIX1'] - 0.5 - (image.shape[1] / 2)
    header['CRPIX1'] = x * (nx / image.shape[1]) + 0.5 + (nx / 2.0)
    y = header['CRPIX2'] - 0.5 - (image.shape[0] / 2)
    header['CRPIX2'] = y * (ny / image.shape[0]) + 0.5 + (ny / 2.0)

    if 'CDELT1' not in header or 'CDELT2' not in header:
        return rebinned, var
    if platescale is not None:
        deg = platescale / 3600.0
        header['CDELT1'] = -deg if header['CDELT1'] < 0 else deg
        header['CDELT2'] = -deg if header['CDELT2'] < 0 else deg
    else:
        header['CDELT1'] *= image.shape[1] / nx
        header['CDELT2'] *= image.shape[0] / ny
    addhist(header, 'CDELT after rebin = [%f,%f]' % (
        header['CDELT1'], header['CDELT2']))

    return rebinned, var


def frame_image(image, shape, header=None, variance=None, border=0, wcs=True):
    """
    Frame an image in the center of a new image and add border

    The size of the new image must be larger than the original in
    both dimensions.  There is a hard-coded border minimum of 5
    pixels.  Unfilled pixels are set to NaN.

    Parameters
    ----------
    image : numpy.ndarray
        input data array (nrow, ncol) to be inserted in the
        center of the output image
    header : astropy.io.fits.header.Header
        input FITS header.  WCS will be updated and HISTORY messages
        will be added.
    shape : 2-tuple of int
        shape of the output image without the border (nrow, ncol)
        NOTE - Python (y, x) order please
    variance : numpy.ndarray
        variance array (nrow, ncol) to update in parallel with image.
    border : int, optional
        default border to add around image.  This will only be applied
        if 'BORDER' is in neither the header nor the drip configuration.
        The minimum border allowed is 5.
    wcs : bool, optional
        If True update the WCS in the header.  Will only update keywords
        that already exist.

    Returns
    -------
    2-tuple
        numpy.ndarray : output data (nrow?, ncol?)
        numpy.ndarray : output variance (nrow?, ncol?)

    Notes
    -----
    The output array shapes are not definitively predicted beforehand.
    `border` merely indicates the minimum size of the output arrays.  If
    the distortion results in an output image that is larger than
    (ny, nx) + 2*border, the border will be expanded to ensure that no
    errors are encountered.  `border` should be set in the drip
    configuration file as that value will override any value supplied to
    sofia_redux.instruments.forcast.undistort or extracted from the header.
    """
    dovar = isinstance(variance, np.ndarray) and variance.shape == image.shape
    var = variance.copy() if dovar else None
    if not dovar and variance is not None:
        msg = 'variance not propagated (invalid variance at frame_image)'
        addhist(header, msg)
        log.error(msg)

    if header is None:
        header = fits.header.Header()
    comment = 'additional border pixels'
    border = getpar(
        header, 'BORDER', comment=comment, dtype=int, default=border)

    shape = np.array(shape).astype(int)
    imshape = np.array(image.shape).astype(int)
    minborder = int(np.ceil((imshape - shape) / 2).max())
    if minborder > border:
        msg = "expanding border from %s to %s" % (border, minborder)
        addhist(header, msg)
        log.info(msg)
        border = minborder
        if 'BORDER' in header:
            header['BORDER'] = border

    shape += 2 * border
    minyx = (shape // 2) - (imshape // 2)
    framed = np.full(shape, np.nan, dtype=image.dtype)
    framed[minyx[0]: minyx[0] + imshape[0],
           minyx[1]: minyx[1] + imshape[1]] = image.copy()

    if dovar:
        newvar = np.full(shape, np.nan, dtype=var.dtype)
        newvar[minyx[0]: minyx[0] + imshape[0],
               minyx[1]: minyx[1] + imshape[1]] = var.copy()
        var = newvar

    header['NAXIS1'] = shape[1]
    header['NAXIS2'] = shape[0]
    if not wcs:
        return framed, var

    if 'CRPIX1' and 'CRPIX2' in header:
        header['CRPIX1'] += minyx[1]
        header['CRPIX2'] += minyx[0]
        addhist(header, "CRPIX after border = [%f,%f]" % (
            header['CRPIX1'], header['CRPIX2']))
    if 'CRVAL1' and 'CRVAL2' in header:
        addhist(header, 'CRVAL = [%f,%f]' % (
            header['CRVAL1'], header['CRVAL2']))

    return framed, var


def find_source(image, header):
    """
    Find single peak in image and update SRCPOS in header

    Parameters
    ----------
    image : numpy.ndarray
        input image (nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        FITS header to update

    Returns
    -------
    None
    """
    search = np.zeros_like(image)
    clip = getpar(header, 'BORDER', update_header=False, default=0,
                  dtype=int) + 10
    search[clip: -clip, clip: -clip] = image[clip: -clip, clip: -clip]
    pos = peakfind(search, npeaks=1, silent=True, coordinates=True)

    if pos is None or len(pos) != 1:
        return
    hdinsert(header, 'SRCPOSX', pos[0][0],
             comment='Source x-position', refkey=kref)
    hdinsert(header, 'SRCPOSY', pos[0][1],
             comment='Source y-position', refkey=kref)


def undistort(data, header=None, pinhole=None, rotate=False,
              variance=None, transform_type='piecewise-affine',
              default_platescale=0.768, extrapolate=False):
    """
    Corrects distortion due to camera optics.

    Resamples data with a polynomial warping calculated from known
    pinhole positions to correct for optical distortions.  Applies
    a border to the image to avoid losing data off the edge of the
    array.  After distortion correction, the WCS keywords are
    updated.  This function is typicalled called without the
    `rotate` parameter and `pinpos` provided by the output of
    discorr_model.  The border width can be set in the `border`
    parameter in the configuration file; default is 128 pixels.
    The border pixels are set to NaN to mark them as non-data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        Input header; will be updated with a HISTORY message
    pinhole : dict or str, optional
        dictionary defining the pinhole model or path to a pinhole
        file.  If None, will generate a default model based on the
        drip configuration file.  If 'default' will use model
        defined in the default_pinpos() function.  See get_pinpos()
        for a description of keys and values.
    rotate : bool, optional
        If True, distortion model data will be rotated by NODANGLE
        (in header) before applying distortion correction.
    variance : numpy.ndarray, optional
        Variance array (nrow, ncol) to update in parallel with the
        data array
    transform_type : str, optional
        See scikit.image.transform for available transformation types.
        Recommended are 'polynomial' and 'piecewise-affine'.  Note that
        if anything other that 'polynomial' is selected, virtual corner
        pins are created to allow for extrapolation.
    default_platescale : float, optional
        If set, CDELT1 and CDELT2 will be set to this value. Set to
        None to automatically update.

    Returns
    -------
    2-tuple
        numpy.ndarray : The distortion-corrected array (nrow, ncol)
        numpy.ndarray : The distortion-corrected variance (nrow, ncol)
    """
    create_header = False
    if not isinstance(header, fits.header.Header):
        create_header = True
        header = fits.header.Header()
    elif 'CRPIX1' not in header or 'CRPIX2' not in header:
        create_header = True

    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, "correction was not applied (Invalid data)")
        log.error("must provide valid data array")
        return

    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    var = variance.copy() if dovar else None
    if not dovar and variance is not None:
        msg = "variance not propagated (invalid variance)"
        addhist(header, msg)
        log.error(msg)

    if create_header:
        addhist(header, 'Created header')
        log.debug("Creating basic header WCS. Using "
                  "CRPIX=data.shape/2, CDELT=-1.0, CROTA2=0.0")
        header['NAXIS1'] = data.shape[1]
        header['NAXIS2'] = data.shape[0]
        header['CDELT1'] = -1.0
        header['CDELT2'] = -1.0
        header['CRPIX1'] = data.shape[1] / 2
        header['CRPIX2'] = data.shape[0] / 2
        header['CROTA2'] = 0.0

    if header.get('NAXIS1') != data.shape[1]:
        header['NAXIS1'] = data.shape[1]
    if header.get('NAXIS2') != data.shape[0]:
        header['NAXIS2'] = data.shape[0]
    orig_crpix1 = header['CRPIX1']
    orig_crpix2 = header['CRPIX2']

    # Get the pinhole model
    pinhole = get_pinpos(header, pinpos=pinhole, rotate=rotate)
    if pinhole is None:
        log.error("failed to find a valid pinhole model")
        return
    x_model = pinhole['model'][:, 0]
    y_model = pinhole['model'][:, 1]
    x_pos = pinhole['pins'][:, 0]
    y_pos = pinhole['pins'][:, 1]

    # Perform transformation
    corrected_image, var, dxy = transform_image(
        data.copy(), x_pos, y_pos, x_model, y_model, header=header,
        variance=var, transform_type=transform_type, order=pinhole['order'],
        get_dxy=True, extrapolate=extrapolate)

    # Rebin image to square pixels
    if default_platescale is None:
        if not isinstance(dxy, dict) or not dxy.get('update_cdelt'):
            default_platescale = 0.768
    factor = pinhole['dy'] / pinhole['dx']
    rebinned, var = rebin_image(
        corrected_image.copy(), factor, header=header, variance=var,
        platescale=default_platescale)

    # Frame the image in an array the size of the original data with
    # an added border (defined in the drip configuration or header)
    # If the border has not been defined, the default is 128 pixels on
    # all sides.
    undistorted, var = frame_image(
        rebinned.copy(), data.shape, header=header, variance=var,
        wcs=isinstance(dxy, dict))

    # Update the product type
    hdinsert(header, 'PRODTYPE', 'UNDISTORTED',
             comment='product type', refkey=kref)

    # If image is a standard and NMC, locate the brightest source
    # and record its position in the basehead.
    mode = readmode(header)
    obstype = getpar(header, 'OBSTYPE', comment='Observation type')
    if mode.lower().strip() == 'nmc' and \
            obstype.lower().strip() == 'standard_flux':
        find_source(undistorted, header)
    log.info("undistorted image shape (x, y): (%i, %i)" %
             (undistorted.shape[1], undistorted.shape[0]))

    # also store the cumulative change to CRPIX
    dcrpix1 = header.get('DCRPIX1', 0.0)
    dcrpix2 = header.get('DCRPIX2', 0.0)
    hdinsert(header, 'DCRPIX1', dcrpix1 + header['CRPIX1'] - orig_crpix1,
             comment='Change in CRPIX before registration', refkey=kref)
    hdinsert(header, 'DCRPIX2', dcrpix2 + header['CRPIX2'] - orig_crpix2,
             comment='Change in CRPIX before registration', refkey=kref)

    return undistorted, var
