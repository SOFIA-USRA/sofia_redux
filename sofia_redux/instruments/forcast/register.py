# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap, kref, hdinsert
from sofia_redux.toolkit.image.adjust import rotate, register_image

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.peakfind import peakfind
from sofia_redux.instruments.forcast.shift import shift

addhist = add_history_wrap('Register')

__all__ = ['coadd_centroid', 'coadd_correlation', 'coadd_header',
           'coadd_user', 'register']


def headererror(header, msg):
    addhist(header, msg)
    log.error(msg)


def coadd_centroid(data, reference, header=None, variance=None,
                   crpix=None, border=0, rot_angle=None, missing=np.nan,
                   shift_order=None, rotation_order=1, get_offsets=False):
    """
    Shift an image for coadding using a centroid algorithm

    Steps are:
        1. Rotate data and variance wrt reference (if rot_angle
           is provided)
        2. Run peakfind on data wrt reference on 1 peak
        3. Return rotated data (if it was rotated) should peakfind
           fail.
        4. Shift image by offsets calculated wrt reference using
           peakfind.
        5. Record the shifts in the header with the COADX0, COADY0
           keywords.

    Parameters
    ----------
    data : numpy.ndarray
        input image array (nrow, ncol)
    reference : numpy.ndarray
        input reference array to compare data to (nrow, ncol)
    header : astropy.fits.header.Header
        FITS header of input data
    border : int
        exclude `border` pixels from the outside edges when
        attempting to find peaks in the image and reference
    shift_order : int, optional
        Order of interpolation for the shift.  The shift order must be
        between 0 and 5, with a default of 3
    rotation_order: int, optional
        Order of interpolaton for the rotation.  The rotation order
        must be between 0 and 5, with a default of 3
    variance : numpy.ndarray
        propagate provided variance if set (nrow, ncol)
    rot_angle : float
        Indicates that the data image is rotated wrt the reference
        image by this amount.  The data image (and variance if supplied)
        will be rotated clockwise by `rot_angle` degrees
    crpix : list, optional
        If provided, will be updated to the new [CRPIX1, CRPIX2]
        values following the shift
    get_offsets : bool, optional
        Do not shift the image.  Only return the (x, y) offset
    missing : float, optional
        Value to represent missing data during shift

    Returns
    -------
    2-tuple
        numpy.ndarray : The shifted image (nrow, ncol)
        numpy.ndarray : The shifted variance (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
        addhist(header, 'Created header')

    var = variance.copy() if isinstance(variance, np.ndarray) else None

    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        headererror(header, "Coadd not applied (invalid data)")
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("invalid variance")

    if not isinstance(reference, np.ndarray) or reference.shape != data.shape:
        headererror(header, "Coadd not applied (invalid reference)")
        return

    image = data.copy()
    ref = reference.copy()
    if border > 0:
        if border * 2 >= data.shape[0] or border * 2 >= data.shape[1]:
            headererror(header, 'Coadd not applied (invalid border)')
            return
        image = image[border: -border, border: -border]
        ref = ref[border: -border, border: -border]

    rot_data = data.copy()
    rot_var = var.copy() if dovar else None
    if rot_angle is not None and abs(rot_angle % 360) != 0:
        image = rotate(image, rot_angle, order=rotation_order)
        rot_data = rotate(data, rot_angle, order=rotation_order)
        if dovar:
            rot_var = rotate(var, rot_angle, order=rotation_order)

    fwhm = getpar(header, 'MFWHM', dtype=float, default=4.5,
                  comment='fwhm for peakfind algorithm')

    shift_coords = peakfind(
        ref, image, fwhm=fwhm, npeaks=1, positive=True, refine=True,
        coordinates=True, silent=True)

    if len(shift_coords) == 0:
        headererror(header, 'Coadd not applied (peakfind failed)')
        return

    offset = shift_coords[0]  # there should only be one peak
    if get_offsets:
        return offset

    addhist(header, 'Used centroid registration to determine shifts')
    addhist(header, 'X, Y shift of image is (%f, %f)' %
            (offset[0], offset[1]))
    log.debug("calculated X,Y shift: (%s, %s)" % (offset[0], offset[1]))
    hdinsert(header, 'COADX0', offset[0], refkey=kref,
             comment='X Shift during coadd process')
    hdinsert(header, 'COADY0', offset[1], refkey=kref,
             comment='Y shift during coadd process')
    return shift(rot_data, offset, variance=rot_var, missing=missing,
                 crpix=crpix, order=shift_order)


def coadd_correlation(data, reference, header=None, variance=None,
                      border=0, rot_angle=None, xydither=None, crpix=None,
                      shift_order=None, rotation_order=1, upsample=100,
                      get_offsets=False, missing=np.nan):
    """
    Shift an image for coaddition using a correlation algorithm

    Parameters
    ----------
    data : numpy.ndarray
        Data to be shifted (nrow, ncol)
    reference : numpy.ndarray
        Data to be compared with (nrow, ncol)
    header : The fits header of the new input data file, optional
    variance : None or numpy.ndarray, optional
        Propagate the provided variance (nrow, ncol)
    border : int, optional
        Remove `border` pixels from the edge of the image before
        correlating
    shift_order : int, optional
        Order of interpolation for the shift.  The shift order must be
        between 0 and 5, with a default of 3
    rotation_order: int, optional
        Order of interpolaton for the rotation.  The rotation order
        must be between 0 and 5, with a default of 3
    xydither : array_like, optional
        Initial x,y shift estimates.  If not set, default is 0, 0
    crpix : array-like, optional
        If provided, will be updated to match image shift_image [x, y]
    upsample : int, optional
        Data will be registered to within 1 / `upsample` of a pixel
    rot_angle : float, optional
        Indicates that the data image is rotated wrt the reference
        image by this amount.  The data image (and variance if supplied)
        will be rotated clockwise by `rot_angle` degrees
    get_offsets : bool, optional
        Only return the (x, y) shift.  Do not shift the data.
    missing : float, optional
        Value to represent missing data during shift

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The shifted image (nrow, ncol)
        The shifted variance (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
        addhist(header, 'Created header')

    var = variance.copy() if isinstance(variance, np.ndarray) else None

    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        headererror(header, "Coadd not applied (invalid data)")
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("Variance must match data: %s" % type(variance))

    if not isinstance(reference, np.ndarray) or reference.shape != data.shape:
        headererror(header, "Coadd not applied (invalid reference)")
        return

    image = data.copy()
    ref = reference.copy()
    if border > 0:
        if border * 2 >= data.shape[0] or border * 2 >= data.shape[1]:
            headererror(header, 'Coadd not applied (invalid border)')
            return
        image = image[border: -border, border: -border]
        ref = ref[border: -border, border: -border]

    rot_data = data.copy()
    rot_var = var.copy() if dovar else None
    if rot_angle is not None and abs(rot_angle % 360) != 0:
        image = rotate(image, rot_angle, order=rotation_order)
        rot_data = rotate(data, rot_angle, order=rotation_order)
        if dovar:
            rot_var = rotate(var, rot_angle, order=rotation_order)

    xyshift = abs(getpar(header, 'XYSHIFT', dtype=float, default=40.0,
                         comment='maximum shift to be applied +/-'))
    offset = register_image(
        image, ref, upsample=upsample, maxshift=xyshift, shift0=xydither)

    log.debug("calculated X,Y shift: (%s, %s)" % (offset[0], offset[1]))

    if np.isnan(offset).any() or np.array(abs(offset) > xyshift).any():
        headererror(header, "Coadd not applied (exceeded maxshift)")
        return

    if get_offsets:
        return offset

    addhist(header, 'Used correlation registration to determine shifts')
    addhist(header, 'X, Y shift of image is (%f, %f)' % (offset[0], offset[1]))
    hdinsert(header, 'COADX0', offset[0], refkey=kref,
             comment='X shift during coadd process')
    hdinsert(header, 'COADY0', offset[1], refkey=kref,
             comment='Y shift during coadd process')
    return shift(rot_data, offset, variance=rot_var, missing=missing,
                 crpix=crpix, order=shift_order)


def coadd_header(data, header, variance=None, crpix=None, get_offsets=False,
                 shift_order=None, missing=np.nan):
    """
    Shift an image for coaddition using header information

    Parameters
    ----------
    data : numpy.ndarray
        Data to be shifted (nrow, ncol)
    header : The fits header of the new input data file, optional
    variance : None or numpy.ndarray, optional
        Propagate the provided variance (nrow, ncol)
    shift_order : int, optional
        Order of interpolation for the shift.  The shift order must be
        between 0 and 5, with a default of 3
    crpix : array-like, optional
        If provided, will be updated to match image shift_image [x, y]
    get_offsets : bool, optional
        Only return the (x, y) shift.  Do not shift the data.
    missing : float, optional
        Value to represent missing data during shift

    Returns
    -------
    2-tuple
        The shifted image (nrow, ncol)
        The shifted variance (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        log.error("invalid header")
        return

    shifts = imgshift_header(header, chop=False, nod=False)
    if 'ditherx' in shifts and 'dithery' in shifts:
        offset = shifts['ditherx'], shifts['dithery']
    else:
        headererror(header, "Coadd failed (invalid header shift)")
        return

    if get_offsets:
        return offset

    addhist(header, 'Used header registration to determine shifts')
    addhist(header, 'X, Y shift of image is (%f, %f)' % (offset[0], offset[1]))
    hdinsert(header, 'COADX0', offset[0], refkey=kref,
             comment='X shift during coadd process')
    hdinsert(header, 'COADY0', offset[1], refkey=kref,
             comment='Y shift during coadd process')

    return shift(data, offset, variance=variance, missing=missing,
                 crpix=crpix, order=shift_order)


def coadd_user(data, reference, position, header=None, variance=None,
               crpix=None, get_offsets=False, shift_order=None,
               missing=np.nan):
    """
    Shift an image for coaddition using header information

    Parameters
    ----------
    data : numpy.ndarray
        Data to be shifted (nrow, ncol)
    reference : array_like of float
        2-element (x, y) reference position in pixels
    position : array_like of float
        2-element (x, y) user position in pixels
    header : The fits header of the new input data file, optional
    variance : None or numpy.ndarray, optional
        Propagate the provided variance (nrow, ncol)
    shift_order : int, optional
        Order of interpolation for the shift.  The shift order must be
        between 0 and 5, with a default of 3
    crpix : array-like, optional
        If provided, will be updated to match image shift_image [x, y]
    get_offsets : bool, optional
        Only return the (x, y) shift.  Do not shift the data.
    missing : float, optional
        Value to represent missing data during shift

    Returns
    -------
    2-tuple
        The shifted image (nrow, ncol)
        The shifted variance (nrow, ncol)
    """
    if not hasattr(reference, '__len__') or len(reference) != 2:
        headererror(header, 'Invalid user reference position')
        return
    elif not hasattr(position, '__len__') or len(position) != 2:
        headererror(header, 'Invalid user position')
        return
    offset = np.array(reference) - np.array(position)
    maxshift = getpar(header, 'MAXREGSH', dtype=int, default=2000,
                      update_header=False)
    absoff = np.abs(offset)
    if np.array(absoff > maxshift).any():
        log.warning('Shift (%f, %f) larger than %i. No shift applied' %
                    (offset[0], offset[1], maxshift))
        offset *= 0

    offset[absoff < 1e-4] = 0
    if get_offsets:
        return offset

    if header is not None:
        addhist(header, 'Applied user shifts')
        addhist(header, 'X, Y shift of image is (%f, %f)' %
                (offset[0], offset[1]))
        hdinsert(header, 'COADX0', offset[0], refkey=kref,
                 comment='X shift during coadd process')
        hdinsert(header, 'COADY0', offset[1], refkey=kref,
                 comment='Y shift during coadd process')

    return shift(data, offset, variance=variance, missing=missing,
                 crpix=crpix, order=shift_order)


def register(data, header, reference=None, variance=None, crpix=None,
             position=None, get_offsets=False, missing=np.nan,
             algorithm=None):
    """
    Use dither data to shift_image input image to a reference image.

    This function shifts dithered observations of the same object into
    the same pixel coordinate system, in preparation for coaddition
    of the images.  No rotation or scaling is performed.  The method
    for determining the shift_image is read from the configuration file.

    In the configuration file, if CORCOADD is set to to CENTROID, then a
    centroiding algorithm is used to determine the shift_image.  If CORCOADD is
    XCOR, a cross-correlation algorithm is used.  If CORCOADD is HEADER,
    ir no reference data is provided, then dither information from the
    FITS header is used to determine the shift_image.  If CORCOADD is NOSHIFT,
    then no shift_image is performed.  If the centroiding algorithm is selected
    and it fails for any reason, then a header shift_image algorithm is used
    instead.  If a reference source position and new source position are
    provided in <parameter: reference> and <parameter: position>, they will
    be used to determine the shift_image, regardless of the value of CORCOADD.

    The order of the interpolation used to shift_image the data is determined
    by the SHIFTORD keyword in the configuration file.  SHIFTORD=1 or 3 will
    interpolate to perform sub-pixel shifts; SHIFTORD=0 will shift_image by
    integer pixels (no interpolation).

    Parameters
    ----------
    data : numpy.ndarray
        The image to register (nrow, ncol)
    header : astropy.io.fits.header.Header
        The FITS header of the input data file
    reference : numpy.ndarray or 2-tuple, optional
        The reference image or data to register to.  May be an image
        array (for centroiding or cross-correlation) or a source position
        (x, y).  If a reference source position is provided in REFDATA,
        a new source position must be provided in <parameter: position>.
        If <parameter: reference> is not specified, and the algorithm is not
        NOSHIFT, then header data will be used to register the image.
    variance : numpy.ndarray, optional
        Variance array (nrow, ncol) to update in parallel with the
        data array.
    crpix : array_like, optional
        2-element (x, y) array to update
    position : 2-tuple, optional
        Position (x, y) to register to a reference position provided
        in <parameter: reference>, used with the interactive USER registration
        algorithm.
    get_offsets : bool, optional
        If True, only return the (x, y) shift.  Headers will still be updated
    missing : float, optional
        Value to represent missing data during shift
    algorithm : str, optional
        Algorithm to use when registering images.  Default is to read from
        the configuration.

    Returns
    -------
    2-tuple
        numpy.ndarray : Registered image array (nrow, ncol)
        numpy.ndarray : Registered variance array (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        if header is None:
            header = fits.header.Header()
            addhist(header, "Created header")
        else:
            log.error("Coadd failed (invalid header)")
            return

    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        headererror(header, "Coadd failed (invalid data)")
        return

    var = variance.copy() if isinstance(variance, np.ndarray) else None
    dovar = var is not None and var.shape == data.shape
    if variance is not None and not dovar:
        headererror(header, "Variance not propagated (invalid variance)")

    if algorithm is None:
        algorithm = getpar(header, 'CORCOADD', default='HEADER')
    if position is not None:
        algorithm = 'USER'
    elif reference is None and algorithm != 'NOSHIFT':
        algorithm = 'HEADER'

    if algorithm == 'XCOR':
        s = imgshift_header(header, chop=False, nod=False)
        result = coadd_correlation(
            data, reference, header=header, variance=var, missing=missing,
            xydither=(s['ditherx'], s['dithery']), get_offsets=get_offsets,
            crpix=crpix)
    elif algorithm == 'CENTROID':
        result = coadd_centroid(
            data, reference, header=header, variance=var, missing=missing,
            get_offsets=get_offsets)
    elif algorithm == 'HEADER':
        log.debug('Using header shift to register frame')
        result = coadd_header(data, header, variance=var, missing=missing,
                              get_offsets=get_offsets, crpix=crpix)
    elif algorithm == 'USER':
        result = coadd_user(
            data, reference, position, header=header, missing=missing,
            variance=var, get_offsets=get_offsets, crpix=crpix)
    elif algorithm == 'NOSHIFT':
        log.debug("No shift applied")
        addhist(header, "No shift applied")
        if get_offsets:
            result = (0., 0.)
        else:
            result = data.copy(), var
    else:
        headererror(header, 'Unknown registration algorithm: %s' %
                    repr(algorithm))
        result = None

    if result is None:
        headererror(header, 'Coadd registration failed')
    return result
