# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.getpar import getpar

__all__ = ['imgshift_header']


def imgshift_header(header, chop=True, nod=True, dither=True,
                    dripconf=True):
    """
    Calculates the shift_image in the pixel frame for merging an image

    Reads the chop/nod/dither amplitudes, angles, and transform from the
    sky to the detector frame depending on the coordinate system.

    The chop/nod angle was defined in the opposite sense in flights
    earlier than 103.  This distinction is recorded in the ANGLCONV
    keyword value in the header.  If "positive", as for 99 and earlier,
    use the angle as written in the header; otherwise, angles
    will need to be converted to negative values.

    The chop, nod, and dither coordinate systems (CHPCRSYS, NODCRSYS,
    DTHCRSYS in the header) may take values of SIRF (science instrument
    reference frame), TARF (telescope assembly reference frame), or
    ERF (equatorial reference frame).  If these keywords are not
    available in the header, we check the CHPCOORD, NODCOORD, and
    DTHRCS) keywords.  The COORD/CS keyword values are integers where
    0=SIRF, 1=TARF, and 2=ERF.  If the coordinate system cannot be
    be determined, SIRF is used by default.  TARF is seen as equivalent
    to SIRF for the purposes of this algorithm.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        A FITS header
    chop : bool, optional
        Indicates whether to perform the calculation of chop shift_image
    nod : bool, optional
        Indicates whether to perform the calculation of nod shift_image
    dither : bool, optional
        Indicates whether to perform the calculation of dither shift_image
    dripconf : bool, optional
        Indicates whether configuration values can overwrite header
        keyword values.

    Returns
    -------
    dict
        A dict containing the X and Y shifts necessary to merge or
        coadd an image based on the header keywords.
    """

    shift = {'chopx': 0., 'chopy': 0., 'chopcoord': '',
             'nodx': 0., 'nody': 0., 'nodcoord': '',
             'ditherx': 0., 'dithery': 0., 'dithercoord': '',
             'sky_angle': 0.}

    if not isinstance(header, fits.header.Header):
        log.error("header is not %s" % fits.header.Header)
        return shift

    # Get plate scale - assume square pixels
    # may be in degrees, in pltscale
    pixsize = header.get('PLTSCALE', None)
    if pixsize is None:
        telescope = header.get('TELESCOP', 'SOFIA').strip().upper()
        if telescope == 'PIXELS':
            plate_scale = [1.0, 1.0]
        else:
            # assume forcast default
            plate_scale = [0.768, 0.768]
    else:
        pixsize *= 3600.
        plate_scale = [pixsize, pixsize]

    sky_angle = getpar(header, 'SKY_ANGL', dripconf=dripconf,
                       dtype=float, default=0.0)
    shift['sky_angle'] = sky_angle
    sky_angle = np.radians(sky_angle)
    cosa, sina = np.cos(sky_angle), np.sin(sky_angle)
    skyrot = np.array(((cosa, -sina), (sina, cosa))).T

    # Determine angle convention +1/-1
    angle_convention = getpar(
        header, 'ANGLCONV', dripconf=dripconf, dtype=str,
        default='UNKNOWN',
        comment="Chopping/Nodding angle conventions").strip().lower()
    angle_convention = 1 if angle_convention == 'positive' else -1

    # Get chop distances
    if int(header.get('CHOPPING', 0)) and chop:
        dchop = getpar(header, 'CHPAMP1', dtype=float, default=0) * 2
        chop_angle = getpar(header, 'CHPANGLE', dtype=float,
                            default=0, dripconf=dripconf)
        chop_angle = np.radians(chop_angle) * angle_convention
        shift['chopx'] = dchop * np.sin(chop_angle) / plate_scale[0]
        shift['chopy'] = dchop * np.cos(chop_angle) / plate_scale[1]
        coordsys = getpar(header, 'CHPCRSYS', dripconf=dripconf,
                          default='unknown', dtype=str).upper().strip()
        if coordsys not in ['ERF', 'SIRF']:
            val = getpar(header, 'CHPCOORD', default=0,
                         dtype=int, dripconf=dripconf)
            coordsys = 'ERF' if val == 2 else 'SIRF'
        if coordsys == 'ERF':
            shift['chopx'], shift['chopy'] = \
                skyrot @ (shift['chopx'], shift['chopy'])
        shift['chopcoord'] = coordsys

    # Get nod distances
    if int(header.get('NODDING', 0)) and nod:
        dnod = getpar(header, 'NODAMP', default=0,
                      dtype=float, dripconf=dripconf)
        nod_angle = getpar(header, 'NODANGLE', default=0,
                           dtype=float, dripconf=dripconf)
        nod_angle = np.radians(nod_angle) * angle_convention
        shift['nodx'] = dnod * np.sin(nod_angle) / plate_scale[0]
        shift['nody'] = dnod * np.cos(nod_angle) / plate_scale[1]
        coordsys = getpar(header, 'NODCRSYS', dripconf=dripconf,
                          default='unknown', dtype=str).upper().strip()
        if coordsys not in ['ERF', 'SIRF']:
            val = getpar(header, 'NODCOORD', default=0,
                         dtype=int, dripconf=dripconf)
            coordsys = 'ERF' if val == 2 else 'SIRF'
        if coordsys == 'ERF':
            shift['nodx'], shift['nody'] = \
                skyrot @ (shift['nodx'], shift['nody'])
        shift['nodcoord'] = coordsys

    # Get dither distances
    if int(header.get('DITHER', 0)) and dither:
        x = getpar(header, 'DITHERX', dtype=float,
                   default=0, dripconf=dripconf)
        y = getpar(header, 'DITHERY', dtype=float,
                   default=0, dripconf=dripconf)

        # from newest coordinate system header keyword to oldest
        for dthcrsys in ['DTHCRSYS', 'DTHRCS', 'DITHERCS']:
            if dthcrsys in header:
                coordsys = str(header.get(dthcrsys)).strip().upper()
                break
        else:
            coordsys = 'unknown'
        if coordsys not in ['ERF', 'SIRF']:
            coordsys = 'ERF' if coordsys == '2' else 'SIRF'

        if coordsys == 'SIRF':
            rot_angle = np.pi - sky_angle
            cosa, sina = np.cos(rot_angle), np.sin(rot_angle)
            # clockwise rotation
            sirf_x = (x * cosa) + (y * sina)
            sirf_y = (-x * sina) + (y * cosa)
            # x dither in the SIRF coorsys has the opposite sign
            x, y = sirf_x, sirf_y
        else:
            # ERF coordsys as dither values in arcseconds
            x /= -plate_scale[0]
            y /= plate_scale[1]

        shift['ditherx'], shift['dithery'] = x, y
        shift['dithercoord'] = coordsys

    return shift
