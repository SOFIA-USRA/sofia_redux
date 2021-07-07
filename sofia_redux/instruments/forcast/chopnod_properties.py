# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap

from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('Chopnodprop')

__all__ = ['chopnod_properties']


def chopnod_properties(header, update_header=False):
    """
    Returns the chop nod properties in the detector reference frame

    Finds the chop and nod amplitudes in the detector frame.  It also
    returns if the chop/nod configuration was NMC.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    update_header : bool, optional
        Update the header with plate scale (in HISTORY)

    Returns
    -------
    dict
        chop: dict
        nod: dict
        nmc: bool
        c2nc2 : int

    Notes
    -----
    - This function is only useful in the case of C2 or C2N observations
    - Assume SOFIA plate scale = 0.768 arcsec/pixel
    """
    telescope = getpar(header, 'TELESCOP')
    plate_scale = 1.0 if telescope == 'PIXELS' else 0.768

    # pixels are square after undistort [x, y]
    plate_scale = np.array([plate_scale, plate_scale])
    if update_header:
        addhist(header, 'X,Y plate scales are %s,%s' %
                (plate_scale[0], plate_scale[1]))

    chop, nod = {}, {}
    # find distances and angles
    keys = ['CHPAMP1', 'CHPANGLE', 'NODAMP', 'NODANGLE']
    for key in keys:
        value = getpar(header, key, dtype=float, default=0.)
        if key == 'CHPAMP1':
            value *= 2
        d = chop if 'CHP' in key else nod
        if 'ANGLE' in key:
            d['angle'] = np.radians(value)
        else:
            d['distance'] = value
    for d in chop, nod:
        d['dxdy'] = [np.sin(d['angle']), np.cos(d['angle'])]
        d['dxdy'] *= d['distance'] / plate_scale

    # find coordinate system
    # If present, use [CHP/NOD]CRSYS: 'ERF' or 'SIRF'
    # Otherwise, use [CHP/NOD]COORD: 0=SIRF, 1=TARF, 2=ERF
    # NOTE: I think TARF is ignored - don't know why
    for cn in ['CHP', 'NOD']:
        d = chop if cn == 'CHP' else nod
        value = getpar(header, '%sCRSYS' % cn, dtype=str).upper().strip()
        if value not in ['ERF', 'SIRF']:
            value = getpar(header, 'CHPCOORD', dtype=int, default=0)
            value = 'ERF' if value == 2 else 'SIRF'
        d['coordsys'] = value

    sky_angle = getpar(header, 'SKY_ANGL', dtype=float, default=0.)
    sky_angle = np.radians(sky_angle)
    cosa, sina = np.cos(sky_angle), np.sin(sky_angle)

    for d in chop, nod:
        if d['coordsys'] == 'ERF':
            dx = (d['dxdy'][0] * cosa) + (d['dxdy'][1] * sina)
            dy = (d['dxdy'][1] * cosa) - (d['dxdy'][0] * sina)
            d['dxdy'][:] = [dx, dy]

    # Check for Nod-match-chop mode:
    cnmode = getpar(header, 'SKYMODE', dtype=str,
                    default='NONE').strip().upper()
    nmc = cnmode == 'NMC'
    # More detailed checks for nmc
    if cnmode == 'NONE':
        # check for NMC
        nod_angle = getpar(header, 'CHPANGLR', dtype=float, default=0.)
        chop_angle = getpar(header, 'NODANGLR', dtype=float, default=0.)
        # chop amplitude = nod amplitude AND chop angle - nod angle = 180 deg
        if np.round(chop['distance']) == np.round(nod['distance']):
            d_angle = abs(chop_angle - nod_angle)
            if d_angle == 180 or d_angle == 0:
                nmc = True

    # check for C2NC2 value
    c2nc2 = 0
    instmode = getpar(header, 'INSTMODE', dtype=str).strip().upper()
    header_c2nc2 = getpar(header, 'C2NC2', dtype=int, default=0,
                          update_header=False)
    if cnmode in ['C2NC2', 'C2', 'NXCAC'] or (
            header_c2nc2 and instmode == 'C2'):
        c2nc2 = 1
        if (cnmode == 'NXCAC') or getpar(
                header, 'NAXIS3', default=0, dtype=int) == 4:
            c2nc2 = 2

    return {'chop': chop, 'nod': nod, 'nmc': nmc, 'c2nc2': c2nc2}
