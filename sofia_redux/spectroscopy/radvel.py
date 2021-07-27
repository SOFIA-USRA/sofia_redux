# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import astropy.constants as const
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.units import imperial

from sofia_redux.spectroscopy.earthvelocity import earthvelocity

__all__ = ['radvel']


def radvel(header, equinox='J2000'):
    """
    Calculate the expected extrinsic radial velocity wavelength shift.

    The procedure is:
        1. Read date and RA/Dec from header.
        2. Compute barycentric and LSR velocity along line of sight.
        3. Return velocities / speed of light

    Parameters
    ----------
    header : fits.Header
        FITS header of the observation, including DATE-OBS, TELRA, and
        TELDEC
    equinox : str, optional
        Equinox of TELRA, TELDEC coordinates

    Returns
    -------
    float, float
        delta lambda / lamdba (positive = shift toward blue).  First
        value is barycentric shift, second value is an additional
        shift to correct to the local standard of rest (LSR)
    """
    if not isinstance(header, fits.Header):
        log.error("Invalid header")
        return
    for required_key in ['DATE-OBS', 'TELRA', 'TELDEC', 'LAT_STA',
                         'LON_STA', 'ALTI_STA']:
        if required_key not in header:
            log.error("%s not found in header" % required_key)
            return
    try:
        time = Time(header['DATE-OBS'])
    except (ValueError, AttributeError, TypeError):
        log.error("Unable to convert %s to Julian Date" % header['DATE-OBS'])
        return

    ra = header['TELRA'] * u.hourangle
    dec = header['TELDEC'] * u.deg
    height = header['ALTI_STA'] * imperial.ft
    lat = header['LAT_STA'] * u.deg
    lon = header['LON_STA'] * u.deg

    vel = earthvelocity(ra, dec, time, lat=lat, lon=lon, height=height,
                        center='barycentric', equinox=equinox)
    vhelio = vel['vhelio']
    vsun = vel['vsun']

    speed_of_light = const.c.to(vhelio.unit)
    dw_bary = vhelio / speed_of_light
    dw_lsr = vsun / speed_of_light

    log.debug("Julian Date %s" % time.jd)
    log.debug("Velocity of Earth wrt the Sun is %s" % vhelio)
    log.debug("Velocity of solar motion wrt LSR is %s" % vsun)
    log.debug("Net radial velocity of Earth wrt LSR is %s" % vel['vlsr'])
    log.debug("Barycentric delta lambda over lambda is "
              "%s toward blue" % dw_bary)
    log.debug("Additional shift to LSR is %s toward blue" % dw_lsr)
    log.debug("Net shift is %s toward blue" % (vel['vlsr'] / speed_of_light))
    return dw_bary.value, dw_lsr.value
