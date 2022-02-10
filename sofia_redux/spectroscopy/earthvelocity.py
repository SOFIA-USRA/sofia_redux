# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import astropy.units as u
from astropy.coordinates import (
    SkyCoord, Angle, EarthLocation, Latitude, Longitude)
from astropy.coordinates.representation import CartesianRepresentation
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

__all__ = ['cartesian_lsr', 'cartesian_helio', 'earthvelocity']


def cartesian_lsr(definition='kinematic'):
    """
    Find the radial LSR velocity towards sky coordinates.

    The Local Standard of Rest allows for two different definitions:

        - 'kinematic' uses 20 km/s towards 18:00:00, +30:00:00 ('J1900')
        - 'dynamical' uses the IAU definition of 9, 12, and 7 km/s along
          the x, y, and z axis in Galactic cartesian coordinates.

    Parameters
    ----------
    definition : str, optional
        One of {'kinematic', 'dynamical'}

    Returns
    -------
    lsr_vel : CartesianRepresentation
        Cartesian representation of the LSR velocity in km / s
    """
    if definition == 'kinematic':
        lsr_vel = SkyCoord(18 * u.Unit('hourangle'), 30 * u.Unit('deg'),
                           frame='fk5', equinox='J1900')
        lsr_vel = lsr_vel.icrs.represent_as('unitspherical').to_cartesian()
        lsr_vel *= 20 * u.Unit('km/s')
    elif definition == 'dynamical':
        lsr_vel = CartesianRepresentation([9, 12, 7]) * u.Unit('km/s')
    else:
        raise ValueError("invalid LSR definition: %s" % definition)
    return lsr_vel


def cartesian_helio(time, center='barycentric', location=None):
    """
    Calculate the Cartesian velocity of the Sun.

    Parameters
    ----------
    time : str or float or astropy.time.Time
        Date and time of observation.  By default this is set to isot
        ('2000-01-01T00:00:00.000') in UTC.
    center : str, optional
        One of {'heliocentric', 'barycentric'}.  Compute the heliocentric
        or barycentric velocity which respectively define the Sun or the
        Sun's barycenter as the center of the Solar System.
    location : EarthLocation
        The location of the observation.

    Returns
    -------
    vhelio : CartesianRepresentation
        Cartesian representation of the velocity of the center of the
        Solar System.
    """
    vhelio = get_body_barycentric_posvel('earth', time)[1]
    if center == 'barycentric':
        pass
    elif center == 'heliocentric':
        vhelio -= get_body_barycentric_posvel('sun', time)[1]
    else:
        raise ValueError("unknown velocity correction: %s" % center)

    # Add rotational velocity if we have a location
    if isinstance(location, EarthLocation):
        vhelio += location.get_gcrs_posvel(time)[1]

    return vhelio


def parse_inputs(ra, dec, time=None, time_format='isot', time_scale='utc',
                 lat=None, lon=None, height=None,
                 equinox='J2000', frame='FK5'):
    """
    Parse arguments and parameters into SkyCoord.

    Parameters
    ----------
    ra : str or float or Angle or array_like
    dec : str or float or Angle or array_like
    time : str or float or Time or array_like, optional
    time_format : str, optional
    time_scale : str, optional
    lat : str or float or Latitude or array_like, optional
    lon : str or float or Longitude or array_like, optional
    height : str or float or Quantity, optional
    equinox : str, optional
    frame : str, optional

    Returns
    -------
    coordinates : SkyCoord
    """

    ra = ra.to('hourangle') if isinstance(ra, Angle) else \
        Angle(ra, unit='hourangle')
    dec = dec.to('deg') if isinstance(dec, Angle) \
        else Angle(dec, unit='deg')
    time = Time(time, format=time_format, scale=time_scale)

    if None not in [lat, lon, height]:
        lat = lat.to('deg') if isinstance(lat, Latitude) else \
            Latitude(lat, unit='deg')
        lon = lon.to('deg') if isinstance(lon, Longitude) else \
            Longitude(lon, unit='deg')
        height = height.to('m') if isinstance(height, u.Quantity) else \
            height * u.Unit('m')
        location = EarthLocation(lat=lat, lon=lon, height=height)
    else:
        location = None

    coords = SkyCoord(ra=ra, dec=dec, obstime=time, location=location,
                      equinox=str(equinox).strip().lower(),
                      frame=str(frame).strip().lower())
    return coords


def earthvelocity(ra, dec, time, equinox='J2000', frame='FK5',
                  time_format=None, time_scale=None,
                  lat=None, lon=None, height=0.0,
                  center='heliocentric', definition='kinematic'):
    """
    Provide velocities of the Earth towards a celestial position.

    The Local Standard of Rest allows for two different definitions
    defined by the `definition` keyword:

        - 'kinematic' uses 20 km/s towards 18:00:00, +30:00:00 ('J1900')
        - 'dynamical' uses the IAU definition of 9, 12, and 7 km/s along
          the x, y, and z axis in Galactic cartesian coordinates.

    Parameters
    ----------
    ra : str or float or Angle or array_like
        The right ascension[s] of the position[s].  Default unit is hourangle.
    dec : str or float or Angle or array_like
        The declination[s] of the position[s].  Default unit is degrees.
    time : str or float or Time or array_like
        Time of the observation.
    equinox : str, optional
        Coordinate frame equinox
    frame : str, optional
        Type of coordinate frame represented.
    time_format : str, optional
        Format of the input `time`.  See `astropy.time` for available
        formats.
    time_scale : str, optional
        Time scale of `time`.  See `astropy.time` for available scales.
    lat : str or float or Latitude or array_like, optional
        The latitude of the observation.  Default unit is degrees.
    lon : str or float or Longitude or array_like, optional
        The longitude of the observation.  Default unit is degrees.
    height : str or float or Quantity, optional
        The height of the observation.  Default unit is meters.
    center : str, optional
        Defines the center of the Solar System.  One of
        {'heliocentric', 'barycentric'}.
    definition : str, optional
        Defines the Local Standard of Rest (LSR) definition as described
        above.  One of {'kinematic', 'dynamical'}.

    Returns
    -------
    velocities : dict
        `vhelio` -> Velocity of the Earth wrt the Sun towards (ra, dec)
        'vsun' -> Velocity of the Solar motion wrt LSR towards (ra, dec)
        'vlsr' -> Net radial velocity of the Earth wrt LSR towards (ra, dec)

    Note if arrays were passed in, then arrays will be passed out.  However,
    all values will be astropy.units.Unit.Quantity with units in km /s.
    """
    sc = parse_inputs(ra, dec, time=time, lat=lat, lon=lon, height=height,
                      equinox=equinox, frame=frame, time_format=time_format,
                      time_scale=time_scale)

    icrs = sc.icrs.represent_as('unitspherical').to_cartesian()

    if lat is None or lon is None:
        # Cartesian earth/sun velocity from position velocities
        vhelio = cartesian_helio(sc.obstime, location=sc.location,
                                 center=center)
        vhelio = icrs.dot(vhelio).to('km/s')
    else:
        # more accurate version including relativistic effects
        try:
            vhelio = sc.radial_velocity_correction(kind=center).to('km/s')
        except ValueError as err:
            log.warning('Error encountered in radial velocity correction; '
                        'attempting offline calculation.')
            log.debug(f'Error from astropy: {str(err)}')
            log.warning('Correction value may not be accurate.')
            from astropy.utils.iers import iers
            with iers.conf.set_temp('auto_max_age', None):
                vhelio = sc.radial_velocity_correction(kind=center).to('km/s')

    vlsr = cartesian_lsr(definition=definition)
    vsun = icrs.dot(vlsr).to('km/s')
    return {'vhelio': vhelio, 'vsun': vsun, 'vlsr': vsun + vhelio}
