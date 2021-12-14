# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Query the Horizons database at JPL for object information"""

import sys

try:
    import astroquery.jplhorizons as aj
except ImportError:  # pragma: no cover
    print('\nAstroquery.jplhorizons not found.')
    print('Install with:')
    print('\tconda install -c astropy astroquery\n')
    sys.exit()

from astropy.time import Time


def asteroid_query(target, date, time):
    """
    Queries the JPL Horizons database for asteroid information.

    Asteroids require additional handling beyond what's needed
    for planets, so they need a separate function.

    Parameters
    ----------
    target : str
        Name of asteroid
    date : datetime.Date, str
        Date of observation (UTC)
    time : datetime.Time, str
        Time of observation (UTC)

    Returns
    -------
    params : dict
        Dictionary a of query results. Keys incude:
            - julian : Julian date of observation
            - albedo : Geometric albedo
            - gmag : Magnitude slope parameter
            - radius : Radius of the asteroid [km]
            - delta : Distance between asteroid and Earth [AU]
            - r : Distance between asteroid and Sun [AU]
            - phi : Phase angle (S-E-O) [deg]
    """

    configure_astroquery()

    obj, timepoint = horizons_object(target, date, time)

    elements = get_elements(obj)

    ephem = get_ephemerides(obj)

    returned_name = verify_object_name(obj)
    print(f'\nQuery: {date}, {time}, Target = {returned_name}')

    params = parse_elements(elements, ephem, timepoint)

    return params


def simple_query(target, date, time):
    """
    Query the JPL Horizons database to get the distance to the target.

    Parameters
    ----------
    target : str
        Name of the object
    date : str
        Date of observation (UTC)
    time : time
        Time of observation (UTC)

    Returns
    -------
    params : dict
        Dictionary of results. Keys included:
            - julian : Julian date of obsevation
            - delta : Distance between Earth and target [AU]
    """

    configure_astroquery()

    t = Time(f'{date}T{time}', format='isot', scale='utc')

    # Set up the query
    # To avoid all confusions, use the id number for each target
    codes = dict()
    codes['uranus'] = '799'
    codes['neptune'] = '899'
    codes['ganymede'] = '503'
    codes['callisto'] = '504'
    if target.lower() in codes:
        code = codes[target.lower()]
    else:
        print('Unknown target. Attempting anyways, but verify')
        print('output before using it.')
        code = target
    obj = aj.Horizons(id=code, location='399', epochs=t.jd)

    # Get the orbital elements and ephemerides
    ephem = obj.ephemerides()

    # Parse out the distance from Earth to the target, which is
    # in a table section whose start is denoted by $$SOE and end
    # is denoted by $$EOE
    delta = ephem['delta'].data[0]

    params = dict()
    params['julian'] = t.jd
    params['delta'] = delta

    return params


def configure_astroquery():
    """
    Configure query to JPL Hoizons.

    This sets up what quantities are returned:
    1 = Astrometric RA & DEC
    9 = Visual mag & surface brightness
    19 = Heliocentric range and range rate
    20 = Observer range and range rate
    24 = Sun-Target-Observer phase angle
    43 = Phase angle and bisector

    Returns
    -------
    None

    """
    # Set up the query for JPL Horizons
    aj.Conf.eph_quantities = '"1,9,19,20,23,24,43"'


def horizons_object(target, date, time):
    """
    Configure an object to query Horizons.

    Parameters
    ----------
    target : str
        Name of the astronomical object.
    date : str, timedate.date
        Date of observation.
    time : str, timdate.time
        Time of observation.

    Returns
    -------
    obj : astroquery.jplhorizons.core.HorizonsClass
        Horizons object ready for query.
    t : astropy.time.core.Time
        Time of observation in ISO format.

    """
    timestring = f'{date}T{time}'
    t = Time(timestring, format='isot', scale='utc')

    # Set up the query
    if 'ceres' in target.lower():
        target = 'Ceres'
    elif 'vesta' in target.lower():
        target = 'Vesta'
    obj = aj.Horizons(id=target, location='399', epochs=t.jd)
    return obj, t


def get_elements(obj):
    """
    Query Horizons for orbital elements.

    Parameters
    ----------
    obj : astroquery.jplhorizons.core.HorizonsClass
        Horizons object ready for query.

    Returns
    -------
    elements : str
        Raw output of query.

    """
    elements = obj.elements(get_raw_response=True)
    return elements


def get_ephemerides(obj):
    """
    Query Horizons for ephemerides.

    Parameters
    ----------
    obj : astroquery.jplhorizons.core.HorizonsClass
        Horizons object ready for query.

    Returns
    -------
    ephem : astropy.table.table.Table
        Ephemerides of object at observation.

    """
    ephem = obj.ephemerides()
    return ephem


def verify_object_name(obj):
    """
    Double-check that the Horizons query was for the correct object.

    Parameters
    ----------
    obj : astroquery.jplhorizons.core.HorizonsClass
        Horizons object ready for query.

    Returns
    -------
    returned_name : str
        Name of object pulled from query.

    """
    # Get the target name and print it for user verification
    aj.Conf.eph_quantities = '"1,9,19,20,23,24"'
    e = obj.ephemerides()
    returned_name = e['targetname'].data[0]
    return returned_name


def parse_property(elements, prop):
    """
    Parse orbital elements for physical property.

    Parameters
    ----------
    elements : str
        Raw output of query to Horizons for orbital elements.
    prop : str
        Name of the property to search for.

    Returns
    -------
    value : float
        Value of `prop` contained in `elements`.

    """
    lines = [ln for ln in elements.split('\n')
             if f' {prop.lower()}=' in ln.lower()]
    value = float(lines[0].split(f'{prop.upper()}=')[1].split()[0])
    return value


def parse_elements(elements, ephem, timepoint):
    """
    Parse out key fields from Horzions query results.

    Parameters
    ----------
    elements : str
        Raw output of query to Horizons for orbital elements.
    ephem : astropy.table.table.Table
        Output of query to Horizons for ephemerides.
    timepoint : astropy.time.core.Time
        Timepoint of observation.

    Returns
    -------
    params : dict
        Key values of interest needed for generating a
        thermal model of the object.

    """

    # Parse out albedo
    albedo = parse_property(elements, 'albedo')
    gmag = parse_property(elements, 'g')
    radius = parse_property(elements, 'rad')

    # r, delta, and phi are in a table section whose start
    # is denoted by $$SOE and end is denoted by $$EOE
    r = ephem['r'].data[0]
    delta = ephem['delta'].data[0]
    phi = ephem['alpha'].data[0]

    params = dict()
    params['julian'] = timepoint.jd
    params['albedo'] = albedo
    params['gmag'] = gmag
    params['radius'] = radius
    params['delta'] = delta
    params['r'] = r
    params['phi'] = phi

    return params
