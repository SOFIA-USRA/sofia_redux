# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy.testing as npt
from sofia_redux.calibration.standard_model import horizons
import astroquery.jplhorizons as jpl
from astropy.time import Time
from astropy.table import Table, Column


@pytest.fixture
def elements():
    return (
        '*********************************************************************'
        '**********\nJPL/HORIZONS                       1 Ceres               '
        '  2019-Oct-10 20:13:04\nRec #:       1 (+COV) Soln.date: 2019-Jun-05_'
        '16:22:15   # obs: 1002 (1995-2019)\n \nIAU76/J2000 helio. ecliptic os'
        'c. elements (au, days, deg., period=Julian yrs):\n \n  EPOCH=  245403'
        '3.5 ! 2006-Oct-25.00 (TDB)         Residual RMS= .22345\n   EC= .0798'
        '7906346370539  QR= 2.544709153978707   TP= 2453193.6614275328\n   OM='
        ' 80.40846590069125   W=  73.1893463033331    IN= 10.58671483589909\n '
        '  A= 2.76562466186023     MA= 179.9741090118086   ADIST= 2.9865401697'
        '41752\n   PER= 4.59937            N= .214296068           ANGMOM= .02'
        '8515965\n   DAN= 2.68593            DDN= 2.81296            L= 153.32'
        '35262\n   B= 10.1294158           MOID= 1.57962           TP= 2004-Ju'
        'l-07.1614275328\n \nAsteroid physical parameters (km, seconds, rotati'
        'onal period in hours):\n   GM= 62.6284             RAD= 469.7        '
        '      ROTPER= 9.07417\n   H= 3.34                 G= .120            '
        '     B-V= .713\n                           ALBEDO= .090            ST'
        'YP= C\n \nASTEROID comments: \n1: soln ref.= JPL#46, OCC=0           '
        'radar(60 delay, 0 Dop.)\n2: source=ORB\n*****************************'
        '**************************************************\n \r\n \r\n*******'
        '*********************************************************************'
        '***\nEphemeris / WWW_USER Thu Oct 10 20:13:04 2019 Pasadena, USA     '
        ' / Horizons    \n****************************************************'
        '***************************\nTarget body name: 1 Ceres               '
        '          {source: JPL#46}\nCenter body name: Earth (399)            '
        '         {source: DE431}\nCenter-site name: BODY CENTER\n************'
        '*******************************************************************\n'
        'Start time      : A.D. 2017-Sep-28 05:51:00.0000 TDB\nStop  time     '
        ' : A.D. 2017-Sep-28 05:51:00.5000 TDB\nStep-size       : 0 steps\n***'
        '*********************************************************************'
        '*******\nCenter geodetic : 0.00000000,0.00000000,0.0000000 {E-lon(deg'
        '),Lat(deg),Alt(km)}\nCenter cylindric: 0.00000000,0.00000000,0.000000'
        '0 {E-lon(deg),Dxy(km),Dz(km)}\nCenter radii    : 6378.1 x 6378.1 x 63'
        '56.8 km     {Equator, meridian, pole}    \nKeplerian GM    : 8.887692'
        '4451256332E-10 au^3/d^2\nSmall perturbers: Yes                       '
        '      {source: SB431-N16}\nOutput units    : AU-D, deg, Julian Day Nu'
        'mber (Tp)                            \nOutput type     : GEOMETRIC os'
        'culating elements\nOutput format   : 10\nReference frame : ICRF/J2000'
        '.0                                                 \nCoordinate systm'
        ': Ecliptic and Mean Equinox of Reference Epoch                 \n****'
        '*********************************************************************'
        '******\nInitial IAU76/J2000 heliocentric ecliptic osculating elements'
        ' (au, days, deg.):\n  EPOCH=  2454033.5 ! 2006-Oct-25.00 (TDB)       '
        '  Residual RMS= .22345        \n   EC= .07987906346370539  QR= 2.5447'
        '09153978707   TP= 2453193.6614275328      \n   OM= 80.40846590069125 '
        '  W=  73.1893463033331    IN= 10.58671483589909       \n  Equivalent '
        'ICRF heliocentric equatorial cartesian coordinates (au, au/d):\n   X='
        ' 2.626536679271237E+00  Y=-1.003038764756320E+00  Z=-1.00729359115881'
        '5E+00\n  VX= 4.202952273775981E-03 VY= 8.054172339518143E-03 VZ= 2.93'
        '8175156440994E-03\nAsteroid physical parameters (km, seconds, rotatio'
        'nal period in hours):        \n   GM= 62.6284             RAD= 469.7 '
        '             ROTPER= 9.07417             \n   H= 3.34                '
        ' G= .120                 B-V= .713                   \n              '
        '             ALBEDO= .090            STYP= C                     \n**'
        '*********************************************************************'
        '********\n            JDTDB,            Calendar Date (TDB),         '
        '            EC,                     QR,                     IN,      '
        '               OM,                      W,                     Tp,   '
        '                   N,                     MA,                     TA,'
        '                      A,                     AD,                     '
        'PR,\n****************************************************************'
        '*********************************************************************'
        '*********************************************************************'
        '*********************************************************************'
        '*******************************************************************\n'
        '$$SOE\n2458024.743750000, A.D. 2017-Sep-28 05:51:00.0000,  1.26269608'
        '3543516E+06,  2.384401426988268E+00,  8.988039361713625E+00,  9.88870'
        '1959962594E+01,  5.750884338585384E+01,  2.458101644457025E+06,  6.58'
        '2574997226868E+05, -5.062046713348336E+07,  3.250199172438903E+02, -1'
        '.888343003836598E-06,  6.684586453809735E+91,  1.157407291666667E+95,'
        '\n$$EOE\n************************************************************'
        '*********************************************************************'
        '*********************************************************************'
        '*********************************************************************'
        '*********************************************************************'
        '**\nCoordinate system description:\n\n  Ecliptic and Mean Equinox of '
        'Reference Epoch\n\n    Reference epoch: J2000.0\n    XY-plane: plane '
        'of the Earth\'s orbit at the reference epoch\n              Note: obl'
        'iquity of 84381.448 arcseconds wrt ICRF equator (IAU76)\n    X-axis  '
        ': out along ascending node of instantaneous plane of the Earth\'s\n  '
        '            orbit and the Earth\'s mean equator at the reference epoc'
        'h\n    Z-axis  : perpendicular to the xy-plane in the directional (+ '
        'or -) sense\n              of Earth\'s north pole at the reference ep'
        'och.\n\n  Symbol meaning [1 au= 149597870.700 km, 1 day= 86400.0 s]:'
        '\n\n    JDTDB    Julian Day Number, Barycentric Dynamical Time\n     '
        ' EC     Eccentricity, e                                              '
        '     \n      QR     Periapsis distance, q (au)                       '
        '                 \n      IN     Inclination w.r.t XY-plane, i (degree'
        's)                           \n      OM     Longitude of Ascending No'
        'de, OMEGA, (degrees)                     \n      W      Argument of P'
        'erifocus, w (degrees)                                \n      Tp     T'
        'ime of periapsis (Julian Day Number)                             \n  '
        '    N      Mean motion, n (degrees/day)                              '
        '        \n      MA     Mean anomaly, M (degrees)                     '
        '                    \n      TA     True anomaly, nu (degrees)        '
        '                                \n      A      Semi-major axis, a (au'
        ')                                           \n      AD     Apoapsis d'
        'istance (au)                                            \n      PR   '
        '  Sidereal orbit period (day)                                       '
        '\n\nGeometric states/elements have no aberrations applied.\n\n Comput'
        'ations by ...\n     Solar System Dynamics Group, Horizons On-Line Eph'
        'emeris System\n     4800 Oak Grove Drive, Jet Propulsion Laboratory\n'
        '     Pasadena, CA  91109   USA\n     Information: http://ssd.jpl.nasa'
        '.gov/\n     Connect    : telnet://ssd.jpl.nasa.gov:6775  (via browser'
        ')\n                  http://ssd.jpl.nasa.gov/?horizons\n             '
        '     telnet ssd.jpl.nasa.gov 6775    (via command-line)\n     Author '
        '    : Jon.D.Giorgini@jpl.nasa.gov\n**********************************'
        '*********************************************\n\n!$$SOF\nTABLE_TYPE ='
        ' ELEMENTS\nMAKE_EPHEM = YES\nOUT_UNITS = AU-D\nCOMMAND = "Ceres;"\nCE'
        'NTER = \'399\'\nCSV_FORMAT = YES\nELEM_LABELS = YES\nOBJ_DATA = YES\n'
        'REF_SYSTEM = J2000\nREF_PLANE = ECLIPTIC\nTP_TYPE = ABSOLUTE\nTLIST ='
        ' 2458024.74375\n')


@pytest.fixture
def timepoint():
    return Time('2017-09-28T05:51:00')


@pytest.fixture
def obj(timepoint):
    target = 'Ceres'
    return jpl.Horizons(id=target, epochs=timepoint.jd, location='399')


@pytest.fixture
def ephem():
    t = Table()
    t['r'] = Column([2.628947608794], unit='AU')
    t['delta'] = Column([2.91004694372708], unit='AU')
    t['alpha'] = Column([20.0253], unit='deg')
    return t


@pytest.mark.parametrize('target', ['Ceres', 'Vesta'])
def test_asteroid_query(timepoint, capsys, target):
    date = timepoint.datetime.date()
    time = timepoint.datetime.time()
    param = horizons.asteroid_query(target, date, time)
    captured = capsys.readouterr()
    assert target in captured.out
    assert not captured.err
    assert len(param) == 7


@pytest.mark.parametrize('target,delta,flag', [('Neptune', 29.0221851, False),
                                               ('Callisto', 6.37027832, False),
                                               ('Ceres', 2.91004694, True)])
def test_simple_query(timepoint, target, delta, capsys, flag):
    params = horizons.simple_query(target, timepoint.datetime.date(),
                                   timepoint.datetime.time())
    captured = capsys.readouterr()
    npt.assert_approx_equal(params['delta'], delta, significant=5)
    if flag:
        assert 'Unknown target' in captured.out
    else:
        assert not captured.err
    assert len(params) == 2


def test_configure_astroquery():
    horizons.configure_astroquery()
    assert '43' in jpl.Conf.eph_quantities
    assert len(jpl.Conf.eph_quantities.split(',')) == 7


def test_get_elements(obj, elements):
    elem = horizons.get_elements(obj)
    tags = ['JPL/HORIZONS', 'Start time', 'Center geodetic',
            'EPOCH=', 'GM=', 'Computations by ...']
    for tag in tags:
        assert tag in elem
    assert isinstance(elem, str)


@pytest.mark.parametrize('field', ['r', 'delta', 'alpha'])
def test_get_ephemerides(obj, ephem, field):
    e = horizons.get_ephemerides(obj)
    assert isinstance(e, Table)
    npt.assert_approx_equal(e[field], ephem[field])


def test_verify_object_name(obj):
    name = horizons.verify_object_name(obj)
    assert name in ['1 Ceres', '1 Ceres (A801 AA)']


@pytest.mark.parametrize('prop,expected', [('albedo', 0.09), ('g', 0.12),
                                           ('rad', 469.7)])
def test_parse_property(elements, prop, expected):
    value = horizons.parse_property(elements, prop)
    npt.assert_almost_equal(value, expected, decimal=2)


@pytest.mark.parametrize('key,value', [('julian', 2458024.74375),
                                       ('albedo', 0.09),
                                       ('gmag', 0.12),
                                       ('radius', 469.7),
                                       ('delta', 2.91004),
                                       ('r', 2.62894),
                                       ('phi', 20.0253)])
def test_parse_elements(elements, ephem, timepoint, key, value):
    param = horizons.parse_elements(elements, ephem, timepoint)
    npt.assert_approx_equal(param[key], value, significant=4)
