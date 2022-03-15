# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.time import Time
from astropy.coordinates import FK4, FK5, SkyCoord
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
from astropy.stats import gaussian_sigma_to_fwhm
from copy import deepcopy
import numpy as np
import pytest

from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import utils


def test_constants():
    assert isinstance(utils.UNKNOWN_INT_VALUE, int)
    assert isinstance(utils.UNKNOWN_STRING_VALUE, str)
    assert isinstance(utils.UNKNOWN_FLOAT_VALUE, float)
    assert isinstance(utils.UNKNOWN_BOOL_VALUE, bool)


def test_get_string():
    get_string = utils.get_string
    assert get_string('abc') == 'abc'
    assert get_string(utils.UNKNOWN_STRING_VALUE, default=None) is None

    class BadString(object):
        def __str__(self):
            raise ValueError("nope")

    assert get_string(BadString(), default=None) is None
    assert get_string(1) == '1'


def test_get_bool():
    get_bool = utils.get_bool
    assert get_bool(True)
    assert not get_bool(False)
    assert get_bool(None, default=None) is None
    assert get_bool('', default=None) is None
    for false_value in [0, 'N', 'F']:
        assert not get_bool(false_value)
    assert get_bool(1)


def test_get_int():
    get_int = utils.get_int
    assert get_int(1) == 1
    assert get_int(2.5) == 2
    assert get_int(None, default=0) == 0
    assert get_int('abc', default=0) == 0
    assert get_int(utils.UNKNOWN_INT_VALUE, default=0) == 0


def test_get_float():
    get_float = utils.get_float
    x = get_float(1)
    assert x == 1 and isinstance(x, float)
    assert get_float(None, default=0.0) == 0
    assert get_float('abc', default=0.0) == 0
    assert np.isnan(get_float(utils.UNKNOWN_FLOAT_VALUE, default=np.nan))


def test_get_range():
    get_range = utils.get_range
    assert get_range(None, default=None) is None
    test_range = Range(1, 2)
    new_range = get_range(test_range)
    assert new_range == test_range and new_range is not test_range
    new_range = get_range((1, 3))
    assert new_range.min == 1 and new_range.max == 3
    assert get_range(['a', 'b'], default=None) is None
    new_range = get_range('1:2')
    assert new_range.min == 1 and new_range.max == 2
    new_range = get_range('1-2', is_positive=True)
    assert new_range.min == 1 and new_range.max == 2
    assert get_range('x-3', default=None) is None


def test_get_list():
    get_list = utils.get_list
    result = get_list(None)
    assert isinstance(result, list) and len(result) == 0
    assert get_list(result) is result
    result = get_list('a string')
    assert isinstance(result, list) and len(result) == 1


def test_get_string_list():
    get_string_list = utils.get_string_list
    assert get_string_list(None, default=1) == 1
    result = get_string_list('a,b,c,d,e', delimiter=',')
    assert len(result) == 5 and result[3] == 'd'


def test_get_float_list():
    get_float_list = utils.get_float_list
    assert get_float_list(None, default=1) == 1
    result = get_float_list('1,2,3,4', delimiter=',')
    assert np.allclose(np.arange(4) + 1, result)
    assert isinstance(result[0], float)


def test_get_int_list():
    get_int_list = utils.get_int_list
    assert get_int_list(None, default=1) == 1
    result = get_int_list('1,2,3,4', delimiter=',')
    assert np.allclose(np.arange(4) + 1, result)
    assert isinstance(result[0], int)

    result = get_int_list('1:3,5,6')
    assert np.allclose(result, [1, 2, 3, 5, 6])
    result = get_int_list('-1:3, 5, 6', is_positive=False)
    assert np.allclose(result, [-1, 0, 1, 2, 3, 5, 6])


def test_to_int_list():
    to_int_list = utils.to_int_list
    assert np.allclose(to_int_list(np.arange(5) / 2), np.arange(5) // 2)
    assert np.allclose(to_int_list(['1-5', 6]), [1, 2, 3, 4, 5, 6])
    assert np.allclose(to_int_list(['-1:2'], is_positive=False), [-1, 0, 1, 2])
    assert np.allclose(to_int_list(['1']), [1])


def test_parse_time():
    ha = units.Unit('hourangle')
    h = units.Unit('hour')
    parse_time = utils.parse_time
    assert parse_time('1h30m0s', angle=False) == 1.5 * h
    assert parse_time('01:30:00', angle=True) == 1.5 * ha
    with pytest.raises(ValueError) as err:
        parse_time('abc')
    assert 'Cannot parse abc' in str(err.value)
    with pytest.raises(ValueError) as err:
        parse_time('a:b:c')
    assert 'Cannot parse a:b:c' in str(err.value)


def test_parse_angle():
    deg = units.Unit('degree')
    parse_angle = utils.parse_angle
    assert parse_angle('1:30:0') == 1.5 * deg
    assert parse_angle('-1:30:0') == -1.5 * deg
    with pytest.raises(ValueError) as err:
        parse_angle('a:b:c')
    assert 'Cannot parse a:b:c' in str(err.value)


def test_get_dms_angle():
    get_dms_angle = utils.get_dms_angle
    deg = units.Unit('degree')
    half = 30 * units.Unit('arcminute')

    with pytest.raises(ValueError) as err:
        get_dms_angle(1, default='a')
    assert "Cannot convert default" in str(err.value)

    with pytest.raises(ValueError) as err:
        get_dms_angle(1, default=1 * units.Unit('meter'))
    assert "Cannot convert default" in str(err.value)

    result = get_dms_angle(None, default=1.0)
    assert result.unit == deg
    assert result.value == 1
    result = get_dms_angle(None, default=half)
    assert result.unit == deg
    assert result.value == 0.5

    result = get_dms_angle(1 * units.Unit('meter'), default=half)
    assert result == half

    result = get_dms_angle(utils.UNKNOWN_FLOAT_VALUE, default=half)
    assert result == half

    result = get_dms_angle(1.0, default=half)
    assert result == 1 * deg

    result = get_dms_angle('1:30:00', default=half)
    assert result == 1.5 * deg

    result = get_dms_angle('abc', default=half)
    assert result == half


def test_get_hms_time():
    get_hms_time = utils.get_hms_time
    h = units.Unit('hour')
    ha = units.Unit('hourangle')
    half = 30 * units.Unit('minute')
    half_a = 0.5 * units.Unit('hourangle')

    with pytest.raises(ValueError) as err:
        get_hms_time(1, default='a')
    assert "Cannot convert default" in str(err.value)

    with pytest.raises(ValueError) as err:
        get_hms_time(1, default=1 * units.Unit('meter'))
    assert "Cannot convert default" in str(err.value)

    result = get_hms_time(None, default=1.0, angle=False)
    assert result.unit == h
    assert result.value == 1
    result = get_hms_time(None, default=half_a, angle=True)
    assert result.unit == ha
    assert result.value == 0.5

    result = get_hms_time(1 * units.Unit('meter'), default=half)
    assert result == half

    result = get_hms_time(utils.UNKNOWN_FLOAT_VALUE, default=half)
    assert result == half

    result = get_hms_time(1.0, default=half)
    assert result == 1 * h

    result = get_hms_time('1:30:00', default=half)
    assert result == 1.5 * h

    result = get_hms_time('abc', default=half)
    assert result == half


def test_get_sign():
    get_sign = utils.get_sign
    assert get_sign(0) == 0
    assert get_sign(100) == 1
    assert get_sign(-np.inf) == -1
    assert get_sign(np.nan) == 0

    for value in ['+', 'pos', 'positive', 'plus']:
        assert get_sign(value) == 1
    for value in ['-', 'neg', 'negative', 'minus']:
        assert get_sign(value) == -1
    for value in ['*', 'any']:
        assert get_sign(value) == 0

    assert get_sign('-1') == -1
    assert get_sign('20') == 1
    assert get_sign('0') == 0
    assert get_sign(None, default=1) == 1
    assert get_sign('abc', default=-1) == -1


def test_get_epoch():
    get_epoch = utils.get_epoch
    epoch = get_epoch('J2000')
    assert epoch.equinox.value.startswith('J2000')
    assert epoch.equinox.mjd == 51544.5
    assert isinstance(epoch, FK5)

    epoch = get_epoch(Time('1970-01-01'))
    assert isinstance(epoch, FK4)
    epoch = get_epoch(Time('2020-01-01'))
    assert isinstance(epoch, FK5)

    epoch2 = get_epoch(epoch)
    assert epoch2 is epoch

    epoch = get_epoch('2020')
    assert isinstance(epoch, FK5)
    assert epoch.equinox.value.startswith('J2020')
    epoch = get_epoch('1900')
    assert isinstance(epoch, FK4)
    assert epoch.equinox.value.startswith('B1900')


def test_valid_value():
    valid_value = utils.valid_value
    assert valid_value(True)
    assert valid_value(False)
    assert valid_value(1)
    assert not valid_value(utils.UNKNOWN_INT_VALUE)
    assert valid_value(1.0)
    assert not valid_value(utils.UNKNOWN_FLOAT_VALUE)
    assert not valid_value(np.nan)
    assert valid_value('foo')
    assert not valid_value(utils.UNKNOWN_STRING_VALUE)
    assert not valid_value(None)


def test_robust_mean():
    values = np.linspace(0, 100, 101)
    robust_mean = utils.robust_mean
    assert robust_mean(values) == 50
    jy = units.Unit('Jansky')
    values = values * jy
    assert robust_mean(values) == 50 * jy
    assert robust_mean(values, tails=0.1) == 50 * jy
    assert np.isnan(robust_mean(values, tails=0.51))
    values[51:] = np.nan
    assert robust_mean(values) == 25 * jy


def test_robust_sigma_clip():
    np.random.seed(1)
    values = np.random.random((32, 32)) - 0.5
    weight = 1 / (np.std(values) ** 2)
    weights = np.full(values.shape, weight)
    mask = np.full(values.shape, True)

    # Add some outliers at known locations
    outlier_inds = tuple((np.random.random((2, 10)) * 32).astype(int))

    positive = values[outlier_inds] > 0
    outliers = 100 + (np.random.random(10) * 10)
    outliers[~positive] *= -1
    values[outlier_inds] = outliers
    n_outliers = outliers.size

    # Test standard performance with all inputs
    out_mask = utils.robust_sigma_clip_mask(values, weights=weights,
                                            mask=mask, sigma=5.0, verbose=True,
                                            max_iterations=5)

    assert np.sum(~out_mask) == n_outliers
    assert not out_mask[outlier_inds].any()
    assert not np.allclose(mask, out_mask)

    # Test units work as expected
    s = units.Unit('second')
    out_mask = utils.robust_sigma_clip_mask(values * s, weights=weights * s,
                                            mask=mask, sigma=5.0, verbose=True,
                                            max_iterations=5)
    expected_mask = out_mask.copy()

    assert np.sum(~out_mask) == n_outliers
    assert not out_mask[outlier_inds].any()

    # Test no mask
    out_mask = utils.robust_sigma_clip_mask(values, weights=weights,
                                            sigma=5.0, max_iterations=5)
    assert np.sum(~out_mask) == n_outliers
    assert not out_mask[outlier_inds].any()

    # Test no weights
    out_mask = utils.robust_sigma_clip_mask(values, sigma=5.0,
                                            max_iterations=5)
    assert np.sum(~out_mask) == n_outliers
    assert not out_mask[outlier_inds].any()

    # Test no valid data iteration break
    out_mask = utils.robust_sigma_clip_mask(values * np.nan)
    assert not out_mask.any()

    # Test mask works as expected
    valid = np.nonzero(expected_mask)
    first_valid = valid[0][0], valid[1][0]
    mask[first_valid] = False
    out_mask = utils.robust_sigma_clip_mask(values, mask=mask)
    assert (out_mask.size - out_mask.sum()) == (n_outliers + 1)
    assert not out_mask[first_valid]
    assert not out_mask[outlier_inds].any()

    # Test weights work as expected
    weights[expected_mask] = 0
    out_mask = utils.robust_sigma_clip_mask(values, weights=weights)
    # Nothing should be blanked since median and variance were calculated
    # using the outliers.
    assert out_mask.all()


def test_roundup_ratio():
    assert utils.roundup_ratio(5, 2) == 3
    assert utils.roundup_ratio(6, 2) == 3
    assert utils.roundup_ratio(2.5, 2) == 1

    assert np.allclose(utils.roundup_ratio(np.arange(10), 2),
                       [0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

    assert np.allclose(utils.roundup_ratio(10, np.arange(10) + 1),
                       [10, 5, 4, 3, 2, 2, 2, 2, 2, 1])


def test_rotate():
    positions = np.asarray([[1.0, 1.0], [-1, 1], [-1, -1], [1, -1]])
    deg45 = 45 * units.Unit('degree')
    d = np.sqrt(2)

    # Test single angle rotation
    rad45 = deg45.to('radian').decompose().value
    rotated = utils.rotate(positions, deg45)
    assert np.allclose(rotated, [[0, d], [-d, 0], [0, -d], [d, 0]])
    rotated = utils.rotate(positions, rad45)
    assert np.allclose(rotated, [[0, d], [-d, 0], [0, -d], [d, 0]])

    # Test multiple angle rotation
    angle = [45, -45, -135, 135] * units.Unit('degree')
    rotated = utils.rotate(positions, angle)
    assert rotated.shape == (4, 2)
    assert np.allclose(rotated[:, 0], 0)
    assert np.allclose(rotated[:, 1], d)


def test_log2round():
    assert utils.log2round(1024) == 10
    result = utils.log2round(np.arange(-5, 5) + 1024)
    assert np.allclose(result, 10)
    assert result.size == 10
    half = int(2 ** 9.5)
    assert utils.log2round(half) == 9
    assert utils.log2round(half + 1) == 10


def test_pow2round():
    assert utils.pow2round(1024) == 1024
    half = int(2 ** 9.5)
    assert utils.pow2round(half) == 512
    assert utils.pow2round(half + 1) == 1024
    result = utils.pow2round(np.arange(half - 9, half + 11))
    assert np.allclose(result[:10], 512)
    assert np.allclose(result[10:], 1024)


def test_pow2floor():
    assert utils.pow2floor(1024) == 1024
    assert utils.pow2floor(1023) == 512
    assert utils.pow2floor(1025) == 1024
    result = utils.pow2floor(np.arange(1014, 1034))
    assert np.allclose(result[:10], 512)
    assert np.allclose(result[10:], 1024)


def test_pow2ceil():
    assert utils.pow2ceil(1024) == 1024
    assert utils.pow2ceil(1023) == 1024
    assert utils.pow2ceil(1025) == 2048
    result = utils.pow2ceil(np.arange(1014, 1034))
    assert np.allclose(result[:11], 1024)
    assert np.allclose(result[11:], 2048)


def test_skycoord_insert_blanks():
    ra = np.arange(10) * units.Unit('degree')
    dec = np.arange(10) * units.Unit('degree')
    coords = SkyCoord(ra=ra, dec=dec)
    result = utils.skycoord_insert_blanks(coords, np.full(3, 8))
    assert np.allclose(result[8:11].ra, 0)
    assert np.allclose(result[8:11].dec, 0)
    original_inds = np.arange(10)
    original_inds[8:] += 3
    original = result[original_inds]
    assert np.allclose(original.ra, coords.ra)
    assert np.allclose(original.dec, coords.dec)


def test_dict_intersection():
    dict1 = {'level1': {
        'level2': {
            'value1': 1,
            'value2': 2,
            'value3': 3}}}

    dict2 = {'level1': {
        'level2': {
            'value1': 1,
            'value2': 2,
            'value3': 4}}}

    result = utils.dict_intersection(dict1, dict2)
    assert 'level1' in result and 'level2' in result['level1']
    assert result['level1']['level2']['value1'] == 1
    assert result['level1']['level2']['value2'] == 2
    assert 'value3' not in result['level1']['level2']


def test_dict_difference():
    dict1 = {'level1': {
        'level2': {
            'value1': 1,
            'value2': 2,
            'value3': 3}}}

    dict2 = {'level1': {
        'level2': {
            'value1': 1,
            'value3': 4}}}

    result = utils.dict_difference(dict1, dict2)
    assert 'level1' in result and 'level2' in result['level1']
    assert result['level1']['level2']['value2'] == 2
    assert result['level1']['level2']['value3'] == 3
    assert 'value1' not in result['level1']['level2']


def test_to_header_quantity():
    unknown = utils.UNKNOWN_FLOAT_VALUE
    ud = units.dimensionless_unscaled
    for value in [unknown, np.nan, np.inf, 'NaN', None]:
        assert utils.to_header_quantity(value) == unknown
        assert utils.to_header_quantity(value, unit=3 * ud) == unknown
        if value != 'NaN' and value is not None:
            assert np.isclose(utils.to_header_quantity(value, keep=True),
                              value, equal_nan=True)
            assert np.isclose(utils.to_header_quantity(
                value, unit=3 * ud, keep=True), value, equal_nan=True)

    with pytest.raises(ValueError) as err:
        _ = utils.to_header_quantity(complex(1, 1))
    assert 'Value must be' in str(err.value)

    assert utils.to_header_quantity(1, 'arcsec') == 1 * units.Unit('arcsec')
    d = utils.to_header_quantity(1 * units.Unit('arcmin'), unit='arcsec')
    assert d.value == 60 and d.unit == 'arcsec'

    d = utils.to_header_quantity(2 * ud)
    assert isinstance(d, float) and d == 2

    d = utils.to_header_quantity(1 * units.Unit('arcsec'))
    assert d.unit == 'arcsec' and d.value == 1

    d = utils.to_header_quantity(1 * units.Unit('arcmin'),
                                 unit=2 * units.Unit('arcsec'))
    assert d.unit == 'arcsec' and d.value == 120

    d = utils.to_header_quantity(1 * units.Unit('arcsec'), unit=ud)
    assert d.unit == 'arcsec' and d.value == 1

    d = utils.to_header_quantity(np.nan * units.Unit('arcsec'))
    assert d == -9999 * units.Unit('arcsec')
    d = utils.to_header_quantity(np.nan * units.Unit('arcsec'), unit=ud)
    assert d == -9999 * units.Unit('arcsec')
    d = utils.to_header_quantity(np.nan, unit='arcsec')
    assert d == -9999 * units.Unit('arcsec')


def test_to_header_float():
    unknown = utils.UNKNOWN_FLOAT_VALUE
    assert np.allclose(utils.to_header_float(None), unknown)
    assert np.allclose(utils.to_header_float(np.nan), unknown)
    test_value = 1 * units.Unit('degree')
    value = utils.to_header_float(test_value)
    assert isinstance(value, float) and value == 1
    value = utils.to_header_float(test_value, unit='arcsec')
    assert value == 3600
    assert utils.to_header_float(5) == 5


def test_convolve_beams():
    # Check standard convolution
    arcsec = units.Unit('arcsec')
    beam1 = Gaussian2D(x_stddev=4 * arcsec, y_stddev=3 * arcsec)
    beam2 = Gaussian2D(x_stddev=3 * arcsec, y_stddev=4 * arcsec)
    beam = utils.combine_beams(beam1, beam2)
    assert beam.x_stddev == 5 * arcsec
    assert beam.y_stddev == 5 * arcsec
    assert beam.theta == 0

    # Check no convolution
    beam = utils.combine_beams(beam1, None)
    assert beam1.x_stddev == beam.x_stddev
    assert beam1.y_stddev == beam.y_stddev
    assert beam.theta == 0

    # Check theta
    beam1.theta = np.pi / 4
    beam2.theta = -np.pi / 4
    beam = utils.combine_beams(beam1, beam2)
    assert np.isclose(beam.theta.value, np.pi / 8)
    assert not np.isclose(beam.x_stddev.value, beam.y_stddev.value)
    assert beam.x_stddev.value > beam1.x_stddev.value
    assert beam.y_stddev.value > beam1.y_stddev.value

    # Check no units
    beam1 = Gaussian2D(x_stddev=4, y_stddev=3)
    beam2 = Gaussian2D(x_stddev=3, y_stddev=4)
    beam = utils.combine_beams(beam1, beam2)
    assert beam.x_stddev == 5 and beam.y_stddev == 5

    # Check bad values
    beam = utils.combine_beams(beam1, beam2, deconvolve=True)
    assert beam.x_stddev.value > 0
    assert beam.y_stddev.value == 0

    # Check deconvolution
    beam1 = Gaussian2D(x_stddev=5, y_stddev=5)
    beam2 = Gaussian2D(x_stddev=4, y_stddev=4)
    beam = utils.combine_beams(beam1, beam2, deconvolve=True)
    assert beam.x_stddev.value == 3
    assert beam.y_stddev.value == 3
    assert beam.theta.value == 0

    beam = utils.combine_beams(beam2, beam1, deconvolve=True)
    assert beam.x_stddev.value == 0
    assert beam.y_stddev.value == 0


def test_convolve_beam():
    arcsec = units.Unit('arcsec')
    beam1 = Gaussian2D(x_stddev=4 * arcsec, y_stddev=3 * arcsec)
    beam2 = Gaussian2D(x_stddev=3 * arcsec, y_stddev=4 * arcsec)
    beam = utils.convolve_beam(beam1, beam2)
    assert beam.x_stddev == 5 * arcsec
    assert beam.y_stddev == 5 * arcsec
    assert beam.theta == 0


def test_deconvolve_beam():
    beam1 = Gaussian2D(x_stddev=5, y_stddev=5)
    beam2 = Gaussian2D(x_stddev=4, y_stddev=4)
    beam = utils.deconvolve_beam(beam1, beam2)
    assert beam.x_stddev.value == 3
    assert beam.y_stddev.value == 3
    assert beam.theta.value == 0


def test_encompass_beam():

    class Beam(object):
        def __init__(self, x_stddev, y_stddev, theta=0.0):
            self.x_stddev = x_stddev
            self.y_stddev = y_stddev
            self.theta = theta

        def copy(self):
            return deepcopy(self)

    # Completely encompass
    arcsec = units.Unit('arcsec')
    degree = units.Unit('degree')
    beam1 = Gaussian2D(x_stddev=5 * arcsec, y_stddev=4 * arcsec)
    beam2 = Gaussian2D(x_stddev=6 * arcsec, y_stddev=5 * arcsec)
    beam = utils.encompass_beam(beam1, beam2)
    assert beam.x_stddev == 6 * arcsec
    assert beam.y_stddev == 5 * arcsec

    # Do not encompass
    beam2 = Gaussian2D(x_stddev=4 * arcsec, y_stddev=3 * arcsec)
    beam = utils.encompass_beam(beam1, beam2)
    assert beam.x_stddev == 5 * arcsec
    assert beam.y_stddev == 4 * arcsec

    # Rotation
    beam2 = Gaussian2D(x_stddev=5 * arcsec, y_stddev=6 * arcsec,
                       theta=90 * degree)
    beam = utils.encompass_beam(beam1, beam2)
    assert np.isclose(beam.x_stddev, 6 * arcsec)
    assert np.isclose(beam.y_stddev, 5 * arcsec)

    # Units
    beam2 = Gaussian2D(x_stddev=6 * degree / 3600, y_stddev=5 * degree / 3600)
    beam = utils.encompass_beam(beam1, beam2)
    assert np.isclose(beam.x_stddev, 6 * arcsec)
    assert np.isclose(beam.y_stddev, 5 * arcsec)

    # Different class of object
    beam1 = Beam(x_stddev=5.0, y_stddev=4.0)
    beam2 = Beam(x_stddev=6.0, y_stddev=5.0)
    beam = utils.encompass_beam(beam1, beam2)
    assert np.isclose(beam.x_stddev, 6)
    assert np.isclose(beam.y_stddev, 5)


def test_encompass_beam_fwhm():
    arcsec = units.Unit('arcsec')
    beam1 = Gaussian2D(x_stddev=3 * arcsec, y_stddev=4 * arcsec)
    beam = utils.encompass_beam_fwhm(beam1,
                                     5 * arcsec * gaussian_sigma_to_fwhm)
    assert beam.x_stddev == 5 * arcsec
    assert beam.y_stddev == 5 * arcsec

    # No beam
    beam = utils.encompass_beam_fwhm(None, 5 * arcsec * gaussian_sigma_to_fwhm)
    assert beam.x_stddev == 5 * arcsec
    assert beam.y_stddev == 5 * arcsec


def test_get_beam_area():
    arcsec = units.Unit('arcsec')
    x, y = 3 * arcsec, 4 * arcsec
    beam = Gaussian2D(x_stddev=x, y_stddev=y)
    area = utils.get_beam_area(beam)
    expected = 12 * 2 * np.pi * units.Unit('arcsec2')
    assert np.isclose(area, expected)

    # Non-standard beam
    class Beam(object):
        def __init__(self, x_stddev, y_stddev, theta=0.0):
            self.x_stddev = x_stddev
            self.y_stddev = y_stddev
            self.theta = theta

    beam = Beam(3, 4)
    area = utils.get_beam_area(beam)
    assert np.isclose(area, expected.value)

    # No beam
    assert utils.get_beam_area(None) == 0


def test_get_header_quantity():
    header = fits.Header()
    header['TEST'] = 1.0, 'a test value (degree)'
    degree = units.Unit('degree')
    arcsec = units.Unit('arcsec')

    result = utils.get_header_quantity(header, 'TEST')
    assert result == 1 * degree

    header['TEST'] = 1.0, 'a test value'
    result = utils.get_header_quantity(header, 'TEST', default_unit='arcsec')
    assert result == 1 * arcsec

    del header['TEST']
    result = utils.get_header_quantity(header, 'TEST', default=np.nan,
                                       default_unit=arcsec)
    assert np.isnan(result) and result.unit == arcsec

    header['TEST'] = 1.0, 'a test value (__dne__)'
    result = utils.get_header_quantity(header, 'TEST', default_unit=arcsec)
    assert result == 1 * arcsec


def test_ascii_file_to_frame_data(tmpdir):
    filename = tmpdir.mkdir("ascii_file_test").join("test.tms")
    n_channels = 10
    n_frames = 20
    frame_values = np.arange(n_frames, dtype=float)
    with open(str(filename), 'w') as f:
        f.write('# Header\n')
        for channel in range(n_channels):
            if channel == 5:
                line = '\t'.join(['---' for _ in range(n_frames)])
            else:
                line = '\t'.join([str(x) for x in frame_values])

            f.write(line + '\n')

    result = utils.ascii_file_to_frame_data(str(filename))
    select = np.full(n_channels, True)
    select[5] = False
    assert result.shape == (10, 20)
    assert np.isnan(result[5]).all()
    valid = result[select]
    assert np.allclose(valid, frame_values[None])


def test_insert_into_header():
    h = fits.Header()
    utils.insert_into_header(h, 'A', 1, comment='first reference')
    utils.insert_into_header(h, 'B', 2, comment='second reference')
    utils.insert_into_header(h, 'C', 3, comment='third reference')
    utils.insert_into_header(h, 'HISTORY', 'The first history message')
    utils.insert_into_header(h, 'HISTORY', 'The second history message',
                             refkey='')
    utils.insert_into_header(h, 'HISTORY', 'The third history message',
                             refkey='')
    assert list(h.keys()) == ['A', 'B', 'C', 'HISTORY', 'HISTORY', 'HISTORY']

    # Replace a value but not the comment
    utils.insert_into_header(h, 'B', 1.9, refkey='C')
    assert tuple(h.cards[1]) == ('B', 1.9, 'second reference')

    # Replace the second history message
    utils.insert_into_header(h, 'HISTORY', 'The second history message',
                             refkey='', delete_special=True)
    assert list(h.keys()) == ['A', 'B', 'C', 'HISTORY', 'HISTORY', 'HISTORY']
    assert tuple(h.cards[-1]) == ('HISTORY', 'The second history message', '')

    utils.insert_into_header(h, 'D', 4, comment='')
    assert list(h.keys()) == ['A', 'B', 'C', 'D'] + ['HISTORY'] * 3
    assert h.comments['D'] == ''
    utils.insert_into_header(h, 'D', 4, comment='another comment')
    assert h.comments['D'] == 'another comment'


def test_insert_info_into_header():
    h1 = fits.Header()
    h_insert = fits.Header()
    h1['A'] = 1
    h1['B'] = 2
    h1['C'] = 3
    h1['D'] = 4

    h_insert['I1'] = -1
    h_insert['I2'] = -2
    h_insert['I3'] = -3

    utils.insert_info_in_header(h1, h_insert, refkey='C')
    assert list(h1.keys()) == ['A', 'B', 'I1', 'I2', 'I3', 'C', 'D']
    assert list(h1.values()) == [1, 2, -1, -2, -3, 3, 4]

    h2 = h1.copy()
    utils.insert_info_in_header(h1, fits.Header())
    assert h2 == h1


def test_to_header_cards():
    assert utils.to_header_cards(None) == []
    h = fits.Header()
    h['A'] = 1, 'comment a'
    h['B'] = 2, 'comment b'
    h['COMMENT'] = 'A full comment'
    h['HISTORY'] = 'A history message'
    cards = utils.to_header_cards(h)
    assert cards == [('A', 1, 'comment a'),
                     ('B', 2, 'comment b'),
                     ('COMMENT', 'A full comment', None),
                     ('HISTORY', 'A history message', None)]

    d = {'A': 1,
         'B': (2, 'this is B comment'),
         'C': 3}
    cards = utils.to_header_cards(d)
    assert cards == [('A', 1, None),
                     ('B', 2, 'this is B comment'),
                     ('C', 3, None)]

    x = ['A,1,a comment', 'B', ['C', 2, 'the c comment'], ['D', 3]]
    cards = utils.to_header_cards(x)
    assert cards == [('A', '1', 'a comment'),
                     ('C', 2, 'the c comment'),
                     ('D', 3, None)]

    assert utils.to_header_cards(1) == []


def test_round_values():
    assert utils.round_values(1) == 1
    assert utils.round_values(1.5) == 2
    assert utils.round_values(2.5) == 3
    assert utils.round_values(1.0) == 1

    # Single valued arrays
    x = utils.round_values(np.asarray(1))
    assert isinstance(x, np.ndarray) and x == 1 and x.dtype == int
    x = utils.round_values(np.asarray(1.5))
    assert isinstance(x, np.ndarray) and x == 2 and x.dtype == int
    x = utils.round_values(np.asarray(2.0))
    assert isinstance(x, np.ndarray) and x == 2 and x.dtype == int

    # Single valued quantities
    s = units.Unit('second')
    x = utils.round_values(1 * s)
    assert x.value == 1 and x.unit == 's'
    x = utils.round_values(1.5 * s)
    assert x.value == 2 and x.unit == 's'
    x = utils.round_values(2 * s)
    assert x.value == 2 and x.unit == 's'
    x = utils.round_values(-2.5 * s)
    assert x.value == -3 and x.unit == 's'

    # Numpy arrays
    x = np.arange(11)
    assert utils.round_values(x) is x
    x = (x - 5) / 2
    r = utils.round_values(x)
    assert np.allclose(r, [-3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3])
    expected = r.copy()

    # Quantities
    x = x * s
    r = utils.round_values(x)
    assert np.allclose(r.value, expected)
    assert r.unit == 's'

    # Check shape is preserved
    x = np.arange(10).reshape(2, 5)
    x = (x - 5) / 2
    r = utils.round_values(x)
    assert np.allclose(r, [[-3, -2, -2, -1, -1],
                           [0, 1, 1, 2, 2]])


def test_calculate_position_angle():
    p1 = np.zeros(2)
    p2 = np.full(2, np.pi)
    pa = utils.calculate_position_angle(p1[0], p1[1], p2[0], p2[1])
    assert pa == 315 * units.Unit('degree')

    pa = utils.calculate_position_angle(p2[0] / 2, p2[1] / 2, p1[0], p1[1])
    assert pa == 270 * units.Unit('degree')


def test_get_comment_unit():
    assert utils.get_comment_unit('this is in (arcsec)') == 'arcsec'
    assert utils.get_comment_unit('or perhaps [deg]') == 'degree'
    assert utils.get_comment_unit('foo', default=None) is None
    assert utils.get_comment_unit('foo', default=units.Unit('m')) == 'meter'
    assert utils.get_comment_unit(None) is None
    assert utils.get_comment_unit('a bad format (degree') is None
    assert utils.get_comment_unit('this is not (a unit)') is None
