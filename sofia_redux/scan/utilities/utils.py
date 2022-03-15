# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.time import Time
from astropy.coordinates import FK4, FK5, BaseCoordinateFrame
from astropy.coordinates import concatenate as astro_concat
from astropy.coordinates.angles import Angle
from astropy.modeling.functional_models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
import bottleneck as bn
from copy import deepcopy
import math
import numpy as np
import random
import re
import string
import warnings

from sofia_redux import scan as sofia_scan_module
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.toolkit.utilities.func import clear_numba_cache as \
    toolkit_clear_numba_cache

__all__ = ['UNKNOWN_STRING_VALUE', 'UNKNOWN_FLOAT_VALUE',
           'UNKNOWN_INT_VALUE', 'UNKNOWN_BOOL_VALUE', 'get_string',
           'get_bool', 'get_int', 'get_float', 'get_range',
           'get_list', 'get_string_list', 'get_float_list',
           'get_int_list', 'to_int_list', 'parse_time',
           'parse_angle', 'get_dms_angle', 'get_hms_time',
           'get_sign', 'get_epoch', 'valid_value', 'robust_mean',
           'robust_sigma_clip_mask', 'roundup_ratio', 'rotate',
           'log2round', 'pow2round', 'pow2floor', 'pow2ceil',
           'skycoord_insert_blanks', 'dict_intersection',
           'dict_difference', 'to_header_quantity', 'to_header_float',
           'combine_beams', 'convolve_beam', 'deconvolve_beam',
           'encompass_beam', 'encompass_beam_fwhm', 'get_beam_area',
           'get_header_quantity', 'ascii_file_to_frame_data',
           'insert_into_header', 'insert_info_in_header', 'to_header_cards',
           'clear_numba_cache', 'round_values', 'get_comment_unit']

UNKNOWN_STRING_VALUE = ''
UNKNOWN_FLOAT_VALUE = -9999.0
UNKNOWN_INT_VALUE = -9999
UNKNOWN_BOOL_VALUE = False

ieee_remainder = np.vectorize(math.remainder)


def get_string(value, default=None):
    """
    Return a string representation of the given value.

    Parameters
    ----------
    value : thing
        The value to resolve.
    default : str, optional
        The default result to return if a conversion is not possible,
        or `value` matches the UNKNOWN_STRING_VALUE.

    Returns
    -------
    str
    """
    if value is None or value == UNKNOWN_STRING_VALUE:
        return default
    try:
        result = str(value)
    except (ValueError, TypeError):
        result = default
    return result


def get_bool(value, default=False):
    """
    Return a bool representation of the given value.

    Parameters
    ----------
    value : thing
        The value to resolve.
    default : bool, optional
        The default result to return if a conversion is not possible,
        or `value` matches the UNKNOWN_BOOL_VALUE.

    Returns
    -------
    bool
    """
    if value is None:
        return default
    value = str(value).strip().lower()
    if len(value) == 0:
        return default
    if value[0] in ['0', 'n', 'f']:
        return False
    return True


def get_int(value, default=0):
    """
    Return an integer representation of the given value.

    Parameters
    ----------
    value : thing
        The value to resolve.
    default : int, optional
        The default result to return if a conversion is not possible,
        or `value` matches the UNKNOWN_INT_VALUE.

    Returns
    -------
    int
    """
    if value is None:
        return default
    try:
        value = int(value)
    except (ValueError, TypeError):
        return default

    if value == UNKNOWN_INT_VALUE:
        return default
    return value


def get_float(value, default=np.nan):
    """
    Return a float representation of the given value.

    Parameters
    ----------
    value : thing
        The value to resolve.
    default : float, optional
        The default result to return if a conversion is not possible,
        or `value` matches the UNKNOWN_FLOAT_VALUE.

    Returns
    -------
    int
    """
    if value is None:
        return default
    try:
        value = float(value)
    except (ValueError, TypeError):
        return default

    if np.isclose(value, UNKNOWN_FLOAT_VALUE, equal_nan=True):
        return default
    return value


def get_range(value, default=Range(), is_positive=False):
    """
    Return a Range object generated from the given value.

    Parameters
    ----------
    value : thing
        The thing to resolve.
    default : Range, optional
        The default result to return if a Range could not be resolved.
    is_positive : bool, optional
        If `True`, all values in the range are considered positive and any
        '-' character in `spec` will be treated as a delimiter rather than
        a minus sign.

    Returns
    -------
    Range
    """
    if value is None:
        return default
    elif isinstance(value, Range):
        return value.copy()
    elif not isinstance(value, str) and len(value) == 2:
        try:
            return Range(min_val=value[0], max_val=value[1])
        except ValueError:
            return default

    try:
        return Range.from_spec(value, is_positive=is_positive)
    except ValueError:
        return default


def get_list(value):
    """
    Resolve the given value into a list.

    Parameters
    ----------
    value : list or thing

    Returns
    -------
    list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_string_list(value, delimiter=',', default=None):
    """
    Return a list of strings from the given single string.

    Parameters
    ----------
    value : str
        The string to convert into a list.
    delimiter : str, optional
        The string delimiter used to separate one element from the next.
    default : list, optional
        The result to return if the value passed in is `None`.

    Returns
    -------
    list (str)
    """
    if value is None:
        return default
    if isinstance(value, str):
        value = [s.strip() for s in re.split(delimiter, value)]
    return value


def get_float_list(value, delimiter=',', default=None):
    """
    Return a list of floats from a single string.

    Parameters
    ----------
    value : str
        The string to convert into a list of floats.
    delimiter : str, optional
        The string delimiter used to separate one element from the next.
    default : list, optional
        The result to return if the value passed in is `None`.

    Returns
    -------
    list (float)
    """
    value = get_string_list(value, delimiter=delimiter, default=None)
    if value is None:
        return default
    return [get_float(x) for x in value]


def get_int_list(value, delimiter=',', default=None, is_positive=True):
    """
    Return a list of ints from a single string.

    Parameters
    ----------
    value : str
        The string to convert into a list of integers.
    delimiter : str, optional
        The string delimiter used to separate one element from the next.
    default : list, optional
        The result to return if the value passed in is `None`.
    is_positive : bool, optional
        If `True`, ranges may be specified using both ':' and '-' characters in
        a string.  Otherwise, the '-' character will imply a negative value.

    Returns
    -------
    list (int)
    """
    values = get_string_list(value, delimiter=delimiter, default=None)
    if values is None:
        return default
    return to_int_list(values, is_positive=is_positive)


def to_int_list(values, is_positive=True):
    """
    Convert all elements in a list to integers.

    String representations of integers may also be included as ranges.
    The default behaviour is to treat the '-' character as range rather than
    a minus sign.  Ranges should generally be specified using the ':'
    character. To allow negative values, set `is_positive=False`.

    For example
    >>> print(to_int_list(['1-3']))
    [1, 2, 3]
    >>> print(to_int_list(['-1:3'], is_positive=False))
    [-1, 0, 1, 2, 3]

    If is_positive is not set, the meaning of '-' is ambiguous and can result
    in errors.

    Parameters
    ----------
    values : list
    is_positive : bool, optional
        If `True`, ranges may be specified using both ':' and '-' characters in
        a string.  Otherwise, the '-' character will imply a negative value.

    Returns
    -------
    list (int)
    """
    result = []
    for value in values:
        if isinstance(value, str):
            splitter = r'[-:]' if is_positive else r'[:]'
            if re.search(splitter, value) is not None:
                start, stop = re.split(splitter, value)
                result.extend(list(np.arange(int(start), int(stop) + 1)))
            else:
                result.append(int(value))
        else:
            result.append(int(value))
    return result


def parse_time(hms_string, angle=False):
    """
    Returns H:M:S string as time Quantity.

    Parameters
    ----------
    hms_string : string or float
    angle : bool, optional
        If `True`, return an hour angle unit instead of hour unit.

    Returns
    -------
    time : units.Quantity
        An hour if angle is `False`, otherwise an hour angle.
    """
    hms = re.split(r'[\t\n\r:hmsHMS]', str(hms_string))
    hms = [x.strip() for x in hms]
    hms = [x for x in hms if x != '']
    unit = units.Unit('hourangle') if angle else units.Unit('hour')
    if len(hms) != 3:
        raise ValueError(f"Cannot parse {hms_string} as an hh:mm:ss string.")

    try:
        hms = [float(x) for x in hms]
    except (ValueError, TypeError):
        raise ValueError(f"Cannot parse {hms_string} as an hh:mm:ss string.")

    return ((hms[0]) + (hms[1] / 60) + (hms[2] / 3600)) * unit


def parse_angle(dms):
    """
    Convert a degree:minute:second string into an angle.

    Parameters
    ----------
    dms : str

    Returns
    -------
    angle : astropy.units.Quantity
    """

    dms = dms.strip()
    negative = dms.startswith('-')
    if negative:
        dms = dms[1:]

    regex = r'[- \t\n\r:dmsDMS]'
    s = re.split(regex, dms)
    s = [x for x in s if s != '']
    try:
        angle = int(s[0]) + (int(s[1]) / 60) + (float(s[2]) / 3600)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot parse {dms} as a dd:mm:ss string.")
    if negative:
        angle *= -1
    return angle * units.Unit('deg')


def get_dms_angle(value, default=np.nan):
    """
    Return a degree angle quantity from a given value.

    The provided value may be a Quantity, float, int, or string value.  If a
    string value is provided, it will be parsed according to
    :func:`parse_angle`.

    Parameters
    ----------
    value : int or float or str or units.Quantity
    default : int or float or units.Quantity, optional
        The default angle to return in cases where the `value` cannot be
        parsed correctly

    Returns
    -------
    angle : units.Quantity
        The resolved angle in degrees.
    """
    degree = units.Unit('deg')

    if not isinstance(default, units.Quantity):
        try:
            default = float(default) * degree
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert default ({default}) to degrees.")
    else:
        try:
            default = default.to(degree)
        except units.UnitConversionError:
            raise ValueError(f"Cannot convert default ({default} to degrees.")

    if isinstance(value, units.Quantity):
        try:
            return value.to(degree)
        except units.UnitConversionError:
            log.warning(f"Cannot convert {value.unit} to {degree}")
            return default

    if value is None:
        return default

    try:
        value = float(value)
        if np.isclose(value, UNKNOWN_FLOAT_VALUE, equal_nan=True):
            return default
        value *= degree
    except (ValueError, TypeError):
        try:
            value = parse_angle(str(value))
        except (ValueError, TypeError):
            log.warning(f"Attempting to parse {value} as a dd:mm:ss string.")
            return default
    return value


def get_hms_time(value, angle=False, default=np.nan):
    """
    Return a time or hour angle quantity from a given value.

    The provided value may be a Quantity, float, int, or string value.  If a
    string value is provided, it will be parsed according to
    :func:`parse_angle`.

    Parameters
    ----------
    value : int or float or str or units.Quantity
    angle : bool, optional
        If `True`, return an hour angle unit instead of hour unit.
    default : int or float or units.Quantity, optional
        The default angle to return in cases where the `value` cannot be
        parsed correctly

    Returns
    -------
    time : units.Quantity
        The resolved time in hours, or as an hour angle.
    """
    unit = units.Unit('hourangle') if angle else units.Unit('hour')

    if not isinstance(default, units.Quantity):
        try:
            default = float(default) * unit
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert default ({default}) to {unit}.")
    else:
        try:
            default = default.to(unit)
        except units.UnitConversionError:
            raise ValueError(f"Cannot convert default ({default} to {unit}.")

    if isinstance(value, units.Quantity):
        try:
            return value.to(unit)
        except units.UnitConversionError:
            log.warning(f"Cannot convert {value.unit} to {unit}")
            return default

    if value is None:
        return default

    try:
        value = float(value)
        if np.isclose(value, UNKNOWN_FLOAT_VALUE, equal_nan=True):
            return default
        value *= unit
    except (ValueError, TypeError):
        try:
            value = parse_time(str(value), angle=angle)
        except (ValueError, TypeError):
            log.warning(f"Attempting to parse {value} as an hh:mm:ss string.")
            return default
    return value


def get_sign(value, default=0):
    """
    Return an integer representation of a sign.

    Parameters
    ----------
    value : int or float or str
        The value to parse.
    default : int, optional
        The default sign to return.

    Returns
    -------
    sign : int
        1 for a positive sign, -1 for a negative sign, and 0 for no sign.
    """
    if isinstance(value, (int, float)):
        if value == 0 or np.isnan(value):
            return 0
        return -1 if value < 0 else 1

    if value is None:
        return default

    str_value = str(value).strip().lower()
    if str_value in ['+', 'pos', 'positive', 'plus']:
        return 1
    elif str_value in ['-', 'neg', 'negative', 'minus']:
        return -1
    elif str_value in ['*', 'any']:
        return 0
    else:
        try:
            value = float(value)
            if value < 0:
                return -1
            elif value > 0:
                return 1
            else:
                return 0
        except ValueError:
            return default


def get_epoch(equinox):
    """
    Return an astropy frame representing the equinox of an observation.

    An FK4 frame will be returned in instances where `equinox` is a string and
    the first letter is 'B' (B1950 for example), or the year is less than 1984.
    In other instances a Julian FK5 frame will be returned.

    Parameters
    ----------
    equinox : BaseCoordinateFrame or Time or float or int or str
        The equinox of the observation

    Returns
    -------
    epoch : BaseCoordinateFrame
        Typically and FK5 or FK5 frame unless another BaseCoordinateFrame was
        passed in as `equinox`.
    """
    if isinstance(equinox, BaseCoordinateFrame):
        return equinox
    elif isinstance(equinox, Time):
        if equinox.datetime.year < 1984:
            return FK4(equinox=equinox)
        else:
            return FK5(equinox=equinox)

    equinox = str(equinox).strip().upper()
    if equinox[0].isalpha():
        prefix = equinox[0]
        year = float(equinox[1:])
    else:
        year = float(equinox)
        prefix = 'B' if year < 1984 else 'J'

    equinox = prefix + str(year)
    frame = FK4 if prefix == 'B' else FK5
    return frame(equinox=Time(equinox))


def valid_value(value):
    """
    Return if a given value is valid in the context of unknown values.

    The unknown values are `utils.UNKNOWN_INT_VALUE`,
    `utils.UNKNOWN_STRING_VALUE`, `utils.UNKNOWN_FLOAT_VALUE`, and NaN.

    Parameters
    ----------
    value : bool or int or str or float
        The value to check against the list of UNKNOWN values.

    Returns
    -------
    valid : bool
        `False` if the value and type matches the corresponding unknown
        value or is not a bool, int, str, or float.  NaNs also return `False`.
        All other values will return `True`.
    """
    if isinstance(value, bool):
        return True
    elif isinstance(value, int):
        return value != UNKNOWN_INT_VALUE
    elif isinstance(value, str):
        return value != UNKNOWN_STRING_VALUE
    elif isinstance(value, float):
        return (value != UNKNOWN_FLOAT_VALUE) and not np.isnan(value)
    else:
        return False


def robust_mean(array, tails=None):
    """
    Performs a mean calculation with optional tails.

    NaNs are automatically excluded from the mean calculation.  The `tails`
    parameter should be between 0 and 0.5, and will exclude
    `tails` * `array.size` from both the beginning and end of the sorted
    array values.

    Parameters
    ----------
    array : numpy.ndarray or Quantity
    tails : float, optional
        The fraction of tails to discount from the mean calculation.  The
        default (`None`), does not exclude any tails and considers the entire
        population.

    Returns
    -------
    mean : float or Quantity
        The mean of the array.
    """
    if isinstance(array, units.Quantity):
        u = array.unit
    else:
        u = None

    if tails is None:
        result = bn.nanmean(array)
        return result if u is None else result * u

    array = np.asarray(array)
    dn = int(np.round(array.size * tails))
    result = bn.nanmean(np.sort(array)[dn:-dn])
    return result if u is None else result * u


def robust_sigma_clip_mask(values, weights=None, mask=None, sigma=5.0,
                           verbose=False, max_iterations=5):
    """
    Return a masking array indicating sigma clipped elements of an input array.

    Iteratively identifies outliers in a given input array based on the number
    of standard deviations (`sigma`) of each datum from the median value of
    the given `values`.

    On each iteration, values that are more than `sigma` * std(`values`) away
    from the median are ignored on subsequent iterations in which new medians
    and standard deviations are calculated.

    Note that while any multi-dimensional arrays may be provided, the
    calculated median and standard deviation on each iteration will apply
    to the entirety of the data set.

    The iterations will desist once `max_iterations` iterations have occurred,
    no additional elements are identified as outliers on an iteration, or all
    elements have been masked.

    Parameters
    ----------
    values : units.Quantity or numpy.ndarray (float)
        The values for which to calculated the sigma clipping masked array.
        NaN values will be ignored in all calculations and appear as `False` in
        the output mask array.
    weights : units.Quantity or numpy.ndarray (float), optional
        Optional weighting for the values.  These should typically represent
        the inverse variance of the provided data.  If provided, the calculated
        median will be weighted accordingly using
        :func:`numba_functions.smart_median`, as will the derived standard
        deviation.
    mask : numpy.ndarray (bool), optional
        An optional starting mask where `False` indicates an invalid or clipped
        value.  Will not be updated in-place.
    sigma : float, optional
        The number of standard deviations away from the median that will result
        in a datum being clipped.
    verbose : bool, optional
        If `True`, will output log messages indicating the total number of
        clipped values, the median, and the standard deviation on each
        iteration.
    max_iterations : int, optional
        The maximum number of iterations that will be used to derive the
        masking array.

    Returns
    -------
    mask : np.ndarray (bool)
        The sigma clipped masking array where `False` indicates a clipped value
        and `True` indicates a valid value.
    """
    if isinstance(values, units.Quantity):
        unit = values.unit
        values = values.value
    else:
        unit = None

    if isinstance(weights, units.Quantity):
        weights = weights.value

    if mask is None:
        mask = np.isfinite(values)
    else:
        mask = mask & np.isfinite(values)

    for _ in range(max_iterations):
        if verbose:
            log.info(f"Valid values: {mask.sum()}")
        current_values = values[mask]
        current_weights = None if weights is None else weights[mask]

        median_value = numba_functions.smart_median(
            current_values, weights=current_weights)[0]
        standard_deviation = np.sqrt(
            np.cov(current_values, aweights=current_weights))
        limit = standard_deviation * sigma

        if verbose:
            line = f"Median: {median_value} +/- {standard_deviation}"
            if unit is not None:
                line += f" {unit}"
            log.info(line)

        valid = np.abs(current_values - median_value) <= limit
        if valid.all():
            break

        mask[mask] = valid

        # This is probably redundant and can't be tested easily
        if not valid.any():  # pragma: no cover
            break

    return mask


def roundup_ratio(a, b):
    """
    Returns int((a + b - 1) / b).

    Parameters
    ----------
    a : int or float or numpy.ndarray
    b : int or float or numpy.ndarray

    Returns
    -------
    ratio : int
        The roundup ratio.
    """
    if hasattr(a, '__len__') or hasattr(b, '__len__'):
        a = np.asarray(a)
        b = np.asarray(b)
        return ((a + b - 1) / b).astype(int)

    return numba_functions.roundup_ratio(a, b)


def rotate(positions, angle):
    """
    Rotate 2 dimensional coordinates by a given angle.

    Parameters
    ----------
    positions : numpy.ndarray
        x, y positions of shape (N, 2)
    angle : float or astropy.units.Quantity or numpy.ndarray (float)
        The rotation angle (anti-clockwise) in radians (float) or as an
        astropy unit.  If supplied as an array, should be of shape (N,).

    Returns
    -------
    numpy.ndarray
        Rotated x, y positions
    """
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    result = np.empty_like(positions)
    result[:, 0] = (positions[:, 0] * cos_a) - (positions[:, 1] * sin_a)
    result[:, 1] = (positions[:, 0] * sin_a) + (positions[:, 1] * cos_a)
    return result


def log2round(x):
    """
    Returns the rounded value of log_2(x).

    Examples
    --------
    >>> print(log2round(1024))
    10
    >>> halfway = int(2 ** 9.5)
    >>> print(log2round(halfway))
    9
    >>> print(log2round(halfway + 1))
    10

    Parameters
    ----------
    x : int or float or numpy.ndarray
        The input value(s).

    Returns
    -------
    log2 : int or numpy.ndarray (int)
    """
    result = np.round(np.log2(x))
    if isinstance(result, np.ndarray):
        return result.astype(int)
    return int(result)


def pow2round(x):
    """
    Returns the value rounded to the closest 2^N integer.

    Examples
    --------
    >>> print(pow2round(1023))
    1024
    >>> halfway = int(2 ** 9.5)
    >>> print(pow2round(halfway))
    512
    >>> print(pow2round(halfway + 1))
    1024

    Parameters
    ----------
    x : int or float or numpy.ndarray
       The input value(s).

    Returns
    -------
    result : int or numpy.ndarray (int)
    """
    return 1 << log2round(x)


def pow2floor(x):
    """
    Returns the value floored to the closest 2^N integer.

    Examples
    --------
    >>> print(pow2floor(1023))
    512
    >>> print(pow2floor(1024))
    1024
    >>> print(pow2floor(1025))
    1024

    Parameters
    ----------
    x : int or float or numpy.ndarray
        The input value(s).

    Returns
    -------
    result : int or numpy.ndarray (int)
    """
    l2 = np.log2(x)
    if isinstance(l2, np.ndarray):
        return 2 ** l2.astype(int)
    return 2 ** int(np.log2(x))


def pow2ceil(x):
    """
    Returns the value ceiled to the closest 2^N integer.

    Examples
    --------
    >>> print(pow2ceil(1023))
    1024
    >>> print(pow2ceil(1024))
    1024
    >>> print(pow2ceil(1025))
    2048

    Parameters
    ----------
    x : int or float or numpy.ndarray
        The input value(s).

    Returns
    -------
    result : int or numpy.ndarray (int)
    """
    value = pow2floor(x)
    if not isinstance(value, np.ndarray):
        return value if value == x else value * 2
    value[value != np.asarray(x)] *= 2
    return value


def skycoord_insert_blanks(coordinates, insert_indices):
    """
    Insert zeroed coordinates in an astropy coordinate frame.

    Parameters
    ----------
    coordinates : astropy.coordinates.BaseCoordinateFrame or SkyCoord
    insert_indices : numpy.ndarray (float)
        The indices at which to insert zero values.

    Returns
    -------
    astropy.coordinates.BaseCoordinateFrame
    """
    n_insert = insert_indices.size
    new = astro_concat((coordinates, coordinates))
    new.cache.clear()
    new.data.lat[coordinates.size:] *= 0
    new.data.lon[coordinates.size:] *= 0

    old_indices = np.arange(coordinates.size)
    good_indices = old_indices.copy()
    for index in insert_indices:
        good_indices[index:] += 1
    bad_indices = np.arange(n_insert) + insert_indices

    full_indices = np.zeros(coordinates.size + n_insert, dtype=int)
    full_indices[good_indices] = old_indices
    full_indices[bad_indices] = coordinates.size + np.arange(n_insert)
    return new[full_indices].copy()


def dict_intersection(options1, options2, initialized=False):
    """
    Return the intersection of two dictionaries.

    Parameters
    ----------
    options1 : dict
    options2 : dict
    initialized : bool, optional
        A switch for use on recursive calls.  Don't mess.

    Returns
    -------
    intersection : dict
    """
    if initialized:
        result = dict()
    else:
        result = options1.__class__()
    for k in set(options1.keys()) & set(options2.keys()):
        value1 = options1[k]
        value2 = options2[k]
        if isinstance(value1, dict) and isinstance(value2, dict):
            result[k] = dict_intersection(value1, value2, initialized=True)
        elif value1 == value2:
            result[k] = value1
    return result


def dict_difference(options1, options2, initialized=False):
    """
    Return the difference of two dictionaries.

    Parameters
    ----------
    options1 : dict
    options2 : dict
    initialized : bool, optional
        A switch for use on recursive calls.  Don't mess.

    Returns
    -------
    dict
    """
    if initialized:
        result = dict()
    else:
        result = options1.__class__()

    for key, value in options1.items():
        if key not in options2:
            result[key] = deepcopy(value)
            continue
        other_value = options2[key]
        if isinstance(value, dict) and isinstance(other_value, dict):
            diff = dict_difference(value, other_value, initialized=True)
            if len(diff) > 0:
                result[key] = diff
            continue
        if value != other_value:
            result[key] = value

    return result


def to_header_quantity(value, unit=None, keep=False):
    """
    Convert a value to a Quantity suitable for header entry.

    Converts various input types to a `float` or `units.Quantity` output value.
    If a dimensionless value is input, it will be multiplied by the unit if
    present.  UNKNOWN header quantities (typically -9999) are preserved, and
    non-finite values also return the UNKNOWN float value.

    Parameters
    ----------
    value : int or float or str or units.Quantity or None
        The value to convert.
    unit : str or units.Unit or units.Quantity, optional
        The units of the output value.  If a `units.Quantity` value is
        supplied, the output value will be converted to the quantity unit and
        scaled by the quantity value.  Dimensionless units will have no effect,
        while dimensionless quantities will still scale the value, but result
        in no conversion to an output unit.
    keep : bool, optional
        If `True`, return the original value when invalid instead of the
        UNKNOWN_FLOAT_VALUE.

    Returns
    -------
    header_quantity : float or units.Quantity
    """
    ud = units.dimensionless_unscaled
    if unit is None and isinstance(value, units.Quantity):
        unit = value.unit

    # Determine scaling and unit
    if unit is None:
        scaling = 1.0
    elif isinstance(unit, units.Quantity):
        scaling = unit.value
        unit = unit.unit
    else:
        scaling = 1.0
        unit = units.Unit(unit)

    if isinstance(unit, units.UnitBase) and unit == ud:
        unit = None

    if isinstance(value, (str, int, float)):
        value = float(value)
    elif value is None:
        value = np.nan
    elif not isinstance(value, units.Quantity):
        raise ValueError(f"Value must be {float}, {int}, {str}, {None} or "
                         f"{units.Quantity}.")

    if isinstance(value, units.Quantity):
        if value.unit == ud:
            value = value.value

    if keep:
        bad_return_value = value
    else:
        bad_return_value = UNKNOWN_FLOAT_VALUE

    # Value is either a float or quantity at this point
    # Unit is either None or an actual unit
    # Neither is dimensionless

    if isinstance(value, units.Quantity):
        if value.value == UNKNOWN_FLOAT_VALUE or not np.isfinite(value):
            if unit is None:
                return bad_return_value * value.unit
            else:
                return bad_return_value * unit
        elif isinstance(unit, units.Unit):
            return value.to(unit) * scaling
        else:
            return value * scaling

    elif value == UNKNOWN_FLOAT_VALUE or not np.isfinite(value):
        if isinstance(unit, units.Unit):
            return bad_return_value * unit
        else:
            return bad_return_value

    elif isinstance(unit, units.Unit):
        return value * scaling * unit

    else:
        return value * scaling


def to_header_float(value, unit=None):
    """
    Parse a value to a parsable header float value.

    Parameters
    ----------
    value : float or units.Quantity or None
    unit : str or units.Unit or units.Quantity or int or float, optional
        If a quantity was passed in, convert to this unit first.

    Returns
    -------
    float
    """
    if value is None:
        return UNKNOWN_FLOAT_VALUE
    if np.isnan(value):
        return UNKNOWN_FLOAT_VALUE
    if isinstance(value, units.Quantity):
        if unit is None:
            return value.value
        else:
            return value.to(unit).value
    else:
        return float(value)


def combine_beams(beam1, beam2, deconvolve=False):
    """
    Convolve or deconvolve one beam by another.

    Parameters
    ----------
    beam1 : astropy.modeling.functional_models.Gaussian2D
        The original beam to modify.
    beam2 : astropy.modeling.functional_models.Gaussian2D or None
        The beam to modify `beam1` with.
    deconvolve : bool, optional
        If `True`, indicates a deconvolution rather than a convolution.

    Returns
    -------
    astropy.modeling.functional_models.Gaussian2D
        The combined beam.
    """
    if beam2 is None:
        return beam1.copy()
    a2x = beam1.x_stddev ** 2
    a2y = beam1.y_stddev ** 2
    b2x = beam2.x_stddev ** 2
    b2y = beam2.y_stddev ** 2
    direction = -1 if deconvolve else 1
    a = a2x - a2y
    b = b2x - b2y

    rad = units.Unit('radian')
    theta1 = beam1.theta
    theta2 = beam2.theta
    if not isinstance(theta1, units.Quantity):
        theta1 = theta1 * rad
    if not isinstance(theta2, units.Quantity):
        theta2 = theta2 * rad

    delta = theta2 - theta1
    delta_unit = delta.unit
    delta = ((delta.to(rad).value % np.pi) * rad).to(delta_unit)
    c2 = (a ** 2) + (b ** 2) + (2 * direction * a * b * np.cos(delta))

    if hasattr(beam1.x_stddev, 'unit') and beam1.x_stddev.unit is not None:
        unit = beam1.x_stddev.unit
    else:
        unit = None

    new = beam1.copy()
    if c2 < 0:  # pragma: no cover
        if unit is None:
            new.x_stddev = 0.0
            new.y_stddev = 0.0
        else:
            new.x_stddev = 0.0 * unit
            new.y_stddev = 0.0 * unit
        return new

    c = np.sqrt(c2)
    b2 = a2x + a2y + (direction * (b2x + b2y))
    x2 = 0.5 * (b2 + c)
    y2 = 0.5 * (b2 - c)
    if x2 < 0:  # pragma: no cover
        x2 = 0.0 if unit is None else 0.0 * (unit ** 2)  # pragma: no cover
    if y2 < 0:
        y2 = 0.0 if unit is None else 0.0 * (unit ** 2)

    x_stddev = np.sqrt(x2)
    y_stddev = np.sqrt(y2)
    if unit is not None:
        x_stddev = x_stddev.to(unit)
        y_stddev = y_stddev.to(unit)

    if c == 0 or np.isnan(c):
        theta = 0.0 * delta_unit
    else:
        sin_beta = direction * np.sin(delta).value * b / c
        if isinstance(sin_beta, units.Quantity):
            sin_beta = sin_beta.decompose().value
        theta = (theta1.value + (0.5 * np.arcsin(sin_beta))) * rad
        theta = theta.to(delta_unit)

    new.x_stddev = x_stddev
    new.y_stddev = y_stddev
    new.theta = theta
    return new


def convolve_beam(beam1, beam2):
    """
    Convolve one beam by another.

    Parameters
    ----------
    beam1 : astropy.modeling.functional_models.Gaussian2D
        The Gaussian beam to convolve.
    beam2 : astropy.modeling.functional_models.Gaussian2D
        The Gaussian beam to convolve `beam1` with.

    Returns
    -------
    beam : astropy.modeling.functional_models.Gaussian2D
    """
    return combine_beams(beam1, beam2, deconvolve=False)


def deconvolve_beam(beam1, beam2):
    """
    Deconvolve one beam by another.

    Parameters
    ----------
    beam1 : astropy.modeling.functional_models.Gaussian2D
        The Gaussian beam to deconvolve.
    beam2 : astropy.modeling.functional_models.Gaussian2D
        The Gaussian beam to deconvolve `beam1` by.

    Returns
    -------
    beam : astropy.modeling.functional_models.Gaussian2D
    """
    return combine_beams(beam1, beam2, deconvolve=True)


def encompass_beam(beam1, beam2):
    """
    Encompass one beam by another.

    Parameters
    ----------
    beam1 : astropy.modeling.functional_models.Gaussian2D
        The Gaussian beam to encompass.
    beam2 : astropy.modeling.functional_models.Gaussian2D
        The Gaussian beam to encompass `beam1` with.

    Returns
    -------
    beam : astropy.modeling.functional_models.Gaussian2D
    """
    delta_angle = beam2.theta - beam1.theta
    cos_a = np.cos(delta_angle)
    sin_a = np.sin(delta_angle)

    if hasattr(beam1.x_stddev, 'unit'):
        base_unit = beam1.x_stddev.unit
    else:
        base_unit = None

    min_x = np.hypot(beam2.x_stddev * cos_a, beam2.y_stddev * sin_a)
    min_y = np.hypot(beam2.x_stddev * sin_a, beam2.y_stddev * cos_a)
    if base_unit is not None:
        if base_unit != min_x.unit:
            min_x = min_x.decompose().to(base_unit)
        if base_unit != min_y.unit:
            min_y = min_y.decompose().to(base_unit)

    beam = beam1.copy()
    if beam1.x_stddev < min_x:
        beam.x_stddev = min_x
    if beam1.y_stddev < min_y:
        beam.y_stddev = min_y

    return beam


def encompass_beam_fwhm(beam, fwhm):
    """
    Encompass one beam by a given fwhm.

    Parameters
    ----------
    beam : astropy.modeling.functional_models.Gaussian2D or None
        The Gaussian beam to encompass.
    fwhm : float or units.Quantity
        The full-width-half-max to encompass `beam1` with.

    Returns
    -------
    astropy.modeling.functional_models.Gaussian2D
    """
    stddev = gaussian_fwhm_to_sigma * fwhm
    if beam is None:
        return Gaussian2D(x_stddev=stddev, y_stddev=stddev)

    encompassed_beam = beam.copy()
    if beam.x_stddev < stddev:
        encompassed_beam.x_stddev = stddev
    if beam.y_stddev < stddev:
        encompassed_beam.y_stddev = stddev

    return encompassed_beam


def get_beam_area(psf):
    """
    Return the area of a 2-dimensional Gaussian model.

    Parameters
    ----------
    psf : astropy.modeling.functional_models.Gaussian2D or None

    Returns
    -------
    area : float or units.Quantity
    """
    if psf is None:
        return 0.0

    if hasattr(psf.x_stddev, 'unit'):
        unit = psf.x_stddev.unit
    else:
        unit = None

    area = 2.0 * np.pi * psf.x_stddev * psf.y_stddev
    if unit is not None:
        area = area.to(unit ** 2)
    return area


def get_header_quantity(header, key, default=np.nan,
                        default_unit=units.dimensionless_unscaled):
    """
    Get an astropy quantity from a FITS header value.

    The unit is determined from the FITS comment, where unit should be enclosed
    in [unit] or (unit).

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    key : str
    default : float, optional
        The default value to pass back if the key is not found in the header.
    default_unit : astropy.units.Unit or astropy.units.Quantity or str
        The default unit if cannot be determined from header.  If a quantity
        is provided, will be multiplied by the value.

    Returns
    -------
    astropy.units.Quantity
    """
    value = header.get(key, default)

    if isinstance(default_unit, str):
        default_unit = units.Unit(default_unit)

    if key not in header:
        return value * default_unit

    comment = header.comments[key]
    str_units = re.findall(r'\(.*?\)|\[.*?\]', comment)

    if len(str_units) == 0:
        return value * default_unit

    try:
        header_unit = units.Unit(str_units[0][1:-1])
        return value * header_unit
    except ValueError:
        return value * default_unit


def ascii_file_to_frame_data(ascii_file):
    """
    Convert time stream data from a ascii file to a numpy array.

    Parameters
    ----------
    ascii_file : str
        The filename for the ascii file.  Output from SOFSCAN via the
        "write.ascii" configuration option.

    Returns
    -------
    frame_data : numpy.ndarray (float)
        The frame data of shape (n_frames, n_channels).
    """
    with open(ascii_file, 'r') as f:
        contents = f.readlines()

    contents = [
        c.strip().replace('---', 'NaN').replace('\t\t', '\t').split('\t')
        for c in contents[1:]]

    n_frames = len(contents)
    n_channels = len(contents[0])

    frame_data = np.empty((n_frames, n_channels), dtype=float)
    for frame in range(n_frames):
        frame_data[frame] = contents[frame]

    return frame_data


def insert_into_header(header, key, value, comment=None, refkey='HISTORY',
                       after=False, delete_special=False):
    """
    Insert a key, value into the header at a specific location.

    The key will always be inserted at the desired location.  If it already
    exists in the header, the old value will be removed before insertion.

    Parameters
    ----------
    header : fits.Header
        The header in which to insert the key.
    key : str
        The name of the header key to insert.
    value : str or object
        The value of the header key.
    comment : str, optional
        An optional header comment.  If `None` and the key already exists in
        the header, the old comment will be used instead.
    refkey : str, optional
        A key in the header marking the location at which to insert `key`.  If
        not present in the header, the key will be inserted in the standard
        manner.
    after : bool, optional
        If `True`, insert `key` after `refkey`.  If `False`, insert `key`
        before `refkey`.
    delete_special : bool, optional
        If `True`, allow deletion of previous HISTORY and COMMENT entries
        containing the same value.

    Returns
    -------
    None
    """
    upper_key = key.upper()
    special_key = upper_key in ['COMMENT', 'HISTORY']
    old_key = old_comment = None
    for try_key in [upper_key, key]:
        if try_key in header:
            old_key = try_key
            if not special_key:
                old_comment = header.comments[try_key]
                if old_comment == '':
                    old_comment = None
            else:
                old_comment = None
            break

    in_header = old_key is not None
    if in_header:
        if not special_key:
            del header[old_key]
        elif delete_special:
            delete_indices = []
            previous_special = header[old_key]
            for i, previous_value in enumerate(previous_special):
                if value == previous_value:
                    delete_indices.append(i)
            for delete_index in delete_indices[::-1]:
                del header[old_key, delete_index]

    if special_key or (comment is None and old_comment is None):
        insert_card = (key, value)
    elif comment is not None:
        insert_card = (key, value, comment)
    else:
        insert_card = (key, value, old_comment)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        if refkey not in header:
            header[insert_card[0]] = insert_card[1:]
        else:
            header.insert(refkey, insert_card, after=after)


def insert_info_in_header(header, header_info, refkey='HISTORY', after=False,
                          delete_special=False):
    """
    Insert header information into a header at a given location.

    Parameters
    ----------
    header : fits.Header
        The header for which to insert `header_info`.
    header_info : fits.Header or dict or list or tuple or set.
        The header information to insert.  Will be normalized using
        :func:`to_header_cards`.
    refkey : str, optional
        The key in `header` marking the insertion location.
    after : bool, optional
        If `True`, insert `header_info` after `refkey`.  Otherwise, insert
        before `refkey`.
    delete_special : bool, optional
        If `True`, allow deletion of previous HISTORY and COMMENT entries
        containing the same value.

    Returns
    -------
    None
    """
    cards = to_header_cards(header_info)
    if len(cards) == 0:
        return

    letters = string.ascii_uppercase
    marker = ''.join(random.choice(letters) for _ in range(8))
    while marker in header:  # pragma: no cover
        marker = ''.join(random.choice(letters) for _ in range(8))
    insert_into_header(header, marker, True,
                       comment='Temporary marker for header insertion.',
                       refkey=refkey, after=after)

    for card in cards:
        insert_into_header(header, card[0], card[1], comment=card[2],
                           refkey=marker, after=False,
                           delete_special=delete_special)
    del header[marker]


def to_header_cards(header_info):
    """
    Convert header-like information to a list of cards.

    Standardizes header type information into a list of cards suitable for
    passing into :func:`insert_into_header`.  COMMENT and HISTORY keys will
    automatically set the associated comment to `None`.

    Parameters
    ----------
    header_info : fits.Header or dict or list or tuple or set or None

    Returns
    -------
    cards : list (tuple)
       A list of cards where each element is (key, value, comment).
    """
    cards = []
    if header_info is None:
        return cards

    if isinstance(header_info, fits.Header):
        for key, value in header_info.items():
            comment = header_info.comments[key]
            cards.append((key, value, comment))

    elif isinstance(header_info, dict):
        for key, value in header_info.items():
            if (hasattr(value, '__len__')
                    and not isinstance(value, str)
                    and len(value) == 2
                    and isinstance(value[1], str)):
                comment = value[1]
                value = value[0]
            else:
                comment = None
            cards.append((key, value, comment))

    elif isinstance(header_info, (list, tuple, set)):
        for element in header_info:
            if isinstance(element, str):
                element = [x.strip() for x in element.split(',')]
            if len(element) not in [2, 3]:
                continue
            key, value = element[:2]
            if len(element) == 3 and isinstance(element[-1], str):
                comment = element[-1]
            else:
                comment = None
            cards.append((key, value, comment))

    else:
        log.warning(f'Header info must be {fits.Header}, {dict}, {list}, '
                    f'{set} or {tuple} type.  Received {header_info}')
        return cards

    result = []
    for (key, value, comment) in cards:
        key = str(key)
        if key.upper() in ['COMMENT', 'HISTORY'] or comment in ['', None]:
            comment = None
        else:
            comment = str(comment)
        result.append((key, value, comment))
    return result


def clear_numba_cache():  # pragma: no cover
    """
    Aggressively remove the numba cache for the sofia_scan package.

    This is necessary when numba does not pick up changes in currently cached
    functions.

    Returns
    -------
    None
    """
    toolkit_clear_numba_cache(module=sofia_scan_module)


def round_values(x):
    """
    Round any given value to the expected result.

    The function :func:`np.round` does not round values as one might expect.
    If int(x) is equal to an odd number, #.5 numbers will be rounded down
    rather than up (as expected).  This function attempts to fix that.

    Parameters
    ----------
    x : int or float or numpy.ndarray or units.Quantity
        The value to round.

    Returns
    -------
    rounded : int or numpy.ndarray (int) or units.Quantity
    """
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return numba_functions.round_value(x)

    is_quantity = isinstance(x, units.Quantity)
    if not is_quantity:
        x = np.asarray(x)

    if x.dtype == int:
        return x
    if is_quantity:
        unit, values = x.unit, x.value
    else:
        unit, values = None, x
    if values.shape == ():
        result = np.asarray(numba_functions.round_value(float(values)))
    else:
        result = numba_functions.round_array(values)
    if unit is None:
        return result
    return result * unit


def calculate_position_angle(lon1, lat1, lon2, lat2):
    """
    Position Angle (East of North) between two points on a sphere.

    Parameters
    ----------
    lon1 : float or units.Quantity
        The longitude of position 1.
    lat1: float or units.Quantity
        The latitude of position 1.
    lon2 : float or units.Quantity
        The longitude of position 2.
    lat2 : float or units.Quantity
        The latitude of position 2.

    Returns
    -------
    pa : units.Quantity
        The (positive) position angle of the vector pointing from position
        1 to position 2.  If any of the angles are arrays, this will contain
        an array following the appropriate `numpy` broadcasting rules.
    """
    delta_lon = lon2 - lon1
    cos_lat = np.cos(lat2)

    x = (np.sin(lat2) * np.cos(lat1)
         - cos_lat * np.sin(lat1) * np.cos(delta_lon))
    y = np.sin(delta_lon) * cos_lat

    pa = Angle(np.arctan2(y, x), units.Unit('radian'))
    pa = pa.wrap_at(360 * units.Unit('degree')).to('degree').value
    return pa * units.Unit('degree')


def get_comment_unit(comment, default=None):
    """
    See if a header comment contains a unit and return if found.

    In this case, header units are expected to be enclosed within round ()
    or square [] brackets.  The first valid unit found inside brackets will be
    returned with round brackets overruling square.

    Parameters
    ----------
    comment : str or None
        The header comment which may contain a unit.
    default : str or units.Unit, optional
        The unit to return if not found.

    Returns
    -------
    unit : None or units.Unit
    """
    if default is not None:
        default = units.Unit(default)

    if comment is None or len(comment) == 0:
        return default
    brackets = ['()', '[]']
    slash = '\\'
    for bracket in brackets:
        if bracket[0] not in comment:
            continue
        found = re.search(f'{slash}{bracket[0]}.*?{slash}{bracket[1]}',
                          comment)
        if found is None:
            continue

        try:
            unit = units.Unit(found[0][1:-1])
            return unit
        except ValueError:
            continue
    else:
        return default
