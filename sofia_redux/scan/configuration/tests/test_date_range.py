# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.time import Time
import pytest

from sofia_redux.scan.configuration.dates import DateRange


@pytest.fixture
def date_range():
    """
    Return a DatRange test object.

    Returns
    -------
    DateRange
    """
    return DateRange('2021-12-09--2021-12-10')


def test_init(date_range):
    d = date_range
    assert d.range[0] == Time('2021-12-09', format='isot', scale='utc')
    assert d.range[1] == Time('2021-12-10', format='isot', scale='utc')


def test_copy(date_range):
    d = date_range
    d2 = d.copy()
    assert d.range[0] == d2.range[0] and d.range[0] is not d2.range[0]
    assert d.range[1] == d2.range[1] and d.range[1] is not d2.range[1]


def test_str(date_range):
    d = date_range
    assert str(d) == '2021-12-09T00:00:00.000--2021-12-10T00:00:00.000'
    # Test mjd dates
    d.parse_range('59922.0--60000')
    assert str(d) == '2022-12-09T00:00:00.000--2023-02-25T00:00:00.000'
    # Test unbounded ranges
    d.parse_range('*--2021-12-10')
    assert str(d) == '***********************--2021-12-10T00:00:00.000'
    d.parse_range('2021-12-10--*')
    assert str(d) == '2021-12-10T00:00:00.000--***********************'
    d.parse_range('*--*')
    assert str(d) == '***********************--***********************'


def test_contains(date_range):
    d = date_range
    assert '2021-12-09T12:00:00' in d
    assert '2021-12-08' not in d
    assert '2022-01-01' not in d


def test_parse_range(date_range):
    d = date_range
    with pytest.raises(ValueError) as err:
        d.parse_range('1--2--3')
    assert 'Cannot parse time range' in str(err.value)

    # Test single time
    d.parse_range('2022-12-09')
    t = Time('2022-12-09', format='isot', scale='utc')
    t1 = Time('2022-12-10', format='isot', scale='utc')
    assert d.range[0] == t
    assert d.range[1] == d.range[0]

    # Test unbounded time
    d.parse_range('2022-12-09--*')
    assert d.range[0] == t
    assert d.range[1] is None
    d.parse_range('*--2022-12-09')
    assert d.range[0] is None
    assert d.range[1] == t

    # Test mjd
    d.parse_range('59922.0--59923')
    assert d.range[0] == t
    assert d.range[1] == t1

    # Parse bad date
    with pytest.raises(ValueError) as err:
        d.parse_range('abc--def')
    assert 'Could not parse date' in str(err.value)


def test_to_time():
    t = Time('2021-10-09T12:30:45')
    assert DateRange.to_time(t) == t
    assert DateRange.to_time(t.isot) == t
    # There may be a slight floating point error here...
    assert DateRange.to_time(t.mjd).mjd == t.mjd
    with pytest.raises(ValueError) as err:
        DateRange.to_time({})
    assert 'Input must be of type' in str(err.value)
