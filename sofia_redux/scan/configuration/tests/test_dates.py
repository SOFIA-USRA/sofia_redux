# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.configuration.dates import DateRangeOptions


@pytest.fixture
def date_options():
    d = DateRangeOptions()
    d['2021-12-9--2021-12-10'] = {'add': 'today'}
    d['2021-12-9T12:00:00--2021-12-10T13:00:00'] = {'add': 'tea_time'}
    return d


def test_init():
    d = DateRangeOptions(allow_error=False, verbose=False)
    assert not (d.allow_error or d.verbose)
    d = DateRangeOptions(allow_error=True, verbose=True)
    assert d.allow_error and d.verbose


def test_copy(date_options):
    d = date_options
    d.options['foo'] = 'bar'
    d2 = d.copy()

    for date, ranges in d.ranges.items():
        r1, r2 = ranges.range, d2.ranges[date].range
        assert r1 == r2 and r1 is not r2

    assert d.options['foo'] == d2.options['foo']


def test_set_get_item():
    d = DateRangeOptions()
    d['2021-12-9--2021-12-10'] = {'add': 'today'}
    assert d['2021-12-9'] == {'add': 'today'}


def test_str(date_options):
    d = date_options
    assert str(d) == ('Available date ranges (UTC):\n'
                      '2021-12-09T00:00:00.000--2021-12-10T00:00:00.000\n'
                      '2021-12-09T12:00:00.000--2021-12-10T13:00:00.000')


def test_repr(date_options):
    d = date_options
    assert "DateRangeOptions object" in repr(d)


def test_clear(date_options):
    d = date_options
    d.options['foo'] = 'bar'
    d.clear()
    assert d.size == 0
    assert len(d.ranges) == 0


def test_update():
    d = DateRangeOptions()
    d.allow_error = True
    d.update({})
    assert len(d.ranges) == 0

    with log.log_to_list() as log_list:
        d.update({'date': None})
    assert 'options could not be parsed' in log_list[0].msg

    options = {'2021-12-25--2021-12-26': {'add': 'xmas'},
               'a_bad_date': None}
    options = {'date': options}
    with log.log_to_list() as log_list:
        d.update(options)
    assert 'Could not parse options for date [a_bad_date]' in log_list[0].msg
    assert d['2021-12-25'] == {'add': 'xmas'}


def test_get(date_options):
    d = date_options
    assert d.get('2008-07-30',
                 default={'hello': 'there'}) == {'hello': 'there'}
    assert d.get('2021-12-09T12:30:00') == {'add': ['today', 'tea_time']}


def test_set(date_options):
    d = date_options
    d.allow_error = True
    with log.log_to_list() as log_list:
        d.set('bad date', {'add': 'foo'})
    assert 'Invalid date range key' in log_list[0].msg
    d.set('2021-12-08--2021-12-09', {'add': 'yesterday'})
    d.set('2021-12-08--2021-12-09', {'forget': 'tomorrow'})
    assert d['2021-12-08'] == {'add': 'yesterday', 'forget': 'tomorrow'}


def test_set_date(date_options):
    d = date_options
    c = Configuration()
    d.set_date(c, '2021-12-09')
    assert c.get_bool('today')
    assert d.current_date == '2021-12-09'
