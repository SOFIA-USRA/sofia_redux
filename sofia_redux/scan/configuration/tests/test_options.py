# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.configuration.options import Options
from sofia_redux.scan.configuration.conditions import Conditions

from astropy import log, units
from configobj import ConfigObj
import numpy as np
import pytest


def test_init():
    options = Options(allow_error=True, verbose=True)
    assert options.verbose
    assert options.allow_error
    options = Options(allow_error=False, verbose=False)
    assert not options.verbose
    assert not options.allow_error
    assert isinstance(options.options, dict)


def test_len():
    options = Options()
    options.options = None
    assert len(options) == 0
    options.options = {'abc': 'def'}
    assert len(options) == 1


def test_contains():
    options = Options()
    options.options = None
    assert 'foo' not in options
    options.options = {'foo': 'bar'}
    assert 'foo' in options


def test_getitem():
    options = Options()
    with pytest.raises(KeyError) as err:
        _ = options['foo']
    assert "foo" in str(err.value)
    options.options['foo'] = 'bar'
    assert options['foo'] == 'bar'


def test_delitem():
    options = Options()
    options.options['foo'] = 'bar'
    del options['foo']
    assert 'foo' not in options
    options.options = None
    with pytest.raises(KeyError) as err:
        del options['foo']
    assert "Options are not initialized" in str(err.value)


def test_size():
    options = Options()
    assert options.size == 0
    options.options['foo'] = 'bar'
    assert options.size == 1
    options.options['123'] = '456'
    assert options.size == 2


def test_is_empty():
    options = Options()
    assert options.is_empty
    options.options['foo'] = 'bar'
    assert not options.is_empty


def test_copy():
    options = Options()
    options.options['foo'] = 'bar'
    new = options.copy()
    assert new is not options
    assert new.options == options.options


def test_clear():
    options = Options()
    options.options['foo'] = 'bar'
    options.clear()
    assert options.size == 0


def test_get():
    options = Options()
    options.options['foo'] = 'bar'
    assert options.get('foo') == 'bar'
    assert options.get('baz', default=1) == 1


def test_get_string():
    options = Options()
    options.options['foo'] = 1
    assert options.get_string('foo') == '1'
    assert options.get_string('baz', default='abc') == 'abc'
    assert options.get_string('baz', default=None) is None


def test_get_bool():
    options = Options()
    options.options['foo'] = 1
    assert options.get_bool('foo') is True
    options.options['foo'] = 'f'
    assert options.get_bool('foo') is False
    assert options.get_bool('baz', default=None) is None


def test_get_int():
    options = Options()
    options.options['foo'] = '1'
    assert options.get_int('foo') == 1
    assert options.get_int('baz', default=None) is None


def test_get_float():
    options = Options()
    options.options['foo'] = '1.5e3'
    assert options.get_float('foo') == 1500
    assert options.get_float('baz', default=None) is None


def test_get_range():
    options = Options()
    options.options['foo'] = '1:5'
    r = options.get_range('foo')
    assert r.max == 5
    assert r.min == 1
    options.options['foo'] = '2-6'
    r = options.get_range('foo', is_positive=True)
    assert r.min == 2
    assert r.max == 6
    r = options.get_range('baz')
    assert not r.bounded
    assert options.get_range('baz', default=None) is None


def test_get_list():
    options = Options()
    options.options['foo'] = 1
    v = options.get_list('foo')
    assert isinstance(v, list) and len(v) == 1 and v[0] == 1
    options.options['foo'] = [1, 2, 3]
    assert options.get_list('foo') == [1, 2, 3]
    assert options.get_list('baz') == []
    assert options.get_list('baz', default=None) == []
    assert options.get_list('baz', default=1) == [1]


def test_get_string_list():
    options = Options()
    options.options['foo'] = '1,2,3'
    assert options.get_string_list('foo') == ['1', '2', '3']
    assert options.get_string_list('foo', delimiter='\t') == ['1,2,3']
    assert options.get_string_list('baz', default=[]) == []
    assert options.get_string_list('baz', default=None) is None


def test_get_int_list():
    options = Options()
    options.options['foo'] = '1,2,3'
    assert options.get_int_list('foo') == [1, 2, 3]
    options.options['foo'] = '1-5'
    assert options.get_int_list('foo', is_positive=True) == [1, 2, 3, 4, 5]
    options.options['foo'] = '1,2,5:7'
    assert options.get_int_list('foo') == [1, 2, 5, 6, 7]
    assert options.get_int_list('baz') is None
    options.options['foo'] = '1;2;3'
    assert options.get_int_list('foo', delimiter=';') == [1, 2, 3]


def test_get_float_list():
    options = Options()
    options.options['foo'] = '1.5,2,3'
    assert options.get_float_list('foo') == [1.5, 2, 3]
    options.options['foo'] = '2.5;3;4'
    assert options.get_float_list('foo', delimiter=';') == [2.5, 3, 4]
    assert options.get_float_list('baz') is None


def test_get_dms_angle():
    options = Options()
    options.options['foo'] = '90:30:0'
    assert options.get_dms_angle('foo') == 90.5 * units.Unit('degree')
    angle = options.get_dms_angle('baz')
    assert np.isnan(angle)
    assert angle.unit == 'degree'


def test_get_hms_angle():
    options = Options()
    options.options['foo'] = '12h30m0s'
    assert options.get_hms_time('foo') == 12.5 * units.Unit('hour')
    assert options.get_hms_time('foo', angle=True) == 12.5 * units.Unit(
        'hourangle')
    angle = options.get_hms_time('baz')
    assert np.isnan(angle) and angle.unit == 'hour'


def test_get_sign():
    options = Options()
    options.options['foo'] = '+'
    assert options.get_sign('foo') == 1
    options.options['foo'] = '-'
    assert options.get_sign('foo') == -1
    assert options.get_sign('baz') == 0


def test_handle_error():
    options = Options()
    options.allow_error = False
    options.verbose = False
    with pytest.raises(ValueError) as err:
        options.handle_error('foo')
    assert 'foo' in str(err.value)

    with pytest.raises(IndexError) as err:
        options.handle_error('bar', error_class=IndexError)
    assert 'bar' in str(err.value) and err.type == IndexError

    options.allow_error = True
    options.verbose = True
    with log.log_to_list() as log_list:
        options.handle_error('baz')
    assert len(log_list) == 1
    record = log_list[0]
    assert record.levelno == 30
    assert record.msg == 'baz'


def test_update():
    options = Options()
    options.allow_error = False
    assert len(options.options) == 0

    with pytest.raises(ValueError) as err:
        options.update(None)
    assert "Could not update with options" in str(err.value)

    options.allow_error = True
    options.verbose = True
    with log.log_to_list() as log_list:
        options.update(None)
    assert len(log_list) == 1
    record = log_list[0]
    assert record.levelno == 30
    assert "Could not update with options" in record.msg

    options.update({'foo': {'bar': {'a': 1, 'b': 2}}})
    assert len(options) == 1
    assert options.options['foo']['bar']['a'] == 1
    assert options.options['foo']['bar']['b'] == 2

    options.update({'foo': {'bar': {'a': 3}}})
    assert options.options['foo']['bar']['a'] == 3
    assert options.options['foo']['bar']['b'] == 2


def test_stringify():
    options = Options()
    options.update({'foo': {'bar': {'a': 1, 'b': 2, 'c': [3, 4, 5]}}})
    result = options.stringify(options.options)
    assert result is not options.options
    assert options.options['foo']['bar']['a'] == 1
    assert options.options['foo']['bar']['b'] == 2
    assert result['foo']['bar']['a'] == '1'
    assert result['foo']['bar']['b'] == '2'
    assert result['foo']['bar']['c'] == ['3', '4', '5']


def test_options_to_dict():
    options = Options()
    c = ConfigObj({'a': 1})
    assert options.options_to_dict(c) is c
    d = {'a': 1}
    assert options.options_to_dict(d) is d

    assert options.options_to_dict(1) is None
    assert options.options_to_dict('a', add_singular=False) is None
    assert options.options_to_dict('a', add_singular=True) == {'add': 'a'}
    assert options.options_to_dict('a=b') == {'a': 'b'}
    assert options.options_to_dict('a=b=c', add_singular=True) is None


def test_merge_options():
    d = Conditions()
    current = ConfigObj()
    new = ConfigObj()
    new['str'] = 'value1'
    new['list'] = ['a', 'b', 'c']
    new['dict'] = {'foo': 'bar'}
    new['add'] = 'single_add'
    new['forget'] = ['a', 'b']
    d.merge_options(current, new)
    assert current == new

    new = ConfigObj()
    new['add'] = 'another_add'
    d.merge_options(current, new)
    assert current['add'] == ['single_add', 'another_add']

    new = ConfigObj()
    new['str'] = {'suboption1': 'foo'}
    d.merge_options(current, new)
    assert current['str'] == {'value': 'value1', 'suboption1': 'foo'}

    new['str'] = {'suboption1': 'bar'}
    d.merge_options(current, new)
    assert current['str'] == {'value': 'value1', 'suboption1': 'bar'}

    new = ConfigObj()
    new['dict'] = 'a value'
    d.merge_options(current, new)
    assert current['dict'] == {'foo': 'bar', 'value': 'a value'}
