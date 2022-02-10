# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import pytest

from sofia_redux.scan.configuration.conditions import Conditions
from sofia_redux.scan.configuration.configuration import Configuration


def test_init():
    d = Conditions(allow_error=False, verbose=False)
    assert isinstance(d, Conditions)
    assert not d.allow_error
    assert not d.verbose
    d = Conditions(allow_error=True, verbose=True)
    assert d.allow_error
    assert d.verbose


def test_len(initialized_conditions):
    assert len(initialized_conditions) == 5
    d = Conditions()
    d.options = None
    assert len(d) == 0


def test_set_item():
    d = Conditions()
    d['cond1=True'] = 'foo'
    assert d['cond1=True'] == {'add': 'foo'}


def test_str():
    d = Conditions()
    assert str(d) == 'Contains 0 conditions.'
    d['cond1=True'] = 'foo'
    assert str(d) == 'Contains 1 condition.'


def test_repr():
    d = Conditions()
    s = repr(d)
    assert 'sofia_redux.scan.configuration.conditions.Conditions' in s
    assert 'Contains 0 conditions.' in s


def test_size(initialized_conditions):
    assert initialized_conditions.size == 5


def test_copy(initialized_conditions):
    d = initialized_conditions
    d2 = d.copy()
    assert d.options is not d2.options
    assert d.options == d2.options


def test_set():
    d = Conditions()
    with pytest.raises(ValueError) as err:
        d.set('cond1', None)
    assert 'Could not parse condition' in str(err.value)
    d.allow_error = True
    d.set('cond1', None)
    assert d.size == 0

    d.set('cond1=True', 'set_this')
    assert d.options['cond1=True'] == {'add': 'set_this'}


def test_update():
    d = Conditions()
    d.allow_error = True
    options = {}
    d.update(options)
    assert d.size == 0
    options['conditionals'] = 1

    with log.log_to_list() as log_list:
        d.update(options)
    assert "Supplied conditionals" in log_list[0].msg

    switches = {'a=1': {'add': 'foo'}, 'a>2': {'forget': 'foo'}}
    options['conditionals'] = switches
    d.update(options)
    assert d.options == switches


def test_check_requirement():
    c = Configuration()
    c.options['str'] = 'value'
    c.options['num'] = '5'
    c.options['bool'] = 'True'
    d = Conditions()
    d.allow_error = True

    with log.log_to_list() as log_list:
        assert not d.check_requirement(c, 'str=')
    assert 'Bad conditional requirement' in log_list[0].msg

    assert d.check_requirement(c, 'bool')
    assert not d.check_requirement(c, 'missing')

    # Check string value equal and not equal
    assert d.check_requirement(c, 'str==value')
    assert d.check_requirement(c, 'str=value')
    assert not d.check_requirement(c, 'str=wrong')
    assert d.check_requirement(c, 'str!=wrong')

    # Check number value equal and not equal
    assert d.check_requirement(c, 'num=5.0')
    assert d.check_requirement(c, 'num==5')
    assert d.check_requirement(c, 'num!=4')
    assert d.check_requirement(c, 'num!=value')
    assert not d.check_requirement(c, 'num=4')

    # Check inequality operators
    test_num = [4, 5, 6]
    for num in test_num:
        expected = 5 > num
        assert d.check_requirement(c, f'num>{num}') is expected
        expected = 5 < num
        assert d.check_requirement(c, f'num<{num}') is expected
        expected = 5 >= num
        assert d.check_requirement(c, f'num>={num}') is expected
        expected = 5 <= num
        assert d.check_requirement(c, f'num<={num}') is expected

    # Check missing value
    assert not d.check_requirement(c, 'foo<1')

    # Check bad inequality
    assert not d.check_requirement(c, 'num<a')


def test_get_met_conditions():
    c = Configuration()
    d = Conditions()
    d.options['always'] = 'add=foo'
    d.options['num>4'] = {'add': 'bar', 'forget': 'foo', 'baz': '3'}
    d.options['num>6'] = {'add': 'high_value'}
    c.options['num'] = '5'
    apply = d.get_met_conditions(c)
    assert apply == {'always': [{'add': 'foo'}],
                     'num>4': [{'add': 'bar', 'forget': 'foo', 'baz': '3'}]}

    # Test bad non-requirement condition
    d.options['bad'] = 'add='
    d.allow_error = True
    with log.log_to_list() as log_list:
        apply = d.get_met_conditions(c)
    assert 'Bad condition' in log_list[0].msg
    assert apply == {'always': [{'add': 'foo'}],
                     'num>4': [{'add': 'bar', 'forget': 'foo', 'baz': '3'}]}


def test_process_conditionals():
    c = Configuration()
    d = Conditions()
    c.options['num'] = '4'
    d.options['num>4'] = {'add': 'foo'}
    d.options['foo'] = {'add': 'bar'}
    d.process_conditionals(c)
    assert 'foo' not in c
    assert 'bar' not in c
    c.put('num', '5')
    d.process_conditionals(c)
    assert 'foo' in c
    assert 'bar' in c


def test_update_configuration():
    c = Configuration()
    c.options['num'] = '5'
    d = Conditions()
    assert not d.update_configuration(c)
    seen = set([])
    d.options['num>3'] = {'add': 'foo'}
    d.options['num>4'] = {'add': 'bar'}
    d.options['foo'] = {'key1': 'value1'}
    d.options['bar'] = {'key2': 'value2', 'key3': 'value3'}
    assert d.update_configuration(c, seen=seen)
    assert c['foo']
    assert c['bar']
    assert 'key1' not in c and 'key2' not in c and 'key3' not in c
    assert d.update_configuration(c, seen=seen)
    assert (c['key1'] == 'value1' and c['key2'] == 'value2'
            and c['key3'] == 'value3')
    assert not d.update_configuration(c, seen=seen)
