# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.configuration.objects import ObjectOptions


@pytest.fixture
def object_options():
    o = ObjectOptions()
    o.allow_error = True
    o.set('source1', {'add': 'foo'})
    o.set('source2', {'add': 'bar'})
    o.set('source3', {'options': {'sub_options': {'a': '1'}}})
    o.set('source4', {'options': '2'})
    o.applied_objects.append('source1')
    return o


def test_init():
    o = ObjectOptions(allow_error=True, verbose=True)
    assert o.allow_error and o.verbose
    o = ObjectOptions(allow_error=False, verbose=False)
    assert not (o.allow_error or o.verbose)


def test_copy(object_options):
    o = object_options
    o2 = o.copy()
    assert o.options == o2.options and o.options is not o2.options
    assert o2.applied_objects == []


def test_clear(object_options):
    o = object_options
    o.clear()
    assert o.size == 0
    assert o.applied_objects == []


def test_getitem(object_options):
    o = object_options
    assert o['source3'] == {'options': {'sub_options': {'a': '1'}}}


def test_setitem(object_options):
    o = object_options
    with log.log_to_list() as log_list:
        o['foo'] = None
    assert 'Could not parse source options' in log_list[0].msg
    o['atherius'] = {'blacklist': 'oblivion'}
    assert o['atherius'] == {'blacklist': 'oblivion'}


def test_str(object_options):
    assert str(object_options) == (
        "Available object configurations:\nsource1\nsource2\nsource3\nsource4")


def test_repr(object_options):
    o = object_options
    assert str(o) in repr(o)
    assert "ObjectOptions object" in repr(o)


def test_get(object_options):
    o = object_options
    assert o.get('foo') == {}
    assert o.get('foo', default={'bar': 'baz'}) == {'bar': 'baz'}
    assert o.get('source1') == {'add': 'foo'}


def test_set(object_options):
    o = object_options
    with log.log_to_list() as log_list:
        o.set('foo', None)
    assert 'Could not parse options' in log_list[0].msg
    o.set('foo', {'add': 'bar'})
    assert o['foo'] == {'add': 'bar'}


def test_update(object_options):
    o = object_options
    o.update({})
    assert o.size == 4
    with log.log_to_list() as log_list:
        o.update({'object': None})
    assert 'Could not parse object options' in log_list[0].msg
    o.update({'object': {'foo': {'add': 'bar'}}})
    assert o['foo'] == {'add': 'bar'}


def test_set_object(object_options):
    o = object_options
    c = Configuration()
    o.set_object(c, 'dne')
    assert c.size == 1
    o.set_object(c, 'source3')
    assert c['options.sub_options.a'] == '1'
