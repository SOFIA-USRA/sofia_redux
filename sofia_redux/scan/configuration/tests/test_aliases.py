# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.configuration.aliases import Aliases
from sofia_redux.scan.configuration.configuration import Configuration


@pytest.fixture
def initialized_aliases(initialized_configuration):
    return initialized_configuration.aliases


def test_init():
    a = Aliases()
    assert isinstance(a, Aliases)
    assert a.size == 0


def test_copy(initialized_aliases):
    a = initialized_aliases
    b = a.copy()
    assert a.options == b.options
    assert a.options is not b.options


def test_unalias_dot_string(initialized_aliases):
    a = initialized_aliases
    assert a.unalias_dot_string('o1s1') == 'options1.suboptions1'
    a = Aliases()
    assert a.unalias_dot_string('foo.bar') == 'foo.bar'
    with pytest.raises(ValueError) as err:
        a.unalias_dot_string(1)
    assert "require string input" in str(err.value)
    a = Aliases(allow_error=True)
    assert a.unalias_dot_string(1) is None


def test_unalias_value(fits_configuration):
    c = fits_configuration
    a = c.aliases
    assert a.unalias_value(c, 1) == 1
    assert a.unalias_value(c, 'foo') == 'foo'
    assert a.unalias_value(c, '{?bad_format') == '{?bad_format'
    assert a.unalias_value(c, '{?missing}') == '{?missing}'
    assert a.unalias_value(c, '{?fits.OBSRA}') == '12.5'


def test_unalias_branch(fits_configuration):
    c = fits_configuration
    a = c.aliases
    a.allow_error = True
    assert a.unalias_branch(1) is None
    assert a.unalias_branch({'foo': 1, 'bar': 2}) is None
    o = a.unalias_branch({'o2s2': {'foo': 'bar'}})
    assert o == {'options2': {'suboptions2': {'foo': 'bar'}}}
    assert a.unalias_branch({'foo': 'bar'}) == {'foo': 'bar'}


def test_unalias_branch_values(fits_configuration):
    c = fits_configuration
    a = c.aliases
    branch = {'references': {'refdec': '{?fits.OBSDEC}',
                             'refsrc': '{?fits.OBJECT}'}}
    o = a.unalias_branch_values(c, branch)
    assert o == {'references': {'refdec': '45.0', 'refsrc': 'Asphodel'}}


def test_update():
    a = Aliases()
    a.update({'aliases': {'foo': 'bar.baz'}})
    assert a.options == {'foo': 'bar.baz'}


def test_resolve_configuration():
    c = Configuration()
    a = Aliases()
    a.options['a'] = 'options1.suboptions1.a'
    a.options['ref1'] = 'options1.suboptions1'
    c.options['a'] = '1'
    c.options['ref1'] = {'foo': 'bar'}
    a.resolve_configuration(c)
    del c.options['configpath']
    assert c.options == {'options1': {'suboptions1': {'a': '1', 'foo': 'bar'}}}


def test_call(fits_configuration):
    c = fits_configuration
    a = c.aliases
    assert a('o1s1') == 'options1.suboptions1'
    assert a({'o1s1': {'foo': 'bar'}}) == {
        'options1': {'suboptions1': {'foo': 'bar'}}}
    with pytest.raises(ValueError):
        a(1)
