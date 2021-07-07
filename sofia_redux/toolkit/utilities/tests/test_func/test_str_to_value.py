# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import str_to_value


def test_standard():
    assert str_to_value('a') == 'a'
    i = str_to_value('4')
    assert isinstance(i, int) and i == 4
    i = str_to_value('4.0')
    assert isinstance(i, float) and i == 4.0
    i = str_to_value('1e2')
    assert isinstance(i, float) and i == 100
    i = str_to_value('1.23e2')
    assert isinstance(i, float) and i == 123
