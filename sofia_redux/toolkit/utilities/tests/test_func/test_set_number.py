# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import setnumber


def test_standard():
    y = setnumber(2.0)
    assert isinstance(y, int) and y == 2
    y = setnumber('a', default=2.0)
    assert isinstance(y, int) and y == 2
    y = setnumber('a', default=2.0, dtype=float)
    assert isinstance(y, float) and y == 2


def test_ranges():
    y = setnumber(2, minval=1)
    assert isinstance(y, int) and y == 2
    y = setnumber(2, minval=3)
    assert isinstance(y, int) and y == 3
    y = setnumber(2, maxval=3)
    assert isinstance(y, int) and y == 2
    y = setnumber(2, maxval=1)
    assert isinstance(y, int) and y == 1
