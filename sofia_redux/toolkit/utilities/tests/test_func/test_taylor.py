# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import taylor


def test_standard():

    assert list(taylor(0, 0)) == [()]
    assert list(taylor(0, 1)) == [(0,)]
    assert list(taylor(0, 2)) == [(0, 0)]
    assert list(taylor(1, 1)) == [(0,), (1,)]
    assert list(taylor(1, 2)) == [(0, 0), (0, 1), (1, 0)]
    assert list(taylor(1, 3)) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    assert list(taylor(3, 1)) == [(0,), (1,), (2,), (3,)]
    assert list(taylor(3, 2)) == [(0, 0), (0, 1), (0, 2), (0, 3),
                                  (1, 0), (1, 1), (1, 2),
                                  (2, 0), (2, 1),
                                  (3, 0)]
