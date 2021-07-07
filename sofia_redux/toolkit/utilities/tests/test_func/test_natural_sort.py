# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import natural_sort


def test_success():
    in_order = ['g%i' % i for i in range(21)]
    badsort = sorted(in_order)
    goodsort = natural_sort(in_order)
    assert not in_order == badsort
    assert in_order == goodsort
