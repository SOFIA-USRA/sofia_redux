# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import recursive_dict_update


def test_standard():
    d = {
        'd1': {
            'a': 1,
            'b': 2
        },
        'd2': {
            'a': 1,
            'b': 2,
            'd3': {
                'a': 1,
                'b': 2
            }
        }
    }

    recursive_dict_update(
        d, {
            'd2': {
                'a': 3,
                'b': 4,
                'd3': {'c': -1}
            }
        }
    )

    assert d['d1']['a'] == 1
    assert d['d1']['b'] == 2
    assert d['d2']['a'] == 3
    assert d['d2']['b'] == 4
    assert d['d2']['d3']['a'] == 1
    assert d['d2']['d3']['b'] == 2
    assert d['d2']['d3']['c'] == -1
