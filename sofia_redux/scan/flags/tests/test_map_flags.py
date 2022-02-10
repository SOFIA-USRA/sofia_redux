# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.array_flags import ArrayFlags
from sofia_redux.scan.flags.map_flags import MapFlags


def test_array_flags():
    assert ArrayFlags.letters == MapFlags.letters
    assert ArrayFlags.descriptions == MapFlags.descriptions
    for i in range(3):
        assert (
            ArrayFlags.convert_flag(i).value == MapFlags.convert_flag(i).value)


def test_init():
    f = MapFlags()
    assert isinstance(f, MapFlags)
