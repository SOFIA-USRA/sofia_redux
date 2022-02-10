# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.mounts import Mount


def test_mounts():
    mount = Mount
    names = list(mount.__dict__.keys())
    for name in ['UNKNOWN', 'CASSEGRAIN', 'GREGORIAN', 'PRIME_FOCUS',
                 'LEFT_NASMYTH', 'RIGHT_NASMYTH', 'NASMYTH_COROTATING']:
        assert name in names
