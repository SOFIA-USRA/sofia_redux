# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.array_flags import ArrayFlags

__all__ = ['MapFlags']


class MapFlags(ArrayFlags):
    def __init__(self):
        super().__init__()
