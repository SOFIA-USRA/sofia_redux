# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = ['Eye', 'log']

if not _ASTROPY_SETUP_:   # noqa
    from sofia_redux.visualization.utils.logger import _init_log
    log = _init_log()

    from sofia_redux.visualization.eye import Eye
