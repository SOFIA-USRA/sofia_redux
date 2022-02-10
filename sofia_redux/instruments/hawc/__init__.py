# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = ['DataFits', 'DataParent', 'DataText',
           'StepLoadAux', 'StepMIParent', 'StepMOParent',
           'StepParent']

if not _ASTROPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.
    from sofia_redux.instruments.hawc.datafits import *
    from sofia_redux.instruments.hawc.dataparent import *
    from sofia_redux.instruments.hawc.datatext import *
    from sofia_redux.instruments.hawc.steploadaux import *
    from sofia_redux.instruments.hawc.stepmiparent import *
    from sofia_redux.instruments.hawc.stepmoparent import *
    from sofia_redux.instruments.hawc.stepparent import *
