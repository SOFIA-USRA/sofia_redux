# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = ['Application', 'Chooser', 'Configuration',
           'Interface', 'Parameters', 'ParameterSet',
           'Pipe', 'Reduction', 'Viewer']

if not _ASTROPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.

    # automodapi docs require imports here
    from sofia_redux.pipeline.application import Application
    from sofia_redux.pipeline.chooser import Chooser
    from sofia_redux.pipeline.configuration import Configuration
    from sofia_redux.pipeline.interface import Interface
    from sofia_redux.pipeline.parameters import Parameters, ParameterSet
    from sofia_redux.pipeline.pipe import Pipe
    from sofia_redux.pipeline.reduction import Reduction
    from sofia_redux.pipeline.viewer import Viewer
