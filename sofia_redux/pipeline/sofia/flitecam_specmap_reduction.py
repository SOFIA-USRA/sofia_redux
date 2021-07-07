# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Quicklook pipeline steps"""

from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
    import FORCASTSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_quicklook_parameters \
    import FLITECAMQuicklookParameters


class FLITECAMSpecmapReduction(FORCASTSpectroscopyReduction):
    """
    FLITECAM quicklook reduction steps.

    This reduction object borrows from the FORCAST pipeline to
    make an image or spectral map for final FLITECAM data products.
    It is not a full FLITECAM reduction pipeline.

    See `FORCASTSpectroscopyReduction` for more information.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to FLITECAM
        self.instrument = 'FLITECAM'

    def load(self, data, param_class=None):
        """Call parent load, with FLITECAM parameters."""
        FORCASTSpectroscopyReduction.load(
            self, data, param_class=FLITECAMQuicklookParameters)

        # override recipe for last step only
        self.recipe = ['specmap']
