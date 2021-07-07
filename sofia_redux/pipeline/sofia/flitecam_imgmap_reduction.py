# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Quicklook pipeline steps"""

from sofia_redux.pipeline.sofia.forcast_imaging_reduction \
    import FORCASTImagingReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_quicklook_parameters \
    import FLITECAMQuicklookParameters


class FLITECAMImgmapReduction(FORCASTImagingReduction):
    """
    FLITECAM quicklook reduction steps.

    This reduction object borrows from the FORCAST pipeline to
    make an image or spectral map for final FLITECAM data products.
    It is not a full FLITECAM reduction pipeline.

    See `FORCASTImagingReduction` for more information.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to FLITECAM
        self.instrument = 'FLITECAM'

    def load(self, data, param_class=None):
        """Call parent load, with FLITECAM parameters."""
        FORCASTImagingReduction.load(
            self, data, param_class=FLITECAMQuicklookParameters)

        # override recipe for last step only
        self.recipe = ['imgmap']
