# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Grism Spatcal Reduction pipeline steps"""

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.flitecam
    assert sofia_redux.instruments.flitecam
except ImportError:
    raise SOFIAImportError('FLITECAM modules not installed')


from sofia_redux.pipeline.sofia.flitecam_reduction import FLITECAMReduction
from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
    import FLITECAMSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_spatcal_parameters \
    import FLITECAMSpatcalParameters
from sofia_redux.pipeline.sofia.forcast_spatcal_reduction \
    import FORCASTSpatcalReduction


class FLITECAMSpatcalReduction(FLITECAMSpectroscopyReduction,
                               FORCASTSpatcalReduction):
    r"""
    FLITECAM spectroscopic spatial calibration reduction steps.

    This reduction object defines specialized reduction steps
    for generating spatial calibration data from spectroscopic
    input files.  It is selected by the SOFIA chooser only if a
    top-level configuration flag is supplied (spatcal=True).  The
    final output product from this reduction is a FITS file (\*SCL\*.fits)
    with PRODTYPE = 'spatcal'.  This file can be supplied to the
    standard spectroscopic pipeline, at the make_profiles step,
    to specify a new spatial calibration.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to calibration
        self.name = 'Spatcal'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'make_profiles': 'spatial_profile',
             'fit_traces': 'traces_fit',
             'rectify': 'rectified_image'})
        self.prodnames.update(
            {'spatial_profile': 'PRF',
             'traces_fit': 'TFT',
             'rectified_image': 'RIM'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['check_header', 'correct_linearity', 'make_image',
                       'stack_dithers', 'make_profiles',
                       'locate_apertures', 'trace_continuum',
                       'fit_traces', 'rectify']
        self.processing_steps.update({'fit_traces': 'Fit Trace Positions',
                                      'rectify': 'Verify Rectification'})

    def load(self, data, param_class=None):
        """Call parent load, with spatcal parameters."""
        FLITECAMReduction.load(
            self, data, param_class=FLITECAMSpatcalParameters)
