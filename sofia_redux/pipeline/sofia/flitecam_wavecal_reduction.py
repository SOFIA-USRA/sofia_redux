# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Grism Wavecal Reduction pipeline steps"""

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.flitecam
    assert sofia_redux.instruments.flitecam
except ImportError:
    raise SOFIAImportError('FLITECAM modules not installed')


from sofia_redux.pipeline.sofia.flitecam_reduction import FLITECAMReduction
from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
    import FLITECAMSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_wavecal_parameters \
    import FLITECAMWavecalParameters
from sofia_redux.pipeline.sofia.forcast_wavecal_reduction \
    import FORCASTWavecalReduction


class FLITECAMWavecalReduction(FLITECAMSpectroscopyReduction,
                               FORCASTWavecalReduction):
    r"""
    FLITECAM wavelength calibration reduction steps.

    This reduction object defines specialized reduction steps
    for generating wavelength calibration data from spectroscopic
    input files.  It is selected by the SOFIA chooser only if a
    top-level configuration flag is supplied (wavecal=True).  The
    final output product from this reduction is a FITS file (\*WCL\*.fits)
    with PRODTYPE = 'wavecal'.  This file can be supplied to the
    standard spectroscopic pipeline, at the make_profiles step,
    to specify a new wavelength calibration.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to calibration
        self.name = 'Wavecal'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'make_profiles': 'spatial_profile',
             'extract_summed_spectrum': 'summed_spectrum',
             'identify_lines': 'lines_identified',
             'reidentify_lines': 'lines_reidentified',
             'fit_lines': 'lines_fit',
             'rectify': 'rectified_image'})
        self.prodnames.update(
            {'spatial_profile': 'PRF',
             'summed_spectrum': 'SSM',
             'lines_identified': 'LID',
             'lines_reidentified': 'LRD',
             'lines_fit': 'LFT',
             'rectified_image': 'RIM'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['check_header', 'correct_linearity', 'make_image',
                       'stack_dithers', 'make_profiles',
                       'extract_summed_spectrum',
                       'identify_lines', 'reidentify_lines',
                       'fit_lines', 'rectify']
        self.processing_steps.update(
            {'extract_summed_spectrum': 'Extract First Spectrum',
             'identify_lines': 'Identify Lines',
             'reidentify_lines': 'Reidentify Lines',
             'fit_lines': 'Fit Lines',
             'rectify': 'Verify Rectification'})

    def load(self, data, param_class=None):
        """Call parent load, with spatcal parameters."""
        FLITECAMReduction.load(
            self, data, param_class=FLITECAMWavecalParameters)
