# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Grism Slit Correction Reduction pipeline steps"""

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.flitecam
    assert sofia_redux.instruments.flitecam
except ImportError:
    raise SOFIAImportError('FLITECAM modules not installed')


from sofia_redux.pipeline.sofia.flitecam_reduction import FLITECAMReduction
from sofia_redux.pipeline.sofia.flitecam_spectroscopy_reduction \
    import FLITECAMSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.flitecam_slitcorr_parameters \
    import FLITECAMSlitcorrParameters
from sofia_redux.pipeline.sofia.forcast_slitcorr_reduction \
    import FORCASTSlitcorrReduction


class FLITECAMSlitcorrReduction(FLITECAMSpectroscopyReduction,
                                FORCASTSlitcorrReduction):
    r"""
    FORCAST spesctroscopic slit correction reduction steps.

    This reduction object defines specialized reduction steps
    for generating slit correction calibration data from spectroscopic
    input files.  It is selected by the SOFIA chooser only if a top-level
    configuration flag is supplied (slitcorr=True).  The final
    output product from this reduction is a FITS file (\*SCR\*.fits)
    with PRODTYPE = 'slit_correction'.  This file can be supplied to the
    standard spectroscopic pipeline, at the make_profiles step,
    to specify a new slit response correction.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to calibration
        self.name = 'Slit correction'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'rectify': 'test_rectified_image',
             'make_profiles': 'rectified_image',
             'extract_median_spectra': 'median_spectra',
             'normalize': 'normalized_image',
             'make_slitcorr': 'slit_correction'})
        self.prodnames.update(
            {'rectified_image': 'RIM',
             'median_spectra': 'MSM',
             'normalized_image': 'NIM',
             'slit_correction': 'SCR'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['check_header', 'correct_linearity', 'make_image',
                       'stack_dithers', 'make_profiles',
                       'locate_apertures', 'extract_median_spectra',
                       'normalize', 'make_slitcorr']
        self.processing_steps.update(
            {'make_profiles': 'Make Profiles',
             'extract_median_spectra': 'Extract Median Spectra',
             'normalize': 'Normalize Response',
             'make_slitcorr': 'Make Slit Correction'})

    def load(self, data, param_class=None):
        """Call parent load, with spatcal parameters."""
        FLITECAMReduction.load(
            self, data, param_class=FLITECAMSlitcorrParameters)
