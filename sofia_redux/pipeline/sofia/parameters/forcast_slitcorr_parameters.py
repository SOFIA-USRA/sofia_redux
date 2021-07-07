# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST parameter sets."""

from sofia_redux.pipeline.sofia.parameters.forcast_parameters import DEFAULT
from sofia_redux.pipeline.sofia.parameters.forcast_spectroscopy_parameters \
    import SPECTRAL_DEFAULT, FORCASTSpectroscopyParameters

SLITCORR_DEFAULT = {
    'stack': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'add_frames',
         'hidden': True,
         'name': "Add all frames instead of subtracting",
         'value': True,
         'description': 'Generates a sky image, for diagnostic purposes.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'jbclean',
         'name': "Apply 'jailbar' correction",
         'value': True,
         'description': 'If set, the jailbar pattern will be '
                        'removed after stacking.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bgscale',
         'hidden': True,
         'name': 'Scale frames to common level',
         'value': False,
         'description': 'If set, a multiplicative scaling will be applied.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bgsub',
         'hidden': True,
         'name': 'Subtract residual background',
         'value': False,
         'description': 'If set, an additive background level '
                        'will be removed.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'secctr',
         'hidden': True,
         'name': 'Background section center',
         'value': '',
         'description': "Specify the center point in integers as 'x,y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'secsize',
         'hidden': True,
         'name': 'Background section size',
         'value': '',
         'description': "Specify in integers as 'size_x,size_y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'bgstat',
         'hidden': True,
         'name': 'Residual background statistic',
         'wtype': 'combo_box',
         'options': ['median', 'mode'],
         'option_index': 0,
         'description': 'Select the statistic to use to calculate '
                        'the residual background.'},
    ],
    'stack_dithers': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_stack',
         'name': 'Skip dither stacking',
         'value': False,
         'description': 'Set to skip stacking input files '
                        'and propagate separate images instead.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'ignore_dither',
         'name': 'Ignore dither information from header',
         'value': True,
         'description': 'Set to ignore dither information '
                        'and stack all input.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Combination method',
         'wtype': 'combo_box',
         'options': ['mean', 'median', 'sum'],
         'option_index': 0,
         'description': 'Select the combination method.'},
        {'key': 'weighted',
         'name': 'Use weighted mean',
         'value': True,
         'description': 'If set, the average of the data will be '
                        'weighted by the variance.\n'
                        'Ignored for method=median.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'robust',
         'name': 'Robust combination',
         'value': True,
         'description': 'If set, data will be sigma-clipped '
                        'before combination',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'threshold',
         'name': 'Outlier rejection threshold (sigma)',
         'value': 8.0,
         'description': 'Specify the number of sigma to use in '
                        'sigma clip for robust algorithms.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'maxiters',
         'name': 'Maximum sigma-clipping iterations',
         'value': 5,
         'description': 'Specify the maximum number of outlier '
                        'rejection iterations to use if robust=True.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'make_profiles': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'wavefile',
         'name': 'Wave/space calibration file',
         'value': '',
         'description': 'FITS file containing coordinate calibration data',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'slitfile',
         'hidden': True,
         'name': 'Slit correction file',
         'value': '',
         'description': 'FITS file containing slit correction data',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'fit_order',
         'name': 'Row fit order',
         'value': 3,
         'description': 'Polynomial fit order for rows '
                        '(along spectral dimension).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'bg_sub',
         'name': 'Subtract median background',
         'value': False,
         'description': 'If set, the median value along columns will '
                        'be subtracted from the profile.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'atmosthresh',
         'name': 'Atmospheric transmission threshold',
         'value': 0.0,
         'description': 'Transmission values below this threshold are not '
                        'considered when making the spatial profile.\n'
                        'Values are 0-1.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'simwavecal',
         'name': 'Simulate calibrations',
         'value': False,
         'description': 'If set, the data will not be rectified or '
                        'calibrated.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'testwavecal',
         'hidden': True,
         'name': 'Test new calibrations',
         'value': False,
         'description': 'If set, WAVECAL and SPATCAL extensions will be used '
                        'for rectification.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'locate_apertures': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'hidden': True,
         'name': 'Aperture location method',
         'wtype': 'combo_box',
         'options': ['auto', 'fix to input', 'fix to center', 'step up slit'],
         'option_index': 3,
         'description': 'Select the aperture location method.'},
        {'key': 'num_aps',
         'name': 'Number of apertures',
         'value': 100,
         'description': 'Number of apertures to step up the slit.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'input_position',
         'hidden': True,
         'name': 'Aperture position',
         'value': '',
         'description': 'Starting position(s) for aperture detection, '
                        'comma-separated for apertures, semi-colon '
                        'separated for files.\n'
                        'If method is "fix to input", will be used '
                        'directly.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'fwhm',
         'hidden': True,
         'name': 'Expected aperture FWHM (arcsec)',
         'value': 3.0,
         'description': 'Gaussian FWHM estimate for fit to profile.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'extract_median_spectra': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_1d',
         'name': 'Save extracted 1D spectra',
         'value': False,
         'description': 'Save spectra to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'normalize': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_1d',
         'name': 'Save extracted 1D spectra',
         'value': False,
         'description': 'Save spectra to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'make_slitcorr': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Fit method',
         'wtype': 'combo_box',
         'options': ['1D', '2D'],
         'option_index': 1,
         'description': 'Select the fit method.'},
        {'key': 'weighted',
         'name': 'Weight by spectral error',
         'value': True,
         'description': 'Fit to median spectra will be weighted by errors.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': '2d_params',
         'name': 'Parameters for 2D fit',
         'wtype': 'group'},
        {'key': 'x_fit_order',
         'name': 'Fit order for X (wavelength)',
         'value': 3,
         'description': 'Wavelength order for surface fit',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'y_fit_order',
         'name': 'Fit order for Y (slit position)',
         'value': 2,
         'description': 'Spatial order for surface fit',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': '1d_params',
         'name': 'Parameters for 1D fit',
         'wtype': 'group'},
        {'key': 'y_fit_order_1d',
         'name': 'Fit order for Y (slit position)',
         'value': 2,
         'description': 'Spatial order for surface fit',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'x_width',
         'name': 'Smoothing window for X (pixels)',
         'value': 10,
         'description': 'Boxcar width for smoothing fit in X-direction',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
}


class FORCASTSlitcorrParameters(FORCASTSpectroscopyParameters):
    """Reduction parameters for the FORCAST grism slitcorr pipeline."""
    def __init__(self, default=None, drip_cal_config=None,
                 drip_config=None):
        """
        Initialize parameters with default values.

        The various config files are used to override certain
        parameter defaults for particular observation modes,
        or dates, etc.

        Parameters
        ----------
        drip_cal_config : dict-like, optional
            Reduction mode and auxiliary file configuration mapping,
            as returned from the sofia_redux.instruments.forcast
            `getcalpath` function.
        drip_config : dict-like, optional
            DRIP configuration, as loaded by the
            sofia_redux.instruments.forcast `configuration` function.
        """
        if default is None:
            default = DEFAULT.copy()
            default.update(SPECTRAL_DEFAULT)
            default.update(SLITCORR_DEFAULT)

        super().__init__(default=default,
                         drip_cal_config=drip_cal_config,
                         drip_config=drip_config)

    def to_config(self):
        """
        Read parameter values into a configuration object.

        Section names in the output object are written as
        ``stepindex: stepname`` in order to record the order of
        reduction steps, and to keep any repeated step names uniquely
        identified.  Only the current parameter values are recorded.
        Other information, such as data or widget type or default
        values, is lost.

        Overrides parent function in order to add a slitcorr = True
        flag to the top-level configuration.

        Returns
        -------
        ConfigObj
            The parameter values in a `configobj.ConfigObj` object.
        """
        config = super().to_config()
        # add slitcorr parameter
        config['wavecal'] = False
        config['spatcal'] = False
        config['slitcorr'] = True
        return config

    def make_profiles(self, step_index):
        """
        Modify parameters for the profile step.

        Sets default wavefile according to `drip_cal_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.drip_cal_config is not None:
            # set default wave and slit file from cal config
            if 'wavefile' in self.drip_cal_config:
                self.current[step_index].set_value(
                    'wavefile', self.drip_cal_config['wavefile'])

    def locate_apertures(self, step_index):
        """Override parent behavior to do nothing."""
        pass
