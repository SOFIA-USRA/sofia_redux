# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST parameter sets."""

from sofia_redux.pipeline.sofia.parameters.forcast_parameters import DEFAULT
from sofia_redux.pipeline.sofia.parameters.forcast_spectroscopy_parameters \
    import FORCASTSpectroscopyParameters, SPECTRAL_DEFAULT

WAVECAL_DEFAULT = {
    'make_profiles': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'wavefile',
         'hidden': True,
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
         'hidden': True,
         'name': 'Subtract median background',
         'value': False,
         'description': 'If set, the median value along columns will '
                        'be subtracted from the profile.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'atmosthresh',
         'hidden': True,
         'name': 'Atmospheric transmission threshold',
         'value': 0.0,
         'description': 'Transmission values below this threshold are not '
                        'considered when making the spatial profile.\n'
                        'Values are 0-1.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'simwavecal',
         'hidden': True,
         'name': 'Simulate calibrations',
         'value': True,
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
    'extract_summed_spectrum': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_1d',
         'name': 'Save extracted 1D spectra',
         'value': True,
         'description': 'Save spectra to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Aperture location method',
         'wtype': 'combo_box',
         'options': ['auto', 'fix to input', 'fix to center'],
         'option_index': 2,
         'description': 'Select the aperture location method.'},
        {'key': 'appos',
         'name': 'Aperture position (pixel)',
         'value': '',
         'description': 'Aperture center for initial spectrum.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'aprad',
         'name': 'Aperture radius (pixel)',
         'value': '40',
         'description': 'Aperture width for initial spectrum.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'detrend_order',
         'name': 'Polynomial order for spectrum detrend',
         'value': '',
         'description': 'If set, the low-order shape of the '
                        'spectrum will be removed.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'identify_lines': [
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
        {'key': 'linefile',
         'name': 'Line list',
         'value': '',
         'description': 'Text file containing identifiable spectral lines',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'line_type',
         'name': 'Line type',
         'wtype': 'combo_box',
         'options': ['emission', 'absorption', 'either'],
         'option_index': 0,
         'description': 'Select the type of lines to fit.'},
        {'key': 'window',
         'name': 'Fit window',
         'value': 10.0,
         'description': 'Initial window considered for line ID (pixels)',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'sigma',
         'name': 'Expected line width (pixel)',
         'value': 3.0,
         'description': 'Gaussian width estimate for fit to lines.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'guess_lines',
         'name': 'Guess wavelengths',
         'value': '',
         'description': 'Approximate wavelengths for a starter guess, '
                        'comma-separated.\nIf present, should have at '
                        'least 2 values, matching the guess positions.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'guess_positions',
         'name': 'Guess wavelength position',
         'value': '',
         'description': 'Starting positions for wavelength ID, '
                        'comma-separated.\nIf present, should have at '
                        'least 2 values, matching the guess wavelengths.',
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
    'reidentify_lines': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_1d',
         'name': 'Save extracted 1D spectra',
         'value': True,
         'description': 'Save spectra to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Aperture location method',
         'wtype': 'combo_box',
         'options': ['auto', 'fix to input', 'step up slit'],
         'option_index': 2,
         'description': 'Select the aperture location method.'},
        {'key': 'num_aps',
         'name': 'Number of auto apertures',
         'value': 3,
         'description': 'Number of apertures to look for '
                        'if method=auto.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'appos',
         'name': 'Aperture position (pixel)',
         'value': '',
         'description': 'Manually specify aperture centers, comma-separated.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'step',
         'name': 'Step size',
         'value': 16,
         'description': 'Spatial step for aperture centers (pixels)',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'radius',
         'name': 'Aperture radius',
         'value': 8,
         'description': 'Spatial radius for extraction apertures (pixels)',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'detrend_order',
         'name': 'Polynomial order for spectrum detrend',
         'value': '',
         'description': 'If set, the low-order shape of the '
                        'spectrum will be removed.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'window',
         'name': 'Fit window',
         'value': 10.0,
         'description': 'Wavelength window considered for line ID (pixels)',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 's2n',
         'name': 'Signal-to-noise requirement (sigma)',
         'value': 2,
         'description': 'If greater than zero, a line will not be considered\n'
                        'for fitting unless it exceeds the S/N specified.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'fit_lines': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'x_fit_order',
         'name': 'Fit order for X (wavelength)',
         'value': 2,
         'description': 'Wavelength order for surface fit',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'y_fit_order',
         'name': 'Fit order for Y (slit position)',
         'value': 2,
         'description': 'Spatial order for surface fit',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'weighted',
         'name': 'Weight by line height',
         'value': True,
         'description': 'Fit to lines will be weighted by line height.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'spatfile',
         'name': 'Spatial calibration file',
         'value': '',
         'description': 'FITS file containing coordinate calibration data',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'rotation',
         'hidden': True,
         'name': 'Rotation code for storing wavecal file',
         'value': 0,
         'description': 'FORCAST does not use a 90 degree rotation for '
                        'calibration files.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'rectify': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'wavefile',
         'hidden': True,
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
         'hidden': True,
         'name': 'Row fit order',
         'value': 3,
         'description': 'Polynomial fit order for rows '
                        '(along spectral dimension).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'bg_sub',
         'hidden': True,
         'name': 'Subtract median background',
         'value': False,
         'description': 'If set, the median value along columns will '
                        'be subtracted from the profile.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'atmosthresh',
         'hidden': True,
         'name': 'Atmospheric transmission threshold',
         'value': 0.0,
         'description': 'Transmission values below this threshold are not '
                        'considered when making the spatial profile.\n'
                        'Values are 0-1.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'simwavecal',
         'hidden': True,
         'name': 'Simulate calibrations',
         'value': False,
         'description': 'If set, the data will not be rectified or '
                        'calibrated.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'testwavecal',
         'hidden': True,
         'name': 'Test new calibrations',
         'value': True,
         'description': 'If set, WAVECAL and SPATCAL extensions will be used '
                        'for rectification.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ]
}


class FORCASTWavecalParameters(FORCASTSpectroscopyParameters):
    """Reduction parameters for the FORCAST grism wavecal pipeline."""
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
            default.update(WAVECAL_DEFAULT)
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

        Overrides parent function in order to add a wavecal = True
        flag to the top-level configuration.

        Returns
        -------
        ConfigObj
            The parameter values in a `configobj.ConfigObj` object.
        """
        config = super().to_config()
        # add wavecal parameter
        config['wavecal'] = True
        config['spatcal'] = False
        config['slitcorr'] = False
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
            # set default wave file from cal config
            if 'wavefile' in self.drip_cal_config:
                self.current[step_index].set_value(
                    'wavefile', self.drip_cal_config['wavefile'])

    def identify_lines(self, step_index):
        """
        Modify parameters for the profile step.

        Sets default wavefile according to `drip_cal_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.drip_cal_config is not None:
            # set default wave and line file from cal config
            if 'wavefile' in self.drip_cal_config:
                self.current[step_index].set_value(
                    'wavefile', self.drip_cal_config['wavefile'])
            if 'linefile' in self.drip_cal_config:
                self.current[step_index].set_value(
                    'linefile', self.drip_cal_config['linefile'])

    def fit_lines(self, step_index):
        """
        Modify parameters for the fitting step.

        Sets default spatfile according to `drip_cal_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.drip_cal_config is not None:
            # set default wave file from cal config
            if 'wavefile' in self.drip_cal_config:
                self.current[step_index].set_value(
                    'spatfile', self.drip_cal_config['wavefile'])
