# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST parameter sets."""

from sofia_redux.pipeline.sofia.parameters.forcast_parameters import DEFAULT
from sofia_redux.pipeline.sofia.parameters.forcast_spectroscopy_parameters \
    import SPECTRAL_DEFAULT
from sofia_redux.pipeline.sofia.parameters.forcast_wavecal_parameters \
    import FORCASTWavecalParameters, WAVECAL_DEFAULT

SPATCAL_DEFAULT = {
    'trace_continuum': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'attach_trace_xy',
         'name': 'Attach trace positions table',
         'hidden': True,
         'value': True,
         'description': 'If set, trace x/y positions will be attached\n'
                        'in an additional extension.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Trace method',
         'wtype': 'combo_box',
         'options': ['fix to aperture position', 'fit to continuum'],
         'option_index': 1,
         'description': 'Select the trace method.'},
        {'key': 'fit_order',
         'name': 'Trace fit order',
         'value': 2,
         'description': 'Polynomial fit order for aperture center '
                        '(along spectral dimension).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'fit_thresh',
         'name': 'Trace fit threshold',
         'value': 4.0,
         'description': 'Robust rejection threshold, in sigma.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'step_size',
         'name': 'Fit position step size (pixels)',
         'value': 3,
         'description': 'Step size along trace for fitting locations.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'fit_traces': [
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
         'name': 'Weight by profile height',
         'value': True,
         'description': 'Fit to traces will be weighted by profile height.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'wavefile',
         'name': 'Wavelength calibration file',
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
}


class FORCASTSpatcalParameters(FORCASTWavecalParameters):
    """Reduction parameters for the FORCAST grism spatcal pipeline."""
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
            default.update(SPATCAL_DEFAULT)

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

        Overrides parent function in order to add a spatcal = True
        flag to the top-level configuration.

        Returns
        -------
        ConfigObj
            The parameter values in a `configobj.ConfigObj` object.
        """
        config = super().to_config()
        # add spatcal parameter
        config['wavecal'] = False
        config['spatcal'] = True
        config['slitcorr'] = False
        return config

    def fit_traces(self, step_index):
        """
        Modify parameters for the fitting step.

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
