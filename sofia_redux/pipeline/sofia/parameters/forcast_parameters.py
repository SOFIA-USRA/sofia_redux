# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST parameter sets."""

from copy import deepcopy

from astropy.io import fits

from sofia_redux.pipeline.parameters import Parameters


# Store default values for all parameters here.
# They could equivalently be read from a file, or
# constructed programmatically.  All keys are optional;
# defaults are specified in the ParameterSet object.
# All 'key' values should be unique.
DEFAULT = {
    'checkhead': [
        {'key': 'abort',
         'name': 'Abort reduction for invalid headers',
         'value': True,
         'description': 'If set, the reduction will be '
                        'aborted if the input headers '
                        'do not meet requirements',
         'dtype': 'bool',
         'wtype': 'check_box'}
    ],
    'clean': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'badfile',
         'name': 'Bad pixel map',
         'value': '',
         'description': 'FITS file containing bad pixel locations',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'autoshift',
         'name': 'Automatically detect readout shift',
         'value': True,
         'description': 'If set, data will be checked and '
                        'corrected for a 16-pixel readout shift.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'shiftfile',
         'name': 'Image number to shift (if not auto)',
         'value': '',
         'description': 'Specify "all", or semicolon-separated '
                        'image numbers, starting with 1. \nFor '
                        'example, to shift the 1st and 3rd image, '
                        'specify "1;3".',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'interpolate',
         'name': 'Interpolate over bad pixels',
         'value': False,
         'description': 'If set, bad pixels will be interpolated over.\n'
                        'If not set, they will be propagated as NaN.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'droop': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'fracdroop',
         'name': 'Droop fraction',
         'value': '',
         'description': 'Numerical factor for droop correction amplitude.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'nonlin': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'secctr',
         'name': 'Background section center',
         'value': '',
         'description': "Specify the center point in integers as 'x,y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'secsize',
         'name': 'Background section size',
         'value': '',
         'description': "Specify in integers as 'size_x,size_y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
    'stack': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'add_frames',
         'name': "Add all frames instead of subtracting",
         'value': False,
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
         'name': 'Scale frames to common level',
         'value': False,
         'description': 'If set, a multiplicative scaling will be applied.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bgsub',
         'name': 'Subtract residual background',
         'value': False,
         'description': 'If set, an additive background level '
                        'will be removed.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'secctr',
         'name': 'Background section center',
         'value': '',
         'description': "Specify the center point in integers as 'x,y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'secsize',
         'name': 'Background section size',
         'value': '',
         'description': "Specify in integers as 'size_x,size_y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'bgstat',
         'name': 'Residual background statistic',
         'wtype': 'combo_box',
         'options': ['median', 'mode'],
         'option_index': 0,
         'description': 'Select the statistic to use to calculate '
                        'the residual background.'},
    ],
}


class FORCASTParameters(Parameters):
    """Reduction parameters for the FORCAST pipeline."""
    def __init__(self, default=None,
                 drip_cal_config=None, drip_config=None):
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
        if default is None:  # pragma: no cover
            default = DEFAULT
        super().__init__(default=default)

        self.drip_cal_config = drip_cal_config
        self.drip_config = drip_config

    def copy(self):
        """
        Return a copy of the parameters.

        Overrides default copy to add in config attributes.

        Returns
        -------
        Parameters
        """
        new = super().copy()
        new.drip_cal_config = deepcopy(self.drip_cal_config)
        new.drip_config = deepcopy(self.drip_config)
        return new

    def clean(self, step_index):
        """
        Modify parameters for the clean step.

        Sets default badfile, using `drip_cal_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if (self.drip_cal_config is not None
                and 'badfile' in self.drip_cal_config):
            self.current[step_index].set_value(
                'badfile', self.drip_cal_config['badfile'])

    def droop(self, step_index):
        """
        Modify parameters for the droop step.

        Sets default droop fraction (fracdroop), using
        `drip_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.drip_config is not None:
            from sofia_redux.instruments.forcast.getpar import getpar
            fracdroop = getpar(fits.Header(), 'fracdroop',
                               dtype=float, default=0.0)
            self.current[step_index].set_value('fracdroop', fracdroop)

    def nonlin(self, step_index):
        """
        Modify parameters for the nonlin step.

        Sets default section center and size (secctr, secsize),
        using `drip_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # read background section from config file if possible
        if self.drip_config is not None:
            # assume config has already been
            # loaded into sofia_redux.instruments.forcast.configuration
            from sofia_redux.instruments.forcast.read_section \
                import read_section

            # assume image is standard 256x256 size
            datasec = read_section(256, 256)

            self.current[step_index].set_value(
                'secctr', '{:.0f},{:.0f}'.format(datasec[0], datasec[1]))
            self.current[step_index].set_value(
                'secsize', '{:.0f},{:.0f}'.format(datasec[2], datasec[3]))

    def stack(self, step_index):
        """
        Modify parameters for the stack step.

        Sets default background scaling and subtraction flags
        (bgscale, bgsub) and section location (secctr, secsize),
        using `drip_config` and `drip_cal_config`.

        If the data is grism mode or C2NC2, background
        subtraction and scaling are turned off by default. Otherwise,
        the default is read from the DRIP config file.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # read background settings from config file
        if (self.drip_config is not None
                and self.drip_cal_config is not None):
            from sofia_redux.instruments.forcast.getpar import getpar
            header = fits.Header()
            bgscale = getpar(header, 'BGSCALE', dtype=bool, default=False)
            bgsub = getpar(header, 'BGSUB', dtype=bool, default=False)

            # modify bg params by sky and grism mode
            if (self.drip_cal_config['gmode'] != -1
                    or self.drip_cal_config['cnmode'] in ['C2NC2', 'C2NC4']):
                bgsub = 0
                bgscale = 0

            # set parameter values in current set
            self.current[step_index].set_value('bgscale', bgscale)
            self.current[step_index].set_value('bgsub', bgsub)

            # read section from config, as for nonlin
            from sofia_redux.instruments.forcast.read_section \
                import read_section
            datasec = read_section(256, 256)
            self.current[step_index].set_value(
                'secctr', '{:.0f},{:.0f}'.format(datasec[0], datasec[1]))
            self.current[step_index].set_value(
                'secsize', '{:.0f},{:.0f}'.format(datasec[2], datasec[3]))
