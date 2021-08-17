# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM parameter sets."""

from copy import deepcopy

from sofia_redux.pipeline.parameters import Parameters


# Store default values for all parameters here.
# They could equivalently be read from a file, or
# constructed programmatically.  All keys are optional;
# defaults are specified in the ParameterSet object.
# All 'key' values should be unique.
DEFAULT = {
    'check_header': [
        {'key': 'abort',
         'name': 'Abort reduction for invalid headers',
         'value': False,
         'description': 'If set, the reduction will be '
                        'aborted if the input headers '
                        'do not meet requirements',
         'dtype': 'bool',
         'wtype': 'check_box'}
    ],
    'correct_linearity': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'linfile',
         'name': 'Linearity correction file',
         'value': '',
         'description': 'FITS file containing linearity coefficients',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'saturation',
         'name': 'Saturation level',
         'value': 5000,
         'description': 'Saturation level for raw FLITECAM data',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
}


class FLITECAMParameters(Parameters):
    """Reduction parameters for the FLITECAM pipeline."""
    def __init__(self, default=None,
                 config=None, pipecal_config=None):
        """
        Initialize parameters with default values.

        The various config files are used to override certain
        parameter defaults for particular observation modes,
        or dates, etc.

        Parameters
        ----------
        config : dict-like, optional
            Reduction mode and auxiliary file configuration mapping,
            as returned from the sofia_redux.instruments.flitecam
            `getcalpath` function.
        pipecal_config : dict-like, optional
            Flux calibration and atmospheric correction configuration,
            as returned from the pipecal `pipecal_config` function.
        """
        if default is None:  # pragma: no cover
            default = DEFAULT
        super().__init__(default=default)

        self.config = config
        self.pipecal_config = pipecal_config

    def copy(self):
        """
        Return a copy of the parameters.

        Overrides default copy to add in config attributes.

        Returns
        -------
        Parameters
        """
        new = super().copy()
        new.config = deepcopy(self.config)
        new.pipecal_config = deepcopy(self.pipecal_config)
        return new

    def correct_linearity(self, step_index):
        """
        Modify parameters for the linearity correction step.

        Sets default linfile, using `config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if (self.config is not None
                and 'linfile' in self.config):
            self.current[step_index].set_value(
                'linfile', self.config['linfile'])
