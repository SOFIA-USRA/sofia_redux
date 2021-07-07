# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""SOFIA Redux configuration."""

import configobj
from sofia_redux.pipeline.configuration import Configuration
from sofia_redux.pipeline.sofia.sofia_chooser import SOFIAChooser


class SOFIAConfiguration(Configuration):
    """
    Set Redux configuration for SOFIA pipelines.

    This class sets the reduction object chooser to `SOFIAChooser`
    and sets some default configuration values for output files.

    If desired, all configuration values may be overridden with an
    input configuration file in INI format.  The following
    example would be equivalent to the current default settings::

        output_directory = .
        input_manifest = redux_infiles.txt
        output_manifest = outfiles.txt
        parameter_file = redux_param.cfg
        log_file = "redux_%Y%m%d_%H%M%S.log"
        log_level = DEBUG
        log_format = "%(asctime)s - %(origin)s - %(levelname)s - %(message)s"
        update_display = True
        display_intermediate = False

    """

    def __init__(self, config_file=None):
        """
        Initialize with an optional configuration file.

        Parameters
        ----------
        config_file : str or ConfigObj, optional
            File path to an INI-format configuration file.
        """
        super().__init__(config_file)

        # set the SOFIA chooser to decide reduction
        # objects based on input data
        self.chooser = SOFIAChooser()

        # set some default values in case they were not passed
        default = configobj.ConfigObj(
            {'output_directory': '.',
             'input_manifest': 'redux_infiles.txt',
             'output_manifest': 'outfiles.txt',
             'parameter_file': 'redux_param.cfg',
             'log_file': 'redux_%Y%m%d_%H%M%S.log',
             'log_level': 'DEBUG',
             'log_format': '%(asctime)s - %(origin)s - '
                           '%(levelname)s - %(message)s',
             'absolute_paths': False,
             'update_display': True,
             'display_intermediate': False,
             'config_file_name': 'redux_config.cfg'},
            interpolation=False)

        # merge loaded config into default dictionary
        if self.config is not None:
            default.merge(self.config)

        if self.config_file is not None:
            default.filename = self.config_file
        self.config = default
