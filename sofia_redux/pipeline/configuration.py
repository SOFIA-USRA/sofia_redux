# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Redux configuration."""

import os
import tempfile

from configobj import ConfigObj, ParseError

from sofia_redux.pipeline.chooser import Chooser


class Configuration(object):
    """
    Set Redux configuration.

    Configuration values are set as attributes of this object,
    so that they can be accessed as ``config.value``.  If
    the value is not set in the configuration, None will be
    returned.

    Subclasses should set the chooser attribute as appropriate.
    If unset, this class will use the default `redux.Chooser`.

    Attributes
    ----------
    config_file : str or dict-like
        Initial configuration file.
    config: `ConfigObj`
        Loaded configuration object.
    """
    def __init__(self, config_file=None):
        """
        Initialize with an optional configuration file.

        Parameters
        ----------
        config_file : str or dict-like, optional
            File path to an INI-format configuration file.
            May alternately be any object accepted by the
            ConfigObj constructor (e.g. a dictionary of
            configuration key-value pairs).
        """
        self.chooser = Chooser()
        self.config_file = None
        self.config = ConfigObj()

        if config_file is not None:
            self.load(config_file)

    def __getattr__(self, key):
        """Return chooser or value from config."""
        if self.config is not None and key in self.config:
            return self.config[key]
        else:
            return None

    def __setattr__(self, key, value):
        """Set value in config."""
        if key not in ['chooser', 'config_file', 'config']:
            if self.config is not None:
                self.config[key] = value
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def load(self, config_file):
        """
        Read config file into ConfigObj object.

        Parameters
        ----------
        config_file : str, dict, or ConfigObj
            File path to an INI-format configuration file.  Alternately,
            may be a dict or ConfigObj that can be directly read by
            the ConfigObj constructor, or it may be a string containing
            INI formatted values.
        """
        # allow command line strings for quick config
        if isinstance(config_file, str) and not os.path.isfile(config_file):
            # do nothing if config is set to "None", for default specification
            if config_file.lower().strip() == 'none':
                self.config_file = None
                co = ConfigObj(interpolation=False)
            else:
                # unescape anything escaped by the command line parser
                unescaped = config_file.encode(
                    'utf-8').decode('unicode_escape')
                # write to temporary file for configobj read in
                with tempfile.NamedTemporaryFile() as cfg:
                    cfg.write(unescaped.encode('utf-8'))
                    cfg.seek(0)
                    try:
                        co = ConfigObj(cfg, interpolation=False)
                    except ParseError:
                        raise IOError(f'Unable to read configuration '
                                      f'from: {config_file}') from None
                self.config_file = None
        else:
            # otherwise load as file
            co = config_file
            self.config_file = config_file

        self.config = ConfigObj(co, interpolation=False)

        # fix any "False" top-level keys
        for key in self.config:
            if str(self.config[key]).lower().strip() == 'false':
                self.config[key] = False

    def to_text(self):
        """
        Print the current configuration to a text list.

        Returns
        -------
        list of str
            The parameters in INI-formatted strings.
        """
        if self.config is None:
            return []
        else:
            return self.config.write()

    def update(self, config_file):
        """
        Update configuration from a new config file.

        Parameters
        ----------
        config_file : str, dict, or ConfigObj
            File path to an INI-format configuration file.  Alternately,
            may be a dict or ConfigObj that can be directly read by
            the ConfigObj constructor, or it may be a string containing
            INI formatted values.
        """
        new_config = ConfigObj(config_file, interpolation=False)
        # fix any "False" top-level keys
        for key in new_config:
            if str(new_config[key]).lower().strip() == 'false':
                new_config[key] = False
        self.config.merge(new_config)
