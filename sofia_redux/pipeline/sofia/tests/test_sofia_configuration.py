# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the SOFIA Configuration class."""

import configobj

from sofia_redux.pipeline.sofia.sofia_configuration import SOFIAConfiguration
from sofia_redux.pipeline.sofia.sofia_chooser import SOFIAChooser


class TestSOFIAConfiguration(object):
    def test_get_chooser(self):
        """Test run method."""
        config = SOFIAConfiguration()
        chooser = config.chooser
        assert isinstance(chooser, SOFIAChooser)

    def test_config_file(self):
        conf = {'test_key': 'test_value',
                'test_bool': 'False'}
        conf_file = configobj.ConfigObj(conf)

        config = SOFIAConfiguration(config_file=conf_file)
        assert config.test_key == 'test_value'
        assert config.test_bool is False

    def test_defaults(self):
        config = SOFIAConfiguration()

        default_keys = ['output_directory',
                        'input_manifest',
                        'output_manifest',
                        'parameter_file',
                        'log_file',
                        'log_level',
                        'log_format',
                        'absolute_paths',
                        'update_display',
                        'display_intermediate']
        for key in default_keys:
            assert getattr(config, key) is not None
