# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Configuration class."""

import configobj
import pytest

from sofia_redux.pipeline.configuration import Configuration
from sofia_redux.pipeline.chooser import Chooser


class TestConfiguration(object):
    def test_get_chooser(self):
        configuration = Configuration()
        chooser = configuration.chooser
        assert isinstance(chooser, Chooser)

    def test_config_file(self, tmpdir):
        conf = {'test_key': 'test_value',
                'test_bool': 'False'}
        co = configobj.ConfigObj(conf)
        conf_file = str(tmpdir.join('test.cfg'))
        co.filename = conf_file
        co.write()

        configuration = Configuration(config_file=conf_file)
        assert configuration.test_key == 'test_value'
        assert configuration.test_bool is False
        assert configuration.config_file == conf_file

    def test_config_dict(self):
        conf = {'test_key': 'test_value',
                'test_bool': 'False'}
        conf_obj = configobj.ConfigObj(conf)
        configuration = Configuration(config_file=conf_obj)

        # test get_attr
        assert configuration.test_key == 'test_value'
        assert configuration.test_bool is False

        # test set_attr
        assert 'test_key_2' not in configuration.config
        configuration.test_key_2 = 'test_value_2'
        assert 'test_key_2' in configuration.config
        assert configuration.config['test_key_2'] == 'test_value_2'
        assert configuration.test_key_2 == 'test_value_2'

        # if config is None, attribute is directly set
        configuration.config = None
        configuration.test_attr = 'test_attr_value'
        assert configuration.config is None
        assert configuration.test_attr == 'test_attr_value'

    def test_config_string(self):
        conf = 'test_key=test_value\ntest_bool=False'

        configuration = Configuration(config_file=conf)
        assert configuration.test_key == 'test_value'
        assert configuration.test_bool is False
        assert configuration.config_file is None

        # should raise error for bad format
        conf = 'bad_value'
        with pytest.raises(IOError):
            Configuration(conf)

        # should allow it and do nothing if input is "None:
        configuration = Configuration(config_file='None')
        assert configuration.config_file is None

    def test_config_update(self):
        configuration = Configuration()
        conf = {'test_key': 'test_value',
                'test_bool': 'False'}

        # load, then update with new value: old value should
        # still be there
        configuration.load(conf)
        configuration.update({'test1': 'True'})
        assert configuration.test_key == 'test_value'
        assert configuration.test1 == 'True'

        # 'False' values are converted to bool
        configuration.update({'test2': 'false'})
        assert configuration.test2 is False

    def test_config_to_text(self):
        configuration = Configuration()
        assert configuration.to_text() == []
        configuration.test_key = 'test_value'
        assert configuration.to_text() == ['test_key = test_value']
        configuration.config = None
        assert configuration.to_text() == []
