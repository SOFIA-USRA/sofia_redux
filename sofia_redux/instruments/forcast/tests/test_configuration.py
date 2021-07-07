# Licensed under a 3-clause BSD style license - see LICENSE.rst

from importlib import reload
import os

from configobj import ConfigObj

import sofia_redux.instruments.forcast.configuration as dripconfig


class TestConfiguration(object):
    def test_empty(self):
        """Test the configuration is empty on initial import"""
        test = reload(dripconfig)
        assert test.configuration is None

    def test_load(self):
        """Test the configuration can be loaded"""
        test = reload(dripconfig)
        test.load()
        assert isinstance(test.configuration, ConfigObj)
        assert 'doinhdch' in test.configuration

    def test_load_file(self, tmpdir, capsys):
        # make local config file
        cwd = os.getcwd()
        os.chdir(tmpdir)
        config_name = 'dripconf.txt'
        conffile = tmpdir.join(config_name)
        conffile.write('testval = 1')

        # test loading local dripconf.txt
        assert os.path.isfile(config_name)
        test = reload(dripconfig)
        test.load()
        assert isinstance(test.configuration, ConfigObj)
        assert 'testval' in test.configuration

        # test loading from file name
        os.chdir(cwd)
        assert not os.path.isfile(config_name)
        test = reload(dripconfig)
        test.load(config_file=str(conffile))
        assert isinstance(test.configuration, ConfigObj)
        assert 'testval' in test.configuration

        # test missing file
        test = reload(dripconfig)
        test.load(config_file=config_name)
        assert test.configuration is None
        capt = capsys.readouterr()
        assert 'configuration file does not exist' in capt.err

        # test badly formatted file
        conffile.write('badval')
        test = reload(dripconfig)
        test.load(config_file=str(conffile))
        assert test.configuration is None
        capt = capsys.readouterr()
        assert 'Could not load' in capt.err
