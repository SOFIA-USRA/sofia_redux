# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.setpar import setpar


class TestSetpar(object):

    def test_badparname(self, capsys):
        setpar(None, 'ok value')
        capt = capsys.readouterr()
        assert 'invalid parname' in capt.err

    def test_badvalue(self, capsys):
        setpar('foo', None)
        capt = capsys.readouterr()
        assert 'invalid parameter value' in capt.err

    def test_load_config(self):
        dripconfig.configuration = None
        setpar('foo', 'bar')
        assert dripconfig.configuration['foo'] == 'bar'

    def test_setpar(self):
        dripconfig.load()
        assert 'foo' not in dripconfig.configuration
        setpar('foo', 'bar')
        assert dripconfig.configuration['foo'] == 'bar'
