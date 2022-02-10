# Licensed under a 3-clause BSD style license - see LICENSE.rst

import configobj
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase


class TestStepParent(DRPTestCase):
    def test_siso(self):
        step = StepParent()
        assert step.iomode == 'SISO'
        assert step.procname == 'unk'

        # default just copies in to out
        step.datain = [1, 2, 3]
        step.run()
        assert step.dataout == [1, 2, 3]

    def test_call(self):
        step = StepParent()
        df = DataFits()

        out = step(df)
        assert isinstance(out, DataFits)
        assert out.header['PRODTYPE'] == 'parent'

    def test_runstart(self, capsys):
        step = StepParent()
        step.config = None
        df = DataFits()
        df.config = None
        args = {'TESTPAR': 'Testval'}

        step.runstart(df, args)

        # sets config to empty configobj
        assert isinstance(step.config, configobj.ConfigObj)
        # lowercases arglist keys
        assert step.arglist == {'testpar': 'Testval'}

        # invalid input data
        with pytest.raises(TypeError):
            step.runstart('baddata', args)
        capt = capsys.readouterr()
        assert 'Invalid input' in capt.err

        # invalid config
        df.config = 'badval'
        with pytest.raises(RuntimeError):
            step.runstart(df, args)
        capt = capsys.readouterr()
        assert 'Invalid configuration' in capt.err

    def test_updateheader(self):
        step = StepParent()
        step.config = {}
        step.paramlist = [['key1', 'val1'],
                          ['key2', 'val2']]
        df = DataFits()

        # a couple short messages, and a couple long ones,
        # to verify the duplication check
        long_msg = '1' * 72
        l1 = 'testconf1'
        l2 = 'testconf2'
        df.config_files = ['test.cfg', 'override.cfg',
                           long_msg + l1, long_msg + l2]

        step.updateheader(df)

        # check history update
        hist = str(df.header['HISTORY'])
        assert 'parent: key1=val1' in hist
        assert 'parent: key2=val2' in hist
        assert 'CONFIG: test.cfg' in hist
        assert 'CONFIG: override.cfg' in hist
        # duplication is checked on the first 72 characters,
        # so the second value is read as a duplicate, and does
        # not appear in the header
        assert l1 in hist
        assert l2 not in hist

    def test_getarg(self, capsys):
        step = StepParent()
        step.config = {'parent': {'key2': 'from_config',
                                  'key3': 'from_config',
                                  'key4': ['3', '4'],
                                  'key5': ['5', '6'],
                                  'key6': ['7', '8'],
                                  'key7': '9',
                                  'key8': ['True', '1', 'False', '0'],
                                  'key9': 'True',
                                  'key10': '0'}}
        step.paramlist = [['key1', 'val1'],
                          ['Key2', 'val2'],
                          ['key3', 'val2'],
                          ['key4', [1, 2]],
                          ['key5', ''],
                          ['key6', []],
                          ['key7', [1, 2]],
                          ['key8', [False]],
                          ['key9', False],
                          ['key10', True],
                          ]
        step.arglist = {'key2': 'from_arg'}

        # try to get missing arg
        with pytest.raises(KeyError):
            step.getarg('badkey')
        capt = capsys.readouterr()
        assert 'no parameter named badkey' in capt.err

        # get value from arglist, config, paramlist, in that order
        assert step.getarg('key1') == 'val1'
        assert step.getarg('key2') == 'from_arg'
        assert step.getarg('key3') == 'from_config'

        # type conversion for config value lists
        # is done according to what's in paramlist, if possible
        assert step.getarg('key4') == [3, 4]
        assert step.getarg('key5') == "['5', '6']"
        assert step.getarg('key6') == ['7', '8']
        assert step.getarg('key7') == [9]
        assert step.getarg('key8') == [True, True, False, False]
        assert step.getarg('key9') is True
        assert step.getarg('key10') is False
