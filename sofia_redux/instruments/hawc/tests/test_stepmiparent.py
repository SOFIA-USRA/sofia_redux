# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.datatext import DataText
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase


class TestStepMIParent(DRPTestCase):
    def test_miso(self):
        step = StepMIParent()
        assert step.iomode == 'MISO'
        assert step.procname == 'unk'

        step.datain = [1, 2, 3]
        step.run()
        assert step.dataout == 1

    def test_runstart(self, capsys):
        df1 = DataFits()
        df1.config['data']['filenum'] = r'.*(\d+).*\.fits'
        df1.filename = 'test001.fits'
        df2 = DataText()
        df2.config['data']['filenum'] = r'.*(\d+).*\.txt'
        df2.filename = 'test002.txt'
        datain = [df1, df2]

        # good input
        step = StepMIParent()
        step.runstart(datain, {})
        assert step.filenum == ['1', '2']

        # bad input
        datain = [1, 2, 3]
        with pytest.raises(TypeError):
            step.runstart(datain, {})
        capt = capsys.readouterr()
        assert 'Invalid input data' in capt.err

        datain = 1
        with pytest.raises(TypeError):
            step.runstart(datain, {})
        capt = capsys.readouterr()
        assert 'Invalid input data' in capt.err

        # bad filenum
        df1.config['data']['filenum'] = r'(.*)\.fits'
        step.runstart([df1], {})
        assert step.filenum == []

        # filenum with '-' in it
        df1.filename = 'test_001-002.fits'
        df1.config['data']['filenum'] = r'.*_(\d+-\d+).*\.fits'
        step.runstart([df1], {})
        assert step.filenum == ['001', '002']

    def test_updateheader(self):
        # middle or end of filename
        fnum = r'(?:.*[a-z](\d+)[a-z]+.*\.fits)|(?:.*[a-z]+(\d+)_[A-Z]+\.fits)'
        fbeg = r'.*_'
        fend = r'\.fits'
        conf = {'data': {'filenum': fnum,
                         'filenamebegin': fbeg,
                         'filenameend': fend}}
        step = StepMIParent()
        step.config = conf
        step.filenum = ['1', '4', '2']

        df1 = DataFits()
        df1.config = conf
        df1.filename = os.path.join(
            'path', 'to', 'midnum50val_RAW.fits')
        df2 = DataFits()
        df2.config = conf
        df2.filename = os.path.join(
            'path', 'to', 'endnum101_RAW.fits')

        # file number is replaced with step.filenum range
        step.updateheader(df1)
        assert df1.filename == os.path.join(
            'path', 'to', 'midnum1-4val_UNK.fits')
        step.updateheader(df2)
        assert df2.filename == os.path.join(
            'path', 'to', 'endnum1-4_UNK.fits')
