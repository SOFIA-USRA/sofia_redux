# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime as dt
import os

import configobj
import numpy as np
import pytest

from sofia_redux.instruments.hawc.dataparent import DataParent
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase


class TestDataParent(DRPTestCase):
    def test_load_error(self, capsys):
        badfile = 'badfile.cfg'
        with pytest.raises(IOError):
            DataParent(badfile)
        capt = capsys.readouterr()
        assert 'invalid file name' in capt.err

    def test_getattr(self):
        # empty data, default config
        df = DataParent()
        assert df.data is None
        assert df.filename == ''
        assert df.filenamebegin == ''
        assert df.filenameend == ''
        assert df.filenum is None
        with pytest.raises(AttributeError):
            _ = df.badval

        # set a HAWC-like reduced filename
        df.filename = os.path.join(
            'path', 'to', 'F0001_HA_POL_90000101_HAWDHWPD_MRG_001.fits')
        assert df.filenamebegin == os.path.join(
            'path', 'to', 'F0001_HA_POL_90000101_HAWDHWPD_')
        assert df.filenameend == '_001.fits'
        assert df.filenum == '001'

        # and with a range of file numbers
        df.filename = 'F0001_HA_POL_90000101_HAWDHWPD_MRG_001-003.fits'
        assert df.filenum == '001-003'

        # and a raw-style filename
        df.filename = os.path.join(
            'path', 'to',
            '2020-01-01_HA_F001_042_POL_90000101_HAWD_HWPD_RAW.fits')
        assert df.filenamebegin == os.path.join(
            'path', 'to', '2020-01-01_HA_F001_042_POL_90000101_HAWD_HWPD_')
        assert df.filenameend == '.fits'
        assert df.filenum == '042'

        # now with empty config
        df.config = {}
        assert df.filenamebegin == os.path.splitext(df.filename)[0] + '.'
        assert df.filenameend == '.fits'
        assert df.filenum is None

        # now with unexpected filename
        df.filename = 'testname.prod.txt'
        assert df.filenamebegin == 'testname.'
        assert df.filenameend == '.txt'
        assert df.filenum is None

        df.filename = 'testname'
        assert df.filenamebegin == 'testname'
        assert df.filenameend == ''
        assert df.filenum is None

        # different filenum regex -- no groups
        df.config['data'] = {'filenum': 'test'}
        assert df.filenum == 'test'

    def test_mergeconfig(self, capsys, tmpdir):
        # default behavior covered elsewhere; check a few corners here

        # mergeconfig without first setconfig -- calls setconfig
        df = DataParent()
        df.config = None
        df.mergeconfig()
        assert df.config is not None

        # mergeconfig with date, no override available
        df.mergeconfig(date=dt.datetime(9999, 1, 1))
        capt = capsys.readouterr()
        assert 'No date config file for 9999-01-01' in capt.out

        # config string -- bad path
        with pytest.raises(Exception):
            df.mergeconfig(config='badfile.cfg')
        capt = capsys.readouterr()
        assert 'invalid file name' in capt.err

        # config string -- unreadable format
        conf = tmpdir.join('conf.cfg')
        conf.write('badval')
        with pytest.raises(Exception):
            df.mergeconfig(config=str(conf))
        capt = capsys.readouterr()
        assert 'Error while loading configuration' in capt.err

        # list of good config files
        conf.write('test1 = val1')
        conf2 = tmpdir.join('conf2.cfg')
        conf2.write('test2 = val2')
        df.mergeconfig(config=[str(conf), str(conf2)])
        assert df.config['test1'] == 'val1'
        assert df.config['test2'] == 'val2'

        # dict type config
        df.mergeconfig(config={'test3': 'val3'})
        assert df.config['test3'] == 'val3'

        # bad config type
        with pytest.raises(TypeError):
            df.mergeconfig(config=10)

    def test_mode(self, capsys):
        # no config
        df = DataParent()
        df.config = None
        assert df.get_pipe_mode() is None

        # set a mode in config, no header
        d1 = {'mode_1': {'datakeys': 'TEST1 = 1 | TEST2 = 2'}}
        df.config = configobj.ConfigObj(d1)
        assert df.get_pipe_mode() is None

        # set a bad mode
        d2 = {'mode_1': {'badkey': 'badval'}}
        df.config = configobj.ConfigObj(d2)
        assert df.get_pipe_mode() is None
        capt = capsys.readouterr()
        assert 'In configuration, missing datakeys' in capt.err

        # match a mode
        df.header = {'TEST1': 1, 'TEST2': 2, 'TEST3': 2}
        df.config = configobj.ConfigObj(d1)
        assert df.get_pipe_mode() == '1'

    def test_load(self):
        df = DataParent()
        with pytest.raises(NotImplementedError):
            df.load('test.fits')
        with pytest.raises(NotImplementedError):
            df.save('test.fits')

    def test_copy(self):
        df = DataParent()
        df.filename = 'test1'
        df.rawname = 'test2'
        df.loaded = True
        df.header = {'TEST1': 1, 'TEST2': 2, 'TEST3': 2}
        df.data = np.zeros(10)

        new_df = df.copy()
        assert new_df.filename == df.filename
        assert new_df.rawname == df.rawname
        assert new_df.loaded == df.loaded
        assert new_df.header is not df.header
        assert new_df.header == df.header
        assert new_df.data is not df.data
        assert np.all(new_df.data == df.data)

        df.data = 'test string data'
        new_df = df.copy()
        assert new_df.data == df.data

    def test_mergehead(self):
        # this test borrowed from test_datafits, with
        # modifications for dict headers instead of fits headers

        # make a config for test value merging
        combo = ['FIRST', 'LAST', 'MIN', 'MAX', 'SUM',
                 'OR', 'AND', 'CONCATENATE', 'DEFAULT']
        keys = ['TEST{}'.format(i + 1) for i in range(len(combo))]
        config = {'headmerge': dict(zip(keys, combo))}

        # set up a couple test headers
        val1 = list(range(len(combo)))
        hdr1 = dict(zip(keys, val1))

        val2 = [v + 10 for v in val1]
        hdr2 = dict(zip(keys, val2))

        # merge with no comment, history, config
        df1 = DataParent()
        df2 = DataParent()
        df1.header = hdr1.copy()
        df2.header = hdr2.copy()
        df1.mergehead(df2)
        # without config, all values match first
        assert df1.header == hdr1

        # now add comment, history
        hdr1['COMMENT'] = ['comment 1']
        hdr1['HISTORY'] = ['history 1']
        hdr2['COMMENT'] = ['comment 2']
        hdr2['HISTORY'] = ['history 2']

        # make the dataparent, load config and header
        df1 = DataParent()
        df2 = DataParent()
        df1.header = hdr1.copy()
        df2.header = hdr2.copy()
        df1.config = config
        df2.config = config

        # merge all keys
        df1.mergehead(df2)
        new_hdr = df1.header

        # check values
        # first
        assert new_hdr['TEST1'] == val1[0]
        # last
        assert new_hdr['TEST2'] == val2[1]
        # min
        assert new_hdr['TEST3'] == val1[2]
        # max
        assert new_hdr['TEST4'] == val2[3]
        # sum
        assert new_hdr['TEST5'] == val1[4] + val2[4]
        # or
        assert new_hdr['TEST6'] == val1[5] | val2[5]
        # and
        assert new_hdr['TEST7'] == val1[6] & val2[6]
        # concatenate
        assert new_hdr['TEST8'] == ','.join(sorted([str(val1[7]),
                                                    str(val2[7])]))
        # default
        assert new_hdr['TEST9'] == -9999

        # all history and comments in new header
        assert 'comment 1' in str(new_hdr['COMMENT'])
        assert 'comment 2' in str(new_hdr['COMMENT'])
        assert 'history 1' in str(new_hdr['HISTORY'])
        assert 'history 2' in str(new_hdr['HISTORY'])

        # test concatenation with multiple values, some overlapping
        df1.header['TEST8'] = 'q,a,r,c'
        df2.header['TEST8'] = 'b,d,c'
        df1.mergehead(df2)
        assert df1.header['TEST8'] == 'a,b,c,d,q,r'

        # test all default values
        df1.header['TEST9'] = 10
        df1.mergehead(df2)
        assert type(df1.header['TEST9']) is int
        assert df1.header['TEST9'] == -9999

        df1.header['TEST9'] = 10.0
        df1.mergehead(df2)
        assert type(df1.header['TEST9']) is float
        assert df1.header['TEST9'] == -9999.0

        df1.header['TEST9'] = 'ten'
        df1.mergehead(df2)
        assert type(df1.header['TEST9']) is str
        assert df1.header['TEST9'] == 'UNKNOWN'

    def test_getheadval(self, capsys):
        df = DataParent()
        hdr1 = {'TESTKEY': 'TESTVAL1', 'ALTKEY': 'ALTVAL'}
        hdr2 = {'TESTKEY': 'TESTVAL2'}
        df.header = hdr1

        # get from primary
        df.config = None
        assert df.getheadval('TESTKEY') == 'TESTVAL1'

        # get from config: alternate key
        df.config = {'header': {'TESTKEY': 'ALTKEY'}}
        assert df.getheadval('TESTKEY') == 'ALTVAL'
        df.header = hdr2
        with pytest.raises(KeyError):
            df.getheadval('TESTKEY')
        capt = capsys.readouterr()
        assert 'Missing ALTKEY keyword in header' in capt.err

        # get from config: optional alternate - used only if first
        # key not present in header
        df.header = hdr1
        df.config = {'header': {'TESTKEY': '? ALTKEY',
                                'TESTKEY2': '? ALTKEY'}}
        assert df.getheadval('TESTKEY') == 'TESTVAL1'
        assert df.getheadval('TESTKEY2') == 'ALTVAL'

        # T/F in config translated as True/False
        df.config = {'header': {'TESTKEY': 'T', 'ALTKEY': 'F'}}
        assert df.getheadval('TESTKEY') is True
        assert df.getheadval('ALTKEY') is False

        # int, float val in config
        df.config = {'header': {'TESTKEY': '1', 'ALTKEY': '2.0'}}
        assert isinstance(df.getheadval('TESTKEY'), int)
        assert isinstance(df.getheadval('ALTKEY'), float)

        # num/str values -- stay str
        df.config = {'header': {'TESTKEY': '1a'}}
        assert isinstance(df.getheadval('TESTKEY'), str)

    def test_setheadval(self):
        df = DataParent()
        hdr1 = {'TESTKEY': 'TESTVAL1',
                'HISTORY': ['HIST1']}
        df.header = hdr1.copy()

        # regular key add with comment
        df.setheadval('TEST1', 'VAL1', comment='COM1')
        assert df.header['TEST1'] == 'VAL1'
        assert df.header['COMMENT'] == ['TEST1, COM1']

        # regular key update
        df.setheadval('TESTKEY', 'TESTVAL2')
        assert df.header['TESTKEY'] == 'TESTVAL2'

        # add history
        df.setheadval('HISTORY', 'HIST2')
        assert df.header['HISTORY'] == ['HIST1', 'HIST2']

    def test_delheadval(self):
        df = DataParent()
        hdr1 = {'TESTKEY': 'TESTVAL1',
                'ALTKEY': 'ALTVAL',
                'HISTORY': ['HIST1']}
        df.header = hdr1.copy()

        # delete a single key
        df.delheadval('TESTKEY')
        assert 'TESTKEY' not in df.header

        # delete a couple
        df.delheadval(['ALTKEY', 'HISTORY'])
        assert len(df.header) == 0
