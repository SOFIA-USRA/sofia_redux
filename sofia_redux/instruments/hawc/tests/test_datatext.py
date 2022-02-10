# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import pytest

from sofia_redux.instruments.hawc.datatext import DataText
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase


class TestDataText(DRPTestCase):
    def test_load(self, tmpdir):
        goodfile = tmpdir.join('test.txt')
        goodfile.write('# testkey = testval\n10 \n20\t\n 30\n')

        # load on init
        df = DataText(str(goodfile))
        assert df.filename == str(goodfile)
        assert df.rawname == str(goodfile)
        assert df.loaded is True
        assert len(df.data) == 3
        assert df.data == ['10', '20', '30']
        assert len(df.header) == 1
        assert df.header['testkey'] == 'testval'

        # load from self.filename
        df2 = DataText()
        df2.filename = str(goodfile)
        df2.load()
        assert df2.loaded
        assert df2.data == df.data
        assert df2.header == df.header

    def test_load_error(self, capsys):
        badfile = 'badfile.fits'
        with pytest.raises(IOError):
            DataText(badfile)
        capt = capsys.readouterr()
        assert 'No such file' in capt.err

    def test_sethead(self):
        lines = ['# testkey = testval1',
                 '# testkey = testval2',
                 '# testcomment',
                 'bad line']
        df = DataText()
        for line in lines:
            df.sethead(line)
        assert df.header['testkey'] == 'testval2'
        assert df.header['COMMENT'] == ['testcomment']
        assert 'bad line' not in str(df.header)

    def test_loadhead(self, tmpdir):
        goodfile = tmpdir.join('test.txt')
        goodfile.write('# testkey = testval\n'
                       '# com1\n'
                       '# com2\n'
                       '10 \n20\t\n 30\n')

        # load from filename
        df = DataText()
        df.loadhead(str(goodfile))
        assert df.header['testkey'] == 'testval'
        assert df.header['COMMENT'] == ['com1', 'com2']
        assert df.data is None
        assert df.loaded is False

        # load from self.filename
        df2 = DataText()
        df2.filename = str(goodfile)
        df2.loadhead()
        assert df2.header == df.header

    def test_save(self, tmpdir):
        df = DataText()
        df.header = {'testkey': 'testval',
                     'HISTORY': ['hist1', 'hist2'],
                     'COMMENT': ['com1', 'com2']}
        df.data = ['a', 'b', 'c']
        fn1 = str(tmpdir.join('fn1.txt'))
        fn2 = str(tmpdir.join('fn2.txt'))

        # save to filename
        df.save(fn1)
        assert os.path.isfile(fn1)

        # read back in to test header, data
        df2 = DataText(fn1)
        assert df2.header['testkey'] == 'testval'
        assert df.header['HISTORY'] == df.header['HISTORY']
        assert df.header['COMMENT'] == df.header['COMMENT']
        assert 'Pipeline Version' in df2.header
        assert 'fn1' in df2.header['File Name']
        assert 'File Date' in df2.header
        del df2.header['File Date']
        assert df2.data == df.data

        # overwrite okay
        df.save(fn1)
        assert os.path.isfile(fn1)
        df3 = DataText(fn1)
        assert 'File Date' in df3.header
        del df3.header['File Date']
        assert df3.header == df2.header
        assert df3.data == df2.data

        # save to self.filename
        df.filename = fn2
        df.save()
        assert os.path.isfile(fn2)
        df4 = DataText(fn2)
        assert df4.data == df2.data
        assert 'fn2' in df4.header['File Name']
