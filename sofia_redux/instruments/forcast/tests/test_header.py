# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from astropy import log
from astropy.io import fits
import pandas
import pytest

import sofia_redux.instruments.forcast as drip
import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.header \
    import (hdadd, addparent, hdkeymerge,
            hdmerge, hdinsert, href, kref)
from sofia_redux.instruments.forcast.tests.resources import nmc_testdata

REFLINE = '--------- Pipeline related Keywords --------'


def fake_data():
    hdrs = []
    for i in range(5):
        # make a bare header, as if from readfits
        header = fits.header.Header()

        # before pipeline
        header['FILENAME'] = ('File {}'.format(i + 1), 'File name')
        header['TESTKEY{}'.format(2 * i + 1)] = ('TESTVAL1', 'Test value 1')

        # keywords added by pipeline
        header['COMMENT'] = '--------------------------------------------'
        header['COMMENT'] = REFLINE
        header['COMMENT'] = '--------------------------------------------'
        header.append((kref, 'Keywords reference',
                       'Header reference keyword'), end=True)
        header['TSTKEY{}'.format(2 * i + 2)] = ('TESTVAL2', 'Test value 2')
        header['PARENT1'] = ('parent {}'.format(2 * i + 1), 'Parent 1')
        header['PARENT2'] = ('parent {}'.format(2 * i + 2), 'Parent 2')
        header['TSTKEY{}'.format(2 * i + 3)] = ('TESTVAL3', 'Test value 3')
        header[href] = ('History reference', 'Header reference keyword')

        # history added by pipeline
        header['HISTORY'] = '--------------------------------------------'
        header['HISTORY'] = '---------- PIPELINE HISTORY -----------'
        header['HISTORY'] = '--------------------------------------------'
        header['HISTORY'] = 'Test history'
        hdrs.append(header)
    return hdrs


class TestHdmerge(object):
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def test_hdinsert(self):
        headers = fake_data()
        h = headers[0].copy()

        assert 'FOO' not in h
        hdinsert(h, 'FOO', 'BAR', comment='a foobar')
        assert 'FOO' in h
        assert h['FOO'] == 'BAR'
        assert h.comments['FOO'] == 'a foobar'
        assert h.count('FOO') == 1

        idx = h.index('FOO')
        assert h.cards[idx + 1][0] == 'HISTORY'
        assert h.cards[idx - 1][0] != 'HISTORY'

        hdinsert(h, 'FOO', 'BAR2')
        assert h.count('FOO') == 1
        assert h['FOO'] == 'BAR2'
        assert h.comments['FOO'] == 'a foobar'
        hdinsert(h, 'FOO', 'BAR2', comment='changed my foobar')
        assert h.comments['FOO'] == 'changed my foobar'

        hdinsert(h, 'foo4', 4)
        hdinsert(h, 'foo3', 3, refkey='FOO4')
        hdinsert(h, 'foo2', 2, refkey='FOO', after=True)
        for place, kw in enumerate(['foo2', 'foo3', 'foo4']):
            assert h.cards[idx + place + 1][0] == kw.upper()
            assert h.cards[idx + place + 1][1] == place + 2

    def test_hdadd(self):
        headers = fake_data()

        reference = headers[0].copy()
        header = headers[1].copy()
        refparents = reference.cards['PARENT*']
        hdrparents = header.cards['PARENT*']
        hdadd(header, reference, REFLINE)

        # check the parents have been added
        assert reference['PARENT1'] == refparents[0][1]
        assert reference['PARENT2'] == refparents[1][1]
        assert reference['PARENT3'] == hdrparents[0][1]
        assert reference['PARENT4'] == hdrparents[1][1]
        # check history is updated
        assert not headers[0]['HISTORY'] == reference['HISTORY']

        refnohist = reference.copy()
        reference = headers[0].copy()
        header = headers[1].copy()
        hdadd(header, reference, REFLINE, do_history=True)
        # check we have extra history
        assert len(reference) > len(refnohist)
        appended = reference.cards[len(refnohist):]
        vals = [x[1] for x in appended if x[1] != '']
        # check new history is not blank
        assert not len(vals) == 0

        # check that we can add coad* keywords, but not
        # skipped ones (mrgd*)
        reference = headers[0].copy()
        header = headers[1].copy()
        hdinsert(reference, 'COADX0', 1)
        hdinsert(reference, 'COADY0', -1)
        hdinsert(reference, 'COADX1', 2)
        hdinsert(reference, 'COADY1', -2)
        hdinsert(reference, 'MRGDX0', 1)
        hdinsert(reference, 'MRGDY0', -1)
        header['COADX0'] = 3
        header['COADY0'] = -3
        header['MRGDX0'] = 3
        header['MRGDY0'] = -3
        # add some duplicate keys too -- should take first only
        header.extend([('TEST', 'val1'), ('TEST', 'val2')])
        hdadd(header, reference, REFLINE)
        for f, c in zip([1, -1], ['X', 'Y']):
            for i in range(3):
                assert reference['COAD%s%s' % (c, i)] == f * (i + 1)
                assert reference['COAD%s%s' % (c, i)] == f * (i + 1)
                if i == 0:
                    assert reference['MRGD%s%s' % (c, i)] == f * (i + 1)
                    assert reference['MRGD%s%s' % (c, i)] == f * (i + 1)
                else:
                    assert 'MRGD%s%s' % (c, i) not in reference
        assert reference.count('TEST') == 1
        assert reference['TEST'] == 'val1'

    def test_hdkeymerge(self):
        reference = fits.header.Header()
        ntest = 5
        headerlist = [fits.header.Header() for _ in range(ntest)]
        for idx, header in enumerate(headerlist):
            header['FILENAME'] = 'file %s' % idx
            header['DATE'] = '2018-0%s-01T12:00:00' % (idx + 1)
            header['FIRST'] = idx
            header['LAST'] = idx
            header['MEAN'] = float(idx)
            header['MEDIAN'] = idx
            header['SUM'] = idx
            header['CONCATENATE'] = str(idx)
            header['SAME'] = 'a value'
            header['MIN'] = idx
            header['MAX'] = idx
        ktypes = [k.lower() for k in header.keys()]
        for ktype in ktypes:
            hdkeymerge(headerlist, reference, ktype.upper(), ktype)
        assert 'HISTORY' not in reference
        assert reference['FIRST'] == 0
        assert reference['LAST'] == ntest - 1
        assert reference['MEAN'] == sum(range(ntest)) / ntest
        assert reference['MEDIAN'] == [*range(ntest)][ntest // 2]
        assert reference['SUM'] == sum(range(ntest))
        assert reference['CONCATENATE'] == ','.join([str(x)
                                                     for x in range(ntest)])
        assert reference['SAME'] == 'a value'
        assert reference['MIN'] == 0
        assert reference['MAX'] == ntest - 1

        headtest = fits.header.Header()
        headerlist[0]['DATE'] = 'foo'
        for ktype in ktypes:
            hdkeymerge(headerlist, headtest, ktype.upper(), ktype)
        assert headtest['FIRST'] == 1
        assert len(headtest['HISTORY*']) == 2
        headtest = fits.header.Header()
        headerlist[ntest - 1]['DATE'] = 'foo'
        for ktype in ktypes:
            hdkeymerge(headerlist, headtest, ktype.upper(), ktype)
        assert len(headtest['HISTORY*']) == 4
        assert headtest['FIRST'] == 1
        assert headtest['LAST'] == ntest - 2

        headerlist[ntest - 1]['max'] = 'foo'
        headtest = fits.header.Header()
        hdkeymerge(headerlist, headtest, 'MAX', 'max')
        assert headtest['MAX'] == ntest - 2

        headers = [fits.header.Header() for _ in range(3)]
        reference = fits.header.Header()
        for h in headers:
            h['STRING'] = 'a value'
            h['FLOAT'] = 1
        hdkeymerge(headers, reference, 'STRING', 'multidefstr')
        hdkeymerge(headers, reference, 'FLOAT', 'multidefflt')
        assert reference['STRING'] == 'UNKNOWN'
        assert reference['FLOAT'] == -9999
        headers = [headers[0]]
        hdkeymerge(headers, reference, 'STRING', 'multidefstr')
        hdkeymerge(headers, reference, 'FLOAT', 'multidefflt')
        assert reference['STRING'] == 'UNKNOWN'
        assert reference['FLOAT'] == -9999
        reference = fits.header.Header()
        hdkeymerge(headers, reference, 'STRING', 'multidefstr')
        hdkeymerge(headers, reference, 'FLOAT', 'multidefflt')
        assert 'STRING' not in reference
        assert 'FLOAT' not in reference

    def test_hdkeymerge_errors(self, capsys):
        reference = fits.header.Header()
        ntest = 2
        headerlist = [fits.header.Header() for _ in range(ntest)]
        for idx, header in enumerate(headerlist):
            header['FILENAME'] = 'file %s' % idx
            if idx < len(headerlist) - 1:
                header['DATE'] = '2018-0%s-01T12:00:00' % (idx + 1)
            if idx > 0:
                header['FIRST'] = idx
            header['LAST'] = idx
            header['MEAN'] = 'a'
            header['MEDIAN'] = idx
            header['SUM'] = idx
            header['CONCATENATE'] = str(idx)
            header['SAME'] = 'a value'

        ktypes = [k.lower() for k in header.keys()]
        ktypes.append('max')
        for ktype in ktypes:
            hdkeymerge(headerlist, reference, ktype.upper(), ktype)
            capt = capsys.readouterr()
            if ktype == 'first':
                assert 'keyword is not present' in capt.out
                assert reference['FIRST'] == 1
            elif ktype == 'last':
                assert 'DATE keyword is not present' in capt.err
            elif ktype == 'max':
                assert 'keyword not present in header list' in capt.out
            elif ktype == 'mean':
                assert 'cannot take mean' in capt.err

        # also check for an entirely missing last key
        hdkeymerge(headerlist, reference, 'TEST', 'last')
        capt = capsys.readouterr()
        assert 'keyword not present in header list' in capt.out

    def test_hdmerge(self):
        reference = fits.header.Header()
        headers = [fits.header.Header() for _ in range(5)]
        fkeydef = os.path.join(os.path.dirname(drip.__file__), 'data',
                               'output-key-merge.txt')
        keydef = pandas.read_csv(
            fkeydef, comment=';', delim_whitespace=True,
            names=['keyword', 'merge_type'])
        for idx, header in enumerate(headers):
            header['DATE'] = '2018-0%s-10T10:00:00' % (idx + 1)
            header['FILENAME'] = 'file %s' % idx
            header['PARENT1'] = 'file %s' % idx
            for _, row in keydef.iterrows():
                key, mt = row['keyword'], row['merge_type']
                if mt in ['first', 'last', 'multidefstr', 'concatenate']:
                    def t(x):
                        return str(x).strip()
                else:
                    t = float
                header[key] = t(idx)

        merged = hdmerge(headers, reference)
        firsttypes = ['min', 'first', 'same']
        for t in firsttypes:
            for key in keydef.loc[keydef['merge_type'] == t]['keyword']:
                assert headers[0][key] == merged[key]

        lasttypes = ['max', 'last']
        for t in lasttypes:
            for key in keydef.loc[keydef['merge_type'] == t]['keyword']:
                assert headers[4][key] == merged[key]

        for key in keydef.loc[keydef['merge_type']
                              == 'concatenate']['keyword']:
            assert merged[key] == '0,1,2,3,4'

        assert 'PARENT1' not in merged
        for header in headers:
            hdinsert(header, 'COMMENT', REFLINE, refkey='DATE')
            hdinsert(header, 'COMMENT', '.....', refkey=REFLINE, after=True)
        merged = hdmerge(headers, reference)
        assert len(merged['PARENT*']), 5

        h1 = fits.header.Header()
        h2 = h1.copy()
        for i in range(10):
            h1['H1_KEY%s' % i] = i
            h2['H2_KEY%s' % i] = i
        h1.insert(5, ('COMMENT', '----'))
        h1.insert(6, ('COMMENT', REFLINE))
        h1.insert(7, ('COMMENT', '----'))
        h2.insert(5, ('COMMENT', '----'))
        h2.insert(6, ('COMMENT', REFLINE))
        h2.insert(7, ('COMMENT', '----'))
        reference = fits.header.Header()
        merged = hdmerge([], reference, hdinit=h2)
        assert merged == h2
        merged = hdmerge([], h1, hdinit=reference)
        assert merged is None
        merged = hdmerge([h1], reference, hdinit=h2)
        assert REFLINE in [*merged.values()]
        assert merged[:5] == h2[:5]
        assert merged[8:13] == h2[8:]
        assert merged[13:] == h1[8:]

    def test_hdmerge_errors(self, capsys, tmpdir):
        headers = fake_data()

        # bad header list
        result = hdmerge(None, headers[0])
        capt = capsys.readouterr()
        assert 'invalid header list' in capt.err
        assert result is None

        # bad header in list
        result = hdmerge([headers[0], None], headers[0])
        capt = capsys.readouterr()
        assert 'invalid header list' in capt.err
        assert result is None

        # bad reference header
        result = hdmerge(headers, None)
        capt = capsys.readouterr()
        assert 'invalid reference header' in capt.err
        assert result is None

        # bad init header
        result = hdmerge(headers, headers[0], hdinit=1)
        capt = capsys.readouterr()
        assert 'invalid hdinit header' in capt.err
        assert result is None

        # init header with refline in reference,
        # but not in hdinit (okay)
        hdinit = headers[0].copy()
        del hdinit[[*hdinit.values()].index(REFLINE)]
        result = hdmerge(headers, headers[0], hdinit=hdinit)
        assert result is not None

        # bad keydef paths: issues error, but returns
        # hdadd merged header, without extra keyword merge
        # handling
        dripconfig.load()
        dripconfig.configuration['caldata'] = 'BADPATH'
        result = hdmerge(headers, headers[0], fkeydef='badfile.txt')
        capt = capsys.readouterr()
        assert 'input file and configuration do ' \
               'not define CALDATA' in capt.err
        assert result is not None

        dripconfig.configuration['caldata'] = str(tmpdir)
        result = hdmerge(headers, headers[0])
        capt = capsys.readouterr()
        assert 'file does not exist' in capt.err
        assert result is not None

        bad_file = tmpdir.join('badfile.txt')
        bad_file.write('; no data')
        result = hdmerge(headers, headers[0], fkeydef=str(bad_file))
        capt = capsys.readouterr()
        assert 'No valid lines' in capt.err
        assert result is not None

    def test_one_parent(self):
        header = nmc_testdata()['header']
        assert 'PARENT1' not in header
        addparent('TESTING_ADDPARENT', header)
        assert 'PARENT1' in header
        assert header['PARENT1'] == 'TESTING_ADDPARENT'

    def test_two_parents(self):
        header = nmc_testdata()['header']
        assert 'PARENT1' not in header
        addparent('FOO', header)
        assert 'PARENT1' in header
        # adding a parent with the same value as one that already
        # exists should not result in the addition of a new parent
        addparent('FOO', header)
        assert 'PARENT2' not in header
        # test we can add a parent if the value is new
        addparent('BAR', header)
        assert 'PARENT2' in header

    def test_comments(self):
        header = nmc_testdata()['header']
        assert 'PARENT1' not in header
        addparent('FOO', header)
        # test the default comment
        assert header.comments['PARENT1'] == 'id or name of file ' \
                                             'used in the processing'
        # add a new parent with a non-default message
        addparent('BAR', header, comment='test comment')
        assert 'PARENT2' in header
        assert header.comments['PARENT2'] == 'test comment'
