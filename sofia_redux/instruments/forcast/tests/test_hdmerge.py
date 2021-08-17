# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import configobj
import numpy as np
import pytest

import sofia_redux.instruments.forcast.hdmerge as u
from sofia_redux.instruments.forcast.tests.resources import raw_testdata


class TestHdmerge(object):

    def make_headers(self):
        hdrl = []
        for i in range(5):
            beam = 'B' if (i % 2) == 0 else 'A'
            hdul = raw_testdata()
            hdul[0].header['NODBEAM'] = beam
            hdul[0].header['AOR_ID'] = f'test_{i}'
            hdul[0].header['ALTI_STA'] = i + 1
            hdul[0].header['ALTI_END'] = i + 10
            hdul[0].header['EXPTIME'] = i * 100 + 1
            hdul[0].header['DTHINDEX'] = i
            hdrl.append(hdul[0].header)
        return hdrl

    def test_order_single(self):
        h = fits.Header()
        h['TESTKEY'] = 'foo'
        headers = [h]
        result = u.order_headers(headers)
        assert len(result) == 2
        assert isinstance(result[0], fits.Header)
        assert len(result[1]) == 1
        assert result[0]['TESTKEY'] == 'foo'

    def test_order_multiple(self):
        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h['DATE-OBS'] = '2018-01-01T00:00:%02i' % (59 - idx)
            h['IDX'] = idx
        basehead, oheaders = u.order_headers(headers)
        assert basehead['IDX'] == (len(headers) - 1)
        assert oheaders[0]['IDX'] == (len(headers) - 1)
        assert oheaders[len(headers) - 1]['IDX'] == 0

    def test_order_nods(self):
        headers = [fits.Header() for _ in range(5)]
        for idx, h in enumerate(headers):
            h['DATE-OBS'] = '2018-01-01T00:00:%02i' % (59 - idx)
            h['IDX'] = idx
            h['NODBEAM'] = 'B' if (idx % 2) == 0 else 'A'
        basehead, oheaders = u.order_headers(headers)

        # basehead is earliest A nod
        assert basehead['IDX'] == 3

        # sorted headers are by date-obs
        assert oheaders[0]['IDX'] == 4
        assert oheaders[4]['IDX'] == 0

    def test_success(self):
        hdrl = self.make_headers()

        # reference header is first A file
        ref = hdrl[1]
        # first header is first file
        first = hdrl[0]
        # last header is last file
        last = hdrl[-1]

        merged = u.hdmerge(hdrl)

        # check for a few known reference/first/last keys
        assert merged['AOR_ID'] == ref['AOR_ID']
        assert merged['ALTI_STA'] == first['ALTI_STA']
        assert merged['ALTI_END'] == last['ALTI_END']

        # summed key
        assert merged['EXPTIME'] == np.sum([h['EXPTIME'] for h in hdrl])

        # default key
        assert merged['DTHINDEX'] == -9999

    def test_ref_header(self):
        hdrl = self.make_headers()

        # reference header is specified
        ref = hdrl[3]
        first = hdrl[0]
        last = hdrl[-1]

        merged = u.hdmerge(hdrl, reference_header=ref)

        # check for a few known reference/first/last keys
        assert merged['AOR_ID'] == ref['AOR_ID']
        assert merged['ALTI_STA'] == first['ALTI_STA']
        assert merged['ALTI_END'] == last['ALTI_END']

        # summed key
        assert merged['EXPTIME'] == np.sum([h['EXPTIME'] for h in hdrl])

        # default key
        assert merged['DTHINDEX'] == -9999

    def test_bad_config(self, tmpdir, mocker, capsys):
        mocker.patch('os.path.dirname', return_value=str(tmpdir))

        # missing config file
        hdrl = [fits.Header()]
        with pytest.raises(IOError):
            u.hdmerge(hdrl)
        assert 'missing' in capsys.readouterr().err

        # bad config file
        pth = tmpdir.join('data')
        os.makedirs(pth)
        conf = pth.join('header_merge.cfg')
        conf.write('bad')
        with pytest.raises(configobj.ParseError):
            u.hdmerge(hdrl)
        assert 'Error while loading' in capsys.readouterr().err

    def test_operations(self, tmpdir, mocker, capsys):
        mocker.patch('os.path.dirname', return_value=str(tmpdir))
        pth = tmpdir.join('data')
        os.makedirs(pth)
        conf = pth.join('header_merge.cfg')

        # write config file with all operations, for valid
        # and invalid keys
        conf.write('TEST1 = FIRST\n'
                   'TEST2 = LAST\n'
                   'TEST3 = SUM\n'
                   'TEST4 = SUM\n'
                   'TEST5 = MEAN\n'
                   'TEST6 = MEAN\n'
                   'TEST7 = AND\n'
                   'TEST8 = AND\n'
                   'TEST9 = OR\n'
                   'TEST10 = OR\n'
                   'TEST11 = CONCATENATE\n'
                   'TEST12 = DEFAULT\n'
                   'TEST13 = DEFAULT\n'
                   'TEST14 = DEFAULT\n'
                   'TEST15 = BAD')

        # make headers to combine
        hdr1 = fits.Header({'TEST1': 'A', 'TEST2': 'A', 'TEST3': 1.0,
                            'TEST4': 'A', 'TEST5': 1.0, 'TEST6': 'A',
                            'TEST7': True, 'TEST8': 'A', 'TEST9': True,
                            'TEST10': 'A', 'TEST11': 'A', 'TEST12': 'A',
                            'TEST13': 1, 'TEST14': 1.0, 'TEST15': 'A'})
        hdr2 = fits.Header({'TEST1': 'B', 'TEST2': 'B', 'TEST3': 2.0,
                            'TEST4': 'B', 'TEST5': 2.0, 'TEST6': 'B',
                            'TEST7': False, 'TEST8': 'B', 'TEST9': False,
                            'TEST10': 'B', 'TEST11': 'B', 'TEST12': 'B',
                            'TEST13': 2, 'TEST14': 2.0, 'TEST15': 'B'})
        hdrl = [hdr1, hdr2]

        # merge
        merge = u.hdmerge(hdrl)

        # check values
        # invalid operations default to first header
        assert merge['TEST1'] == 'A'
        assert merge['TEST2'] == 'B'
        assert merge['TEST3'] == 3.0
        assert merge['TEST4'] == 'A'
        assert merge['TEST5'] == 1.5
        assert merge['TEST6'] == 'A'
        assert merge['TEST7'] is False
        assert merge['TEST8'] == 'A'
        assert merge['TEST9'] is True
        assert merge['TEST10'] == 'A'
        assert merge['TEST11'] == 'A,B'
        assert merge['TEST12'] == 'UNKNOWN'
        assert merge['TEST13'] == -9999
        assert merge['TEST14'] == -9999.0
        assert merge['TEST15'] == 'A'

        # check for expected warnings
        capt = capsys.readouterr()
        assert 'SUM operation is invalid for TEST4' in capt.err
        assert 'MEAN operation is invalid for TEST6' in capt.err
        assert 'AND operation is invalid for TEST8' in capt.err
        assert 'OR operation is invalid for TEST10' in capt.err
        assert 'Invalid key merge operation BAD' in capt.err
