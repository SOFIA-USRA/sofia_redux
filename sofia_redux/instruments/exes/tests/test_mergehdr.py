# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import configobj
import numpy as np
import pytest

import sofia_redux.instruments.exes.mergehdr as mh


@pytest.fixture
def headers(rdc_header):
    hdrl = []
    for i in range(5):
        beam = 'B' if (i % 2) == 0 else 'A'
        hdr = rdc_header.copy()
        hdr['NODBEAM'] = beam
        hdr['AOR_ID'] = f'test_{i}'
        hdr['ALTI_STA'] = i + 1
        hdr['ALTI_END'] = i + 10
        hdr['EXPTIME'] = i * 100 + 1
        hdr['DTHINDEX'] = i
        hdrl.append(hdr)
    return hdrl


class TestMergehdr(object):

    def test_success(self, headers):
        # reference header is first A file
        ref = headers[1]
        # first header is first file
        first = headers[0]
        # last header is last file
        last = headers[-1]

        merged = mh.mergehdr(headers)

        # check for a few known reference/first/last keys
        assert merged['AOR_ID'] == ref['AOR_ID']
        assert merged['ALTI_STA'] == first['ALTI_STA']
        assert merged['ALTI_END'] == last['ALTI_END']

        # summed key
        assert merged['EXPTIME'] == np.sum([h['EXPTIME'] for h in headers])

        # default key
        assert merged['DTHINDEX'] == -9999

        # recalculated key
        assert np.allclose(merged['TOTTIME'], 60)

    def test_ref_header(self, headers):
        # reference header is specified
        ref = headers[3]
        first = headers[0]
        last = headers[-1]

        merged = mh.mergehdr(headers, reference_header=ref)

        # check for a few known reference/first/last keys
        assert merged['AOR_ID'] == ref['AOR_ID']
        assert merged['ALTI_STA'] == first['ALTI_STA']
        assert merged['ALTI_END'] == last['ALTI_END']

        # summed key
        assert merged['EXPTIME'] == np.sum([h['EXPTIME'] for h in headers])

        # default key
        assert merged['DTHINDEX'] == -9999

    def test_bad_config(self, tmpdir, mocker, capsys):
        mocker.patch('os.path.dirname', return_value=str(tmpdir))

        # missing config file
        headers = [fits.Header()]
        with pytest.raises(IOError):
            mh.mergehdr(headers)
        assert 'missing' in capsys.readouterr().err

        # bad config file
        pth = tmpdir.join('data', 'header')
        os.makedirs(pth)
        conf = pth.join('header_merge.cfg')
        conf.write('bad')
        with pytest.raises(configobj.ParseError):
            mh.mergehdr(headers)
        assert 'Error while loading' in capsys.readouterr().err

    def test_operations(self, tmpdir, mocker, capsys):
        mocker.patch('os.path.dirname', return_value=str(tmpdir))
        pth = tmpdir.join('data', 'header')
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
        headers = [hdr1, hdr2]

        # merge
        merge = mh.mergehdr(headers)

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
