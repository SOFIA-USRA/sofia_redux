# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.exes.get_resolution import get_resolution
from sofia_redux.instruments.exes.tests.resources \
    import cross_dispersed_flat_header


class TestGetResolution(object):

    @pytest.mark.parametrize('rp', [True, False])
    def test_empty(self, tmpdir, mocker, rp):
        # for present but empty data files and low/med modes,
        # RESOLUN is returned
        header = cross_dispersed_flat_header()
        if not rp:
            del header['RP']

        # mock the file
        os.makedirs(tmpdir.join('data', 'resolution'))
        default = tmpdir.join('data', 'resolution',
                              'long_wave_resolution.txt')
        default.write('# no data present\n')

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.exes.__file__',
                     str(mock_file))

        resw = get_resolution(header)
        assert isinstance(resw, float)
        if not rp:
            assert np.allclose(resw, header['RESOLUN'])
        else:
            assert np.allclose(resw, header['RP'])

        header['INSTCFG'] = 'LOW'
        resw = get_resolution(header)
        assert isinstance(resw, float)
        if not rp:
            assert np.allclose(resw, header['RESOLUN'])
        else:
            assert np.allclose(resw, header['RP'])

    @pytest.mark.parametrize('wavecent,rfile,rp',
                             [(5., 'short_wave', 40000),
                              (12., 'medium_wave', 60000),
                              (20., 'long_wave', 80000)])
    def test_populated(self, tmpdir, mocker, wavecent, rfile, rp):
        # for populated data files and high modes,
        # data files are used
        header = cross_dispersed_flat_header()
        header['WAVECENT'] = wavecent

        # mock the file
        os.makedirs(tmpdir.join('data', 'resolution'))
        longfile = tmpdir.join('data', 'resolution',
                               'long_wave_resolution.txt')
        longfile.write('1     70000\n3    80000\n')
        medfile = tmpdir.join('data', 'resolution',
                              'medium_wave_resolution.txt')
        medfile.write('1     50000\n3    60000\n')
        shortfile = tmpdir.join('data', 'resolution',
                                'short_wave_resolution.txt')
        shortfile.write('1     30000\n3    40000\n')

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.exes.__file__',
                     str(mock_file))

        resw = get_resolution(header)
        assert isinstance(resw, float)
        assert np.allclose(resw, rp)
        assert rfile in header['RESFILE']

        # low/medium modes still use rp/resolun
        header['INSTCFG'] = 'LOW'
        resw = get_resolution(header)
        assert isinstance(resw, float)
        assert np.allclose(resw, header['RP'])

    def test_error(self, capsys):
        # bad header
        with pytest.raises(ValueError):
            get_resolution(None)
        capt = capsys.readouterr()
        assert 'Invalid header' in capt.err

        # now start with empty header
        header = fits.Header()

        # add missing keys one at a time
        header['INSTCFG'] = 'HIGH_LOW'
        for key in ['WAVECENT', 'SLITWID', 'RESOLUN']:
            with pytest.raises(ValueError):
                get_resolution(header)
            capt = capsys.readouterr()
            assert 'Header missing {}'.format(key) in capt.err
            header[key] = -9999.

        # now it has all others; should succeed
        get_resolution(header)
        assert 'short_wave' in header['RESFILE']

    def test_default_table(self, tmpdir, mocker, capsys):
        header = cross_dispersed_flat_header()

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.exes.__file__',
                     str(mock_file))

        with pytest.raises(ValueError):
            get_resolution(header)
        capt = capsys.readouterr()
        assert 'Cannot read resolution file' in capt.err
