# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
from astropy.time import Time
import pytest
import numpy as np

import sofia_redux.instruments.exes.readhdr as rh


def get_default_header(instcfg='HIGH_MED'):
    header = fits.Header({'INSTCFG': instcfg, 'WAVENO0': 1200,
                          'TESTKEY': 'test',
                          'HISTORY': 'Test history'})
    return rh.readhdr(header, check_header=False)


class TestReadHdr(object):
    def test_get_configuration_file(self):
        badfile = '__does_not_exist__.txt'
        baddir = '__does_not_exist__'
        with pytest.raises(ValueError):
            rh._get_configuration_file(badfile, check=True)
        with pytest.raises(ValueError):
            rh._get_configuration_file(badfile, subdir=baddir, check=True)

        r1 = rh._get_configuration_file(badfile, check=False)
        assert badfile in r1
        r2 = rh._get_configuration_file(badfile, subdir=baddir, check=False)
        assert badfile in r2 and len(r2) > len(r1)

        assert os.path.isfile(rh._get_configuration_file('README.rst'))
        assert os.path.isfile(rh._get_configuration_file('headerdef.dat',
                                                         subdir='header'))

    def test_get_header_configuration(self):
        with pytest.raises(ValueError):
            rh._get_header_configuration(config_file='__does_not_exist__')
        with pytest.raises(ValueError):
            rh._get_header_configuration(comments_file='__does_not_exist__')

        df = rh._get_header_configuration()
        assert df.loc['AOR_ID', 'required']

    def test_get_default(self, capsys):
        df = rh._get_header_configuration()
        assert rh._get_default('AOR_ID', df) == 'UNKNOWN'
        assert rh._get_default('ALTI_STA', df) == -9999
        assert rh._get_default('DITHER', df) is False

        # missing default - okay
        assert rh._get_default('NAIF_ID', df) is None
        assert 'not found' not in capsys.readouterr().err

        # missing key - logs error
        assert rh._get_default('BADKEY', df) is None
        assert 'not found' in capsys.readouterr().err

        # check non-robust bool handling
        df.at['DITHER', 'type'] = bool
        df.at['DITHER', 'default'] = True
        assert rh._get_default('DITHER', df) is True

        # check bad typing
        df.at['DITHER', 'type'] = dict
        assert rh._get_default('DITHER', df) is None

        # check bad default
        df.at['DITHER', 'type'] = float
        df.at['DITHER', 'default'] = 'a'
        assert rh._get_default('DITHER', df) == -9999.0
        df.at['DITHER', 'type'] = int
        assert rh._get_default('DITHER', df) == -9999

    def test_checkreq(self, capsys):
        df = rh._get_header_configuration()

        assert rh._checkreq('DATE-OBS', Time.now().isot, df)
        assert capsys.readouterr().err == ''

        assert not rh._checkreq('DATE-OBS', 'cannot parse this', df)
        assert 'wrong value' in capsys.readouterr().err

        assert rh._checkreq('not a keyword', 1.0, df)
        assert 'keyword not found' in capsys.readouterr().err

        assert rh._checkreq('DATE', 'this is not required', df)
        assert capsys.readouterr().err == ''

        # null value
        assert not rh._checkreq('ALTI_STA', None, df)
        assert 'missing value' in capsys.readouterr().err

        # incorrect type
        assert not rh._checkreq('ALTI_STA', 'a', df)
        assert 'wrong type' in capsys.readouterr().err

        # check enum
        assert not rh._checkreq('OBSTYPE', 'FOO', df)
        assert 'wrong value' in capsys.readouterr().err

        # check min
        assert not rh._checkreq('ZA_START', -100000, df)
        assert 'wrong value' in capsys.readouterr().err

        # check max
        assert not rh._checkreq('ZA_START', 91.0, df)
        assert 'wrong value' in capsys.readouterr().err

    def test_set_decimal_date(self, capsys):
        header = fits.Header()
        header['DATE-OBS'] = '2019-07-26T23:07:56'
        header['FDATE'] = 0.0
        rh._set_decimal_date(header)
        assert header['FDATE'] == 19.072623
        assert capsys.readouterr().err == ''

        header['DATE-OBS'] = '2019-07-26'
        rh._set_decimal_date(header)
        assert header['FDATE'] == 19.0726
        assert capsys.readouterr().err == ''

        header['FDATE'] = 0.0
        header['DATE-OBS'] = 'FOO'
        rh._set_decimal_date(header)
        assert header['FDATE'] == 0.0
        assert 'not understood' in capsys.readouterr().err

    def test_standardize_values(self, capsys):
        header = fits.Header()
        header['AOR_ID'] = 'test_aor'
        header['CARDMODE'] = 'none'
        header['INSTMODE'] = 'test_mode'
        header['HRG'] = 1
        header['INSTCFG'] = 'MED'
        header['EXPTIME'] = 2.0
        header['NEXP'] = -9999
        header['DATE-OBS'] = '2019-07-26T23:07:56'
        header['UTCSTART'] = '23:17:45.123'
        header['UTCEND'] = '23:18:56.234'
        rh._standardize_values(header)

        assert header['ASSC_AOR'] == header['AOR_ID']
        assert os.path.isdir(header['PKGPATH'])
        assert os.path.isdir(header['DATAPATH'])
        assert header['CARDMODE'] == 'NONE'
        assert header['INSTMODE'] == 'TEST_MODE'
        assert header['HRG'] == 1 and isinstance(header['HRG'], float)
        assert header['INSTCFG'] == 'MEDIUM'
        assert header['NEXP'] == 1
        assert header['EFL0'] == 3000.0
        assert not header['PINHOLE']
        assert np.allclose(header['TOTTIME'], 71.111)

        # check it can be converted and exists
        _ = Time(header['DATE'])

        # check handling for known bad instcfg values
        header['INSTCFG'] = 'hi-med'
        rh._standardize_values(header)
        assert header['INSTCFG'] == 'HIGH_MED'
        header['INSTCFG'] = 'hi-lo'
        rh._standardize_values(header)
        assert header['INSTCFG'] == 'HIGH_LOW'
        header['INSTCFG'] = 'lo'
        rh._standardize_values(header)
        assert header['INSTCFG'] == 'LOW'
        header['INSTCFG'] = 'cam'
        rh._standardize_values(header)
        assert header['INSTCFG'] == 'CAMERA'
        assert capsys.readouterr().err == ''

        # check bad float value
        header['EXPTIME'] = 'bad'
        rh._standardize_values(header)
        assert header['EXPTIME'] == -9999
        assert 'Unable to convert' in capsys.readouterr().err

    def test_process_instrument_configuration(self):
        header = fits.Header()
        header['XDLRDGR'] = 1.0
        header['XDMRDGR'] = 2.0
        header['ECHELLE'] = 30.0
        header['INSTCFG'] = 'HIGH_MED'
        header['WAVENO0'] = 3.0
        header['XDDGR'] = 4.0
        header['XDDELTA'] = 0.0
        rh._process_instrument_configuration(header)

        assert header['GRATANGL'] == 30
        assert header['XDDGR'] == 2
        assert np.isclose(header['XDR'], 0.5773502691896256)

        header['INSTCFG'] = 'HIGH_LOW'
        rh._process_instrument_configuration(header)
        assert np.isclose(header['XDR'], 0.2886751345948128)
        assert header['XDDGR'] == 1

        # misssing/bad resolun is replaced with default
        assert header['RESOLUN'] == 75000

        header['RESOLUN'] = -9999
        header['INSTCFG'] = 'LOW'
        rh._process_instrument_configuration(header)
        assert header['RESOLUN'] == 2000

        header['RESOLUN'] = -9999
        header['INSTCFG'] = 'MEDIUM'
        rh._process_instrument_configuration(header)
        assert header['RESOLUN'] == 10000

    def test_process_slit_configuration(self, capsys, mocker):
        header = fits.Header()
        header['FDATE'] = 0.0
        header['SDEG'] = 180.0
        rh._process_slit_configuration(header)
        assert 'SLITVAL' in header

        header['SDEG'] = -9999
        rh._process_slit_configuration(header)
        assert 'Slit angle out of range' in capsys.readouterr().err
        assert header['SLITVAL'] == 0.01

        mocker.patch.object(rh, 'goodfile', return_value=False)
        with pytest.raises(ValueError) as err:
            rh._process_slit_configuration(header)
        assert 'Could not read' in str(err)

    def test_process_tort_configuration(self, capsys, mocker):
        header = get_default_header()
        rh._process_tort_configuration(header)
        assert np.isclose(header['HRR'], 9.8)

        # override defaults in header with config values
        override_keys = ['KROT', 'SLITROT', 'HRFL0', 'XDFL0', 'HRR', 'DETROT']
        for key in override_keys:
            header[key] = -9999.
        rh._process_tort_configuration(header)
        for key in override_keys:
            assert header[key] != -9999.

        # pltscale varies by mode
        header['INSTCFG'] = 'MEDIUM'
        rh._process_tort_configuration(header)
        assert header['PLTSCALE'] == 0.201
        header['INSTCFG'] = 'LOW'
        rh._process_tort_configuration(header)
        assert header['PLTSCALE'] == 0.201
        header['INSTCFG'] = 'HIGH_MED'
        rh._process_tort_configuration(header)
        assert header['PLTSCALE'] < 0.201
        header['INSTCFG'] = 'HIGH_LOW'
        rh._process_tort_configuration(header)
        assert header['PLTSCALE'] < 0.201

        # bad/missing tort param uses low version
        capsys.readouterr()
        mocker.patch.object(rh, 'goodfile', return_value=False)
        rh._process_tort_configuration(header)
        capt = capsys.readouterr()
        assert 'Cannot read' in capt.err
        assert 'tortparm_low' in capt.out

    def test_add_configuration_files(self, mocker, tmpdir):
        header = fits.Header()
        header['FDATE'] = 0.0
        rh._add_configuration_files(header)

        for key in ['BPM', 'LINFILE', 'DRKFILE']:
            assert os.path.isfile(header[key])

        # check that fdate at boundary gets the correct values
        mocker.patch.object(rh, 'goodfile', return_value=True)
        mocker.patch.object(os.path, 'dirname', return_value=str(tmpdir))
        data_dir = str(tmpdir.join('data'))
        os.makedirs(data_dir)
        cfg = tmpdir.join('data', 'caldefault.dat')
        cfg.write(
            "22.0504 bpm.fits dark.1s.f863.fits nlcoefs.fits\n"
            "22.0505 bpm.fits dark.1s.f864.fits nlcoefs.fits\n"
            "22.0506 bpm.fits dark.1s.f865.fits nlcoefs.fits\n"
            "22.0507 bpm.fits dark.1s.f866.fits nlcoefs.fits\n"
        )

        header['FDATE'] = 22.0504
        rh._add_configuration_files(header)
        assert header['DRKFILE'] == os.path.join(
            data_dir, 'dark', 'dark.1s.f864.fits')
        header['FDATE'] = 22.050423
        rh._add_configuration_files(header)
        assert header['DRKFILE'] == os.path.join(
            data_dir, 'dark', 'dark.1s.f864.fits')
        header['FDATE'] = 22.0505
        rh._add_configuration_files(header)
        assert header['DRKFILE'] == os.path.join(
            data_dir, 'dark', 'dark.1s.f865.fits')
        header['FDATE'] = 22.050501
        rh._add_configuration_files(header)
        assert header['DRKFILE'] == os.path.join(
            data_dir, 'dark', 'dark.1s.f865.fits')
        header['FDATE'] = 22.0506
        rh._add_configuration_files(header)
        assert header['DRKFILE'] == os.path.join(
            data_dir, 'dark', 'dark.1s.f866.fits')

        # check for bad file
        mocker.patch.object(rh, 'goodfile', return_value=False)
        with pytest.raises(ValueError) as err:
            rh._add_configuration_files(header)
        assert 'Could not read' in str(err)

    def test_add_configuration_files_missing(self, mocker, tmpdir, capsys):
        header = fits.Header()
        header['FDATE'] = 0.0

        # mock config
        mocker.patch.object(os.path, 'dirname', return_value=str(tmpdir))
        data_dir = str(tmpdir.join('data'))
        os.makedirs(data_dir)
        cfg = tmpdir.join('data', 'caldefault.dat')
        cfg.write("22.0504 bpm.fits dark.fits lin.fits\n")

        # mock missing local files
        mocker.patch.object(rh, 'DATA_URL', f'file:///{str(tmpdir)}/')
        rh._add_configuration_files(header)
        capt = capsys.readouterr()
        assert capt.err.count('could not be downloaded') == 3
        assert str(tmpdir) in capt.err
        assert header['DRKFILE'] == 'dark.fits'
        assert header['BPM'] == 'bpm.fits'
        assert header['LINFILE'] == 'lin.fits'

    def test_update_header(self):
        header = get_default_header()

        # with check
        new_header, status = rh.readhdr(header, check_header=True)
        assert isinstance(new_header, fits.Header)
        assert new_header is not header

        # input values are passed to output
        assert new_header['TESTKEY'] == 'test'
        assert new_header['HISTORY'] == 'Test history'

        # without check
        new_header = rh.readhdr(header, check_header=False)
        assert isinstance(new_header, fits.Header)
        assert new_header is not header

        with pytest.raises(ValueError) as err:
            rh.readhdr('bad')
        assert 'Header is not' in str(err)

    def test_read_header_bad_value(self, capsys):
        header = get_default_header()
        header['ALTI_STA'] = 'UNKNOWN'

        # with check
        new_header, status = rh.readhdr(header, check_header=True)
        assert isinstance(new_header, fits.Header)
        assert new_header is not header

        assert status is False
        assert 'wrong type' in capsys.readouterr().err
        assert new_header['ALTI_STA'] == 'UNKNOWN'

    def test_download_cache(self, mocker, tmpdir, capsys):
        mocker.patch.object(rh, 'DATA_URL', f'file:///{str(tmpdir)}/')

        # direct test for missing file: returns basename and warns
        assert rh._download_cache_file('test_file.fits') == 'test_file.fits'
        capt = capsys.readouterr()
        assert 'File test_file.fits could not be downloaded' in capt.err
        assert str(tmpdir) in capt.err
