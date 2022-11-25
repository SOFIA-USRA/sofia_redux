# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.exes import utils


class TestUtils(object):

    def test_get_detsec(self):
        detsec = '[1,2,3,4]'
        header = fits.Header({'DETSEC': detsec})

        assert utils.get_detsec(detsec) == (0, 2, 2, 4)
        assert utils.get_detsec(header) == (0, 2, 2, 4)

        with pytest.raises(Exception) as err:
            utils.get_detsec(1)
        assert "must supply a fits header or string" in str(err).lower()

        with pytest.raises(Exception) as err:
            utils.get_detsec('foobar')
        assert "detsec must be of the format [#,#,#,#]" in str(err).lower()

        with pytest.raises(Exception) as err:
            utils.get_detsec('[1, 2, 3]')
        assert "detsec must be of the format [#,#,#,#]" in str(err).lower()

        with pytest.raises(Exception) as err:
            utils.get_detsec('[0, 1, 2, 3]')
        assert ("starting indices for detsec in x and y "
                "must be positive nonzero"
                in str(err).lower())

    def test_check_data_dimensions(self):
        nz, ny, nx = 4, 5, 6
        params = {'data': np.ones((nz, ny, nx)), 'nx': nx, 'ny': ny}

        assert utils.check_data_dimensions(params=params) == nz
        assert utils.check_data_dimensions(**params) == nz

        params['data'] = params['data'][0]
        assert utils.check_data_dimensions(params=params) == 1
        assert utils.check_data_dimensions(**params) == 1

        params['data'] = params['data'][0]
        with pytest.raises(RuntimeError):
            utils.check_data_dimensions(params=params)
        with pytest.raises(RuntimeError):
            utils.check_data_dimensions(params=params)

    def test_check_variance_dimensions(self):
        nz, ny, nx = 4, 5, 6
        params = {'variance': np.ones((nz, ny, nx)),
                  'nx': nx, 'ny': ny, 'nz': nz}

        assert utils.check_variance_dimensions(**params) is True

        params['variance'] = params['variance'][0]
        with pytest.raises(RuntimeError):
            utils.check_variance_dimensions(**params)
        params['nz'] = 1
        assert utils.check_variance_dimensions(**params) is True

        params['variance'] = params['variance'][0]
        with pytest.raises(RuntimeError):
            utils.check_variance_dimensions(**params)

        params['variance'] = None
        assert utils.check_variance_dimensions(**params) is False

    def test_get_reset_dark(self, tmpdir):
        header = fits.Header()

        # no darkfile specified
        with pytest.raises(ValueError) as err:
            utils.get_reset_dark(header)
        assert 'Cannot open dark file UNKNOWN' in str(err)

        nz, ny, nx = 4, 5, 6
        dark = str(tmpdir.join('dark.fits'))
        fits.writeto(dark, np.ones((nz, ny, nx)), overwrite=True)
        header['DRKFILE'] = dark
        with pytest.raises(ValueError) as err:
            utils.get_reset_dark(header)
        assert 'DETSEC must be' in str(err)

        header['NSPAT'] = nx
        header['NSPEC'] = ny
        dark1s = utils.get_reset_dark(header)
        assert dark1s.shape == (ny, nx)

        dark = str(tmpdir.join('dark2.fits'))
        fits.writeto(dark, np.ones((ny, nx)), overwrite=True)
        header['DRKFILE'] = dark
        dark1s = utils.get_reset_dark(header)
        assert dark1s.shape == (ny, nx)

        dark = str(tmpdir.join('dark3.fits'))
        fits.writeto(dark, np.ones(nx), overwrite=True)
        header['DRKFILE'] = dark
        with pytest.raises(ValueError) as err:
            utils.get_reset_dark(header)
        assert 'Dark file has wrong dimensions' in str(err)

    def test_set_elapsed_time(self, capsys):
        header = fits.Header()
        header['DATE-OBS'] = '2019-07-26T23:07:56'
        header['UTCSTART'] = '23:17:45.123'
        header['UTCEND'] = '23:18:56.234'
        utils.set_elapsed_time(header)
        assert np.allclose(header['TOTTIME'], 71.111)
        assert capsys.readouterr().err == ''
        del header['TOTTIME']

        # missing time okay in date-obs
        header['DATE-OBS'] = '2019-07-26'
        utils.set_elapsed_time(header)
        assert np.allclose(header['TOTTIME'], 71.111)
        assert capsys.readouterr().err == ''
        del header['TOTTIME']

        # bad key
        header['UTCSTART'] = 'FOO'
        utils.set_elapsed_time(header)
        assert 'not understood' in capsys.readouterr().err
        assert 'TOTTIME' not in header

        # start after end
        header['UTCSTART'] = '23:19:56.234'
        utils.set_elapsed_time(header)
        assert 'not understood' in capsys.readouterr().err
        assert 'TOTTIME' not in header

    def test_parse_central_wavenumber(self):
        header = fits.Header()

        # no wavenumber provided
        assert utils.parse_central_wavenumber(header) is None

        # waveno0 only
        header['WAVENO0'] = 510.0
        assert utils.parse_central_wavenumber(header) == 510.0
        header['WAVENO0'] = -9999
        assert utils.parse_central_wavenumber(header) == -9999

        # wno0 only
        del header['WAVENO0']
        header['WNO0'] = 1210.0
        assert utils.parse_central_wavenumber(header) == 1210.0
        header['WNO0'] = 0
        assert utils.parse_central_wavenumber(header) == 0

        # valid waveno, invalid wno
        header['WAVENO0'] = 510.0
        header['WNO0'] = 0
        assert utils.parse_central_wavenumber(header) == 510.0
        header['WNO0'] = -9999
        assert utils.parse_central_wavenumber(header) == 510.0

        # valid wno overrides waveno
        header['WAVENO0'] = 510.0
        header['WNO0'] = 1210.0
        assert utils.parse_central_wavenumber(header) == 1210.0
        header['WAVENO0'] = -9999
        assert utils.parse_central_wavenumber(header) == 1210.0

        # both invalid returns waveno0
        header['WAVENO0'] = 0
        header['WNO0'] = -9999
        assert utils.parse_central_wavenumber(header) == 0
