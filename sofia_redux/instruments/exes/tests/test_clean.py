# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest
from astropy.io import fits

from sofia_redux.instruments.exes import clean as cl
from sofia_redux.toolkit.utilities.fits import set_log_level


class TestClean(object):

    def test_check_inputs(self):
        # nframes, nspec, nspat
        shape = 3, 10, 10
        data = np.empty(shape)
        std = np.empty(shape)
        mask = np.empty(shape, dtype=int)

        with pytest.raises(ValueError) as err:
            cl._check_inputs(data[0, 0], std)
        assert "data must be a 2 or 3 dimensional array" in str(err).lower()

        with pytest.raises(ValueError) as err:
            cl._check_inputs(data, std[0, 0])
        assert "std must be a 2 or 3 dimensional array" in str(err).lower()

        with pytest.raises(ValueError) as err:
            cl._check_inputs(data, std[..., :-1])
        assert "std and data shape mismatch" in str(err).lower()

        with pytest.raises(ValueError) as err:
            cl._check_inputs(data, std, mask[0, 0])
        assert "mask must be a 2 or 3 dimensional array" in str(err).lower()

        with pytest.raises(Exception) as err:
            cl._check_inputs(data, std, mask[..., :-1])
        assert "mask and data shape mismatch" in str(err).lower()

        d, s, m = cl._check_inputs(data, std, mask=mask)
        assert d.shape == shape
        assert s.shape == shape
        assert m.shape == shape

        d, s, m = cl._check_inputs(data, std, None)
        assert d.shape == shape
        assert s.shape == shape
        assert m.shape == shape
        assert np.all(m)

        d, s, m = cl._check_inputs(data[0], std[0], mask=mask[0])
        shape1 = shape[1:]
        assert d.shape == shape1
        assert s.shape == shape1
        assert m.shape == shape1

    def test_apply_badpix_mask(self, tmpdir):

        bpm = os.path.join(str(tmpdir.mkdir('test_clean')), 'test_02')

        newmask = np.full((10, 10), 1)
        newmask[0, 0] = 0
        mask = np.full((10, 10, 10), 1)
        mask[9, 9, 0] = 0

        fits.HDUList(
            fits.PrimaryHDU(data=newmask)).writeto(bpm, overwrite=True)
        header = fits.Header({'BPM': bpm, 'DETSEC': '[1,10,1,10]'})
        assert np.sum(cl._apply_badpix_mask(mask.copy(), header)) == 989
        del header['BPM']
        assert np.sum(cl._apply_badpix_mask(mask.copy(), header)) == 999

        mask = np.full((10, 10), 1)
        mask[9, 9] = 0
        header = fits.Header({'BPM': bpm, 'DETSEC': '[1,10,1,10]'})
        assert np.sum(cl._apply_badpix_mask(mask.copy(), header)) == 98
        del header['BPM']
        assert np.sum(cl._apply_badpix_mask(mask.copy(), header)) == 99

    def test_check_noise(self):

        mask = np.full((10, 10), True)
        std = np.ones((10, 10))

        std[0, 0] = 1e-6
        std[0, 1] = 1e6
        std[0, 2] = np.nan

        stdfac = 20.0

        # Check 'MAP' and 'FILEMAP' modes do not perform this check
        header = fits.Header({'SCAN': 'MAP'})
        sout, mout = cl._check_noise(std, mask, header, stdfac)
        assert sout is std
        assert mout is mask

        header = fits.Header({'SCAN': 'FILEMAP'})
        sout, mout = cl._check_noise(std, mask, header, stdfac)
        assert sout is std
        assert mout is mask

        # Check invalid STDFAC does not perform this check
        header = fits.Header()
        stdfac = -1
        sout, mout = cl._check_noise(std, mask, header, stdfac)
        assert sout is std
        assert mout is mask

        # Assert check is aborted for zero or NaN value std
        stdfac = 1.0
        sout, mout = cl._check_noise(std * 0, mask, header, stdfac)
        assert np.allclose(sout, std * 0, equal_nan=True)
        assert mout is mask
        sout, mout = cl._check_noise(std * np.nan, mask, header, stdfac)
        assert np.isnan(sout).all()
        assert mout is mask

        # Check 1 pixel flagged and new minimum set
        sout, mout = cl._check_noise(std, mask, header, stdfac)
        assert mout.sum() == 99
        assert np.nanmedian(sout) > 1

    def test_clean_frame_interpolate(self, capsys):
        # single frame from input cube
        mask = np.full((10, 10), True)
        std = np.ones((10, 10), dtype=float)
        data = np.arange(10 ** 2, dtype=float).reshape((10, 10))

        # clean mask, no NaNs
        d, s = cl._clean_frame(data, std, mask)
        assert np.allclose(d, data)
        assert np.allclose(s, std)

        # mask a few pixels: should all be interpolated over
        mask[2, 3] = False
        mask[5, 6] = False
        mask[7, 8] = False
        d, s = cl._clean_frame(data.copy(), std, mask)
        assert np.allclose(d, data)
        assert np.allclose(s[mask], 1.0)
        assert s[2, 3] == np.sqrt(2)
        assert s[5, 6] == np.sqrt(2)
        assert s[7, 8] == np.sqrt(2)
        assert 'could not be cleaned' not in capsys.readouterr().err

        # mask a whole edge: cannot be cleaned
        mask = np.full((10, 10), True)
        mask[0] = False
        d, s = cl._clean_frame(data.copy(), std, mask)
        assert np.allclose(d, data)
        assert np.allclose(s, 1.0)
        assert '10 pixels could not be cleaned' in capsys.readouterr().err

        # add some NaNs: should be cleaned, mask is updated
        mask = np.full((10, 10), True)
        nandata = data.copy()
        nandata[2, 3] = np.nan
        nandata[5, 6] = np.nan
        nandata[7, 8] = np.nan
        d, s = cl._clean_frame(nandata, std, mask)
        assert np.allclose(d, data)
        assert np.allclose(s[mask], 1.0)
        assert s[2, 3] == np.sqrt(2)
        assert s[5, 6] == np.sqrt(2)
        assert s[7, 8] == np.sqrt(2)
        assert 'could not be cleaned' not in capsys.readouterr().err

    def test_clean_frame_nan(self, capsys):
        # single frame from input cube
        mask = np.full((10, 10), True)
        std = np.ones((10, 10), dtype=float)
        data = np.arange(10 ** 2, dtype=float).reshape((10, 10))

        # clean mask, no NaNs
        d, s = cl._clean_frame(data, std, mask, propagate_nan=True)
        assert np.allclose(d, data)
        assert np.allclose(s, std)

        # mask a few pixels: should all be marked with NaN
        mask[2, 3] = False
        mask[5, 6] = False
        mask[7, 8] = False
        d, s = cl._clean_frame(data.copy(), std.copy(), mask,
                               propagate_nan=True)
        assert np.allclose(d[mask], data[mask])
        assert np.allclose(s[mask], 1.0)
        assert np.all(np.isnan(d[~mask]))
        assert np.all(np.isnan(s[~mask]))
        assert 'could not be cleaned' not in capsys.readouterr().err

        # mask a whole edge: all marked NaN
        mask = np.full((10, 10), True)
        mask[0] = False
        d, s = cl._clean_frame(data.copy(), std.copy(), mask,
                               propagate_nan=True)
        assert np.allclose(d[mask], data[mask])
        assert np.allclose(s[mask], 1.0)
        assert np.all(np.isnan(d[~mask]))
        assert np.all(np.isnan(s[~mask]))
        assert 'could not be cleaned' not in capsys.readouterr().err

        # add some NaNs: should be left as is, mask is updated
        mask = np.full((10, 10), True)
        nandata = data.copy()
        nandata[2, 3] = np.nan
        nandata[5, 6] = np.nan
        nandata[7, 8] = np.nan
        d, s = cl._clean_frame(nandata, std.copy(), mask, propagate_nan=True)
        assert np.allclose(d[mask], data[mask])
        assert np.allclose(s[mask], 1.0)
        assert np.all(np.isnan(d[~mask]))
        assert np.all(np.isnan(s[~mask]))
        assert 'could not be cleaned' not in capsys.readouterr().err

    @pytest.mark.parametrize('nframe', [4, 1])
    def test_clean(self, rdc_low_hdul, capsys, nframe):
        # same process for 3D or 2D, as long as std matches data
        if nframe > 1:
            data = rdc_low_hdul[0].data
            std = rdc_low_hdul[1].data
        else:
            data = rdc_low_hdul[0].data[0]
            std = rdc_low_hdul[1].data[0]
        header = rdc_low_hdul[0].header
        mask = np.full(data.shape, True)
        with set_log_level('DEBUG'):
            d, s = cl.clean(data.copy(), header, std.copy(), mask)
        assert d.shape == data.shape
        assert s.shape == std.shape

        # bad pixels cleaned from 4 frames
        capt = capsys.readouterr()
        assert capt.out.count('Cleaned 955 bad pixels') == nframe
        assert capt.err.count('2048 pixels could not be cleaned') == nframe
        assert (~mask).sum() == 3003 * nframe

        # good pixels are left alone
        assert np.all(d[mask] == data[mask])
        assert np.all(s[mask] == std[mask])

        # 'bad' pixels should be close to original for synthetic data,
        # error values should be higher
        assert np.allclose(d[~mask], data[~mask], rtol=0.1)
        assert not np.allclose(s[~mask], std[~mask])
        assert np.all(s[~mask] >= std[~mask])

    def test_clean_2dstd(self, rdc_low_hdul, capsys):
        # slightly different handling for 2D std, 3D data
        data = rdc_low_hdul[0].data
        header = rdc_low_hdul[0].header
        mask = np.full(data.shape, True)

        # single frame std
        std = rdc_low_hdul[1].data[0]

        with set_log_level('DEBUG'):
            d, s = cl.clean(data.copy(), header, std.copy(), mask)
        assert d.shape == data.shape
        assert s.shape == std.shape

        # bad pixels cleaned from 4 frames
        capt = capsys.readouterr()
        assert capt.out.count('Cleaned 955 bad pixels') == 4
        assert capt.err.count('2048 pixels could not be cleaned') == 4
        assert (~mask).sum() == 3003 * 4

        # good pixels are left alone
        assert np.all(d[mask] == data[mask])
        assert np.all(s[mask[0]] == std[mask[0]])

        # 'bad' pixels should be close to original for synthetic data,
        # error values should be higher
        assert np.allclose(d[~mask], data[~mask], rtol=0.1)
        assert not np.allclose(s[~mask[0]], std[~mask[0]])
        assert np.all(s[~mask[0]] >= std[~mask[0]])
