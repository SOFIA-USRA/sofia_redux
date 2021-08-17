# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

import sofia_redux.instruments.flitecam.mkspecimg as u
from sofia_redux.instruments.flitecam.tests.resources import raw_specdata
from sofia_redux.toolkit.image.adjust import rotate90


class TestMkspecimg(object):

    def make_data(self, nfiles=2):
        data = []
        for i in range(nfiles):
            beam = 'A' if (i % 2) == 0 else 'B'
            hdul = raw_specdata(dthindex=i + 1,
                                nodbeam=beam, add_ext=True)
            data.append(hdul)
        return data

    def test_no_subtract(self):
        data = self.make_data(nfiles=5)
        flux = [d[0].data.copy() for d in data]
        err = [d[1].data.copy() for d in data]

        result = u.mkspecimg(data, pair_subtract=False)

        assert isinstance(result, list)
        assert len(result) == 5

        # data is rotated, but otherwise unchanged
        for i in range(5):
            assert np.allclose(result[i]['FLUX'].data,
                               rotate90(flux[i], 1))
            assert np.allclose(result[i]['ERROR'].data,
                               rotate90(err[i], 1))
            assert 'BADMASK' not in result[i]

    def test_pair_subtract(self, capsys):
        data = self.make_data(nfiles=5)
        flux = [d[0].data.copy() for d in data]
        err = [d[1].data.copy() for d in data]

        result = u.mkspecimg(data, pair_subtract=True)

        assert isinstance(result, list)

        # two pairs are subtracted, one file is dropped
        assert len(result) == 2
        assert 'Mismatched pairs' in capsys.readouterr().err

        # data is rotated and subtracted
        sub1 = rotate90(flux[0], 1) - rotate90(flux[1], 1)
        assert np.allclose(result[0]['FLUX'].data, sub1)
        sub2 = rotate90(flux[2], 1) - rotate90(flux[3], 1)
        assert np.allclose(result[1]['FLUX'].data, sub2)

        # variance is propagated
        var1 = rotate90(err[0], 1)**2 + rotate90(err[1], 1)**2
        assert np.allclose(result[0]['ERROR'].data, np.sqrt(var1))
        var2 = rotate90(err[2], 1)**2 + rotate90(err[3], 1)**2
        assert np.allclose(result[1]['ERROR'].data, np.sqrt(var2))

        # no badmask
        assert 'BADMASK' not in result[0]
        assert 'BADMASK' not in result[1]

    def test_sign_correct(self, capsys):
        # subtract A - B (ordered by date)
        data = self.make_data(nfiles=2)
        result1 = u.mkspecimg(data, pair_subtract=True)

        # subtract B - A
        data = self.make_data(nfiles=2)
        d1 = data[0][0].header['DATE-OBS']
        d2 = data[1][0].header['DATE-OBS']
        data[0][0].header['DATE-OBS'] = d2
        data[1][0].header['DATE-OBS'] = d1
        result2 = u.mkspecimg(data, pair_subtract=True)

        # result should be the same
        for ext in ['FLUX', 'ERROR']:
            assert np.allclose(result1[0][ext].data, result2[0][ext].data)

    def test_one_file(self, capsys):
        data = self.make_data(nfiles=1)
        flux = data[0][0].data.copy()
        err = data[0][1].data.copy()

        result = u.mkspecimg(data, pair_subtract=True)

        assert isinstance(result, list)
        assert len(result) == 1

        # data is rotated, but otherwise unchanged
        assert np.allclose(result[0]['FLUX'].data, rotate90(flux, 1))
        assert np.allclose(result[0]['ERROR'].data, rotate90(err, 1))
        assert 'BADMASK' not in result[0]

        # warns that pair subtract is not applied
        assert 'turning off pair-subtraction' in capsys.readouterr().err

    def test_filenum(self):
        data = self.make_data(nfiles=5)
        filenum = list(range(5))

        # without pair sub, filenums are just returned
        result = u.mkspecimg(data, pair_subtract=False, filenum=filenum)
        assert isinstance(result, tuple)
        rdata, rfilenum = result
        assert len(rdata) == 5
        assert rfilenum == [0, 1, 2, 3, 4]

        # with pair sub, they're rearranged
        data = self.make_data(nfiles=5)
        result = u.mkspecimg(data, pair_subtract=True, filenum=filenum)
        assert isinstance(result, tuple)
        rdata, rfilenum = result
        assert len(rdata) == 2
        assert rfilenum == [[0, 1], [2, 3]]

    def test_flat(self, tmpdir):
        data = self.make_data(nfiles=2)
        flux = [d[0].data.copy() for d in data]
        err = [d[1].data.copy() for d in data]

        flat = fits.HDUList(fits.PrimaryHDU(
            np.full(data[0][0].data.shape, 2.0)))
        flatfile = str(tmpdir.join('flat.fits'))
        flat.writeto(flatfile, overwrite=True)

        # error if bad flat name provided
        with pytest.raises(ValueError) as verr:
            u.mkspecimg(data, flatfile='badfile.fits')
        assert 'Could not read' in str(verr)

        # data is divided by flat if provided
        result = u.mkspecimg(data, pair_subtract=False, flatfile=flatfile)
        assert isinstance(result, list)
        assert len(result) == 2

        # data is rotated and flat corrected
        for i in range(2):
            assert np.allclose(result[i]['FLUX'].data,
                               rotate90(flux[i], 1) / 2)
            assert np.allclose(result[i]['ERROR'].data,
                               rotate90(err[i], 1) / 2)

        # same with pair sub
        data = self.make_data(nfiles=2)
        result = u.mkspecimg(data, pair_subtract=True, flatfile=flatfile)
        assert isinstance(result, list)
        assert len(result) == 1

        sub = rotate90(flux[0], 1) - rotate90(flux[1], 1)
        var = rotate90(err[0], 1) ** 2 + rotate90(err[1], 1) ** 2
        assert np.allclose(result[0]['FLUX'].data,
                           sub / 2)
        assert np.allclose(result[0]['ERROR'].data,
                           np.sqrt(var) / 2)
