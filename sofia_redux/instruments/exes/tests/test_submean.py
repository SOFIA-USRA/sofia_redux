# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.exes import submean as sm


class TestSubMean(object):

    def test_submean(self):
        nz, ny, nx = 5, 10, 20
        data = np.full((nz, ny, nx), 1.0)
        flat = np.full((ny, nx), 1.0)
        order_mask = np.full((ny, nx), 1)
        header = fits.Header({'NSPAT': nx, 'NSPEC': ny})

        # flat data: corrected to zero
        corrected = sm.submean(data, header, flat, None, order_mask)
        assert np.allclose(corrected, 0)

        # un-flat data: corrected to zero mean by frame
        data = np.arange(nz * ny * nx, dtype=float).reshape((nz, ny, nx))
        corrected = sm.submean(data, header, flat, None, order_mask)
        assert np.allclose(corrected.mean(), 0)
        assert np.allclose(corrected, corrected[0])

    def test_submean_input_errors(self, capsys):
        nz, ny, nx = 5, 10, 20
        data = np.full((nz, ny, nx), 1.0)
        flat = np.full((ny, nx), 1.0)
        illum = np.full((ny, nx), 1)
        order_mask = np.full((ny, nx), 1)
        header = fits.Header({'NSPAT': nx, 'NSPEC': ny})

        # returns data in case of error
        corrected = sm.submean(data[:, 0, :], header, flat, illum, order_mask)
        assert np.allclose(corrected, 1.0)
        assert 'Data has wrong dimensions' in capsys.readouterr().err

        corrected = sm.submean(data, header, flat, illum[0], order_mask)
        assert np.allclose(corrected, 1.0)
        assert 'Illum array has wrong dimensions' in capsys.readouterr().err

    def test_order_average(self):
        ny, nx = 10, 20
        data = np.arange(ny * nx, dtype=float).reshape((ny, nx))
        flat = np.ones((ny, nx), dtype=float)
        good = np.full((ny, nx), True)
        order_mask = np.full((ny, nx), 1)

        # single order, all good, flat weight
        avg = sm._multi_order_avg(data, flat, good, order_mask)
        assert avg.shape == (ny, nx)
        assert np.allclose(avg, np.mean(data, axis=0))

        # down-weight higher values
        flat *= np.arange(ny)[:, None] + 1
        avg = sm._multi_order_avg(data, flat, good, order_mask)
        assert np.all(avg < np.mean(data, axis=0))
        flat = np.ones((ny, nx), dtype=float)

        # mark higher values bad
        good[5:] = False
        avg = sm._multi_order_avg(data, flat, good, order_mask)
        assert np.allclose(avg, np.mean(data[:5], axis=0))
        good = np.full((ny, nx), True)

        # multi order
        order_mask[:5] = 1
        order_mask[5:] = 2
        avg = sm._multi_order_avg(data, flat, good, order_mask)
        assert avg.shape == (ny, nx)
        assert np.allclose(avg[:5], np.mean(data[:5], axis=0))
        assert np.allclose(avg[5:], np.mean(data[5:], axis=0))

        # all data bad: avg is zero
        good[:] = False
        avg = sm._multi_order_avg(data, flat, good, order_mask)
        assert np.all(avg == 0)
