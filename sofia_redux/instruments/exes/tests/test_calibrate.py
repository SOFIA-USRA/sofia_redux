# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.exes import calibrate


class TestCalibrate(object):

    def make_data(self):
        rand = np.random.RandomState(42)

        nx = 10
        ny = 10
        header = fits.Header()
        header['NSPAT'] = nx
        header['NSPEC'] = ny

        data = rand.random((ny, nx)) * 100
        flat = rand.random((ny, nx)) * 0.1
        variance = rand.random((ny, nx)) + 1.0
        flat_var = rand.random((ny, nx)) + 0.01

        return header, data, flat, variance, flat_var

    def test_calibrate(self):
        header, data, flat, variance, flat_var = self.make_data()
        orig_var = variance.copy()

        cal_data, var = calibrate.calibrate(data, header, flat,
                                            variance, flat_var)

        assert cal_data.shape == data.shape
        assert np.nanmean(cal_data) < np.nanmean(data)

        assert var.shape == orig_var.shape
        assert np.nanmean(var) > np.nanmean(orig_var)

        assert np.allclose(cal_data, data * flat)
        assert np.allclose(var, orig_var * flat ** 2 + flat_var * data ** 2)

    def test_input_errors(self, capsys):
        header, data, flat, variance, flat_var = self.make_data()

        # input is returned if error
        header['NSPAT'] = 4
        d, v = calibrate.calibrate(data, header, flat,
                                   variance, flat_var)
        assert d is data
        assert v is variance
        assert 'Data has wrong dimensions' in capsys.readouterr().err

        header['NSPAT'] = 10
        bad_var = np.ones(5)
        d, v = calibrate.calibrate(data, header, flat,
                                   bad_var, flat_var)
        assert d is data
        assert v is bad_var
        assert 'Variance has wrong dimensions' in capsys.readouterr().err

        # if flat var is bad, use zeros
        d, v = calibrate.calibrate(data, header, flat,
                                   variance, bad_var)
        assert np.allclose(d, data * flat)
        assert np.allclose(v, variance * flat ** 2)
        assert 'Flat variance has wrong dimensions' in capsys.readouterr().err
