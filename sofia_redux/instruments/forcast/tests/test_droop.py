# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.droop import addhist, droop


class TestDroop(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Droop: test history message'

    def test_2d_droop(self):
        _, data = np.mgrid[:256, :256]
        data = data.astype(float)
        data[20, 20] = np.nan
        header = fits.header.Header()
        variance = np.full_like(data, 2.0)
        dripconfig.load()
        dripconfig.configuration['nrodroop'] = 16
        dripconfig.configuration['mindroop'] = 30
        dripconfig.configuration['maxdroop'] = 100
        result, var = droop(data, header=header, variance=variance, frac=0.1)
        assert var.shape == data.shape
        assert np.allclose(var, 2)
        assert np.nanmin(result) == 30
        assert np.nanmax(result) == 100
        assert np.isnan(result).sum() == 1
        assert np.isnan(result[20, 20])
        assert 'Applied channel suppression (droop) ' \
               'correction' in repr(header)

    def test_3d_droop(self):
        _, data = np.mgrid[:256, :256]
        data = np.array([data.copy(), data.copy(), data.copy()])
        data = data.astype(float)
        data[1, 20, 20] = np.nan
        header = fits.header.Header()
        variance = np.full_like(data, 2.0)
        dripconfig.load()
        dripconfig.configuration['nrodroop'] = 16
        dripconfig.configuration['mindroop'] = 30
        dripconfig.configuration['maxdroop'] = 100
        result, var = droop(data, header=header, variance=variance, frac=0.1)
        assert var.shape == data.shape
        assert np.allclose(var, 2)
        assert np.nanmin(result) == 30
        assert np.nanmax(result) == 100
        assert np.isnan(result).sum() == 1
        assert np.isnan(result[1, 20, 20])
        assert 'Applied channel suppression (droop) ' \
               'correction' in repr(header)

    def test_errors(self):
        _, data = np.mgrid[:256, :256]
        assert droop(np.array([])) is None
        result, var = droop(data, variance=np.array([]))
        assert var is None
