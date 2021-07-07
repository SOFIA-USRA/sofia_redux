# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for cal factor calculation."""

import numpy as np
import pytest

from sofia_redux.calibration.pipecal_calfac import pipecal_calfac
from sofia_redux.calibration.pipecal_error import PipeCalError


class TestCalfac(object):
    def mock_config(self):
        # Build mock config dictionary
        config = dict()
        config['wref'] = 5.35578
        config['lpivot'] = 5.35551
        config['std_flux'] = 2144.43
        config['std_eflux'] = 5
        config['color_corr'] = 1.00
        return config

    def test_calc(self):
        config = self.mock_config()

        flux = 2000.0
        flux_err = 15.0

        correct_calfac = 0.93255
        correct_ecalfac = 0.047149

        calfac, ecalfac = pipecal_calfac(flux, flux_err, config)

        rtol = 1e-2
        assert np.allclose(calfac, correct_calfac, rtol=rtol)
        assert np.allclose(ecalfac, correct_ecalfac, rtol=rtol)

    def test_errors(self, capsys):
        config = self.mock_config()
        flux = 2000.0
        flux_err = 15.0

        config['wref'] = 0.0
        with pytest.raises(PipeCalError):
            pipecal_calfac(flux, flux_err, config)
        capt = capsys.readouterr()
        assert 'All inputs to pipecal_calfac must be positive' in capt.err

        config['wref'] = -9
        with pytest.raises(PipeCalError):
            pipecal_calfac(flux, flux_err, config)
        capt = capsys.readouterr()
        assert 'All inputs to pipecal_calfac must be positive' in capt.err

        config.pop('wref')
        with pytest.raises(PipeCalError):
            pipecal_calfac(flux, flux_err, config)
        capt = capsys.readouterr()
        assert 'Invalid config' in capt.err
