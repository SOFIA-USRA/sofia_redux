# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for `pipecal_fitpeak` module"""

from importlib import reload
import os

import numpy as np
import pytest

from sofia_redux.calibration.pipecal_fitpeak import pipecal_fitpeak
from sofia_redux.calibration.pipecal_error import PipeCalError


class TestFitPeak(object):

    def model_param(self, wr_col, wr_row, model):
        rowtest, coltest = np.mgrid[0:2 * wr_row, 0:2 * wr_col]
        par = {'baseline': 0.1,
               'dpeak': 20.0,
               'col_mean': 15, 'row_mean': 20,
               'col_sigma': 4, 'row_sigma': 2,
               'theta': 10 * np.pi / 180.}
        if 'moffat' in model:
            par['beta'] = 2.0

        # re-import fitpeak for jit testing reasons
        import sofia_redux.calibration.pipecal_fitpeak as fp
        fp = reload(fp)
        func = getattr(fp, model)
        stamp = func((rowtest, coltest), **par).reshape(2 * wr_row,
                                                        2 * wr_col)
        return stamp, par

    @pytest.mark.parametrize('model',
                             ['elliptical_gaussian',
                              'elliptical_lorentzian',
                              'elliptical_moffat'])
    def test_no_jit(self, model):
        # note: jit decorators on models were removed in v1.0.1,
        # but this test will stay for now, in case they come back
        wr_col = 20
        wr_row = 40

        # run without jit for coverage metrics
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        stamp_no_j, _ = self.model_param(wr_col, wr_row, model)

        # reload module, without jit disable, and rerun
        del os.environ['NUMBA_DISABLE_JIT']
        stamp_j, _ = self.model_param(wr_col, wr_row, model)

        # compare result with and without jit for each model used
        assert np.allclose(stamp_j, stamp_no_j)

    def test_pipecal_fitpeak_gauss(self):
        """Test the peak fitting method for gaussian function."""
        # Window radius
        wr_col = 20
        wr_row = 40
        stamp, par = self.model_param(wr_col, wr_row, 'elliptical_gaussian')

        # close but not exact estimates
        est = {'baseline': 0.,
               'dpeak': 10.,
               'col_mean': 10, 'row_mean': 15,
               'col_sigma': 1, 'row_sigma': 1,
               'theta': 0.0}

        fitp, _, _ = pipecal_fitpeak(stamp, profile='gaussian',
                                     estimates=est)

        # Relative tolerance
        rtol = 0.02

        # should be nearly the same
        same_keys = ['baseline', 'dpeak', 'col_mean', 'row_mean']
        for key in same_keys:
            np.testing.assert_allclose(fitp[key], par[key], rtol=rtol)

        # widths/angles may be 90 degrees swapped
        try:
            np.testing.assert_allclose(fitp['theta'], par['theta'], rtol=rtol)
            np.testing.assert_allclose(fitp['col_sigma'], par['col_sigma'],
                                       rtol=rtol)
            np.testing.assert_allclose(fitp['row_sigma'], par['row_sigma'],
                                       rtol=rtol)
        except AssertionError:
            np.testing.assert_allclose(np.abs(fitp['theta'] - par['theta']),
                                       np.pi / 2, rtol=rtol)
            np.testing.assert_allclose(fitp['col_sigma'], par['row_sigma'],
                                       rtol=rtol)
            np.testing.assert_allclose(fitp['row_sigma'], par['col_sigma'],
                                       rtol=rtol)

    def test_pipecal_fitpeak_moffat(self):
        # Window radius
        wr_col = 20
        wr_row = 40
        stamp, par = self.model_param(wr_col, wr_row, 'elliptical_moffat')

        # close but not exact estimates
        est = {'baseline': 0.,
               'dpeak': 10.,
               'col_mean': 10, 'row_mean': 15,
               'col_sigma': 1, 'row_sigma': 1,
               'theta': 0.0,
               'beta': 3.0}

        fitp, _, _ = pipecal_fitpeak(stamp, profile='moffat',
                                     estimates=est)

        # Relative tolerance
        rtol = 0.02

        # should be nearly the same
        same_keys = ['baseline', 'dpeak', 'col_mean', 'row_mean', 'beta']
        for key in same_keys:
            np.testing.assert_allclose(fitp[key], par[key], rtol=rtol)

        # widths/angles may be 90 degrees swapped
        try:
            np.testing.assert_allclose(fitp['theta'], par['theta'], rtol=rtol)
            np.testing.assert_allclose(fitp['col_sigma'], par['col_sigma'],
                                       rtol=rtol)
            np.testing.assert_allclose(fitp['row_sigma'], par['row_sigma'],
                                       rtol=rtol)
        except AssertionError:
            np.testing.assert_allclose(np.abs(fitp['theta'] - par['theta']),
                                       np.pi / 2, rtol=rtol)
            np.testing.assert_allclose(fitp['col_sigma'], par['row_sigma'],
                                       rtol=rtol)
            np.testing.assert_allclose(fitp['row_sigma'], par['col_sigma'],
                                       rtol=rtol)

    def test_pipecal_fitpeak_lorentzian(self):
        """Test the peak fitting method for lorentzian function."""
        # Window radius
        wr_col = 20
        wr_row = 40
        stamp, par = self.model_param(wr_col, wr_row, 'elliptical_lorentzian')

        # close but not exact estimates
        est = {'baseline': 0.,
               'dpeak': 10.,
               'col_mean': 10, 'row_mean': 15,
               'col_sigma': 1, 'row_sigma': 1,
               'theta': 0.0}

        fitp, _, _ = pipecal_fitpeak(stamp, profile='lorentzian',
                                     estimates=est)

        # Relative tolerance
        rtol = 0.02

        # should be nearly the same
        same_keys = ['baseline', 'dpeak', 'col_mean', 'row_mean']
        for key in same_keys:
            np.testing.assert_allclose(fitp[key], par[key], rtol=rtol)

        # widths/angles may be 90 degrees swapped
        try:
            np.testing.assert_allclose(fitp['theta'], par['theta'], rtol=rtol)
            np.testing.assert_allclose(fitp['col_sigma'], par['col_sigma'],
                                       rtol=rtol)
            np.testing.assert_allclose(fitp['row_sigma'], par['row_sigma'],
                                       rtol=rtol)
        except AssertionError:
            np.testing.assert_allclose(np.abs(fitp['theta'] - par['theta']),
                                       np.pi / 2, rtol=rtol)
            np.testing.assert_allclose(fitp['col_sigma'], par['row_sigma'],
                                       rtol=rtol)
            np.testing.assert_allclose(fitp['row_sigma'], par['col_sigma'],
                                       rtol=rtol)

    def test_fitpeak_errors(self, capsys):
        # Window radius: center source
        wr_col = 15
        wr_row = 20
        stamp, par = self.model_param(wr_col, wr_row, 'elliptical_gaussian')
        stamp_err = 0.1 * stamp

        # close but not exact estimates
        est = {'baseline': 0.,
               'dpeak': 10.,
               'col_mean': 10, 'row_mean': 15,
               'col_sigma': 1, 'row_sigma': 1,
               'theta': 0.0}

        # bad image
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(np.array(10), profile='gaussian',
                            estimates=est)
        capt = capsys.readouterr()
        assert 'Image must be 2-dimensional' in capt.err

        # bad error
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            estimates=est, error=np.array(10))
        capt = capsys.readouterr()
        assert 'Error must be 2-dimensional' in capt.err

        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            estimates=est, error=np.zeros((10, 10)))
        capt = capsys.readouterr()
        assert 'Error must have the same shape' in capt.err

        # bad profile
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='unknown',
                            estimates=est)
        capt = capsys.readouterr()
        assert 'Profile must be one of' in capt.err

        # bad estimates: not a dictionary
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            estimates=list(est.values()))
        capt = capsys.readouterr()
        assert 'Estimates must be a dictionary' in capt.err

        # bad estimates: missing key, not moffat
        bad_est = est.copy()
        del bad_est['theta']
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            estimates=bad_est)
        capt = capsys.readouterr()
        assert 'Estimates missing required keys' in capt.err

        # bad estimates: missing beta key for moffat
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='moffat',
                            estimates=est)
        capt = capsys.readouterr()
        assert 'Estimates missing required keys' in capt.err

        # missing all estimates: no error, should still be reasonable fit
        # for centered source
        result = pipecal_fitpeak(stamp, profile='gaussian', error=stamp_err)
        assert np.allclose(result[0]['col_mean'], par['col_mean'], atol=0.1)
        assert np.allclose(result[0]['row_mean'], par['row_mean'], atol=0.1)
        result = pipecal_fitpeak(stamp, profile='moffat', estimates={})
        assert np.allclose(result[0]['col_mean'], par['col_mean'], atol=0.1)
        assert np.allclose(result[0]['row_mean'], par['row_mean'], atol=0.1)

    def test_fitpeak_bounds(self, capsys):
        # Window radius: center source
        wr_col = 15
        wr_row = 20
        stamp, par = self.model_param(wr_col, wr_row, 'elliptical_gaussian')

        # good bounds
        bounds = dict()
        bounds['baseline'] = [-np.inf, np.inf]
        bounds['dpeak'] = [0, np.inf]
        bounds['col_mean'] = [0, wr_col * 2]
        bounds['row_mean'] = [0, wr_row * 2]
        bounds['col_sigma'] = [0, wr_col * 2]
        bounds['row_sigma'] = [0, wr_row * 2]
        bounds['theta'] = [-np.pi / 2., np.pi / 2.]

        # too restrictive bounds -- not appropriate to fit
        rbounds = dict()
        rbounds['baseline'] = [0, 0.2]
        rbounds['dpeak'] = [0, 21.]
        rbounds['col_mean'] = [14, 16]
        rbounds['row_mean'] = [19, 21]
        rbounds['col_sigma'] = [0, 2]
        rbounds['row_sigma'] = [0, 2]
        rbounds['theta'] = [0, 0.1]

        # bad bounds: not a dict
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            bounds=list(bounds.values()))
        capt = capsys.readouterr()
        assert 'Bounds must be a dictionary' in capt.err

        # bad bounds: missing key, not moffat
        bad_bounds = bounds.copy()
        del bad_bounds['col_mean']
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            bounds=bad_bounds)
        capt = capsys.readouterr()
        assert 'Bounds missing required keys' in capt.err

        # bad bounds: missing beta for moffat
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='moffat',
                            bounds=bad_bounds)
        capt = capsys.readouterr()
        assert 'Bounds missing required keys' in capt.err

        # bad bounds: values not 2-item list
        bad_bounds = bounds.copy()
        bad_bounds['col_mean'] = wr_col * 2
        with pytest.raises(PipeCalError):
            pipecal_fitpeak(stamp, profile='gaussian',
                            bounds=bad_bounds)
        capt = capsys.readouterr()
        assert 'Elements of bounds must be 2-element list' in capt.err

        # test good bounds
        fitp1, _, bestn1 = pipecal_fitpeak(stamp, profile='gaussian',
                                           bounds=bounds)
        for key, val in fitp1.items():
            assert bounds[key][0] <= val <= bounds[key][1]

        # and restrictive bounds
        fitp2, _, bestn2 = pipecal_fitpeak(stamp, profile='gaussian',
                                           bounds=rbounds)
        for key, val in fitp2.items():
            assert rbounds[key][0] <= val <= rbounds[key][1]

        # first fit is better than second
        assert bestn1 < bestn2

    def test_bestnorm(self):
        wr_col = 15
        wr_row = 20

        # test output with bestnorm=True
        stamp, par = self.model_param(wr_col, wr_row, 'elliptical_gaussian')
        fitp, fite, bestn = pipecal_fitpeak(stamp, bestnorm=True)
        assert len(fitp) > 1
        assert len(fite) > 1
        assert bestn > 0

        # test output with bestnorm=False
        fitp, fite, bestn = pipecal_fitpeak(stamp, bestnorm=False)
        assert len(fitp) > 1
        assert len(fite) > 1
        assert bestn is None
