# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay

from sofia_redux.toolkit.utilities.base import Model
from sofia_redux.toolkit.interpolate.interpolate import interp_error


__all__ = ['ConvolveBase']


class ConvolveBase(Model):
    """
    Convolution class allowing error propagation.
    """
    def __init__(self, *args, error=1, mask=None, stats=True,
                 robust=0.0, eps=0.01, maxiter=10, do_error=False,
                 axes=None, ignorenans=True, **kwargs):

        self.do_error = do_error
        self._axes = axes
        self._tri = None
        self._result = None
        self._interpolated_error = None

        super().__init__(*args, error=error, mask=mask, covar=False,
                         stats=stats, robust=robust, eps=eps,
                         maxiter=maxiter, ignorenans=ignorenans,
                         fit_kwargs=kwargs)

    @property
    def result(self):
        return self.reshape(self._result)

    @property
    def error(self):
        """Don't create the error unless asked for or already present"""
        if self._interpolated_error is None:
            if hasattr(self._error, '__len__'):
                self._interpolated_error = self._error.copy()
            else:
                self._interpolated_error = np.full(
                    self.mask.shape, float(self._error))
        return self.reshape(self._interpolated_error)

    @property
    def residuals(self):
        if self.stats is None:
            return
        return self.reshape(self.stats.residuals)

    @property
    def masked(self):
        return self.reshape(self.mask)

    @staticmethod
    def replace_masked_samples(samples, mask, get_tri=False):
        invalid = ~mask
        if not invalid.any():
            result = samples[-1].copy()
            return (result, None) if get_tri else result
        elif not mask.any():
            result = np.full_like(samples[-1], np.nan)
            return (result, None) if get_tri else result

        ndim = samples.shape[0] - 1
        result = samples[-1].copy()
        if ndim > 1:
            tri = Delaunay(samples[:-1, mask].T)
            interpolator = LinearNDInterpolator(tri, result[mask])
            newvals = interpolator(samples[:-1, invalid].T)
        else:
            tri = samples[0, mask]
            interpolator = interp1d(samples[0, mask], result[mask],
                                    fill_value='extrapolate')
            newvals = interpolator(samples[0, invalid])

        # keep old values if they are replaced with NaNs
        bad = np.isnan(newvals)
        newvals[bad] = samples[-1, invalid][bad]
        result[invalid] = newvals
        return (result, tri) if get_tri else result

    def replace_masked_error(self):
        self._interpolated_error = None
        if self._tri is None or not self.do_error:
            return
        if hasattr(self._error, '__len__'):
            result = self._error.copy()
        else:
            result = np.full(self.mask.shape, float(self._error))
        if self.mask.all():
            return result
        invalid = ~self.mask

        ix = self._samples[:-1, invalid]
        if self._ndim > 1:
            ix = ix.T
        else:
            ix = ix.ravel()

        new_error = interp_error(
            self._tri, self.error.ravel()[self.mask], ix)

        bad = np.isnan(new_error)
        new_error[bad] = self.error.ravel()[invalid][bad]
        result[invalid] = new_error
        self._interpolated_error = result

    def _convolve(self, clean_flat_array):
        """Place holder"""
        self._result = clean_flat_array.ravel().copy()

    def initial_fit(self):
        """The initial fit"""
        self._nparam = 0  # fitting the mask, not parameters
        self.refit_mask(self._usermask, covar=False)

    def refit_mask(self, mask, covar=False):
        self.mask = mask.ravel()
        cleaned, self._tri = self.replace_masked_samples(
            self._samples, self.mask, get_tri=True)
        self.replace_masked_error()
        self._convolve(self.reshape(cleaned, copy=False))
        self.success = np.isfinite(self._result).any()
        if not self.success:
            self._interpolated_error = None
        self._fit_statistics()

    def evaluate(self, _, dovar=False):
        return (self._result, None) if dovar else self._result

    def __call__(self, *independent_values, dovar=False):
        pass
