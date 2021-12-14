# Licensed under a 3-clause BSD style license - see LICENSE.rst

from argparse import Namespace
import warnings

import bottleneck as bn
import numpy as np
from scipy.stats import chi2

from sofia_redux.toolkit.utilities.func import stack, remove_sample_nans
from sofia_redux.toolkit.stats.stats import find_outliers


__all__ = ['Model']


class Model(object):
    """
    Base model Class for fitting N-dimensional data

    Attributes
    ----------
    stats : argparse.Namespace
        Contains the following statistics:
            samples : numpy.ndarray (ndim, nsamples)
                The samples used to fit the polynomial.  samples[:-1]
                contain the independent variables for each dimension
                and samples[-1] contains the dependent variables.
                Note that all NaNs will have been stripped from the
                original input samples.
            ndata : int
                The number of samples used to fit the polynomial
            fit : numpy.array (nsamples,)
                The fitted polynomial over the original sample points
            residuals : numpy.ndarray (nsamples,)
                The residual of fit(data) - data
            sigma : numpy.ndarray (ncoeffs,)
                The error of each polynomial coefficient (will only
                be calculated if a covariance matrix exists)
            dof : int
                Degrees of Freedom of the fit
            rms : float
                Root Mean Square deviation of the fit
            chi2 : float
                Chi-Squared
            rchi2 : float
                Reduced Chi-Squared
            q : float
                Goodness of fit, or survival function.  The probability (0->1)
                that one of the `samples` is greater than `chi2` away from the
                `fit`.
    """
    def __init__(self, *args, error=1, mask=None, covar=True, stats=True,
                 robust=0, eps=0.01, maxiter=100, ignorenans=True,
                 fit_kwargs=None, eval_kwargs=None):

        self._samples = None
        self._error = None
        self._interpolated_error = None
        self._usermask = None
        self._model_args = None
        self._model_kwargs = None
        self._initial_shape = None
        self._ignorenans = ignorenans
        self.termination = None
        self._parse_args(error, mask, *args)
        self._parse_model_args()

        self._ndim, self._nsamples = self._samples.shape
        self._ndim -= 1  # only include independent variables
        self._nparam = None
        self._fit_kwargs = fit_kwargs
        self._eval_kwargs = eval_kwargs

        self.success = False
        self.covar = covar
        self.covariance = None
        self.mask = self._usermask
        self.robust = robust
        self.stats = None
        self.dostats = stats or self.robust > 0
        self.maxiter = maxiter
        self.eps = eps

        self.initial_fit()
        self._iteration = 1
        self._state = "initial fit"

        if self.robust > 0:  # pragma: no cover
            if self.covar and self.covariance is not None:
                covariance1 = self.covariance.copy()
            else:
                covariance1 = None
            self._iterate()
            if self._iteration == 1 and covar and self.success:
                self.covariance = covariance1
            else:
                self.refit_mask(self.mask, covar=True)

    @property
    def state(self):
        return self._state

    @property
    def error(self):
        """Don't create the error unless asked for or already present

        This is for the errors of the samples only
        """
        if self._interpolated_error is None:
            if hasattr(self._error, '__len__'):
                self._interpolated_error = self._error
            else:
                self._interpolated_error = np.full(
                    self.mask.shape, float(self._error))
        return self.reshape(self._interpolated_error)

    def reshape(self, flattened_array, copy=True):
        array = flattened_array.reshape(self._initial_shape)
        return array.copy() if copy else array

    def __repr__(self):
        return "%s (%i features, %s parameters)" % (
            self.__class__.__name__, self._ndim, self._nparam)

    def __str__(self):
        s = "Name: %s\n" % self.__class__.__name__
        s += self._stats_string()
        s += self._parameters_string()
        return s

    def _parameters_string(self):
        """Place holder for model parameters"""
        return ''

    def _stats_string(self):
        s = ''
        if self.stats is not None:
            n_in = self.mask.size
            n_nan = self._samples.shape[1] - n_in
            n_outliers = np.sum(~self.mask) - n_nan
            s += "\n         Statistics"
            s += "\n--------------------------------"
            s += "\nNumber of original points : %i" % n_in
            s += "\n           Number of NaNs : %i" % n_nan
            s += "\n       Number of outliers : %i" % n_outliers
            s += "\n     Number of points fit : %i" % self.mask.sum()
            s += "\n       Degrees of freedom : %i" % self.stats.dof
            s += "\n              Chi-Squared : %f" % self.stats.chi2
            s += "\n      Reduced Chi-Squared : %f" % self.stats.rchi2
            s += "\n      Goodness-of-fit (Q) : %f" % self.stats.q
            s += "\n     RMS deviation of fit : %f" % self.stats.rms
            if self.robust > 0:
                s += '\n  Outlier sigma threshold : %s' % self.robust
                s += '\n  eps (delta_sigma/sigma) : %s' % self.eps
                s += '\n               Iterations : %s' % self._iteration
                s += '\n    Iteration termination : %s' % self.termination
            s += '\n'
        return s

    def print_stats(self):
        """
        Print statistical information on the fit to stdout.

        Returns
        -------
        None
        """
        print(self._stats_string())

    def print_params(self):
        """
        Print parameters to stdout.

        Returns
        -------
        None
        """
        print(self._parameters_string())

    def __call__(self, *independent_values, dovar=False):  # pragma: no cover
        """
        Evaluate the model

        Parameters
        ----------
        samples : n-tuple of array_like (shape)
            n-features of independent variables.
        dovar : bool, optional
            If True return the variance of the fit in addition to the
            fit.

        Returns
        -------
        fit, [variance] : numpy.ndarray (shape), [numpy.ndarray (shape)]
            The output fit and optionally, the variance.
        """
        if len(independent_values) != self._ndim:
            raise ValueError(
                "Require %i features of independent values" % self._ndim)
        v = stack(*independent_values)
        r = self.evaluate(v, dovar=dovar)
        fit, var = (r[0], r[1]) if dovar else (r, None)
        isarr = hasattr(independent_values[0], '__len__')
        havevar = var is not None
        test = np.asarray(independent_values[0])
        if test.ndim > 1:
            shape = test.shape
            fit = fit.reshape(shape)
            if havevar:
                var = var.reshape(shape)
        elif not isarr:
            fit = fit[0]
            if havevar:
                var = var[0]

        return (fit, var) if dovar else fit

    @staticmethod
    def _create_coordinates(*args):
        nargs = len(args)
        if nargs < 2:
            raise ValueError(
                "Require at least 2 arguments (f(x), model_args)")
        if nargs >= 3:  # have coorindates as args[0]
            return args
        shape = np.asarray(args[-2]).shape
        c = np.meshgrid(*(np.arange(s, dtype=float) for s in shape),
                        indexing='ij')
        return tuple(c) + args

    def _parse_args(self, error, mask, *args):
        args = self._create_coordinates(*args)
        nargs = len(args)
        self._ndim = nargs - 2
        self._initial_shape = np.asarray(args[-2]).shape
        self._samples = stack(*args[:-1])
        self._model_args = args[-1]
        nsamples = self._samples.shape[1]

        if hasattr(error, '__len__'):
            error = np.asarray(error).astype(float).ravel()
            if error.size != nsamples:
                raise ValueError("Error size does not match number of samples")
        else:
            try:
                error = float(error)
            except (ValueError, TypeError):
                error = 1.0
        self._error = error

        if self._ignorenans:
            nanmask = remove_sample_nans(self._samples, error, mask=True)
        else:
            nanmask = np.full(self._samples[-1].shape, True)
        if hasattr(mask, '__len__'):
            usermask = np.asarray(mask).astype(bool).ravel()
            if mask.size != nsamples:
                raise ValueError("Mask size does not match number of samples")
            usermask &= nanmask
        else:
            usermask = nanmask  # no copy
        self._usermask = usermask

    def _fit_statistics(self):  # pragma: no cover
        if not self.dostats:
            return
        self.stats = Namespace()
        stats = self.stats
        n0 = self.mask.size
        mask = self.mask & self._usermask
        stats.n = n = mask.sum()
        if n > 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                stats.dof = stats.n - self._nparam
                stats.fit = self.evaluate(self._samples[:-1], dovar=False)
                stats.residuals = self._samples[-1] - stats.fit
                r = stats.residuals.compress(mask)
                if self.covariance is not None:
                    stats.sigma = np.sqrt(np.diag(self.covariance))
                else:
                    stats.sigma = None
                stats.rms = bn.nanstd(r) * np.sqrt(n / (n - 1))
                stats.chi2 = bn.nansum((r / self.error.ravel()[mask]) ** 2)
                if stats.dof != 0:
                    stats.rchi2 = stats.chi2 / stats.dof
                else:
                    stats.rchi2 = np.inf
                stats.q = 1 - chi2.sf(stats.chi2, stats.dof)
        else:
            stats.fit = np.full(n0, np.nan)
            stats.residuals = np.full(n0, np.nan)
            stats.dof = 0
            stats.rms = stats.chi2 = stats.rchi2 = stats.q = np.nan

    def _iterate(self):  # pragma: no cover
        """
        Iterates to refine the polynomial fit

        1. Idenitify outliers in the residuals of data - fit
        2. Re-fit the polynomial excluding all outliers
        3. Goto 1.

        The iteration is terminated after a set number of iterations or
        the relative delta between successive residual RMS values is less
        than a set value.

        Notes
        -----
        The covariance calculation is skipped which results in a speed
        increase in most cases.  However, if the covariance is required,
        and only 1 iteration occurs, this results in a slight speed
        decrease.
        """
        self._iteration = 1
        if self.robust <= 0:
            return
        self.termination = "initial"
        last_rms = self.stats.rms
        relative_delta = np.inf
        min_valid_points = max([2, self._nparam])

        for _ in range(self._iteration, self.maxiter):

            if last_rms <= np.finfo(float).eps:  # pragma: no cover
                self.termination = ("solution found to within %s precision"
                                    % float)
                break
            elif relative_delta < self.eps:  # pragma: no cover
                self.termination = "delta_rms/rms = %f" % relative_delta
                break

            last_rms = self.stats.rms
            mask = find_outliers(self.stats.residuals, threshold=self.robust)
            if mask.sum() < min_valid_points:
                self.termination = "insufficient samples remain"
                self.mask.fill(False)
                self.success = False
                break
            elif np.allclose(self.mask, mask):
                self.termination = "delta_rms = 0"
                break

            self.refit_mask(mask, covar=False)

            self._iteration += 1
            if not self.success:  # pragma: no cover
                self.termination = "fit failed"
                break
            delta = abs(self.stats.rms - last_rms)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                relative_delta = delta / last_rms
        else:  # pragma: no cover
            self.termination = "reached maximum iterations"

    def _parse_model_args(self):  # pragma: no cover
        """Place holder to perform operation on model arguments"""
        pass

    def initial_fit(self):  # pragma: no cover
        """Place holder"""
        pass

    def refit_mask(self, mask, covar=False):  # pragma: no cover
        """Place holder"""
        pass

    def evaluate(self, samples, dovar=False):  # pragma: no cover
        """Place holder"""
        if True is not False:
            raise ValueError("Create this method")
        elif self.stats or samples or dovar:
            return []
