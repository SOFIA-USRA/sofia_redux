# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.interpolate import spline

__all__ = ['ExtinctionModel']


class ExtinctionModel(object):
    """
    Extinction model for de-reddening spectra.
    """

    def __init__(self, model='rieke1989', cval=np.nan, sigma=10.0,
                 extrapolate=False):
        """
        Set extinction model.

        Parameters
        ----------
        model : {'rieke1989', 'nishiyama2009'}
            Model to use.  Options are `rieke1989_table` and
            `nishiyama2009_table`.
        cval : float, optional
            Value to fill for missing data.
        sigma : float, optional
            Spline fit tension.
        extrapolate : bool, optional
            If set, missing values will be extrapolated. If not, they
            will be set to `cval`.
        """
        table = getattr(self, model + '_table', None)
        self.default_r_v = None
        if table is None:
            raise AttributeError("%s model not available" % model)
        self.table = table()
        self.range = min(self.table[0]), max(self.table[0])
        self.arange = self.range[0] * 1e4, self.range[1] * 1e4
        self.sigma = sigma
        self.cval = cval
        self.extrapolate = extrapolate

    @staticmethod
    def rieke1989_table():
        """Rieke, Rieke, & Paul (1989 ApJ, 336, 752)"""
        w = [0.365, 0.440, 0.550, 0.700, 0.900, 1.250, 1.600, 2.200, 3.500,
             4.800, 8.000, 8.500, 9.000, 9.500, 10.00, 10.50, 10.60, 11.00,
             11.50, 12.00, 12.50, 13.00]
        a = [1.640, 1.000, 0.000, -0.78, -1.60, -2.22, -2.55, -2.74, -2.91,
             -3.02, -3.03, -2.96, -2.87, -2.83, -2.86, -2.87, -2.93, -2.91,
             -2.95, -2.98, -3.00, -3.01]
        return np.array([w, a])

    @staticmethod
    def nishiyama2009_table():
        """Nishiyama et al. 2009"""
        filters = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'L', 'M',
                   '[8.0]', '[8.5]', '[9.0]', '[9.5]', '[10.0]', '[10.5]',
                   '[11.0]', '[11.5]', '[12.0]', '[12.5]', '[13.0]']
        w = [0.365, 0.445, 0.551, 0.658, 0.806, 1.170, 1.570, 2.120, 3.400,
             4.750, 8.000, 8.500, 9.000, 9.500, 10.00, 10.50, 11.00, 11.50,
             12.00, 12.50, 13.00]
        av = [1.531, 1.324, 1.000, 0.748, 0.482, 0.282, 0.175, 0.112, 0.058,
              0.023, 0.020, 0.043, 0.074, 0.087, 0.083, 0.074, 0.060, 0.047,
              0.037, 0.030, 0.027]
        # Need to normalize normalize to k filter
        a = np.array(av) / av[filters.index('K')]
        return np.array([w, a])

    def __call__(self, wave):
        """
        Generate an extinction model for the given wavelength range.

        The extinction table is fit onto the wavelength range with a spline
        interpolation.

        Parameters
        ----------
        wave : array_like of float or float
            Wavelength values to interpolate onto.

        Returns
        -------
        array_like of float or float
            Matches `wave` dimension and type.
        """
        isarr = hasattr(wave, '__len__')
        if isarr:
            wave = np.array(wave).astype(float)
        else:
            wave = np.array([wave]).astype(float)

        wave *= 1e-4  # convert from angstroms to microns
        if not self.extrapolate:
            valid = (wave >= self.range[0]) & (wave <= self.range[1])
            if not valid.any():
                log.warning("Extinction model range is (%s, %s) angstroms" %
                            self.arange)
                return np.full(wave.shape, self.cval) if isarr else self.cval
        else:
            valid = None

        result = spline(self.table[0], self.table[1], wave, sigma=self.sigma)
        if not self.extrapolate:
            result[~valid] = self.cval

        return result if isarr else result[0]
