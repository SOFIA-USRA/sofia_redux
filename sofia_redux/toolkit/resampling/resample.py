# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import bottleneck as bn
import joblib
import os
import numpy as np
from scipy.special import gamma
import shutil
from tempfile import mkdtemp
import time

from sofia_redux.toolkit.utilities.multiprocessing import multitask
from sofia_redux.toolkit.stats.stats import robust_mask

from .grid import ResampleGrid
from .resample_utils import (scale_coordinates,
                             shaped_adaptive_weight_matrices,
                             scaled_adaptive_weight_matrices,
                             relative_density, solve_fits,
                             convert_to_numba_list)
from .tree import Rtree


__all__ = ['Resample', 'resamp']


_global_resampling_values = {}


class Resample(object):

    def __init__(self, coordinates, data,
                 error=None, mask=None, window=None,
                 order=1, fix_order=True,
                 robust=None, negthresh=None,
                 window_estimate_bins=10,
                 window_estimate_percentile=50,
                 window_estimate_oversample=2.0,
                 leaf_size=40,
                 **distance_kwargs):
        """
        Class to resample data using local polynomial fits.

        Parameters
        ----------
        coordinates : array_like of float
            (n_features, n_samples) array of independent values.  A local
            internal copy will be created if it is not a numpy.float64
            type.
        data : array_like of float
            (n_sets, n_samples) or (n_samples,) array of dependent values.
            multiple (n_sets) sets of data are supplied, then n_sets solutions
            will be calculated at each resampling point.
        error : array_like of float, optional
            (n_sets, n_samples) or (n_samples,) array of error (1-sigma) values
            associated with the `data` array.  `error` will be used to
            weight fits, and be propagated to the output error values.  If not
            supplied, the error may still be calculated from residuals to the
            fit during :func:`Resample.__call__`.
        mask : array_like of bool, optional
            (n_sets, n_data) or (n_data,) array of bool where `True`
            indicates a valid data point that can be included the fitting
            and `False` indicates data points that should be excluded from
            the fit.  Masked points will be reflected in the output counts
            array.  If not supplied, all samples are considered valid.
        window : array_like or float or int, optional
            (n_features,) array or single value specifying the maximum
            distance (distance definition is handled by `distance_kwargs`) of a
            data sample from a resampling point such that it can be included in
            a local fit.  `window` may be declared for each feature.  For
            example, when fitting 2-dimensional (x, y) data, a window of 1.0
            would create a circular fitting window around each resampling
            point, whereas a window of (1.0, 0.5) would create an elliptical
            fitting window with a semi-major axis of 1.0 in x and semi-minor
            axis of 0.5 in y.  If not supplied, `window` is estimated via
            :func:`Resample.estimate_feature_windows`.
        order : array_like or int, optional
            (n_features,) array or single integer value specifying the
            polynomial fit order for each feature.
        fix_order : bool, optional
            In order for local polynomial fitting to occur, the basic
            requirement is that n_samples >= (order + 1) ** n_features,
            where n_samples is the number of data samples within `window`.
            If `fix_order` is True and this condition is not met, then
            local fitting will be aborted for that point, and a value of
            `cval` will be returned instead.  If `fix_order` is False,
            then `order` will be reduced to the maximum value where this
            condition can be met.  NOTE: this is only available if
            `order` is symmetrical. i.e. it was passed in as a single
            integer to be applied across all features.  Otherwise, it is
            unclear as to which feature order should be reduced to meet
            the condition.
        robust : float, optional
            Specifies an outlier rejection threshold for `data`. A data point
            is identified as an outlier if abs(x_i - x_med)/MAD > robust, where
            x_med is the median, and MAD is the Median Absolute Deviation
            defined as 1.482 * median(abs(x_i - x_med)).
        negthresh : float, optional
            Specifies a negative value rejection threshold such that
            data < (-stddev(data) * negthresh) will be excluded from the fit.
        window_estimate_bins : int, optional
            Used to estimate the `window` if not supplied using
            :func:`Resample.estimate_feature_windows`.
        window_estimate_percentile : int or float, optional
            Used to estimate the `window` if not supplied using
            :func:`Resample.estimate_feature_windows`.
        window_estimate_oversample : int or float, optional
            Used to estimate the `window` if not supplied using
            :func:`Resample.estimate_feature_windows`.
        leaf_size : int, optional
            Number of points at which to switch to brute-force during the
            ball tree query algorithm.  See `sklearn.neighbours.BallTree`
            for further details.
        distance_kwargs : dict, optional
            Optional keyword arguments passed into
            :func:`sklearn.neighbors.DistanceMetric`.  The default is to use
            the "minkowski" definition with `p=2`, i.e., the Euclidean
            definition.  This is important in determining which samples lie
            inside the window region of a resampling point, and when deriving
            distance weighting factors.

        Raises
        ------
        ValueError : Invalid inputs to __init__ or __call__
        """
        coordinates, data, error, mask = self._check_input_arrays(
            coordinates, data, error=error, mask=mask)
        self._n_features, self._n_samples = coordinates.shape
        order = self.check_order(order, self._n_features, self._n_samples)

        self._n_sets = 0
        self._multiset = None
        self._valid_set = None
        self.data = None
        self.error = None
        self._error_valid = None
        self.mask = None

        self._process_input_data(
            data, error=error, mask=mask, negthresh=negthresh, robust=robust)

        scaled_coordinates = self._scale_to_window(
            coordinates, radius=window, order=order,
            feature_bins=window_estimate_bins,
            percentile=window_estimate_percentile,
            oversample=window_estimate_oversample
        ).astype(np.float64)

        self.sample_tree = Rtree(
            scaled_coordinates, build_type='all', leaf_size=leaf_size,
            **distance_kwargs)

        self.sample_tree.set_order(order, fix_order=fix_order)
        self.sample_tree.precalculate_phi_terms()
        self._fit_settings = None
        self.iteration = 0

    @property
    def features(self):
        """int : number of data features (dimensions)"""
        return self._n_features

    @property
    def multi_set(self):
        """bool : True if solving for multiple data sets"""
        return self._multiset

    @property
    def n_sets(self):
        """int : The number of data sets to fit."""
        return self._n_sets

    @property
    def n_samples(self):
        """int : The number of samples in each data set."""
        return self._n_samples

    @property
    def window(self):
        """numpy.ndarray (n_features,) : Window radius in each dimension."""
        if self._radius is None:  # pragma: no cover
            return None
        return self._radius.copy()

    @property
    def order(self):
        """int or numpy.ndarray (n_features,) : Orders of polynomial fit."""
        return self.sample_tree.order

    @property
    def fit_settings(self):
        r"""dict : Fit reduction settings applied during last call"""
        return self._fit_settings

    @classmethod
    def _check_input_arrays(cls, coordinates, data, error=None, mask=None):
        """Checks the validity of arguments to __init__

        Checks that sample coordinates, values, error, and mask have compatible
        dimensions.

        Raises a ValueError if an argument or parameter is not valid.
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = coordinates[None]
        if coordinates.ndim != 2:
            raise ValueError("Coordinates array must have 1 (n_samples,) "
                             "or 2 (n_features, n_samples) axes.")

        ndata = coordinates.shape[-1]

        data = np.asarray(data, dtype=np.float64)
        shape = data.shape
        if shape[-1] != ndata:
            raise ValueError("Data sample size does not match coordinates")
        if data.ndim not in [1, 2]:
            raise ValueError(
                "Data must have 1 or 2 (multi-set) dimensions")

        if error is not None:
            error = np.atleast_1d(np.asarray(error, dtype=np.float64))
            if error.size == 1:
                error = np.full_like(data, error[0])
            if error.shape != shape:
                raise ValueError("Error shape does not match data")

        if mask is not None:
            mask = np.asarray(mask, dtype=np.bool)
            if mask.shape != shape:
                raise ValueError("Mask shape does not match data")

        return coordinates, data, error, mask

    @staticmethod
    def check_order(order, n_features, n_samples):
        r"""
        Check the order is of correct format and enough samples exist.

        Parameters
        ----------
        order : int or array_like of int (n_features,)
            The polynomial order to fit, either supplied as an integer to be
            applied over all dimensions, or as an array to give the order
            in each dimension.
        n_features : int
            The number of features (dimensions) of the sample coordinates.
        n_samples : int
            The number of samples.

        Returns
        -------
        order : int or numpy.ndarray of int (n_features,)
            The formatted polynomial order.

        Raises
        ------
        ValueError
            If there are too few samples for the desired order, or `order` is
            not formatted correctly.
        """
        order = np.asarray(order, dtype=np.int64)
        if order.ndim > 1:
            raise ValueError(
                "Order should be a scalar or 1-D array")
        elif order.ndim == 1 and order.size != n_features:
            raise ValueError(
                "Order vector does not match number of features")

        if order.ndim == 0:
            min_points = (order + 1) ** n_features
        else:
            min_points = np.product(order + 1)
        if n_samples < min_points:
            raise ValueError("Too few data samples for order")

        return order

    def _process_input_data(self, data, error=None, mask=None,
                            negthresh=None, robust=None):
        """Formats the input data, error, and mask for subsequent use.

        Sets the data, mask, and error attributes to numpy arrays of shape
        (n_sets, n_samples).

        The output mask will be a union of the input mask (if there is one) and
        finite data values and nonzero error values.  If the user has provided
        `robust` or `negthresh` then the mask will be updated to reflect this.
        See :func:`Resample.__init__` for further details.
        """
        data = np.asarray(data).astype(np.float64, order='F')

        if data.ndim == 2:
            self._multiset = True
            self._n_sets = data.shape[0]
        else:
            self._multiset = False
            self._n_sets = 1

        data = np.atleast_2d(data)

        if mask is not None:
            mask = np.atleast_2d(np.asarray(mask).astype(bool))

        mask = robust_mask(data, robust, mask=mask, axis=-1)

        if negthresh is not None:
            invalid = np.logical_not(mask)
            data[invalid] = np.nan
            rms = bn.nanstd(data, ddof=1, axis=-1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mask &= data > (-rms * negthresh)

        invalid = np.logical_not(mask)

        self._error_valid = error is not None
        if self._error_valid:
            error = np.atleast_2d(error).astype(np.float64, order='F')
            if error.shape[1] != data.shape[1] and error.shape[1] != 1:
                raise ValueError("Error must be a single value or an array "
                                 "matching the data shape.")

            invalid |= ~np.isfinite(error)
            invalid |= error == 0
            if error.shape[1] == data.shape[1]:
                error[invalid] = np.nan

        else:
            error = np.empty((data.shape[0], 0), dtype=np.float64)

        data[invalid] = np.nan

        self._valid_set = np.any(np.isfinite(data), axis=1)
        if not self._valid_set.any():
            raise ValueError("All data has been flagged as invalid")

        self.data = data
        self.error = error
        self.mask = np.logical_not(invalid, order='F')

    def _scale_to_window(self, coordinates, radius=None, order=1,
                         feature_bins=10, percentile=50, oversample=2.0):
        r"""
        Scale input coordinates to units of the resampling window.

        Coordinates (:math:`x`) are scaled such that for dimension :math:`k`,
        the coordinates stored in the resampling tree (see
        :func:`resampling.tree.Rtree`) are set as:

        .. math::

            x_k^{\prime} = \frac{x_k - min(x_k)}{\Omega_k}

        where :math:`\Omega` is the resampling window radius.  If the window
        radius (or principle axes) are unknown, an attempt is made to determine
        one using `:func:`Resampler.estimate_feature_windows`.  If so, an order
        should be supplied, otherwise the default of order = 1 is used.

        Parameters
        ----------
        coordinates : numpy.ndarray (n_features, n_coordinates)
            The sample coordinates.
        radius : numpy.ndarray (n_features,), optional
            The radius of the window around each fitting point used to
            determine sample selection for fit.  If not supplied, will be
            estimated using :func:`Resample.estimate_feature_windows`.
        order : int or numpy.ndarray, optional
            The order of polynomial fit to perform.
        feature_bins : int, optional
            When estimating `radius`, gives the number of bins in each
            dimension by which to equally divide (by ordinate, not count) the
            feature coordinates for the purposes of finding the sample density.
        percentile : int or float, optional
            When estimating `radius`, the percentile used to define a
            representative value for samples per bin.  The default (50), gives
            the median of all bin populations.
        oversample : int or float, optional
            When estimating `radius`, the oversampling factor for the window
            region.  A value of one will result in a window that should provide
            the exact number of samples required for a polynomial fit of the
            given order assuming uniform density of the samples.

        Returns
        -------
        scaled_coordinates : numpy.ndarray (n_features, n_coordinates)
            The coordinates scaled to units of the window radius.
        """
        if radius is None:
            radius = self.estimate_feature_windows(coordinates, order,
                                                   feature_bins=feature_bins,
                                                   percentile=percentile,
                                                   oversample=oversample)
        else:
            radius = np.atleast_1d(radius)
            if radius.size < self._n_features:
                radius = np.full(self._n_features, radius[0])
        self._radius = radius.astype(np.float64)
        self._scale_offsets = coordinates.min(axis=1)

        return scale_coordinates(
            coordinates, self._radius, self._scale_offsets)

    @staticmethod
    def estimate_feature_windows(coordinates, order,
                                 feature_bins=10, percentile=50,
                                 oversample=2.0):
        r"""
        Estimates the radius of the fitting window for each feature.

        The estimate of the window is given as the minimum required to
        theoretically allow for a polynomial fit based on the number of samples
        within such a window.  Since the window is a constant of the
        resampling algorithm, it is difficult to calculate a precise value
        that will allow all fits to occur at every point within (or close to)
        the sample distribution.

        Therefore, the sample distribution is divided up into `feature_bins`
        n-dimensional boxes over each feature.  For example, if
        `feature_bins=10` for 2-dimensional data, we would divide the sample
        distribution into a total of 100 (10 * 10) equal boxes before counting
        the number of samples inside each box.

        The number of samples used to then calculate the final window radius
        is determined from a given `percentile` (default = 50) of the box
        counts.

        For a fit of polynomial order :math:`p`, the minimum number of samples
        required for fitting is:

        .. math::

            N = (p + 1)^K

        for :math:`K` features if the order is "symmetrical" (i.e., the same
        for each feature), or

        .. math::

            N_{required} = \prod_{k=1}^{K}{(p_k + 1)}

        if the fit order varies by feature.  Coordinates are then scaled so
        that:

        .. math::

            x_k^{\prime} = \text{feature\_bins} \times \frac{x_k - min(x_k)}
                           {\beta_k}

        where the scaling factor :math:`\beta_k = max(x_k) - min(x_k)`. In this
        scheme, the coordinates have been normalized such that the width of the
        bin in each dimension is 1, and has a volume equal to one.  Therefore,
        the sample density (:math:`\rho`) of the bin is equal to the number of
        samples it contains.  The volume of a unit radius spheroid is then
        calculated as:

        .. math::

            V_{r=1} = \frac{\pi^{K/2}}{\Gamma(\frac{n}{2} + 1)}

        and the expected number of samples expected to fall inside the spheroid
        is given as:

        .. math::

            N_{r=1} = \rho V_{r=1}

        We can then set the radius of the spheroid to give the required number
        of points as:

        .. math::

            r_{scaled} = \left( \frac{N_{required} \nu}{N_{r=1}}
                         \right)^{\frac{1}{K}} + \epsilon

        where :math:`\nu` is the `oversample` factor and :math:`\epsilon` is
        added to ensure that if resampling on a uniform grid of samples,
        fitting at a point between two samples will always result in enough
        samples available to perform a fit.  :math:`\epsilon` is given as:

        .. math::

            \epsilon = 0.5 \rho^{\frac{-1}{K}}

        or half the average spacing between samples.  We can then define the
        final window radius for dimension :math:`k` as:

        .. math::

            \Omega_k = \frac{\beta_k r_{scaled}}{\text{feature\_bins}}


        Parameters
        ----------
        coordinates : numpy.ndarray (n_features, n_coordinates)
            The coordinates of all samples to be fit.
        order : int or numpy.ndarray of int (n_features,)
            The polynomial order to fit for all or each feature.
        feature_bins : int, optional
            The number of bins to divide the sample coordinates into, per
            feature, when determining the sample density (default = 10).
        percentile : int or float, optional
            The percentile used to define a representative value for samples
            per bin.  The default (50), gives the median of all bin
            populations.
        oversample : int or float, optional
            The oversampling factor for the window region.  A value of one will
            result in a window that should provide the exact number of samples
            required for a polynomial fit of the given order assuming
            uniform density of the samples.

        Returns
        -------
        window : numpy.ndarray (n_features,)
            The principle axes of an ellipsoid used to create a fitting
            region around each resampling point.
        """
        # Scale all coordinates between 0 and 1 (inclusive)
        x = np.asarray(coordinates).astype(float)
        scale = np.ptp(x, axis=1)
        x -= np.min(x, axis=1)[:, None]
        x /= scale[:, None]
        features, n_samples = x.shape

        feature_bins = np.max((int(feature_bins), 1))
        m = 1 if feature_bins == 1 else feature_bins - 1
        if feature_bins == 1:
            x0 = np.zeros(x.shape, dtype=int)
        else:
            x0 = np.floor(x * m).astype(int)
        i = np.ravel_multi_index(
            x0, [feature_bins] * features, mode='raise')
        counts = np.bincount(i)

        bin_population = np.percentile(counts[counts > 0], percentile)
        o = np.asarray(order)
        if o.shape == ():
            o = np.full(features, int(o))
        # required samples for each local fit
        required_samples = np.product(o + 1)

        unit_spheroid_volume = (np.pi ** (features / 2)
                                ) / gamma((features / 2) + 1)
        bin_volume = 1.0  # in units of feature_bins (just for my notes)
        bin_density = bin_population / bin_volume
        unit_spheroid_count = unit_spheroid_volume * bin_density

        unit_radius = (required_samples * oversample
                       / unit_spheroid_count) ** (1 / features)

        unit_sample_spacing = bin_density ** (-1 / features)
        max_offset = 0.5 * unit_sample_spacing
        unit_radius += max_offset

        scaled_radius = unit_radius * scale / feature_bins

        return scaled_radius

    def reduction_settings(self, smoothing=0.0, relative_smooth=False,
                           adaptive_threshold=None,
                           adaptive_algorithm='scaled', error_weighting=True,
                           fit_threshold=0.0, cval=np.nan,
                           edge_threshold=0.0, edge_algorithm='distribution',
                           order_algorithm='bounded', is_covar=False,
                           estimate_covariance=False, jobs=None,
                           adaptive_region_coordinates=None):
        r"""
        Define a set of reduction instructions based on user input.

        This method is responsible for determining, formatting, and checking
        a number variables required for the resampling algorithm based on
        user input.  For detailed descriptions of user options, please see
        :func:`Resample.__call__`.

        Parameters
        ----------
        smoothing : float or array_like (n_features,), optional
        relative_smooth : bool, optional
        adaptive_threshold : float or array_like (n_features,), optional
        adaptive_algorithm : str, optional
        error_weighting : bool, optional
        fit_threshold : float, optional
        cval : float, optional
        edge_threshold : float or array_like (n_features,), optional
        edge_algorithm : str, optional
        order_algorithm : str, optional
        is_covar : bool, optional
        estimate_covariance : bool, optional
        jobs : int, optional
        adaptive_region_coordinates : array_like, optional

        Returns
        -------
        settings : dict
            The reduction settings.  Also, stored as
            :func:`Resample.fit_settings`.
        """
        n_features = self.sample_tree.features

        adaptive_algorithm = str(adaptive_algorithm).lower().strip()
        if (adaptive_algorithm != 'none'
                and adaptive_threshold is not None
                and not np.allclose(adaptive_threshold, 0)):
            if not error_weighting:
                raise ValueError("Error weighting must be enabled for "
                                 "adaptive smoothing")
            if self.sample_tree.order is None or np.allclose(
                    self.sample_tree.order, 0):
                raise ValueError("Adaptive smoothing cannot be applied for "
                                 "polynomial fit of zero order.")
            else:
                adaptive = True
                adaptive_algorithm = str(adaptive_algorithm).strip().lower()
                if adaptive_algorithm == 'shaped':
                    shaped = n_features > 1  # only relevant in 2+ dimensions
                elif adaptive_algorithm == 'scaled':
                    shaped = False
                else:
                    raise ValueError("Adaptive algorithm must be one of "
                                     "{None, 'shaped', 'scaled'}.")
        else:
            adaptive = False
            shaped = False
            adaptive_threshold = 0.0

        distance_weighting = smoothing is not None or adaptive
        error_weighting = adaptive or error_weighting
        if adaptive and not self._error_valid:
            raise ValueError(
                "Errors must be provided for adaptive smoothing")
        else:
            error_weighting &= self._error_valid

        if distance_weighting:

            # In the case that adaptive smoothing is set, but no distance
            # weighting is selected.  Smoothing of 1/3 is reasonable.
            if smoothing is None:
                smoothing = np.full(
                    n_features, 1 / 3 if relative_smooth else self.window / 3)

            alpha = np.atleast_1d(np.asarray(smoothing, dtype=np.float64))
            if alpha.size not in [1, n_features]:
                raise ValueError(
                    "Smoothing size does not match number of features")

            if adaptive:
                adaptive_threshold = np.atleast_1d(
                    np.asarray(adaptive_threshold, dtype=np.float64))

                if alpha.size != n_features:
                    alpha = np.full(n_features, alpha[0])

                if adaptive_threshold.size not in [1, n_features]:
                    raise ValueError("Adaptive smoothing size does not "
                                     "match number of features")
                elif adaptive_threshold.size != n_features:
                    adaptive_threshold = np.full(
                        n_features, adaptive_threshold[0])

            if not relative_smooth:
                if alpha.size != n_features:  # alpha size = 1
                    alpha = np.full(n_features, alpha[0])
                if not adaptive:
                    alpha = 2 * (alpha ** 2) / (self.window ** 2)
                else:
                    alpha /= self.window  # sigma in terms of window

            if not adaptive and alpha.size == 1 or np.unique(alpha).size == 1:
                # Symmetrical across dimensions - use single value
                alpha = np.atleast_1d(np.float64(alpha[0]))

        else:
            alpha = np.asarray([0.0])

        if edge_threshold is None or \
                (np.atleast_1d(edge_threshold) == 0).all():
            edge_threshold = 0.0

        edge_threshold = np.atleast_1d(edge_threshold).astype(np.float64)
        if edge_threshold.size not in [1, n_features]:
            raise ValueError("Edge threshold size does not match number of "
                             "features.")
        elif edge_threshold.size == 1 and edge_threshold.size != n_features:
            edge_threshold = np.full(n_features, edge_threshold[0])

        edge_func_lookup = {
            'none': 0,
            'distribution': 1,
            'ellipsoid': 2,
            'box': 3,
            'range': 4,
        }
        edge_algorithm = str(edge_algorithm).lower().strip()
        if edge_algorithm not in edge_func_lookup:
            raise ValueError("Unknown edge algorithm: %s" % edge_algorithm)
        edge_algorithm_idx = edge_func_lookup[edge_algorithm]

        if np.any(edge_threshold < 0):
            raise ValueError("Edge threshold must positive valued.")

        upper_edge_limit = np.inf if edge_algorithm == 'distribution' else 1
        for x in np.atleast_1d(edge_threshold):
            if x < 0 or x >= upper_edge_limit:
                raise ValueError("Edge threshold must be less than %s "
                                 "for %s algorithm" %
                                 (upper_edge_limit, edge_algorithm))

        order = self.sample_tree.order
        order_varies = self.sample_tree.order_varies
        order_symmetry = self.sample_tree.order_symmetry
        order_algorithm = str(order_algorithm).lower().strip()
        order_func_lookup = {
            'none': 0,
            'bounded': 1,
            'extrapolate': 2,
            'counts': 3
        }
        if order_algorithm not in order_func_lookup:
            raise ValueError("Unknown order algorithm: %s" % order_algorithm)
        order_algorithm_idx = order_func_lookup[order_algorithm]

        if order_symmetry:
            order_minimum_points = (order + 1) ** n_features
        else:
            order_minimum_points = np.prod(order + 1)

        if is_covar:
            mean_fit = True
        else:
            mean_fit = order_symmetry and order == 0

        check_fit = True if fit_threshold else False
        if check_fit:
            fit_threshold = np.float64(fit_threshold)
        else:
            fit_threshold = np.float64(0.0)

        if adaptive:
            region_coordinates = adaptive_region_coordinates
        else:
            region_coordinates = None

        self._fit_settings = {
            'n_features': n_features,
            'error_weighting': error_weighting,
            'distance_weighting': distance_weighting,
            'alpha': alpha,
            'adaptive_threshold': adaptive_threshold,
            'shaped': shaped,
            'adaptive_alpha': np.empty((0, 0, 0, 0), dtype=np.float64),
            'order': np.atleast_1d(order),
            'order_varies': order_varies,
            'order_algorithm': order_algorithm,
            'order_algorithm_idx': order_algorithm_idx,
            'order_symmetry': order_symmetry,
            'order_minimum_points': order_minimum_points,
            'fit_threshold': fit_threshold,
            'is_covar': is_covar,
            'mean_fit': mean_fit,
            'cval': np.float64(cval),
            'edge_threshold': edge_threshold,
            'edge_algorithm': edge_algorithm,
            'edge_algorithm_idx': edge_algorithm_idx,
            'jobs': jobs,
            'relative_smooth': relative_smooth,
            'estimate_covariance': estimate_covariance,
            'adaptive_region_coordinates': region_coordinates
        }

        return self._fit_settings

    def _check_call_arguments(self, *args):
        r"""
        Check the fitting coordinates have the correct dimensions.

        Parameters
        ----------
        args : N-tuple
            The input coordinates for each feature.  Either arrays or
            scalar values for N features.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of features present in the fitting coordinates does
            not match the sample coordinates.
        """
        nargs = len(args)
        if len(args) not in [1, self.features]:
            raise ValueError(
                "%i-feature coordinates passed to %i-feature Resample"
                % (nargs, self.features))

    def calculate_adaptive_smoothing(self, settings):
        r"""
        Calculate the adaptive distance weighting kernel.

        Calculates a weighting kernel for each sample in the reduction.  i.e,
        each sample will have a different weighting factor based on its
        distance (and optionally, direction) from each point at which a fit
        is required.  This is done by performing a fit centered on each sample,
        and measuring how that fit deviates from the actual observed sample
        value.  The first step is to decide upon the distance weighting factor
        used to perform the initial fit.

        It is assumed that the coordinate error is known (or approximately
        known), and supplied as a :math:`1 \sigma` error as the 'alpha'
        keyword in `settings`.  For example, the error in astronomical
        observations taken using a Gaussian beam with known FWHM is:

        .. math::

            \sigma = \frac{\text{FWHM}}{2 \sqrt{2 \ln{2}}}

        However, adaptive weighting may be calculated over select dimensions,
        leaving some fixed.  In this case, the `alpha` keyword will always
        apply a weighting factor in a "fixed" dimension :math:`k` as:

        .. math::

            w_k = exp \left(
                  \frac{-\Delta x_k^2}{\alpha_{fixed, k} \Omega_k^2}
                  \right)

        where :math:`\Delta x_k` is the distance from the resampling point to
        the sample in question, and :math:`\Omega_k` is the window principle
        axis in dimension :math:`k`.  Note that in the weighting function,
        :math:`\sigma` is related to :math:`\alpha` by:

        .. math::

            \alpha_k = 2 \sigma_k^2

        and weighting is applied using the definition of :math:`w_k` above.  To
        signify that :math:`\alpha_{fixed, k}` should be used instead of
        :math:`\sigma` for dimension :math:`k`, the "adaptive_threshold" array
        in `settings` should be set to zero for the corresponding dimension.
        For example, if we had::

            settings['alpha'] = [0.3, 0.3]
            settings['adaptive_threshold'] = [1, 0]

        the first dimension would have :math:`\alpha_0=0.18`
        (:math:`2 \times 0.3^2`), and the second would have
        :math:`\alpha_1=0.3`.  In this example, :math:`\alpha_0` would be
        allowed to vary per sample, while :math:`\alpha_1` would be fixed for
        each sample at 0.3.  An initial fit is then performed at each sample
        coordinate using a test distance weighting parameter:

        .. math::

            \sigma_{test, k} = \frac{\pi \sigma_k}{\sqrt{2 ln{2}}}

            \alpha_{test, k} = 2 \sigma_{test, k}^2

        Using this initial weighting parameter, the adaptive kernels are
        derived using either
        :func:`resample_utils.shaped_adaptive_weight_matrix` or
        :func:`resample_utils.scaled_adaptive_weight_matrix` depending on
        whether the "shaped" keyword in `settings` is set to `True` or `False`
        respectively.

        The weighting kernels are stored in the "adaptive_alpha" keyword value
        in `settings`.

        Parameters
        ----------
        settings : dict
            Reduction settings, as returned by
            :func:`Resample.reduction_settings`.

        Returns
        -------
        None
        """

        adaptive = settings['adaptive_threshold']
        do_adaptive = adaptive is not None
        if do_adaptive:
            adaptive = np.atleast_1d(adaptive).astype(float)
            if adaptive.size == 0 or np.allclose(adaptive, 0):
                do_adaptive = False

        if not do_adaptive:
            settings['adaptive_threshold'] = None
            settings['adaptive_alpha'] = np.empty((0, 0, 0, 0))
            return

        sigma = np.atleast_1d(settings['alpha'])
        if sigma.size != self.features:
            sigma = np.full(self.features, sigma[0])

        fixed = adaptive == 0

        nyquist = (np.pi / np.sqrt(2 * np.log(2)))
        test_sigma = sigma * nyquist
        test_sigma[fixed] = sigma[fixed].copy()

        scaled_test_sigma = test_sigma.copy()
        scaled_test_sigma[~fixed] *= adaptive[~fixed]

        scaled_test_alpha = 2 * (scaled_test_sigma ** 2)

        if settings['relative_smooth']:
            scaled_test_alpha[fixed] = sigma[fixed].copy()
        else:
            scaled_test_alpha[fixed] = 2 * (sigma[fixed] ** 2)

        shaped = settings['shaped']

        test_reduction = self.__call__(
            scale_coordinates(self.sample_tree.coordinates.copy(),
                              self._radius, self._scale_offsets, reverse=True),
            smoothing=scaled_test_alpha,
            relative_smooth=True,
            adaptive_threshold=None,
            fit_threshold=0.0,  # No fit checking
            cval=np.nan,  # NaN on failure
            edge_threshold=0.0,  # No edge checking
            error_weighting=settings['error_weighting'],
            order_algorithm=settings['order_algorithm'],
            estimate_covariance=settings['estimate_covariance'],
            get_error=False,
            get_counts=shaped,
            get_weights=False,
            get_distance_weights=shaped,
            get_rchi2=True,
            get_cross_derivatives=shaped,
            get_offset_variance=shaped,
            is_covar=False,
            jobs=settings['jobs'],
            adaptive_region_coordinates=settings[
                'adaptive_region_coordinates'])

        if shaped:
            counts = np.atleast_2d(test_reduction[1])
            distance_weights = np.atleast_2d(test_reduction[2])
            rchi2 = np.atleast_2d(test_reduction[3])
            gradient_mscp = test_reduction[4]
            distribution_offset_variance = test_reduction[5]
            if gradient_mscp.ndim == 3:
                gradient_mscp = gradient_mscp[None]
            rho = relative_density(scaled_test_sigma, counts, distance_weights)

        else:
            rchi2 = np.atleast_2d(test_reduction[1])
            gradient_mscp = None
            rho = None
            distribution_offset_variance = None

        # Here the Nyquist level sigma is used, thereby implementing the
        # requested scaling "adaptive" factor.
        if shaped:
            gmat = shaped_adaptive_weight_matrices(
                test_sigma, rchi2, gradient_mscp,
                density=rho,
                variance_offsets=distribution_offset_variance,
                fixed=fixed)

        else:
            gmat = scaled_adaptive_weight_matrices(
                test_sigma, rchi2, fixed=fixed)

        settings['adaptive_alpha'] = np.swapaxes(gmat, 0, 1).copy()

        self._fit_settings = settings

    def __call__(self, *args, smoothing=0.0, relative_smooth=False,
                 adaptive_threshold=None, adaptive_algorithm='scaled',
                 fit_threshold=0.0, cval=np.nan, edge_threshold=0.0,
                 edge_algorithm='distribution', order_algorithm='bounded',
                 error_weighting=True, estimate_covariance=False,
                 is_covar=False, jobs=None,
                 get_error=False, get_counts=False, get_weights=False,
                 get_distance_weights=False, get_rchi2=False,
                 get_cross_derivatives=False, get_offset_variance=False,
                 adaptive_region_coordinates=None):
        """
        Resample data defined during initialization onto new coordinates

        Parameters
        ----------
        args : array_like or n-tuple of array_like
            args can take one of the following formats:

                - grid : n-tuple of array_like
                  Here `n` is the number of dimensions and each array should
                  be of shape (n_data,).  This indicates that resampling
                  should occur on a grid where the first argument
                  corresponds to the coordinates along the first dimension.
                - single point : n-tuple of float
                  `n` is the number of dimensions.  The coordinate to
                  resample onto.
                - irregular : array_like
                  An array of shape (n_dimensions, n_ndata) defining a
                  set of coordinates onto which to resample.

        smoothing : float or array_like of float, optional
            If set, weights the polynomial fitting by distance from the
            resampling coordinate to the sample coordinate.  The weighting
            factor applied is exp(-dx^2 / `smoothing`).  A value may be
            defined for each dimension by providing `smoothing` as an array of
            shape (n_dimensions,).  However, if `adaptive_threshold` > 0 for a
            certain feature, the corresponding smoothing element has a
            different meaning.  In this case, it gives the 1-sigma uncertainty
            in the coordinate position for that feature.  As a reference, for
            astronomical observations using a Gaussian beam with known FWHM,
            `smoothing` should be set to `FWHM / (2 * sqrt(2 * log(2)))`
            so long as adaptive weighting is enabled.
        relative_smooth : bool, optional
            If `True`, the supplied `smoothing` value is defined in units
            of `window`.  Otherwise, `smoothing` is in the same units as the
            supplied coordinates.
        adaptive_threshold : float or array_like (n_features,)
            If a single value is supplied, each feature will use this value.
            Otherwise, each feature should specify an adaptive threshold
            as an element of `adaptive_threshold`.  If a non-zero value is
            supplied, adaptive weighting will be enabled for that feature.
            These values define the size of the initial weighting kernel
            used during fitting.  The nominal value is 1.
        adaptive_algorithm : str, optional
            May be one of {'scaled', 'shaped', None}. `None` disables adaptive
            weighting.  The "scaled" algorithm allows the weighting kernel
            to change in size only.  The "shaped" algorithm allows the kernel
            to change in size, rotate, and stretch each primary axis.  Please
            see :func:`resample_utils.scaled_adaptive_weight_matrix` and
            :func:`resample_utils.shaped_adaptive_weight_matrix` for further
            details.
        fit_threshold : float, optional
            If nonzero, rejects a polynomial fit if it deviates by
            `|fit_threshold|` * the RMS of the samples.  If it is rejected,
            that value will be replaced by the mean (error weighted if set),
            if `fit_threshold > 0` or NaN if `fit_threshold < 0`.
        cval : float, optional
            During fitting, any fit that does not meet the `order_algorithm`
            requirement will be set to this value. This will be NaN by default.
        edge_threshold : float or array_like or float
            If set to a value > 0, edges of the fit will be masked out
            according to `edge_algorithm`. Values close to zero will result in
            a low degree of edge clipping, while values close to 1 clip edges
            to a greater extent.  The exact definition of `edge_threshold`
            depends on the algorithm.  For further details, please see
            :func:`resampling.resample_utils.check_edges`.
        edge_algorithm : str, optional
            Describes how to clip edges if edge_threshold is non-zero. The
            available algorithms are:

                - 'distribution' (default): Statistics on sample distributions
                  are calculated, and if the resampling point is > 1/threshold
                  standard deviations away from the sample mean, it will be
                  clipped.
                - 'ellipsoid': If the samples used to fit a
                  resampling point deviate from the resampling point
                  location by more than this amount, it will be clipped.
                - 'box': If the flattened 1-dimensional distribution
                  of samples center-of-mass deviates from the resampling
                  point location in any dimension, it will be clipped.
                - 'range': Over each dimension, check the distribution of
                  points is greater than edge_threshold to the "left" and
                  "right" of the resampling point.

        order_algorithm : str, optional
            The type of check to perform on whether the sample distribution
            for each resampling point is adequate to derive a polynomial fit.
            Depending on `order` and `fix_order`, if the distribution does
            not meet the criteria for `order_algorithm`, either the fit will
            be aborted, returning a value of `cval`, or the fit order will be
            reduced.  Available algorithms are:

                - 'bounded': Require that there are `order` samples in both
                  the negative and positive directions of each feature
                  from the resampling point.
                - 'counts': Require that there are (order + 1) ** n_features
                  samples within the `window` of each resampling point.
                - 'extrapolate': Attempt to fit regardless of the sample
                  distribution.

            Note that 'bounded' is the most robust mode as it ensures
            that no singular values will be encountered during the
            least-squares fitting of polynomial coefficients.
        error_weighting : bool, optional
            If `True` (default), weight polynomial fitting by the `error`
            values of each sample.
        estimate_covariance : bool, optional
            If `True`, calculate errors on the fit using
            :func:`resample_utils.estimated_covariance_matrix_inverse`.
            Otherwise, use :func:`resample_utils.covariance_matrix_inverse`.
        is_covar : bool, optional
            If True, the input data is treated as a covariance instead of
            a flux, and is propagated as if through a weighted mean.
        jobs : int, optional
            Specifies the maximum number of concurrently running jobs.
            Values of 0 or 1 will result in serial processing.  A negative
            value sets jobs to `n_cpus + 1 + jobs` such that -1 would use
            all cpus, and -2 would use all but one cpu.
        get_error : bool, optional
            If `True`, If True returns the error which is given as the weighted
            RMS of the samples used for each resampling point.
        get_counts : bool, optional
            If `True` returns the number of samples used to fit each resampling
            point.
        get_weights : bool, optional
            If `True`, returns the sum of all sample weights (error and
            distance) used in the fit at each resampling point.
        get_distance_weights : bool, optional
            If `True`, returns the sum of all sample distance weights (not
            including error) used in the fit at each resampling point.
        get_rchi2 : bool, optional
            If `True`, returns the reduced chi-squared statistic of the fit
            at each resampling point.  Note that this is only valid if errors
            were supplied during :func:`Resample.__init__`.
        get_cross_derivatives : bool, optional
            If `True`, returns the derivative mean-squared-cross-product of
            the fit derivatives at each resampling point.
        get_offset_variance : bool, optional
            If `True`, returns the offset of the resampling point from the
            center of the sample distribution used to generate the fit as
            a variance.  i.e., a return value of 9 indicates a 3-sigma
            deviation of the resampling point from the sample distribution.

        Returns
        -------
        fit_data, [fit_error], [fit_counts]
            The data fit at `args` and optionally, the error associated with
            the fit, and the number of samples used to generate each point.
            Will be of a shape determined by `args`.  `fit_data` and
            `fit_error` will be of type np.float64, and `fit_counts` will
            be of type np.int64.
        """

        self._check_call_arguments(*args)

        settings = self.reduction_settings(
            smoothing=smoothing,
            relative_smooth=relative_smooth,
            adaptive_threshold=adaptive_threshold,
            adaptive_algorithm=adaptive_algorithm,
            error_weighting=error_weighting,
            fit_threshold=fit_threshold,
            cval=cval,
            edge_threshold=edge_threshold,
            edge_algorithm=edge_algorithm,
            order_algorithm=order_algorithm,
            is_covar=is_covar,
            estimate_covariance=estimate_covariance,
            jobs=jobs,
            adaptive_region_coordinates=args)

        self.calculate_adaptive_smoothing(settings)

        fit_grid = ResampleGrid(
            *args, tree_shape=self.sample_tree.shape,
            build_tree=True, scale_factor=self._radius,
            scale_offset=self._scale_offsets, dtype=np.float64)

        fit_tree = fit_grid.tree
        if adaptive_region_coordinates is not None:
            region_grid = ResampleGrid(
                *adaptive_region_coordinates,
                tree_shape=self.sample_tree.shape,
                build_tree=True, scale_factor=self._radius,
                scale_offset=self._scale_offsets, dtype=np.float64)
            skip_blocks = region_grid.tree.hood_population == 0
            fit_tree.block_population[skip_blocks] = 0

        if settings['order_symmetry']:
            o = settings['order'][0]
        else:
            o = settings['order']

        fit_tree.set_order(o, fix_order=not settings['order_varies'])
        fit_tree.precalculate_phi_terms()

        (fit, error, counts, weights, distance_weights, sigma,
         derivative_cross_products, distribution_offset_variance
         ) = self.block_loop(
            self.data, self.error, self.mask, fit_tree, self.sample_tree,
            settings, self.iteration, get_error=get_error,
            get_counts=get_counts,
            get_weights=get_weights, get_distance_weights=get_distance_weights,
            get_rchi2=get_rchi2, get_cross_derivatives=get_cross_derivatives,
            get_offset_variance=get_offset_variance, jobs=jobs)

        self.iteration += 1

        if not self.multi_set:
            fit = fit[0]
            if get_error:
                error = error[0]
            if get_counts:
                counts = counts[0]
            if get_weights:
                weights = weights[0]
            if get_distance_weights:
                distance_weights = distance_weights[0]
            if get_rchi2:
                sigma = sigma[0]
            if get_cross_derivatives:
                derivative_cross_products = derivative_cross_products[0]
            if get_offset_variance:
                distribution_offset_variance = distribution_offset_variance[0]

        fit = fit_grid.reshape_data(fit)
        if (get_error or get_counts or get_weights
                or get_rchi2 or get_distance_weights):
            result = (fit,)
            if get_error:
                result += (fit_grid.reshape_data(error),)
            if get_counts:
                result += (fit_grid.reshape_data(counts),)
            if get_weights:
                result += (fit_grid.reshape_data(weights),)
            if get_distance_weights:
                result += (fit_grid.reshape_data(distance_weights),)
            if get_rchi2:
                result += (fit_grid.reshape_data(sigma),)

            if get_cross_derivatives:

                if fit_grid.singular:
                    if self.multi_set:
                        result += (derivative_cross_products[:, 0],)
                    else:
                        result += (derivative_cross_products[0],)
                else:
                    features = self.sample_tree.features
                    shape = fit_grid.shape + (features, features)
                    if self.multi_set:
                        shape = (derivative_cross_products.shape[0],) + shape
                    result += (derivative_cross_products.reshape(shape),)

            if get_offset_variance:
                result += (fit_grid.reshape_data(
                    distribution_offset_variance),)

            return result
        else:
            return fit

    @staticmethod
    def block_loop(sample_values, error, mask, fit_tree, sample_tree, settings,
                   iteration, get_error=True, get_counts=True,
                   get_weights=True, get_distance_weights=True, get_rchi2=True,
                   get_cross_derivatives=True, get_offset_variance=True,
                   jobs=None):
        r"""
        Perform resampling reduction in parallel or series.

        Utility function to allow the resampling algorithm to process blocks
        of data in series or parallel, recombining the data once complete.
        Please see :func:`Resample.__call__` for descriptions of the arguments.

        Parameters
        ----------
        sample_values : numpy.ndarray
        error : numpy.ndarray
        mask : numpy.ndarray
        fit_tree : resampling.tree.Rtree object
        sample_tree : resampling.tree.Rtree object
        settings : dict
        iteration : int
        get_error : bool, optional
        get_counts : bool, optional
        get_weights : bool, optional
        get_distance_weights : bool, optional
        get_rchi2 : bool, optional
        get_cross_derivatives : bool, optional
        get_offset_variance : bool, optional
        jobs : int, optional

        Returns
        -------
        combined_results : 8-tuple of numpy.ndarray
            In order: fit, error, counts, weights, distance weights,
                reduced chi-squared, MSCP derivatives, distribution offset.
        """
        n_sets = sample_values.shape[0]
        n_fits = fit_tree.n_members
        n_features = sample_tree.features

        block_population = fit_tree.block_population
        hood_population = sample_tree.hood_population

        args = (sample_values, error, mask, fit_tree, sample_tree,
                get_error, get_counts, get_weights, get_distance_weights,
                get_rchi2, get_cross_derivatives, get_offset_variance,
                settings)

        _global_resampling_values['args'] = args
        _global_resampling_values['iteration'] = iteration

        if jobs is not None and jobs > 1:
            cache_dir = mkdtemp()
            filename = os.path.join(
                cache_dir, 'joblib_resampling_cache_%s.%s' %
                           (id(args), time.time()))
            joblib.dump(args, filename)
        else:
            cache_dir = filename = None

        _global_resampling_values['filename'] = iteration

        task_args = filename, iteration

        blocks = multitask(
            Resample.process_block, range(fit_tree.n_blocks), task_args, None,
            jobs=jobs, skip=(block_population == 0) | (hood_population == 0))

        if 'args' in _global_resampling_values:
            del _global_resampling_values['args']

        if cache_dir is not None:
            shutil.rmtree(cache_dir)

        return Resample.combine_blocks(
            blocks, n_sets, n_fits, n_features, settings['cval'],
            get_error=get_error,
            get_counts=get_counts,
            get_weights=get_weights,
            get_distance_weights=get_distance_weights,
            get_rchi2=get_rchi2,
            get_cross_derivatives=get_cross_derivatives,
            get_offset_variance=get_offset_variance)

    @staticmethod
    def process_block(args, block):
        r"""
        Run :func:`solve_fits` on each block.

        Utility function that parses the settings and tree objects to something
        usable by the numba JIT compiled resampling functions.  This is not
        meant to be called directly.

        Parameters
        ----------
        args : 13-tuple
            Please just read the code for each parameter.
        block : int
            The block index to process.

        Returns
        -------
        results : 9-tuple of numpy.ndarray
            The first element contains the fit point indices to be fit.  For
            the remaining elements, please see :func:`solve_fits` return
            values.
        """
        filename, iteration = args

        # This cannot be covered on tests as it occurs on other CPUs.
        load_args = False
        if filename is not None:  # pragma: no cover
            if 'args' not in _global_resampling_values:
                load_args = True
            elif 'iteration' not in _global_resampling_values:
                load_args = True
            elif iteration != _global_resampling_values.get('iteration'):
                load_args = True
            elif filename != _global_resampling_values.get('filename'):
                load_args = True

        if load_args:
            _global_resampling_values['args'] = joblib.load(filename)
            _global_resampling_values['iteration'] = iteration
            _global_resampling_values['filename'] = filename

        (sample_values, sample_error, sample_mask, fit_tree, sample_tree,
         get_error, get_counts, get_weights,
         get_distance_weights, get_rchi2, get_cross_derivatives,
         get_offset_variance, settings) = _global_resampling_values['args']

        fit_indices, fit_coordinates, fit_phi_terms = \
            fit_tree.block_members(block, get_locations=True, get_terms=True)

        sample_indices = convert_to_numba_list(sample_tree.query_radius(
            fit_coordinates, 1.0, return_distance=False))

        return (fit_indices,
                *solve_fits(
                    sample_indices, sample_tree.coordinates,
                    sample_tree.phi_terms,
                    sample_values, sample_error, sample_mask,
                    fit_coordinates, fit_phi_terms, settings['order'],
                    settings['alpha'], settings['adaptive_alpha'],
                    is_covar=settings['is_covar'],
                    mean_fit=settings['mean_fit'],
                    cval=settings['cval'],
                    fit_threshold=settings['fit_threshold'],
                    error_weighting=settings['error_weighting'],
                    estimate_covariance=settings['estimate_covariance'],
                    order_algorithm_idx=settings['order_algorithm_idx'],
                    order_term_indices=sample_tree.term_indices,
                    derivative_term_map=sample_tree.derivative_term_map,
                    edge_algorithm_idx=settings['edge_algorithm_idx'],
                    edge_threshold=settings['edge_threshold'],
                    minimum_points=settings['order_minimum_points'],
                    get_error=get_error, get_counts=get_counts,
                    get_weights=get_weights,
                    get_distance_weights=get_distance_weights,
                    get_rchi2=get_rchi2,
                    get_cross_derivatives=get_cross_derivatives,
                    get_offset_variance=get_offset_variance))

    @staticmethod
    def combine_blocks(blocks, n_sets, n_fits, n_dimensions, cval,
                       get_error=True, get_counts=True, get_weights=True,
                       get_distance_weights=True, get_rchi2=True,
                       get_cross_derivatives=True, get_offset_variance=True):
        r"""
        Combines the results from multiple reductions into one set.

        The resampling reduction may be performed in serial or parallel over
        multiple "blocks", where each block contains a set of spatially
        close fit coordinates and all samples necessary to perform a fit
        over all points.

        Parameters
        ----------
        blocks : n-tuple of processed reductions for n blocks.
        n_sets : int
            The number of data sets in the reduction.  Each set is contains
            the same sample coordinates as all other sets, but the sample
            values may vary.
        n_fits : int
            The number of fitting points over all blocks.
        n_dimensions : int
            The number of coordinate dimensions.
        cval : float
            The fill value for missing data in the output fit value arrays.
        get_error : bool, optional
            If `True`, indicates that errors on the fit were calculated.
        get_counts : bool, optional
            If `True`, indicates that the number of samples used for each
            fit should be returned.
        get_weights : bool, optional
            If `True`, indicates that the total weight sum of all samples used
            in each fit should be returned.
        get_distance_weights : bool, optional
            If `True`, indicates that the distance weight sum of all samples
            used in each fit should be returned.
        get_rchi2 : bool, optional
            If `True`, indicates that the reduced chi-squared statistic for
            each fit should be returned.
        get_cross_derivatives : bool, optional
            If `True`, indicates that the derivative MSCP should be returned.
        get_offset_variance : bool, optional
            If `True`, indicates that the offset variance of the fit from the
            sample distribution should be returned.

        Returns
        -------
        results : 8-tuple of numpy.ndarray
            results[0] = fitted values
            results[1] = error on the fit
            results[2] = counts
            results[3] = total weight sums
            results[4] = total distance weight sums
            results[5] = reduced chi-squared statistic
            results[6] = derivative MSCP
            results[7] = offset variance

        Notes
        -----
        The return value is always an 8-tuple, and the get_* keywords indicate
        whether the calculated values in the block reductions are valid.  If
        `False`, the corresponding output array will have the correct number of
        axes, but be of zero size.
        """

        fit = np.full((n_sets, n_fits), float(cval))

        if get_error:
            error = np.full((n_sets, n_fits), np.nan)
        else:
            error = np.empty((0, 0), dtype=float)

        if get_counts:
            counts = np.zeros((n_sets, n_fits), dtype=int)
        else:
            counts = np.empty((0, 0), dtype=int)

        if get_weights:
            weights = np.zeros((n_sets, n_fits), dtype=float)
        else:
            weights = np.empty((0, 0), dtype=float)

        if get_distance_weights:
            distance_weights = np.zeros((n_sets, n_fits), dtype=float)
        else:
            distance_weights = np.empty((0, 0), dtype=float)

        if get_rchi2:
            rchi2 = np.full((n_sets, n_fits), np.nan)
        else:
            rchi2 = np.empty((0, 0), dtype=float)

        if get_cross_derivatives:
            cross_derivatives = np.full(
                (n_sets, n_fits, n_dimensions, n_dimensions), np.nan,
                dtype=float)
        else:
            cross_derivatives = np.empty((1, 0, 0, 0), dtype=float)

        if get_offset_variance:
            offsets = np.full((n_sets, n_fits), np.nan, dtype=float)
        else:
            offsets = np.empty((0, 0), dtype=float)

        for block in blocks:
            fit_indices = block[0]
            fit[:, fit_indices] = block[1]
            if get_error:
                error[:, fit_indices] = block[2]
            if get_counts:
                counts[:, fit_indices] = block[3]
            if get_weights:
                weights[:, fit_indices] = block[4]
            if get_distance_weights:
                distance_weights[:, fit_indices] = block[5]
            if get_rchi2:
                rchi2[:, fit_indices] = block[6]
            if get_cross_derivatives:
                cross_derivatives[:, fit_indices] = block[7]
            if get_offset_variance:
                offsets[:, fit_indices] = block[8]

        return (fit, error, counts, weights, distance_weights, rchi2,
                cross_derivatives, offsets)


def resamp(coordinates, data, *locations,
           error=None, mask=None, window=None,
           order=1, fix_order=True,
           robust=None, negthresh=None,
           window_estimate_bins=10,
           window_estimate_percentile=50,
           window_estimate_oversample=2.0,
           leaf_size=40,
           smoothing=0.0, relative_smooth=False,
           adaptive_threshold=None, adaptive_algorithm='scaled',
           fit_threshold=0.0, cval=np.nan, edge_threshold=0.0,
           edge_algorithm='distribution', order_algorithm='bounded',
           error_weighting=True, estimate_covariance=False,
           is_covar=False, jobs=None,
           get_error=False, get_counts=False, get_weights=False,
           get_distance_weights=False, get_rchi2=False,
           get_cross_derivatives=False, get_offset_variance=False,
           **distance_kwargs):
    r"""
    Resample data using local polynomial fitting.

    Initializes and then calls the :class:`Resample` class.  For further
    details on all available parameters, please see :func:`Resample.__init__`
    and :func:`Resample.__call__`.

    Parameters
    ----------
    coordinates
    data
    locations
    error
    mask
    window
    order
    fix_order
    robust
    negthresh
    window_estimate_bins
    window_estimate_percentile
    window_estimate_oversample
    leaf_size
    smoothing
    relative_smooth
    adaptive_threshold
    adaptive_algorithm
    fit_threshold
    cval
    edge_threshold
    edge_algorithm
    order_algorithm
    error_weighting
    estimate_covariance
    is_covar
    jobs
    get_error
    get_counts
    get_weights
    get_distance_weights
    get_rchi2
    get_cross_derivatives
    get_offset_variance
    distance_kwargs

    Returns
    -------
    results : float or numpy.ndarray or n-tuple of (float or numpy.ndarray)
        If a fit is performed at a single location, the output will consist
        of int or float scalar values.  Multiple fits result in numpy arrays.
        The exact output shape depends on the number of data sets, number of
        fitted points, dimensions of the fit locations.  Assuming that all
        get_* keywords are set to `True`, the output order is:

            results[0] = fitted values
            results[1] = error on the fit
            results[2] = sample counts for each fit
            results[3] = total weight of all samples in fit
            results[4] = total distance weight sum of all samples in fit
            results[5] = reduced chi-squared statistic of the fit
            results[6] = derivative mean squared cross products
            results[7] = offset variance of fit from sample distribution
    """

    resampler = Resample(coordinates, data,
                         error=error, mask=mask, window=window,
                         order=order, fix_order=fix_order,
                         robust=robust, negthresh=negthresh,
                         window_estimate_bins=window_estimate_bins,
                         window_estimate_percentile=window_estimate_percentile,
                         window_estimate_oversample=window_estimate_oversample,
                         leaf_size=leaf_size,
                         **distance_kwargs)

    return resampler(*locations,
                     smoothing=smoothing,
                     relative_smooth=relative_smooth,
                     adaptive_threshold=adaptive_threshold,
                     adaptive_algorithm=adaptive_algorithm,
                     fit_threshold=fit_threshold,
                     cval=cval,
                     edge_threshold=edge_threshold,
                     edge_algorithm=edge_algorithm,
                     order_algorithm=order_algorithm,
                     error_weighting=error_weighting,
                     estimate_covariance=estimate_covariance,
                     is_covar=is_covar,
                     jobs=jobs,
                     get_error=get_error,
                     get_counts=get_counts,
                     get_weights=get_weights,
                     get_distance_weights=get_distance_weights,
                     get_rchi2=get_rchi2,
                     get_cross_derivatives=get_cross_derivatives,
                     get_offset_variance=get_offset_variance)
