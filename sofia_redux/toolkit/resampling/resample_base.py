# Licensed under a 3-clause BSD style license - see LICENSE.rst

import bottleneck as bn
import numba
import numpy as np
import os
from scipy.special import gamma
import shutil
from tempfile import mkdtemp
import time
import warnings

from sofia_redux.toolkit.utilities.multiprocessing import (
    multitask, relative_cores, pickle_object)
from sofia_redux.toolkit.stats.stats import robust_mask
from sofia_redux.toolkit.resampling.grid.base_grid import BaseGrid
from sofia_redux.toolkit.resampling.resample_utils import scale_coordinates
from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree


_global_resampling_values = {}
__all__ = ['ResampleBase', '_global_resampling_values']


class ResampleBase(object):

    def __init__(self, coordinates, data,
                 error=None, mask=None, window=None,
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
            fit during :func:`ResamplePolynomial.__call__`.
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
            :func:`ResamplePolynomial.estimate_feature_windows`.
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
            :func:`ResamplePolynomial.estimate_feature_windows`.
        window_estimate_percentile : int or float, optional
            Used to estimate the `window` if not supplied using
            :func:`ResamplePolynomial.estimate_feature_windows`.
        window_estimate_oversample : int or float, optional
            Used to estimate the `window` if not supplied using
            :func:`ResamplePolynomial.estimate_feature_windows`.
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
        self._n_sets = 0
        self._n_features = 0
        self._n_samples = 0
        self._multiset = None
        self._valid_set = None
        self.data = None
        self.coordinates = None
        self.error = None
        self.mask = None
        self._error_valid = None
        self._fit_settings = None
        self.sample_tree = None
        self.fit_grid = None
        self.iteration = 0

        self._process_input_data(data, coordinates,
                                 error=error, mask=mask, negthresh=negthresh,
                                 robust=robust)

        self.set_sample_tree(
            self.coordinates, radius=window,
            window_estimate_bins=window_estimate_bins,
            window_estimate_percentile=window_estimate_percentile,
            window_estimate_oversample=window_estimate_oversample,
            leaf_size=leaf_size, **distance_kwargs)

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
    def fit_settings(self):
        r"""dict : Fit reduction settings applied during last call"""
        return self._fit_settings

    @property
    def fit_tree(self):
        """
        Return the fitting tree representative of points to fit.

        Returns
        -------
        BaseTree
        """
        if self.fit_grid is None:
            return None
        return self.fit_grid.tree

    @property
    def grid_class(self):
        """
        Return the grid class of the resampler

        Returns
        -------
        BaseGrid subclass
        """
        return self.get_grid_class()

    @classmethod
    def global_resampling_values(cls):
        """
        Return the global resampling values.

        The global resampling values are of main importance when performing
        multiprocessing, and allows each process to gain fast access to the
        necessary data.

        Returns
        -------
        dict
        """
        return _global_resampling_values

    def get_grid_class(self):
        """
        Return the appropriate grid class for the resampler.

        Returns
        -------
        BaseGrid subclass
        """
        return BaseGrid.get_class_for(self)

    def _process_input_data(self, data, coordinates, error=None, mask=None,
                            negthresh=None, robust=None):
        """Formats the input data, error, and mask for subsequent use.

        Sets the data, mask, and error attributes to numpy arrays of shape
        (n_sets, n_samples).

        The output mask will be a union of the input mask (if there is one) and
        finite data values and nonzero error values.  If the user has provided
        `robust` or `negthresh` then the mask will be updated to reflect this.
        See :func:`ResamplePolynomial.__init__` for further details.
        """
        coordinates, data, error, mask = self._check_input_arrays(
            coordinates, data, error=error, mask=mask)
        self._n_features, self._n_samples = coordinates.shape
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
            if error.size == 1:
                error = np.full((self._n_sets, 1), error.ravel()[0])
            else:
                error = np.atleast_2d(error)

            error = error.astype(np.float64, order='F')
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
        self.coordinates = coordinates
        self.error = error
        self.mask = np.logical_not(invalid, order='F')

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
            # Error may either be scalar value applied to all, or an array
            error = np.atleast_1d(np.asarray(error, dtype=np.float64))

            if error.shape != shape and error.size != 1:
                if (data.ndim == 2 and error.ndim == 1
                        and error.size == shape[0]):
                    # Error contains a single value for each data set
                    error = error[:, None]
                else:
                    raise ValueError(
                        "Error must be a single value, an array matching the "
                        "data shape, or an array containing a single value "
                        "for each data set.")

        if mask is not None:
            mask = np.asarray(mask, dtype=np.bool)
            if mask.shape != shape:
                raise ValueError("Mask shape does not match data")

        return coordinates, data, error, mask

    def set_sample_tree(self, coordinates,
                        radius=None,
                        window_estimate_bins=10,
                        window_estimate_percentile=50,
                        window_estimate_oversample=2.0,
                        leaf_size=40,
                        **distance_kwargs):
        """
        Build the sample tree from input coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray (float)
            The input coordinates of shape (n_features, n_samples).
        radius :  float or sequence (float), optional
            The radius of the window around each fitting point used to
            determine sample selection for fit.  If not supplied, will be
            estimated using
            :func:`ResamplePolynomial.estimate_feature_windows`.
        window_estimate_bins : int, optional
            Used to estimate the `window` if not supplied using
            :func:`ResamplePolynomial.estimate_feature_windows`.
        window_estimate_percentile : int or float, optional
            Used to estimate the `window` if not supplied using
            :func:`ResamplePolynomial.estimate_feature_windows`.
        window_estimate_oversample : int or float, optional
            Used to estimate the `window` if not supplied using
            :func:`ResamplePolynomial.estimate_feature_windows`.
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

        Returns
        -------
        None
        """
        scaled_coordinates = self._scale_to_window(
            coordinates,
            radius=radius,
            feature_bins=window_estimate_bins,
            percentile=window_estimate_percentile,
            oversample=window_estimate_oversample
        ).astype(np.float64)

        tree_class = BaseTree.get_class_for(self)
        self.sample_tree = tree_class(
            scaled_coordinates, build_type='all', leaf_size=leaf_size,
            **distance_kwargs)

    def _scale_to_window(self, coordinates, radius=None,
                         feature_bins=10, percentile=50,
                         oversample=2.0):
        r"""
        Scale input coordinates to units of the resampling window.

        Coordinates (:math:`x`) are scaled such that for dimension :math:`k`,
        the coordinates stored in the resampling tree (see
        :func:`resampling.tree.BaseTree`) are set as:

        .. math::

            x_k^{\prime} = \frac{x_k - min(x_k)}{\Omega_k}

        where :math:`\Omega` is the resampling window radius.  If the window
        radius (or principle axes) are unknown, an attempt is made to determine
        one using `:func:`Resampler.estimate_feature_windows`.

        Parameters
        ----------
        coordinates : numpy.ndarray (n_features, n_coordinates)
            The sample coordinates.
        radius : numpy.ndarray (n_features,), optional
            The radius of the window around each fitting point used to
            determine sample selection for fit.  If not supplied, will be
            estimated using
            :func:`ResamplePolynomial.estimate_feature_windows`.
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
            the exact number of samples required for a fit assuming uniform
            density of the samples.

        Returns
        -------
        scaled_coordinates : numpy.ndarray (n_features, n_coordinates)
            The coordinates scaled to units of the window radius.
        """
        if radius is None:
            radius = self.estimate_feature_windows(
                coordinates,
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

    def estimate_feature_windows(self, coordinates, feature_bins=10,
                                 percentile=50, oversample=2.0):
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

        unit_spheroid_volume = (np.pi ** (features / 2)
                                ) / gamma((features / 2) + 1)
        bin_volume = 1.0  # in units of feature_bins (just for my notes)
        bin_density = bin_population / bin_volume
        unit_spheroid_count = unit_spheroid_volume * bin_density

        required_samples = self.calculate_minimum_points()
        unit_radius = (required_samples * oversample
                       / unit_spheroid_count) ** (1 / features)

        unit_sample_spacing = bin_density ** (-1 / features)
        max_offset = 0.5 * unit_sample_spacing
        unit_radius += max_offset

        scaled_radius = unit_radius * scale / feature_bins

        return scaled_radius

    def calculate_minimum_points(self):
        """
        Return the minimum number of points for a fit.

        Parameters
        ----------
        args : n-tuple
            Input arguments used to determine the number of points required for
            a fit.

        Returns
        -------
        minimum_points : int
            The minimum number of points for a fit
        """
        return 1

    def reduction_settings(self, error_weighting=True,
                           fit_threshold=0.0, cval=np.nan,
                           edge_threshold=0.0, edge_algorithm='distribution',
                           jobs=None, use_threading=None, use_processes=None,
                           **kwargs):
        r"""
        Define a set of reduction instructions based on user input.

        This method is responsible for determining, formatting, and checking
        a number variables required for the resampling algorithm based on
        user input.  For detailed descriptions of user options, please see
        :func:`ResamplePolynomial.__call__`.

        Parameters
        ----------
        error_weighting : bool, optional
        fit_threshold : float, optional
        cval : float, optional
        edge_threshold : float or array_like (n_features,), optional
        edge_algorithm : str, optional
        jobs : int, optional
        use_threading : bool, optional
            If `True`, force use of threads during multiprocessing.
        use_processes : bool, optional
            If `True`, force use of sub-processes during multiprocessing.
        kwargs : dict
            Optional keyword arguments to the reduction settings.

        Returns
        -------
        settings : dict
            The reduction settings.  Also, stored as
            :func:`ResamplePolynomial.fit_settings`.
        """
        n_features = self.sample_tree.features
        error_weighting &= self._error_valid

        if (edge_threshold is None
                or (np.atleast_1d(edge_threshold) == 0).all()):
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
            raise ValueError(f"Unknown edge algorithm: {edge_algorithm}")
        edge_algorithm_idx = edge_func_lookup[edge_algorithm]

        if np.any(edge_threshold < 0):
            raise ValueError("Edge threshold must positive valued.")

        upper_edge_limit = np.inf if edge_algorithm == 'distribution' else 1
        for x in np.atleast_1d(edge_threshold):
            if x < 0 or x >= upper_edge_limit:
                raise ValueError(f"Edge threshold must be less than "
                                 f"{upper_edge_limit} for "
                                 f"{edge_algorithm} algorithm")

        check_fit = True if fit_threshold else False
        if check_fit:
            fit_threshold = np.float64(fit_threshold)
        else:
            fit_threshold = np.float64(0.0)

        use_processes = bool(use_processes)
        use_threading = bool(use_threading)
        if use_processes and use_threading:
            raise ValueError("Can use either thread or process based "
                             "multiprocessing; not both.")
        elif not use_processes and not use_threading:
            use_processes = True

        self._fit_settings = {
            'n_features': n_features,
            'error_weighting': error_weighting,
            'fit_threshold': fit_threshold,
            'cval': np.float64(cval),
            'edge_threshold': edge_threshold,
            'edge_algorithm': edge_algorithm,
            'edge_algorithm_idx': edge_algorithm_idx,
            'jobs': jobs,
            'use_processes': use_processes,
            'use_threading': use_threading
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
                f"{nargs}-feature coordinates passed to "
                f"{self.features}-feature resampler.")

    def __call__(self, *args,
                 fit_threshold=0.0, cval=np.nan, edge_threshold=0.0,
                 edge_algorithm='distribution', error_weighting=True,
                 jobs=None, use_threading=None, use_processes=None,
                 get_error=False, get_counts=False, get_weights=False,
                 get_distance_weights=False, get_rchi2=False,
                 get_cross_derivatives=False, get_offset_variance=False,
                 **kwargs):
        """
        Resample data defined during initialization onto new coordinates.

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
        error_weighting : bool, optional
            If `True` (default), weight polynomial fitting by the `error`
            values of each sample.
        is_covar : bool, optional
            If True, the input data is treated as a covariance instead of
            a flux, and is propagated as if through a weighted mean.
        jobs : int, optional
            Specifies the maximum number of concurrently running jobs.  An
            attempt will be made to parallel process using a thread-pool if
            available, but will otherwise revert to the "loky" backend.
            Values of 0 or 1 will result in serial processing.  A negative
            value sets jobs to `n_cpus + 1 + jobs` such that -1 would use
            all cpus, and -2 would use all but one cpu.
        use_threading : bool, optional
            If `True`, force use of threads during multiprocessing.
        use_processes : bool, optional
            If `True`, force use of sub-processes during multiprocessing.
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
            were supplied during :func:`ResamplePolynomial.__init__`.
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
        fit, [optional return values]
            The data fit at `args`.  Optional return values may also be
            returned if any of the get_* options are `True`.  If all are set
            to `True`, the return order is: fit, error, counts, weights,
            distance_weights, rchi2, cross_derivatives, offset_variance.
        """
        self._check_call_arguments(*args)

        settings = self.reduction_settings(
            error_weighting=error_weighting,
            fit_threshold=fit_threshold,
            cval=cval,
            edge_threshold=edge_threshold,
            edge_algorithm=edge_algorithm,
            jobs=jobs,
            use_threading=use_threading,
            use_processes=use_processes,
            **kwargs)

        self.pre_fit(settings, *args)

        (fit, error, counts, weights, distance_weights, sigma,
         derivative_cross_products, distribution_offset_variance
         ) = self.block_loop(
            self.data, self.error, self.mask, self.fit_tree, self.sample_tree,
            settings, self.iteration, get_error=get_error,
            get_counts=get_counts, get_weights=get_weights,
            get_distance_weights=get_distance_weights, get_rchi2=get_rchi2,
            get_cross_derivatives=get_cross_derivatives,
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

        fit = self.fit_grid.reshape_data(fit)
        if (get_error or get_counts or get_weights
                or get_rchi2 or get_distance_weights):
            result = (fit,)
            if get_error:
                result += (self.fit_grid.reshape_data(error),)
            if get_counts:
                result += (self.fit_grid.reshape_data(counts),)
            if get_weights:
                result += (self.fit_grid.reshape_data(weights),)
            if get_distance_weights:
                result += (self.fit_grid.reshape_data(distance_weights),)
            if get_rchi2:
                result += (self.fit_grid.reshape_data(sigma),)

            if get_cross_derivatives:

                if self.fit_grid.singular:
                    if self.multi_set:
                        result += (derivative_cross_products[:, 0],)
                    else:
                        result += (derivative_cross_products[0],)
                else:
                    features = self.sample_tree.features
                    shape = self.fit_grid.shape + (features, features)
                    if self.multi_set:
                        shape = (derivative_cross_products.shape[0],) + shape
                    result += (derivative_cross_products.reshape(shape),)

            if get_offset_variance:
                result += (self.fit_grid.reshape_data(
                    distribution_offset_variance),)

            return result
        else:
            return fit

    def pre_fit(self, settings, *args):
        """
        Perform pre-fitting steps and build the fitting tree.

        Parameters
        ----------
        settings : dict
            Settings calculated via `reduction_settings` to be applied
            if necessary.
        args : n-tuple
            The call input arguments.

        Returns
        -------
        None
        """
        self.fit_grid = self.grid_class(
            *args, tree_shape=self.sample_tree.shape,
            build_tree=True, scale_factor=self._radius,
            scale_offset=self._scale_offsets, dtype=np.float64)

    @classmethod
    def block_loop(cls, sample_values, error, mask, fit_tree, sample_tree,
                   settings, iteration, get_error=True, get_counts=True,
                   get_weights=True,
                   get_distance_weights=True, get_rchi2=True,
                   get_cross_derivatives=True, get_offset_variance=True,
                   jobs=None):
        r"""
        Perform resampling reduction in parallel or series.

        Utility function to allow the resampling algorithm to process blocks
        of data in series or parallel, recombining the data once complete.
        Please see :func:`ResamplePolynomial.__call__` for descriptions of the
        arguments.

        Parameters
        ----------
        sample_values : numpy.ndarray
        error : numpy.ndarray
        mask : numpy.ndarray
        fit_tree : BaseTree
        sample_tree : BaseTree
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
        args = (sample_values, error, mask, fit_tree, sample_tree,
                get_error, get_counts, get_weights, get_distance_weights,
                get_rchi2, get_cross_derivatives, get_offset_variance,
                settings)
        kwargs = None

        blocks = cls.process_blocks(args, kwargs, settings, sample_tree,
                                    fit_tree, jobs, iteration)

        n_sets = sample_values.shape[0]
        n_fits = fit_tree.n_members
        n_features = sample_tree.features

        return ResampleBase.combine_blocks(
            blocks, n_sets, n_fits, n_features, settings['cval'],
            get_error=get_error,
            get_counts=get_counts,
            get_weights=get_weights,
            get_distance_weights=get_distance_weights,
            get_rchi2=get_rchi2,
            get_cross_derivatives=get_cross_derivatives,
            get_offset_variance=get_offset_variance)

    @classmethod
    def process_blocks(cls, args, kwargs, settings, sample_tree, fit_tree,
                       jobs, iteration):
        """
        Wrapper for handling block resampling in a multiprocessing environment.

        Parameters
        ----------
        args : n-tuple
            The arguments to pass into :func:`multitask`.
        kwargs : dict or None
            The keyword arguments to pass into :func:`multitask`.
        settings : dict
            Reduction settings.
        sample_tree : BaseTree
            The resampling tree in sample space.
        fit_tree : BaseTree
            The fitting tree.
        jobs : int
            The number of jobs to perform in parallel.
        iteration : int
            The current resampling iteration.  This is simply used as a marker
            to distinguish pickled files on each resampling run.

        Returns
        -------
        blocks : list
            A list of the return values from the :method:`process_block` method
            for each block.
        """

        block_population = fit_tree.block_population
        hood_population = sample_tree.hood_population

        cores = relative_cores(jobs)
        if cores > 1:
            cache_dir = mkdtemp()
            filename = os.path.join(
                cache_dir,
                f'resampling_cache_'
                f'{id(args)}.{id(fit_tree)}.{time.time()}')
            filename = pickle_object(args, filename)
        else:
            cache_dir = filename = None

        _global_resampling_values['args'] = args
        _global_resampling_values['iteration'] = iteration
        _global_resampling_values['filename'] = filename
        task_args = filename, iteration

        old_threading = numba.config.THREADING_LAYER
        numba.config.THREADING_LAYER = 'threadsafe'
        force_threading = settings.get('use_threading', False)
        force_processes = settings.get('use_processes', False)

        blocks = multitask(
            cls.process_block, range(fit_tree.n_blocks), task_args, kwargs,
            jobs=cores, skip=(block_population == 0) | (hood_population == 0),
            force_threading=force_threading,
            force_processes=force_processes)

        numba.config.THREADING_LAYER = old_threading

        if 'args' in _global_resampling_values:
            del _global_resampling_values['args']
            del _global_resampling_values['iteration']
            del _global_resampling_values['filename']

        if cache_dir is not None:
            shutil.rmtree(cache_dir)

        return blocks

    @classmethod
    def process_block(cls, args, block):
        r"""
        Run :func:`solve_fits` on each block.

        Utility function that parses the settings and tree objects to something
        usable by the numba JIT compiled resampling functions.  This is not
        meant to be called directly.

        Parameters
        ----------
        args : 2-tuple
            A tuple of form (filename, iteration) where the filename is a
            string pointing towards a previously saved pickle file containing
            the relevant information for the reduction if required.  If set to
            `None`, the arguments are retrieved from the
            `_global_resampling_values` global parameter.
        block : int
            The block index to process.

        Returns
        -------
        results : 9-tuple of numpy.ndarray
            The first element contains the fit point indices to be fit.  For
            the remaining elements, please see :func:`solve_fits` return
            values.
        """
        return None

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
