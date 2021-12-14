# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

from sofia_redux.toolkit.resampling.resample_utils import (
    scale_coordinates,
    shaped_adaptive_weight_matrices,
    scaled_adaptive_weight_matrices,
    relative_density, solve_fits)
from sofia_redux.toolkit.resampling.resample_base import (
    ResampleBase, _global_resampling_values)
from sofia_redux.toolkit.utilities.multiprocessing import unpickle_file


__all__ = ['ResamplePolynomial', 'resamp']


class ResamplePolynomial(ResampleBase):

    def __init__(self, coordinates, data,
                 error=None, mask=None, window=None,
                 robust=None, negthresh=None,
                 window_estimate_bins=10,
                 window_estimate_percentile=50,
                 window_estimate_oversample=2.0,
                 leaf_size=40,
                 order=1, fix_order=True,
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
        self._order = order
        self._fix_order = fix_order
        super().__init__(coordinates, data, error=error, mask=mask,
                         window=window, robust=robust, negthresh=negthresh,
                         window_estimate_bins=window_estimate_bins,
                         window_estimate_percentile=window_estimate_percentile,
                         window_estimate_oversample=window_estimate_oversample,
                         leaf_size=leaf_size, **distance_kwargs)

    @property
    def order(self):
        """
        Return the order of polynomial fit.

        Returns
        -------
        order : int or numpy.ndarray (int)
            A symmetrical order, or the order for each feature.
        """
        if self.sample_tree is None:
            return self._order
        return self.sample_tree.order

    @property
    def fit_tree(self):
        """
        Return the fitting tree representative of points to fit.

        Returns
        -------
        PolynomialTree
        """
        return super().fit_tree

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
        self._order = self.check_order(self._order, self._n_features,
                                       self._n_samples)
        super().set_sample_tree(
            coordinates,
            radius=radius,
            window_estimate_bins=window_estimate_bins,
            window_estimate_percentile=window_estimate_percentile,
            window_estimate_oversample=window_estimate_oversample,
            leaf_size=leaf_size,
            **distance_kwargs)
        self.sample_tree.set_order(self._order, fix_order=self._fix_order)
        self.sample_tree.precalculate_phi_terms()

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

    def calculate_minimum_points(self):
        """
        Return the minimum number of points for a polynomial fit.

        Returns
        -------
        minimum_points : int
            The minimum number of points for a polynomial fit
        """
        o = np.asarray(self.order)
        if o.shape == ():
            o = np.full(self.features, int(o))
        return np.product(o + 1)

    def reduction_settings(self, smoothing=0.0, relative_smooth=False,
                           adaptive_threshold=None,
                           adaptive_algorithm='scaled', error_weighting=True,
                           fit_threshold=0.0, cval=np.nan,
                           edge_threshold=0.0, edge_algorithm='distribution',
                           order_algorithm='bounded', is_covar=False,
                           estimate_covariance=False, jobs=None,
                           adaptive_region_coordinates=None,
                           use_threading=None,
                           use_processes=None):
        r"""
        Define a set of reduction instructions based on user input.

        This method is responsible for determining, formatting, and checking
        a number variables required for the resampling algorithm based on
        user input.  For detailed descriptions of user options, please see
        :func:`ResamplePolynomial.__call__`.

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
        use_threading : bool, optional
            If `True`, force use of threads during multiprocessing.
        use_processes : bool, optional
            If `True`, force use of sub-processes during multiprocessing.

        Returns
        -------
        settings : dict
            The reduction settings.  Also, stored as
            :func:`ResamplePolynomial.fit_settings`.
        """
        settings = super().reduction_settings(
            error_weighting=error_weighting,
            fit_threshold=fit_threshold,
            cval=cval,
            edge_threshold=edge_threshold,
            edge_algorithm=edge_algorithm,
            jobs=jobs,
            use_threading=use_threading,
            use_processes=use_processes)

        settings['estimate_covariance'] = estimate_covariance

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
                    # only relevant in 2+ dimensions
                    shaped = self.features > 1
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
                smoothing = np.full(self.features, 1 / 3 if relative_smooth
                                    else self.window / 3)

            alpha = np.atleast_1d(np.asarray(smoothing, dtype=np.float64))
            if alpha.size not in [1, self.features]:
                raise ValueError(
                    "Smoothing size does not match number of features")

            if adaptive:
                adaptive_threshold = np.atleast_1d(
                    np.asarray(adaptive_threshold, dtype=np.float64))

                if alpha.size != self.features:
                    alpha = np.full(self.features, alpha[0])

                if adaptive_threshold.size not in [1, self.features]:
                    raise ValueError("Adaptive smoothing size does not "
                                     "match number of features")
                elif adaptive_threshold.size != self.features:
                    adaptive_threshold = np.full(
                        self.features, adaptive_threshold[0])

            if not relative_smooth:
                if alpha.size != self.features:  # alpha size = 1
                    alpha = np.full(self.features, alpha[0])
                if not adaptive:
                    alpha = 2 * (alpha ** 2) / (self.window ** 2)
                else:
                    alpha /= self.window  # sigma in terms of window

            if not adaptive and alpha.size == 1 or np.unique(alpha).size == 1:
                # Symmetrical across dimensions - use single value
                alpha = np.atleast_1d(np.float64(alpha[0]))

        else:
            alpha = np.asarray([0.0])

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
            raise ValueError(f"Unknown order algorithm: {order_algorithm}")
        order_algorithm_idx = order_func_lookup[order_algorithm]

        if order_symmetry:
            order_minimum_points = (order + 1) ** self.features
        else:
            order_minimum_points = np.prod(order + 1)

        if is_covar:
            mean_fit = True
        else:
            mean_fit = order_symmetry and order == 0

        if adaptive:
            region_coordinates = adaptive_region_coordinates
        else:
            region_coordinates = None

        settings['distance_weighting'] = distance_weighting
        settings['alpha'] = alpha
        settings['adaptive_threshold'] = adaptive_threshold
        settings['shaped'] = shaped
        settings['adaptive_alpha'] = np.empty((0, 0, 0, 0), dtype=np.float64)
        settings['order'] = np.atleast_1d(order)
        settings['order_varies'] = order_varies
        settings['order_algorithm'] = order_algorithm
        settings['order_algorithm_idx'] = order_algorithm_idx
        settings['order_symmetry'] = order_symmetry
        settings['order_minimum_points'] = order_minimum_points
        settings['is_covar'] = is_covar
        settings['mean_fit'] = mean_fit
        settings['relative_smooth'] = relative_smooth
        settings['adaptive_region_coordinates'] = region_coordinates
        self._fit_settings = settings
        return settings

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
            :func:`ResamplePolynomial.reduction_settings`.

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

    def pre_fit(self, settings, *args, adaptive_region_coordinates=None):
        """
        Perform pre-fitting steps and build the fitting tree.

        Parameters
        ----------
        settings : dict
            Settings calculated via `reduction_settings` to be applied
            if necessary.
        args : n-tuple
            The call input arguments.
        adaptive_region_coordinates : numpy.ndarray (float), optional
            The coordinates determined from a previous adaptive smoothing
            algorithm.


        Returns
        -------
        None
        """
        settings['adaptive_region_coordinates'] = args
        self.calculate_adaptive_smoothing(settings)
        super().pre_fit(settings, *args)

        if adaptive_region_coordinates is not None:
            region_grid = self.grid_class(
                *adaptive_region_coordinates,
                tree_shape=self.sample_tree.shape,
                build_tree=True, scale_factor=self._radius,
                scale_offset=self._scale_offsets, dtype=np.float64)
            skip_blocks = region_grid.tree.hood_population == 0
            self.fit_tree.block_population[skip_blocks] = 0

        if settings['order_symmetry']:
            o = settings['order'][0]
        else:
            o = settings['order']

        self.fit_tree.set_order(o, fix_order=not settings['order_varies'])
        self.fit_tree.precalculate_phi_terms()

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
        filename, iteration = args

        # Loading cannot be covered in tests as it occurs on other CPUs.
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

        if load_args:  # pragma: no cover
            _global_resampling_values['args'], _ = unpickle_file(filename)
            _global_resampling_values['iteration'] = iteration
            _global_resampling_values['filename'] = filename

        (sample_values, sample_error, sample_mask, fit_tree, sample_tree,
         get_error, get_counts, get_weights,
         get_distance_weights, get_rchi2, get_cross_derivatives,
         get_offset_variance, settings) = _global_resampling_values['args']

        fit_indices, fit_coordinates, fit_phi_terms = \
            fit_tree.block_members(block, get_locations=True, get_terms=True)

        sample_indices = nb.typed.List(sample_tree.query_radius(
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

    def __call__(self, *args, smoothing=0.0, relative_smooth=False,
                 adaptive_threshold=None, adaptive_algorithm='scaled',
                 fit_threshold=0.0, cval=np.nan, edge_threshold=0.0,
                 edge_algorithm='distribution', order_algorithm='bounded',
                 error_weighting=True, estimate_covariance=False,
                 is_covar=False, jobs=None, use_threading=None,
                 use_processes=None, adaptive_region_coordinates=None,
                 get_error=False, get_counts=False, get_weights=False,
                 get_distance_weights=False, get_rchi2=False,
                 get_cross_derivatives=False, get_offset_variance=False,
                 **kwargs):
        """
        Resample data defined during initialization to new coordinates.

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
        fit_data, [fit_error], [fit_counts]
            The data fit at `args` and optionally, the error associated with
            the fit, and the number of samples used to generate each point.
            Will be of a shape determined by `args`.  `fit_data` and
            `fit_error` will be of type np.float64, and `fit_counts` will
            be of type np.int64.
        """
        return super().__call__(
            *args,
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
            adaptive_region_coordinates=adaptive_region_coordinates,
            use_threading=use_threading,
            use_processes=use_processes,
            get_error=get_error,
            get_counts=get_counts,
            get_weights=get_weights,
            get_distance_weights=get_distance_weights,
            get_rchi2=get_rchi2,
            get_cross_derivatives=get_cross_derivatives,
            get_offset_variance=get_offset_variance,
            **kwargs)


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
    ResamplePolynomial data using local polynomial fitting.

    Initializes and then calls the :class:`ResamplePolynomial` class.  For
    further details on all available parameters, please see
    :func:`ResamplePolynomial.__init__` and
    :func:`ResamplePolynomial.__call__`.

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

    resampler = ResamplePolynomial(
        coordinates, data,
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
