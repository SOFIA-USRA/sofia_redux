# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba
import numpy as np


from sofia_redux.toolkit.resampling.resample_utils import (
    scale_coordinates)
from sofia_redux.toolkit.resampling.resample_kernel_utils import (
    solve_kernel_fits)
from sofia_redux.toolkit.resampling.resample_base import (
    ResampleBase, _global_resampling_values)
from sofia_redux.toolkit.resampling.tree.kernel_tree import KernelTree
from sofia_redux.toolkit.utilities.multiprocessing import unpickle_file


__all__ = ['ResampleKernel']


class ResampleKernel(ResampleBase):

    def __init__(self, coordinates, data, kernel,
                 error=None, mask=None,
                 robust=None, negthresh=None,
                 kernel_spacing=None,
                 kernel_offsets=None,
                 kernel_weights=None,
                 limits=None,
                 degrees=3,
                 smoothing=0.0,
                 knots=None,
                 knot_estimate=None,
                 eps=1e-8,
                 fix_knots=None,
                 tolerance=1e-3,
                 max_iteration=20,
                 exact=False,
                 reduce_degrees=False,
                 imperfect=False,
                 leaf_size=40,
                 **distance_kwargs):
        """
        Class to resample data using kernel convolution.

        The kernel resampler may take regular or irregular spaced data, an
        irregular or regular kernel, and use kernel convolution to resample
        onto arbitrary coordinates where each output value is the convolution
        of the original sample data and the given kernel.  Here, irregular
        refers values that have specified coordinates.  i.e., they do not
        necessarily fall onto a regular grid.

        Generally, this class was designed to process irregular data when
        convolution is required.  If both the kernel and samples exist on
        regularly spaced grids, there are many alternatives that could be used
        to perform such a convolution much more quickly such as
        :func:`scipy.ndimage.convolve`.

        Convolution is achieved by creating a spline representation of the
        kernel which is then used to interpolate values of the kernel at the
        required locations.  Note that while the default settings assume that
        the kernel is perfect, noisy kernels may also be supplied and smoothed
        if required.  Please see the :class:`Spline` class for further details.

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
        kernel : array_like of float
            An n_features-dimensional array containing a regularly spaced
            resampling kernel or an irregular 1-D kernel array.  If the kernel
            is irregular, `kernel_offsets` must be provided.
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
        robust : float, optional
            Specifies an outlier rejection threshold for `data`. A data point
            is identified as an outlier if abs(x_i - x_med)/MAD > robust, where
            x_med is the median, and MAD is the Median Absolute Deviation
            defined as 1.482 * median(abs(x_i - x_med)).
        negthresh : float, optional
            Specifies a negative value rejection threshold such that
            data < (-stddev(data) * negthresh) will be excluded from the fit.
        kernel_spacing : float or numpy.ndarray (float), optional
            The spacing between kernel elements for all or each feature.  If
            an array is supplied, should be of shape (n_features,).  This
            feature may only be used for regular grids with n_features
            dimensions.  Either `kernel_spacing` or `kernel_offsets` must be
            supplied.
        kernel_offsets : tuple or array_like, optional
            If the kernel is regular, should be an n-dimensional tuple
            containing the grid indices in each dimension.  Otherwise, should
            be an array of shape (n_dimensions, kernel.size).  Either
            `kernel_spacing` or `kernel_offsets` must be supplied.
        kernel_weights : numpy.ndarray, optional
            Optional weights to supply to the spline fit for each data point.
            Should be the same shape as the supplied kernel.
        limits : numpy.ndarray (float), optional
            An array of shape (n_dimensions, 2) that may be supplied to set the
            minimum and maximum coordinate values used during the spline fit.
            For example, limits[1, 0] sets the minimum knot value in the second
            dimensions and limits[1, 1] sets the maximum knot value in the
            second dimension.  By default this is set to the minimum and
            maximum values of the coordinates in each dimension.
        degrees : int or numpy.ndarray (int), optional
            The degree of spline to fit in each dimension.  Either a scalar can
            be supplied pertaining to all dimensions, or an array of shape
            (n_dimensions,) can be used.
        smoothing : float, optional
            Used to specify the smoothing factor.  If set to `None`, the
            smoothing will be determined based on user settings or input data.
            If `exact` is `True`, smoothing will be disabled (zero).  If
            `exact`is `False`, smoothing will be set to n - sqrt(2 * n) where
            n is the number of data values.  If supplied, smoothing must be
            greater than zero.  See above for further details.  Note that if
            smoothing is zero, and the degrees are not equal over each
            dimension, smoothing will be set to `eps` due to numerical
            instabilities.
        knots : list or tuple or numpy.ndarray, optional
            A set of starting knot coordinates for each dimension.  If a list
            or tuple is supplied it should be of length n_dimensions where
            element i is an array of shape (n_knots[i]) for dimension i.  If
            an array is supplied, it should be of shape
            (n_dimension, max(n_knots)). Note that there must be at least
            2 * (degree + 1) knots for each dimension.  Unused or invalid
            knots may be set to NaN, at the end of each array.  Knot
            coordinates must also be monotonically increasing in each
            dimension.
        knot_estimate : numpy.ndarray (int), optional
            The maximum number of knots required for the spline fit in each
            dimension and of shape (n_dimensions,).  If not supplied, the knot
            estimate will be set to
            int((n / n_dimensions) ** n_dimensions^(-1)) or n_knots if
            knots were supplied and fixed.
        eps : float, optional
            A value where 0 < eps < 1.  This defines the magnitude used to
            identify singular values in the spline observation matrix (A).  If
            any row of A[:, 0] < (eps * max(A[:,0])) it will be considered
            singular.
        fix_knots : bool, optional
            If `True`, do not attempt to modify or add knots to the spline fit.
            Only the initial supplied user knots will be used.
        tolerance : float, optional
            A value in the range 0 < tolerance < 1 used to determine the exit
            criteria for the spline fit.  See above for further details.
        max_iteration : int, optional
            The maximum number of iterations to perform when solving for the
            spline fit.
        exact : bool, optional
            If `True`, the initial knots used will coincide with the actual
            input coordinates and smoothing will be set to zero.  No knots
            should be supplied by the user in this instance.
        reduce_degrees : bool, optional
            Only relevant if `exact` is `True`.  If set to `True`, the maximum
            allowable degree in each dimension will be limited to
            (len(unique(x)) // 2) - 1 where x are the coordinate values in any
            dimension.
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
        self.kernel_args = {
            'kernel': kernel,
            'kernel_spacing': kernel_spacing,
            'kernel_offsets': kernel_offsets,
            'weights': kernel_weights,
            'limits': limits,
            'degrees': degrees,
            'smoothing': smoothing,
            'knots': knots,
            'knot_estimate': knot_estimate,
            'eps': eps,
            'fix_knots': fix_knots,
            'tolerance': tolerance,
            'max_iteration': max_iteration,
            'exact': exact,
            'reduce_degrees': reduce_degrees,
            'imperfect': imperfect,
        }
        self._distance_kwargs = distance_kwargs
        super().__init__(coordinates, data, error=error, mask=mask,
                         robust=robust, negthresh=negthresh,
                         leaf_size=leaf_size, **distance_kwargs)

    @property
    def kernel(self):
        """
        Return the resampling kernel.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self.sample_tree.kernel

    @property
    def kernel_spacing(self):
        """
        Return the spacing between kernel grid points.

        Returns
        -------
        numpy.ndarray (float)
        """
        spacing = self.sample_tree.kernel_spacing
        if spacing is None:
            return None
        return scale_coordinates(spacing, self.window, self.window * 0,
                                 reverse=True)

    @property
    def kernel_offsets(self):
        """
        Return the coordinate offsets for the kernel.

        Returns
        -------
        numpy.ndarray (float)
        """
        offsets = self.sample_tree.kernel_coordinates
        return scale_coordinates(offsets, self.window, self.window * 0,
                                 reverse=True)

    @property
    def degrees(self):
        """
        Return the degrees of the spline fit to the kernel.

        Returns
        -------
        numpy.ndarray (int)
        """
        return self.sample_tree.degrees

    @property
    def exit_code(self):
        """
        Return the exit code of the spline fit.

        Please see the :class:`Spline` class for further details on
        the meanings of each code.

        Returns
        -------
        int
        """
        return self.sample_tree.exit_code

    @property
    def exit_message(self):
        """
        Return the spline exit message.

        Returns
        -------
        str
        """
        return self.sample_tree.exit_message

    @property
    def spline(self):
        """
        Return the spline object for the kernel fit.

        Returns
        -------
        Spline
        """
        return self.sample_tree.spline

    def set_sample_tree(self, coordinates,
                        leaf_size=40,
                        **distance_kwargs):
        """
        Build the sample tree from input coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray (float)
            The input coordinates of shape (n_features, n_samples).
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
        super().set_sample_tree(
            coordinates,
            radius=self.estimate_feature_windows(),
            leaf_size=leaf_size, **self._distance_kwargs)
        self.set_kernel(**self.kernel_args)

    def estimate_feature_windows(self, *args, **kwargs):
        r"""
        Estimates the radius of the fitting window for each feature.

        The window for the resampling algorithm will be set to encompass the
        kernel extent over all dimensions.  Unlike the standard resampling
        algorithm, the coordinates of the data samples are irrelevant.

        The window radius is based on the kernel extent in this implementation.

        Since the resampling algorithm uses an ellipsoid window to determine
        possible candidates for fitting, the window is determined as an
        ellipsoid which circumscribes the cuboid kernel array.  Although there
        are an infinite number of possible ellipsoids, this particular method
        uses the constraints that principle axes of the ellipsoid and cuboid
        widths are at a constant ratio.  For example, in two dimensions where
        a and b are the principle axes of an ellipsoid, and w and h are the
        widths of the kernel:

        a/w = b/h; (w / 2a)^2 + (h / 2b)^2 = 1.

        This leads to us setting the principle axes as:

        a_i = sqrt(n / 4) * w_i + delta

        where w_i is the width of the kernel in dimension i in n-dimensions,
        and delta = spacing/1e6 so that any edge cases can be included safely.
        This is also valid in one dimension (a_i = w_i / 2).  Note that kernels
        are assumed to be exactly centered, so the width of a kernel in any
        dimension will be:

        w_i = spacing_i * (kernel.shape[i] - 1)

        Note that this implementation takes the width as 2 * the maximum
        absolute offset of the kernel for each dimension, so that irregular
        kernels may be safely used.

        Returns
        -------
        window : numpy.ndarray (n_features,)
            The principle axes of an ellipsoid used to create a fitting
            region around each resampling point.
        """
        temp_shape = tuple([1] * self.features)
        temp_tree = KernelTree(temp_shape)
        temp_tree.parse_kernel(
            self.kernel_args['kernel'],
            kernel_spacing=self.kernel_args['kernel_spacing'],
            kernel_offsets=self.kernel_args['kernel_offsets'])
        max_width = np.max(np.abs(temp_tree.extent), axis=1)

        typical_spacing = np.empty(self.features, dtype=float)
        for feature in range(self.features):
            typical_spacing[feature] = np.nanmedian(np.diff(np.unique(
                temp_tree.kernel_coordinates[feature])))

        window = max_width * np.sqrt(self.features)
        window += typical_spacing * 1e-6
        return window

    def set_kernel(self, kernel, kernel_spacing=None, kernel_offsets=None,
                   weights=None, limits=None, degrees=3, smoothing=0.0,
                   knots=None, knot_estimate=None, eps=1e-8, fix_knots=None,
                   tolerance=1e-3, max_iteration=20, exact=False,
                   reduce_degrees=False, imperfect=False):
        """
        Set the kernel for subsequent fitting.

        During this process, a spline will be fit to describe the kernel at
        all intermediate points.

        Parameters
        ----------
        kernel : numpy.ndarray (float)
            The kernel to set.  Must have n_features dimensions.
        kernel_spacing : float or numpy.ndarray (float), optional
            The spacing between each kernel element in units of the
            coordinates. Either supplied as a single value for all features,
            or as an array of shape (n_features,) giving the kernel spacing
            for each feature.
        kernel_offsets : tuple or array_like, optional
            If the kernel is regular, should be an n-dimensional tuple
            containing the grid indices in each dimension.  Otherwise, should
            be an array of shape (n_dimensions, kernel.size).
        weights : numpy.ndarray, optional
            Optional weights to supply to the spline fit for each data point.
            Should be the same shape as the supplied kernel.
        limits : numpy.ndarray (float), optional
            An array of shape (n_dimensions, 2) that may be supplied to set the
            minimum and maximum coordinate values used during the spline fit.
            For example, limits[1, 0] sets the minimum knot value in the second
            dimensions and limits[1, 1] sets the maximum knot value in the
            second dimension.  By default this is set to the minimum and
            maximum values of the coordinates in each dimension.
        degrees : int or numpy.ndarray (int), optional
            The degree of spline to fit in each dimension.  Either a scalar can
            be supplied pertaining to all dimensions, or an array of shape
            (n_dimensions,) can be used.
        smoothing : float, optional
            Used to specify the smoothing factor.  If set to `None`, the
            smoothing will be determined based on user settings or input data.
            If `exact` is `True`, smoothing will be disabled (zero).  If
            `exact` is `False`, smoothing will be set to n - sqrt(2 * n)
            where n is the number of data values.  If supplied, smoothing
            must be greater than zero.  See above for further details.  Note
            that if smoothing is zero, and the degrees are not equal over
            each dimension, smoothing will be set to `eps` due to numerical
            instabilities.
        knots : list or tuple or numpy.ndarray, optional
            A set of starting knot coordinates for each dimension.  If a list
            or tuple is supplied it should be of length n_dimensions where
            element i is an array of shape (n_knots[i]) for dimension i.  If
            an array is supplied, it should be of shape
            (n_dimension, max(n_knots)). Note that there must be at least
            2 * (degree + 1) knots for each dimension.  Unused or invalid
            knots may be set to NaN, at the end of each array.  Knot
            coordinates must also be monotonically increasing in each
            dimension.
        knot_estimate : numpy.ndarray (int), optional
            The maximum number of knots required for the spline fit in each
            dimension and of shape (n_dimensions,).  If not supplied, the knot
            estimate will be set to
            int((n / n_dimensions) ** n_dimensions^(-1)) or n_knots if
            knots were supplied and fixed.
        eps : float, optional
            A value where 0 < eps < 1.  This defines the magnitude used to
            identify singular values in the spline observation matrix (A).  If
            any row of A[:, 0] < (eps * max(A[:,0])) it will be considered
            singular.
        fix_knots : bool, optional
            If `True`, do not attempt to modify or add knots to the spline fit.
            Only the initial supplied user knots will be used.
        tolerance : float, optional
            A value in the range 0 < tolerance < 1 used to determine the exit
            criteria for the spline fit.  See above for further details.
        max_iteration : int, optional
            The maximum number of iterations to perform when solving for the
            spline fit.
        exact : bool, optional
            If `True`, the initial knots used will coincide with the actual
            input coordinates and smoothing will be set to zero.  No knots
            should be supplied by the user in this instance.
        reduce_degrees : bool, optional
            Only relevant if `exact` is `True`.  If set to `True`, the maximum
            allowable degree in each dimension will be limited to
            (len(unique(x)) // 2) - 1 where x are the coordinate values in any
            dimension.
        imperfect : bool, optional
            If a spline fit to the kernel is allowed to be imperfect (`True`),
            will only raise an error on spline fitting if a major error was
            encountered.  Otherwise, fits will be permitted so long as a
            solution was reached, even if that solution did not meet
            expectations.

        Returns
        -------
        None
        """
        if kernel_spacing is None and kernel_offsets is None:
            raise ValueError("Kernel spacing or offsets must be supplied.")

        # Scale offsets or spacings
        # Kernel coordinates are always relative to the center.

        # offsets supersede spacings
        if kernel_offsets is not None:
            scaled_offsets = []
            kernel_spacing = None
            for feature in range(self.features):
                scaled_offsets.append(np.asarray(kernel_offsets[feature])
                                      / self.window[feature])
            kernel_offsets = scaled_offsets
        else:
            kernel_offsets = None
            kernel_spacing = np.atleast_1d(kernel_spacing).astype(float)
            if kernel_spacing.size == 1 and self.features > 1:
                kernel_spacing = np.full(self.features, kernel_spacing[0])
            kernel_spacing /= self.window

        self.sample_tree.set_kernel(kernel,
                                    kernel_spacing=kernel_spacing,
                                    kernel_offsets=kernel_offsets,
                                    weights=weights,
                                    limits=limits,
                                    degrees=degrees,
                                    smoothing=smoothing,
                                    knots=knots,
                                    knot_estimate=knot_estimate,
                                    eps=eps,
                                    fix_knots=fix_knots,
                                    tolerance=tolerance,
                                    max_iteration=max_iteration,
                                    exact=exact,
                                    reduce_degrees=reduce_degrees,
                                    imperfect=imperfect)

    def reduction_settings(self, error_weighting=True,
                           absolute_weight=None,
                           fit_threshold=0.0, cval=np.nan,
                           edge_threshold=0.0, edge_algorithm='distribution',
                           is_covar=False, jobs=None, use_threading=None,
                           use_processes=None, **kwargs):
        r"""
        Define a set of reduction instructions based on user input.

        This method is responsible for determining, formatting, and checking
        a number variables required for the resampling algorithm based on
        user input.  For detailed descriptions of user options, please see
        :func:`ResamplePolynomial.__call__`.

        Parameters
        ----------
        error_weighting : bool, optional
            If `True` (default), weight polynomial fitting by the `error`
            values of each sample.
        absolute_weight : bool, optional
            If the kernel weights are negative, can lead to almost zero-like
            divisions in many of the algorithms.  If set to `True`, the sum of
            the absolute weights are used for normalization.
        fit_threshold : float, optional
            Not implemented for kernel fitting.
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
        kwargs : dict
            Optional keyword arguments to the reduction settings.

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
        settings['is_covar'] = is_covar

        if absolute_weight is None:
            settings['absolute_weight'] = np.any(self.spline.values < 0)
        else:
            settings['absolute_weight'] = absolute_weight

        self._fit_settings = settings
        return settings

    def __call__(self, *args,
                 cval=np.nan, edge_threshold=0.0,
                 edge_algorithm='distribution',
                 error_weighting=True, absolute_weight=None, normalize=True,
                 is_covar=False, jobs=None, use_threading=None,
                 use_processes=None,
                 get_error=False, get_counts=False, get_weights=False,
                 get_distance_weights=False, get_rchi2=False,
                 get_offset_variance=False, **kwargs):
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
        absolute_weight : bool, optional
            If the kernel weights are negative, can lead to almost zero-like
            divisions in many of the algorithms.  If set to `True`, the sum of
            the absolute weights are used for normalization.
        normalize : bool, optional
            If `True`, the data will be convolved with a normalized kernel
            such that sum(kernel) = 1.  Note that this may be sum(abs(kernel))
            if `absolute_weight` is `True`.
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
            absolute_weight=absolute_weight,
            cval=cval,
            edge_threshold=edge_threshold,
            edge_algorithm=edge_algorithm,
            is_covar=is_covar,
            jobs=jobs,
            use_threading=use_threading,
            use_processes=use_processes,
            **kwargs)

        self.pre_fit(settings, *args)

        if not normalize:
            return_distance_weights = get_distance_weights
            get_distance_weights = True
        else:
            return_distance_weights = get_distance_weights

        (fit, error, counts, weights, distance_weights, sigma,
         distribution_offset_variance) = self.block_loop(
            self.data, self.error, self.mask, self.fit_tree, self.sample_tree,
            settings, self.iteration, get_error=get_error,
            get_counts=get_counts, get_weights=get_weights,
            get_distance_weights=get_distance_weights, get_rchi2=get_rchi2,
            get_offset_variance=get_offset_variance, jobs=jobs)

        self.iteration += 1

        if not normalize:
            nzi = distance_weights != 0
            fit[nzi] *= distance_weights[nzi]
            fit[~nzi] = cval

        get_distance_weights = return_distance_weights

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
            if get_offset_variance:
                result += (self.fit_grid.reshape_data(
                    distribution_offset_variance),)

            return result
        else:
            return fit

    @classmethod
    def block_loop(cls, sample_values, error, mask, fit_tree, sample_tree,
                   settings, iteration, get_error=True, get_counts=True,
                   get_weights=True, get_distance_weights=True, get_rchi2=True,
                   get_offset_variance=True, jobs=None, **kwargs):
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
        fit_tree : resampling.tree.PolynomialTree object
        sample_tree : resampling.tree.PolynomialTree object
        settings : dict
        iteration : int
        get_error : bool, optional
        get_counts : bool, optional
        get_weights : bool, optional
        get_distance_weights : bool, optional
        get_rchi2 : bool, optional
        get_offset_variance : bool, optional
        jobs : int, optional
        kwargs : dict, optional
            For consistency with the resampler base only.

        Returns
        -------
        combined_results : 8-tuple of numpy.ndarray
            In order: fit, error, counts, weights, distance weights,
                reduced chi-squared, MSCP derivatives, distribution offset.
        """
        args = (sample_values, error, mask, fit_tree, sample_tree,
                get_error, get_counts, get_weights, get_distance_weights,
                get_rchi2, get_offset_variance, settings)
        kwargs = None

        blocks = cls.process_blocks(args, kwargs, settings, sample_tree,
                                    fit_tree, jobs, iteration)

        n_sets = sample_values.shape[0]
        n_fits = fit_tree.n_members
        n_features = sample_tree.features

        return cls.combine_blocks(
            blocks, n_sets, n_fits, n_features, settings['cval'],
            get_error=get_error,
            get_counts=get_counts,
            get_weights=get_weights,
            get_distance_weights=get_distance_weights,
            get_rchi2=get_rchi2,
            get_offset_variance=get_offset_variance)

    @staticmethod
    def combine_blocks(blocks, n_sets, n_fits, n_dimensions, cval,
                       get_error=True, get_counts=True, get_weights=True,
                       get_distance_weights=True, get_rchi2=True,
                       get_offset_variance=True, **kwargs):
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
        get_offset_variance : bool, optional
            If `True`, indicates that the offset variance of the fit from the
            sample distribution should be returned.
        kwargs : dict, optional
            For consistency with ResampleBase only (not used).

        Returns
        -------
        results : 7-tuple of numpy.ndarray
            results[0] = fitted values
            results[1] = error on the fit
            results[2] = counts
            results[3] = total weight sums
            results[4] = total distance weight sums
            results[5] = reduced chi-squared statistic
            results[6] = offset variance

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
            if get_offset_variance:
                offsets[:, fit_indices] = block[7]

        return fit, error, counts, weights, distance_weights, rchi2, offsets

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
         get_error, get_counts, get_weights, get_distance_weights, get_rchi2,
         get_offset_variance, settings) = _global_resampling_values['args']

        fit_indices, fit_coordinates = \
            fit_tree.block_members(block, get_locations=True)

        sample_indices = numba.typed.List((sample_tree.query_radius(
            fit_coordinates, 1.0, return_distance=False)))

        (knots, coefficients, degrees, panel_mapping, panel_steps, knot_steps,
         nk1, spline_mapping, n_knots) = sample_tree.resampling_arguments

        return (fit_indices,
                *solve_kernel_fits(
                    sample_indices, sample_tree.coordinates,
                    sample_values, sample_error, sample_mask,
                    fit_coordinates, knots, coefficients,
                    degrees, panel_mapping, panel_steps,
                    knot_steps, nk1, spline_mapping, n_knots,
                    is_covar=settings['is_covar'],
                    cval=settings['cval'],
                    error_weighting=settings['error_weighting'],
                    absolute_weight=settings['absolute_weight'],
                    edge_algorithm_idx=settings['edge_algorithm_idx'],
                    edge_threshold=settings['edge_threshold'],
                    get_error=get_error,
                    get_counts=get_counts,
                    get_weights=get_weights,
                    get_distance_weights=get_distance_weights,
                    get_rchi2=get_rchi2,
                    get_offset_variance=get_offset_variance)
                )
