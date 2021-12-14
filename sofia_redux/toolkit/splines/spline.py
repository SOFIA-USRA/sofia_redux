import itertools
import numpy as np

from sofia_redux.toolkit.splines.spline_utils import (
    flat_index_mapping, build_observation,
    find_knots, create_ordering, solve_observation,
    add_knot, knot_fit, calculate_minimum_bandwidth, check_input_arrays,
    determine_smoothing_spline, perform_fit)

__all__ = ['Spline']


class Spline(object):

    def __init__(self, *args, weights=None, limits=None, degrees=3,
                 smoothing=None, knots=None, knot_estimate=None, eps=1e-8,
                 fix_knots=None, tolerance=1e-3, max_iteration=20, exact=False,
                 reduce_degrees=False, solve=True):
        """
        Initialize a Spline object.

        This is a Python implementation of the Fortran fitpack spline based
        routines (Diercchx 1993) but is not limited to a maximum of
        2-dimensions.  Fast numerical processing is achieved using the `numba`
        Python package (Lam et. al., 2015).

        The actual spline fitting (representation) is performed during
        initialization from user supplied data values and coordinates and other
        parameters (see below).  Spline evaluations at other coordinates may
        then be retrieved using the __call__() method.

        Spline coefficients and knots are derived iteratively, and will be
        deemed acceptable once:

            abs(sum(residuals^2) - smoothing) <= tolerance * smoothing

        However, iterations may cease in a variety of scenarios.  Exit
        conditions should be examined prior to evaluating the spline and can
        be retrieved from the `exit_code` attribute or `exit_message` property.

        References
        ----------
        Dierckx, P. Curve and Surface Fitting with Splines (Oxford Univ. Press,
            1993).
        Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A llvm-based
            python jit compiler. In Proceedings of the Second Workshop on
            the LLVM Compiler Infrastructure in HPC (pp. 1–6).

        Parameters
        ----------
        args : n-tuple (numpy.ndarray) or numpy.ndarray
            The input arguments of the form (c1, ..., cn, d) or d where c
            signifies data coordinates and d are the data values.  If a single
            data array is passed in, the coordinates are derived from the data
            dimensions.  For example if d is an array of shape (a, b, c), c1
            will range from 0 -> a - 1 etc.  If coordinates are specified, the
            coordinates for each dimension and the data array should be
            one-dimensional.
        weights : numpy.ndarray, optional
            Optional weights to supply to the spline fit for each data point.
            Should be the same shape as the supplied data values.
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
            element i is an array of shape (n_knots[i]) for dimension i.
            If an array is supplied, it should be of shape
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
        solve : bool, optional
            If `True`, solve for the knots and spline coefficients.  Otherwise,
            leave for later processing.
        """
        self.coordinates = None  # float: (n_dimensions, n_data)
        self.values = None  # float: (n_data,)
        self.weights = None  # float: (n_data,)
        self.limits = None  # float: (n_dimensions, 2)
        self.knots = None  # list (n_dimensions) of float: (knot_size,)
        self.degrees = None  # int (n_dimensions,)
        self.k1 = None  # int (n_dimensions,)
        self.knot_estimate = None  # int (n_dimensions,)
        self.knot_indices = None  # int : (n_dimensions, n_data)
        self.knot_coordinates = None  # float : (n_dimensions, knot_estimate)
        self.knot_weights = None  # float : (n_dimensions, knot_estimate)
        self.panel_mapping = None  # int : (n_dimensions, n_panels)
        self.panel_steps = None  # int : (n_dimensions,)
        self.knot_mapping = None  # int : (n_dimensions, nk1)
        self.knot_steps = None  # int : (n_dimensions,)
        self.panel_indices = None  # int : (n_data,)
        self.amat = None  # float : (max_possible_knots, n_coefficients)
        self.beta = None  # float : (max_possible_knots,)
        self.spline_steps = None  # int : (n_dimensions,)
        self.spline_mapping = None  # int : (n_dimensions, bandwidth)
        self.n_knots = None  # int : (n_dimensions,)
        self.start_indices = None  # int : (n_data,)
        self.next_indices = None  # int : (n_data,)
        self.splines = None  # float : (n_dimensions, n_data, max(k1))
        self.coefficients = None  # float : (n_coefficients,)
        self.n_dimensions = 0
        self.dimension_permutations = None
        self.dimension_order = None
        self.permutation = None
        self.change_order = False

        self.smoothing = 0.0
        self.accuracy = 0.0  # absolute tolerance
        self.fix_knots = False
        self.eps = 1e-8
        self.tolerance = 1e-3
        self.max_iteration = 20
        self.exit_code = 20  # Return code
        self.panel_shape = None  # panel dimensions (solution space) (nxx)
        self.nk1 = None  # Last valid knot (n_dimensions,)
        self.iteration = -1
        self.smoothing_difference = np.nan

        self.n_panels = 0  # nreg
        self.n_intervals = 0  # nrint
        self.sum_square_residual = 0.0
        self.initial_sum_square_residual = np.nan
        self.bandwidth = 0  # iband
        self.n_coefficients = 0
        self.rank = 0
        self.fitted_knots = None
        self.fit_coordinates = None
        self.fit = None
        self.grid_reduction = None
        self.exact = False

        self.parse_inputs(
            *args, weights=weights, limits=limits, degrees=degrees,
            smoothing=smoothing, knots=knots, knot_estimate=knot_estimate,
            eps=eps, tolerance=tolerance, max_iteration=max_iteration,
            fix_knots=fix_knots, exact=exact, reduce_degrees=reduce_degrees)

        if solve:
            self.iterate()
            self.final_reorder()
            if self.fitted_knots is None:
                self.fit_knots()

    @property
    def exit_message(self):
        """
        Returns an exit message for the spline fit.  Error codes in the range
        -2 -> 0 generally indicate a successful fit.

        Returns
        -------
        message : str
        """
        if self.exit_code == 0:
            msg = (f"The spline has a residual sum of squares fp such that "
                   f"abs(fp-s)/s <= {self.tolerance}")
        elif self.exit_code == -1:
            msg = "The spline is an interpolating spline (fp=0)"
        elif self.exit_code == -2:
            msg = (f"The spline is a weighted least-squares polynomial of "
                   f"degree {self.degrees}. fp gives the upper bound for fp0 "
                   f"for the smoothing factor s = {self.smoothing}.")
        elif self.exit_code == -3:
            msg = ("Warning.  The coefficients of the spline have been "
                   "computed as the minimal norm least-squares solution of "
                   "a rank deficient system.")
        elif self.exit_code < 0:
            if self.rank >= self.n_coefficients:
                msg = f"Rank={self.rank} (full rank)"
            else:
                msg = (f"Rank={self.rank} (rank deficient "
                       f"{self.rank}/{self.n_coefficients})")
        elif self.exit_code == 1:
            msg = ("The required storage space exceeds the available storage "
                   "space. Probable causes: knot_estimate too small or s is "
                   "too small. (fp>s)")
        elif self.exit_code == 2:
            msg = (f"A theoretically impossible result when finding a "
                   f"smoothing spline with fp=s.  Probable causes: s too "
                   f"small or badly chosen eps."
                   f"(abs(fp-s)/s>{self.tolerance})")
        elif self.exit_code == 3:
            msg = (f"The maximal number of iterations ({self.max_iteration}) "
                   f"allowed for finding smoothing spline with fp=s has been "
                   f"reached.  Probable cause: s too small."
                   f"(abs(fp-s)/s>{self.tolerance})")
        elif self.exit_code == 4:
            msg = ("No more knots can be added because the number of B-spline"
                   "coefficients already exceeds the number of data points m."
                   "Probable causes: either s or m too small. (fp>s)")
        elif self.exit_code == 5:
            msg = ("No more knots can be added because the additional knot "
                   "would coincide with an old one.  Probable cause: s too "
                   "small or too large a weight to an inaccurate data point. "
                   "(fp>s)")
        elif self.exit_code == 20:
            msg = "Knots are not initialized."
        else:
            msg = "An unknown error occurred."

        return msg

    @property
    def size(self):
        """
        Return the number of values used for the spline fit.

        Returns
        -------
        n : int
        """
        return self.values.size

    @property
    def knot_size(self):
        """
        Return the number of knots in each dimension.

        Returns
        -------
        n_knots : numpy.ndarray (int)
            An array of shape (n_dimensions,).
        """
        n_knots = np.zeros(self.n_dimensions, dtype=int)  # nx, ny
        for dimension in range(self.n_dimensions):
            knot = self.knots[dimension]
            for i in range(knot.size):
                if np.isnan(knot[i]):
                    break
                n_knots[dimension] += 1
        return n_knots

    def parse_inputs(self, *args, weights=None, limits=None, degrees=3,
                     smoothing=None, knots=None, fix_knots=None,
                     knot_estimate=None, exact=False, reduce_degrees=False,
                     eps=1e-8, tolerance=1e-3, max_iteration=20):
        """
        Parse and apply user inputs to the spline fit.

        Parameters
        ----------
        args : n-tuple (numpy.ndarray) or numpy.ndarray
            The input arguments of the form (c1, ..., cn, d) or d where c
            signifies data coordinates and d are the data values.  If a single
            data array is passed in, the coordinates are derived from the data
            dimensions.  For example if d is an array of shape (a, b, c), c1
            will range from 0 -> a - 1 etc.  If coordinates are specified, the
            coordinates for each dimension and the data array should be
            one-dimensional.
        weights : numpy.ndarray, optional
            Optional weights to supply to the spline fit for each data point.
            Should be the same shape as the supplied data values.
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
            must be greater than zero.  See __init__() for further details.
            Note that if smoothing is zero, and the degrees are not equal
            over each dimension, smoothing will be set to `eps` due to
            numerical instabilities.
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
        fix_knots : bool, optional
            If `True`, do not attempt to modify or add knots to the spline fit.
            Only the initial supplied user knots will be used.
        knot_estimate : numpy.ndarray (int), optional
            The maximum number of knots required for the spline fit in each
            dimension and of shape (n_dimensions,).  If not supplied, the knot
            estimate will be set to
            int((n / n_dimensions) ** n_dimensions^(-1)) or n_knots
            if knots were supplied and fixed.
        exact : bool, optional
            If `True`, the initial knots used will coincide with the actual
            input coordinates and smoothing will be set to zero.  No knots
            should be supplied by the user in this instance.
        reduce_degrees : bool, optional
            Only relevant if `exact` is `True`.  If set to `True`, the maximum
            allowable degree in each dimension will be limited to
            (len(unique(x)) // 2) - 1 where x are the coordinate values in any
            dimension.
        eps : float, optional
            A value where 0 < eps < 1.  This defines the magnitude used to
            identify singular values in the spline observation matrix (A).  If
            any row of A[:, 0] < (eps * max(A[:,0])) it will be considered
            singular.
        tolerance : float, optional
            A value in the range 0 < tolerance < 1 used to determine the exit
            criteria for the spline fit.  See __init__() further details.
        max_iteration : int, optional
            The maximum number of iterations to perform when solving for the
            spline fit.

        Returns
        -------
        None
        """
        if len(args) == 1:
            # Assume a regularly spaced grid of data values
            indices = np.indices(np.asarray(args[0]).shape)[::-1]
            self.coordinates = np.stack(
                [np.asarray(x, dtype=float).ravel() for x in indices])
        else:
            self.coordinates = np.stack(
                [np.asarray(x, dtype=float).ravel() for x in args[:-1]])

        self.n_dimensions = self.coordinates.shape[0]

        self.values = np.asarray(args[-1], dtype=float).ravel()
        if weights is None:
            self.weights = np.ones(self.size, dtype=float)
        else:
            self.weights = np.asarray(weights, dtype=float).ravel()

        self.degrees = np.atleast_1d(np.asarray(degrees, dtype=int))
        if self.degrees.size != self.n_dimensions:
            self.degrees = np.full(self.n_dimensions, self.degrees[0])
        self.k1 = self.degrees + 1

        if exact:
            if knots is not None:
                raise ValueError(
                    "Cannot use the 'exact' option if knots are supplied")
            knots = []
            for dimension in range(self.n_dimensions):
                knots.append(np.unique(self.coordinates[dimension]))
            if smoothing is None:
                smoothing = 0.0
            self.exact = True
            if reduce_degrees:
                n_knots = np.asarray([knot_line.size for knot_line in knots])
                max_degrees = (n_knots // 2) - 1
                self.degrees = np.clip(self.degrees, None, max_degrees)
                self.k1 = self.degrees + 1
        else:
            self.exact = False

        self.dimension_permutations = np.asarray(
            list(itertools.permutations(np.arange(self.n_dimensions))))
        self.check_array_inputs()

        if not (0 < eps < 1):
            raise ValueError(f"eps not in range (0 < eps < 1): {eps}")
        self.eps = float(eps)

        if not (0 < tolerance < 1):
            raise ValueError(
                f"tolerance not in range (0 < tolerance < 1): {tolerance}")
        self.tolerance = float(tolerance)

        if self.size < np.prod(self.k1):
            raise ValueError("Data size >= product(degrees + 1) "
                             "not satisfied.")

        if smoothing is None:
            self.smoothing = self.size - np.sqrt(2 * self.size)
        else:
            if smoothing < 0:
                raise ValueError(f"smoothing must be >= 0: {smoothing}")
            self.smoothing = float(smoothing)

        if fix_knots is None:
            fix_knots = knots is not None
        self.fix_knots = bool(fix_knots)
        self.knots = []

        if knots is None and self.fix_knots:
            raise ValueError('Knots must be supplied if fixed.')

        if limits is None:
            if not self.fix_knots:
                self.limits = np.stack(
                    [np.array([x.min(), x.max()], dtype=float)
                     for x in self.coordinates])
            else:
                self.limits = np.stack(
                    [np.array([min(k), max(k)], dtype=float)
                     for k in knots])
        else:
            self.limits = np.atleast_2d(limits).astype(float)
            if self.limits.shape != (self.n_dimensions, 2):
                raise ValueError(
                    f"limits must be of shape ({self.n_dimensions}, 2).")

        if knots is None:
            for dimension in range(self.n_dimensions):
                self.knots.append(np.pad(self.limits[dimension],
                                         self.degrees[dimension],
                                         mode='edge'))
        else:
            for dimension in range(self.n_dimensions):
                knot_line = np.unique(np.asarray(knots[dimension],
                                                 dtype=float))
                self.knots.append(knot_line)

        self.n_knots = self.knot_size
        if self.fix_knots:
            for dimension in range(self.n_dimensions):
                if self.n_knots[dimension] < (2 * self.k1[dimension]):
                    raise ValueError(
                        f"There must be at least 2 * (degree + 1) knots in "
                        f"dimension {dimension} for fixed knots. "
                        f"knots={self.n_knots[dimension]}, "
                        f"degree={self.degrees[dimension]}.")
            self.knot_estimate = self.n_knots.copy()

        elif knot_estimate is not None:
            self.knot_estimate = np.asarray(
                np.atleast_1d(knot_estimate), dtype=int)
            if self.knot_estimate.size == 1 < self.n_dimensions:
                self.knot_estimate = np.full(
                    self.n_dimensions, self.knot_estimate[0])
            elif self.knot_estimate.size != self.n_dimensions:
                raise ValueError(f"Knot estimate must be a scalar or have "
                                 f"size {self.n_dimensions}")

        elif self.smoothing == 0:
            add = (3 * self.size) ** (1 / self.n_dimensions)
            self.knot_estimate = (self.degrees + add).astype(int)

        else:
            add = (self.size / self.n_dimensions) ** (1 / self.n_dimensions)
            self.knot_estimate = (self.degrees + add).astype(int)

        self.knot_estimate = np.clip(self.knot_estimate,
                                     (2 * self.degrees) + 3, None)

        # Expand knot arrays if necessary
        max_k1 = np.max(self.k1)
        max_estimate = np.max(self.knot_estimate)
        max_knot = np.max(self.knot_estimate + self.k1)
        new_knots = np.full((self.n_dimensions, max_estimate), np.nan)

        for dimension in range(self.n_dimensions):
            knot_line = self.knots[dimension]
            new_knots[dimension, :knot_line.size] = knot_line.copy()

        self.knots = new_knots
        self.sum_square_residual = 0.0
        self.max_iteration = int(max_iteration)
        self.exit_code = -2

        # Initialize some work arrays

        self.knot_coordinates = np.zeros((2, max_knot))
        self.knot_weights = np.zeros((2, max_knot))
        self.splines = np.zeros((self.n_dimensions, self.size, max_k1),
                                dtype=float)
        self.accuracy = self.tolerance * self.smoothing
        self.dimension_order = np.arange(self.n_dimensions)

    def check_array_inputs(self):
        """
        Remove zero weights and invalid data points.

        Invalid data points are those that contain NaN values, weights, or
        coordinates, or zero weights.

        Returns
        -------
        None
        """
        valid = check_input_arrays(self.values, self.coordinates, self.weights)
        if valid.all():
            return
        self.values = self.values[valid]
        self.weights = self.weights[valid]
        self.coordinates = self.coordinates[:, valid]

    def initialize_iteration(self):
        """
        Initialize the iteration for the number of current panels.

        Creates array maps that represents N-dimensional data flattened to
        a single dimension for fast access and the ability to pass these
        structures to numba JIT compiled functions.

        Returns
        -------
        None
        """
        # The number of panels in which the approximation domain is divided.
        self.panel_shape = self.n_knots - (2 * self.k1) + 1
        self.n_panels = int(np.prod(self.panel_shape))
        self.n_intervals = int(np.sum(self.panel_shape))
        self.nk1 = self.n_knots - self.k1  # last valid knot index

        self.n_coefficients = int(np.prod(self.nk1))

        # Find the bandwidth of the observation matrix (amat)
        # # Never change
        # self.dimension_permutations = self.dimension_permutations[0][None]

        self.bandwidth, self.permutation, self.change_order = (
            calculate_minimum_bandwidth(
                self.degrees, self.n_knots, self.dimension_permutations))
        if self.change_order:
            # Reordering dimensions creates the mapping indices.
            self.reorder_dimensions(self.permutation)
        else:
            self.create_mapping_indices()

    def create_mapping_indices(self):
        """
        Mapping indices allow 1-D representation of N-D data.

        Returns
        -------
        None
        """
        # index mapping
        self.panel_mapping, _, self.panel_steps = flat_index_mapping(
            self.panel_shape)
        self.knot_mapping, _, self.knot_steps = flat_index_mapping(self.nk1)

        self.spline_mapping, _, self.spline_steps = flat_index_mapping(self.k1)

        # find the panel and knot indices for the data values.
        self.order_points()

    def reorder_dimensions(self, order):
        """
        Re-order the dimensions in various attributes.

        Occasionally it is beneficial to re-order the dimensions of the
        data structures such that a minimal bandwidth for the observation
        matrix is achievable.  This reduces the amount of processing time
        required to reach a solution.

        Parameters
        ----------
        order : numpy.ndarray (int)
            An array of shape (n_dimensions,) indicating the new order of the
            dimensions.  E.g., to re-order dimensions [1, 2, 3] to [3, 1, 2],
            order should be [1, 2, 0].

        Returns
        -------
        None
        """
        if self.n_dimensions < 2:
            return

        if np.allclose(order, np.arange(self.n_dimensions)):
            return

        old_mapping, _, old_steps = flat_index_mapping(self.nk1)
        new_mapping, _, new_steps = flat_index_mapping(self.nk1[order])
        reverse_order = np.argsort(order)
        c_order = np.sum(new_mapping[reverse_order] * old_steps[:, None],
                         axis=0)

        # The easy to reorder attributes...
        for attribute in ['coordinates', 'limits', 'degrees', 'k1', 'nk1',
                          'knots', 'n_knots', 'knot_estimate',
                          'knot_coordinates', 'knot_weights', 'panel_shape',
                          'splines']:
            value = getattr(self, attribute)
            if not isinstance(value, np.ndarray):
                continue
            if value.shape[0] != self.n_dimensions:
                continue
            setattr(self, attribute, value[order])

        self.n_knots = self.knot_size

        # The attributes that are dependent on a specific coefficient order...
        for attribute in ['coefficients', 'amat', 'beta']:
            value = getattr(self, attribute)
            if not isinstance(value, np.ndarray):
                continue
            diff = self.n_coefficients - value.shape[0]
            if diff > 0:
                padding = [(0, 0)] * value.ndim
                padding[0] = (0, diff)
                value = np.pad(value, padding, mode='constant')

            setattr(self, attribute, value[c_order])

        self.dimension_order = self.dimension_order[order].copy()
        self.create_mapping_indices()

    def final_reorder(self):
        """
        Re-order the dimensions of various arrays to match those of the inputs.

        The dimensions of the various data structures may have changed during
        the course of processing to reduce the bandwidth of the observation
        matrix.  This step correctly reorders all dimensions such that they
        match those initially provided by the user.

        Returns
        -------
        None
        """
        self.reorder_dimensions(np.argsort(self.dimension_order))

    def order_points(self):
        """
        Sort the data points according to which panel they belong to.

        This is based on the fporde Fortran fitpack routine.

        Returns
        -------
        None
        """
        self.knot_indices = find_knots(
            coordinates=self.coordinates,
            knots=self.knots,
            valid_knot_start=self.degrees,
            valid_knot_end=self.nk1)

        self.panel_indices = self.knot_indices_to_panel_indices(
            self.knot_indices)

        start_indices, next_indices = create_ordering(
            self.panel_indices, self.size)
        self.n_panels = np.nonzero(start_indices != -1)[0][-1] + 1

        self.start_indices = start_indices
        self.next_indices = next_indices

    def knot_indices_to_panel_indices(self, knot_indices):
        """
        Convert knot indices to flat panel indices.

        Parameters
        ----------
        knot_indices : numpy.ndarray (int)
            An array of shape (n_dimensions, n_knots).

        Returns
        -------
        panel_indices : numpy.ndarray (int)
            The flat 1-D panel indices for the knots.
        """
        panel_indices = knot_indices - self.degrees[:, None]
        panel_indices *= self.panel_steps[:, None]
        return np.sum(panel_indices, axis=0)

    def panel_indices_to_knot_indices(self, panel_indices):
        """
        Convert panel indices to dimensional knot indices.

        Parameters
        ----------
        panel_indices : numpy.ndarray (int)
            An array of shape (n_knots,).

        Returns
        -------
        panel_indices : numpy.ndarray (int)
            knot.
        """
        return self.panel_mapping[:, panel_indices] + self.degrees[:, None]

    def iterate(self):
        """
        Iteratively determine the spline fit.

        Calculates the splines and coefficients for the provided data.  If
        this cannot be accomplished before reaching the maximum number of
        iterations, a smoothing spline will be calculated instead.

        Returns
        -------
        None
        """
        self.iteration = -1

        for iteration in range(1, self.size + 1):
            self.iteration = iteration
            if not self.next_iteration():
                break
        else:  # pragma: no cover
            # This should never happen - but just in case...
            self.determine_smoothing_spline()
            return

        if self.fix_knots:
            return

        if abs(self.smoothing_difference) <= self.accuracy:
            return
        if self.smoothing_difference < 0:
            self.determine_smoothing_spline()

    def next_iteration(self):
        """
        Perform a single iteration of the spline fit.

        During each iteration, the observation matrix is built and solved.  An
        exit code will be generated in cases where no further modifications to
        the solution are appropriate.  Additional knots will also be added if
        required and possible.

        Returns
        -------
        continue_iterations : bool
            If `False` no further iterations should occur due to an acceptable
            solution being reached or due to a given limitation.  If `True`,
            subsequent iterations are deemed appropriate.
        """
        self.initialize_iteration()
        amat, beta, splines, ssr = build_observation(
            coordinates=self.coordinates,
            values=self.values,
            weights=self.weights,
            n_coefficients=self.n_coefficients,
            bandwidth=self.bandwidth,
            degrees=self.degrees,
            knots=self.knots,
            knot_steps=self.knot_steps,
            start_indices=self.start_indices,
            next_indices=self.next_indices,
            panel_mapping=self.panel_mapping,
            spline_mapping=self.spline_mapping)

        self.amat = amat
        self.beta = beta
        self.splines = splines
        self.sum_square_residual = ssr

        self.coefficients, self.rank, ssr_solve = solve_observation(
            amat=self.amat, beta=self.beta, n_coefficients=self.n_coefficients,
            bandwidth=self.bandwidth, eps=self.eps)

        self.sum_square_residual += ssr_solve
        if self.exit_code == -2:
            self.initial_sum_square_residual = self.sum_square_residual

        if self.fix_knots:  # do not find knots
            self.exit_code = -self.rank
            return False

        # Test whether the lsq spline is acceptable
        self.smoothing_difference = (
            self.sum_square_residual - self.smoothing)

        if abs(self.smoothing_difference) <= self.accuracy:

            if self.sum_square_residual <= 0:
                self.exit_code = -1
                self.sum_square_residual = 0.0

            if self.n_coefficients != self.rank:
                self.exit_code = -self.rank
            return False

        # Test whether we can accept the choice of knots
        if self.smoothing_difference < 0:
            return False  # Do smoothing

        if self.n_coefficients > self.size:
            self.exit_code = 4
            return False

        # Add a new knot
        self.exit_code = 0
        self.fit_knots()
        self.exit_code = add_knot(
            knot_weights=self.knot_weights,
            knot_coords=self.knot_coordinates,
            panel_shape=self.panel_shape,
            knots=self.knots,
            n_knots=self.n_knots,
            knot_estimate=self.knot_estimate,
            k1=self.k1)

        return self.exit_code == 0  # False if an error was encountered

    def fit_knots(self):
        """
        Derive fits at the current knot locations.

        In addition to finding the value of the function at each knot,
        the knot weights and weight normalized coordinates are also determined.
        These a subsequently used to decide where a new knot should be placed.

        Returns
        -------
        None
        """
        fitted_knots, knot_weights, knot_coordinates = knot_fit(
            splines=self.splines,
            coefficients=self.coefficients,
            start_indices=self.start_indices,
            next_indices=self.next_indices,
            panel_mapping=self.panel_mapping,
            spline_mapping=self.spline_mapping,
            knot_steps=self.knot_steps,
            panel_shape=self.panel_shape,
            k1=self.k1,
            weights=self.weights,
            values=self.values,
            coordinates=self.coordinates)
        self.fitted_knots = fitted_knots
        self.knot_weights = knot_weights
        self.knot_coordinates = knot_coordinates

    def determine_smoothing_spline(self, smoothing=None):
        """
        Smooth the interpolating spline to the required level.

        Parameters
        ----------
        smoothing : float, optional
            Used to specify the an alternate smoothing factor.  Note that this
            should be very close to the original smoothing factor in order to
            succeed.

        Returns
        -------
        None
        """
        if smoothing is not None and smoothing != self.smoothing:
            change_smoothing = True
        else:
            change_smoothing = False

        if self.exit_code == -2 and not change_smoothing:
            return

        if change_smoothing:
            if smoothing < 0:
                raise ValueError(f"smoothing must be >= 0: {smoothing}")

            # Calculate this before setting new smoothing...
            self.smoothing = float(smoothing)
            self.smoothing_difference = (
                self.sum_square_residual - self.smoothing)
            self.accuracy = self.tolerance * self.smoothing

        coefficients, sq, exit_code, fitted_knots = determine_smoothing_spline(
            knots=self.knots,
            n_knots=self.n_knots,
            knot_estimate=self.knot_estimate,
            degrees=self.degrees,
            initial_sum_square_residual=self.initial_sum_square_residual,
            smoothing=self.smoothing,
            smoothing_difference=self.smoothing_difference,
            n_coefficients=self.n_coefficients,
            bandwidth=self.bandwidth,
            amat=self.amat,
            beta=self.beta,
            max_iteration=self.max_iteration,
            knot_steps=self.knot_steps,
            knot_mapping=self.knot_mapping,
            eps=self.eps,
            splines=self.splines,
            start_indices=self.start_indices,
            next_indices=self.next_indices,
            panel_mapping=self.panel_mapping,
            spline_mapping=self.spline_mapping,
            coordinates=self.coordinates,
            values=self.values,
            weights=self.weights,
            panel_shape=self.panel_shape,
            accuracy=self.accuracy)
        self.coefficients = coefficients
        self.sum_square_residual = sq
        self.exit_code = exit_code
        self.fitted_knots = fitted_knots

    def __call__(self, *args):
        """
        Evaluate the spline at given coordinates.

        Parameters
        ----------
        args : tuple (numpy.ndarray) or numpy.ndarray (float)
            The coordinate arguments.  If supplied as a tuple, should be of
            length n_dimensions where each element of the tuple defines grid
            coordinates along the (x, y, z,...) dimensions.  Arbitrary
            coordinates may be supplied as an array of shape (n_dimensions, n)
            where n is the number of coordinates.  A singular coordinate may
            also be supplied as an array of shape (n_dimensions,).

        Returns
        -------
        fit : float or numpy.ndarray (float)
            An (x[n], x[n-1], ..., x[0]) shaped array of values if args was
            provided in tuple form, or an array of shape (n,) if a
            2-dimensional array of arbitrary coordinates were provided.
            If a single coordinate was provided, the resulting output will
            be a float.
        """
        if len(args) == 1:
            # In cases where an array of coordinates are supplied.
            fit_coordinates = np.asarray(np.atleast_2d(args[0]), dtype=float)
            if fit_coordinates.shape[0] != self.n_dimensions:

                if (fit_coordinates.shape[0] == 1
                        and fit_coordinates.shape[1] == self.n_dimensions):
                    singular = True
                    fit_coordinates = fit_coordinates.T
                else:
                    raise ValueError("Coordinate shape[0] does not match "
                                     "number of spline dimensions.")
            else:
                singular = False

            self.grid_reduction = False
            self.fit_coordinates = fit_coordinates
            out_shape = fit_coordinates.shape

        elif len(args) != self.n_dimensions:
            raise ValueError("Number of arguments does not match number of "
                             "spline dimensions.")

        else:
            # In cases where grid coordinates are provided
            self.grid_reduction = True
            self.fit_coordinates = np.vstack(
                [np.asarray(x, dtype=float).ravel() for x in
                 np.meshgrid(*args[::-1], indexing='ij')[::-1]])
            out_shape = tuple([len(x) if hasattr(x, '__len__') else 1
                               for x in args[::-1]])
            singular = np.allclose(out_shape, 1)

        self.fit = perform_fit(
            coordinates=self.fit_coordinates,
            knots=self.knots,
            coefficients=self.coefficients,
            degrees=self.degrees,
            panel_mapping=self.panel_mapping,
            panel_steps=self.panel_steps,
            knot_steps=self.knot_steps,
            nk1=self.nk1,
            spline_mapping=self.spline_mapping,
            n_knots=self.n_knots)  # This is ok

        if self.grid_reduction:
            self.fit = self.fit.reshape(out_shape)

        if singular:
            self.fit = self.fit.ravel()[0]

        return self.fit.copy()
