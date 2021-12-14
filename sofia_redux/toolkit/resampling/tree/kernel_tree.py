# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
from sofia_redux.toolkit.splines.spline import Spline

__all__ = ['KernelTree']


class KernelTree(BaseTree):

    def __init__(self, argument, shape=None, build_type='all',
                 leaf_size=40, kernel=None, kernel_spacing=1.0,
                 kernel_offsets=None, smoothing=0.0,
                 imperfect=False, degrees=3, spline_kwargs=None,
                 **distance_kwargs):
        r"""
        Create a tree structure for use with the kernel resampling algorithm.

        The resampling tree is primarily responsible for deriving and
        storing all independent variables necessary for kernel fitting,
        as well as allowing fast access to those variables that belong to
        coordinates within a certain radius of a given point.

        TREE STRUCTURE AND ACCESS

        The tree itself is divided into N-dimensional blocks, each of which
        is allocated a set of coordinates.  The width of these blocks should
        correspond to the `window` (:math:`\Omega`) defined in the resampling
        algorithm, and coordinates should be scaled accordingly.  For example,
        if the window radius is set to :math:`\Omega=4` in (arbitrary) units
        for the purposes of resampling 1-dimensional data, and the independent
        values are:

            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        They should be supplied to the tree as :math:`x^\prime = x / \Omega`.

        .. math::

            x^\prime = \frac{x}{\Omega} =
                [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

        The tree defines blocks by grouping all coordinates with the same
        floored values into a single block.  Therefore, in this case the tree
        will contain 3 blocks.  The first contains [0.25, 0.5, 0.75], the
        second contains [1, 1.25, 1.5, 1.75], and the third contains
        [2, 2.25, 2.5].

        The reasoning behind the whole tree structure is to allow for easy
        extraction of all coordinates within range of a user supplied
        coordinate.  This is done in two stages:  The first is to find out
        which block the user supplied coordinate belongs to.  We can then
        quickly narrow down the search by recognizing that coordinates in the
        tree population inside the window region of the supplied coordinate
        must either belong to the same block, or to immediately neighboring
        blocks since each block of the tree is the same width as the window
        radius.

        Once all candidates have been identified, the next step is to keep
        only those that are within a radius :math:`\Omega` of the user supplied
        coordinate.  This can be accomplished quickly using the ball-tree
        algorithm (see :func:`sklearn.neighbors.BallTree`.

        In practice, the resampling algorithm loops through each block of
        the tree in parallel.  For each block, all user supplied coordinates
        (points at which a fit is required) within that block, and all tree
        members within the neighborhood (the block and all adjacent blocks
        including diagonals) are evaluated in one step by the ball-tree
        algorithm so that for each point, we quickly get all tree members
        within that point's window region.

        Parameters
        ----------
        argument : numpy.ndarray (n_features, n_samples) or n-tuple
            Either the independent coordinates of samples in n_features-space,
            or the shape defining the skeleton of the tree.
        shape : n-tuple, optional
            If coordinates were supplied with `argument`, the shape of the
            tree to build.  Otherwise, the shape will be determined from the
            coordinate values in each dimension as
            floor(max(coordinates[i])) + 1 for dimension i.
        build_type : str, optional
            Must be one of {'hood', 'balltree', 'all', None}.  Defines the
            type of tree structures to create.
        order : int or array_like (n_features,), optional
           The symmetrical or asymmetrical orders respectively.  Symmetrical
           orders are selected by supplying an integer to this parameter.
        fix_order : bool, optional
            If `order` is symmetrical, allow for a varying order in an attempt
            to pass the order validation algorithm (the order can only
            decrease).
        balltree_metric : str or sklearn.neighbors.DistanceMetric object
            The distance metric to use for the tree. Default=’minkowski’ with
            p=2 (that is, a euclidean metric). See the documentation of the
            :func:`sklearn.neighbors.DistanceMetric` class for a list of
            available metrics. ball_tree.valid_metrics gives a list of the
            metrics which are valid for BallTree.
        leaf_size : int, optional
            If `build_type` was set to 'all' or 'balltree', defines the leaf
            size of the BallTree.  Please see
            :func:`sklearn.neighbors.BallTree` for further details.
        kernel : numpy.ndarray (float)
            The kernel to apply as an array with n_features dimensions.
        kernel_spacing : float or numpy.ndarray (float), optional
            The spacing between kernel elements for all or each feature.  If
            an array is supplied, should be of shape (n_features,).
        kernel_offsets : tuple or array_like, optional
            If the kernel is regular, should be an n-dimensional tuple
            containing the grid indices in each dimension.  Otherwise, should
            be an array of shape (n_dimensions, kernel.size).
        smoothing : float, optional
            Used to specify the smoothing factor.  If set to `None`, the
            smoothing will be determined based on user settings or input data.
            If `exact` is `True`, smoothing will be disabled (zero).  If
            `exact` is `False`, smoothing will be set to n - sqrt(2 * n)
            where n is the number of data values.  For interpolation, smoothing
            should be set to zero.  Smoothing must be >= 0.
        imperfect : bool, optional
            If a spline fit to the kernel is allowed to be imperfect (`True`),
            will only raise an error on spline fitting if a major error was
            encountered.  Otherwise, fits will be permitted so long as a
            solution was reached, even if that solution did not meet
            expectations.
        degrees : int or numpy.ndarray (int), optional
            The degree of spline to fit in each dimension.  Either a scalar can
            be supplied pertaining to all dimensions, or an array of shape
            (n_dimensions,) can be used.
        spline_kwargs : dict, optional
            Optional keyword arguments for spline initialization.  Please see
            :class:`Spline` for further details.
        distance_kwargs : dict, optional
            Optional keyword arguments passed into
            :func:`sklearn.neighbors.DistanceMetric`.  The default is to use
            the "minkowski" definition with `p=2`, i.e., the Euclidean
            definition.
        """
        super().__init__(argument, shape=shape, build_type=build_type,
                         leaf_size=leaf_size, **distance_kwargs)
        self.kernel = None
        self.kernel_spacing = None
        self.kernel_coordinates = None
        self.spline = None
        self.imperfect = imperfect
        if kernel is not None:
            if spline_kwargs is None:
                spline_kwargs = {}
            self.set_kernel(kernel, kernel_spacing=kernel_spacing,
                            kernel_offsets=kernel_offsets,
                            smoothing=smoothing, degrees=degrees,
                            **spline_kwargs)

    @property
    def degrees(self):
        """
        Return the degree of the spline fit.

        Returns
        -------
        numpy.ndarray (int)
            The spline degrees of shape (n_dimensions,) for each dimension
            in (x, y, z, ...) order.
        """
        if self.spline is None:
            return None
        return self.spline.degrees

    @property
    def smoothing(self):
        """
        Return the spline smoothing factor.

        Returns
        -------
        float
        """
        if self.spline is None:
            return None
        return self.spline.smoothing

    @property
    def exit_code(self):
        """
        Return the spline exit code.

        Please see :class:`Spline` for further details on all codes.

        Returns
        -------
        int
        """
        if self.spline is None:
            return None
        return self.spline.exit_code

    @property
    def exit_message(self):
        """
        Return the spline exit message.

        Please see :class:`Spline` for further details on all codes.

        Returns
        -------
        str
        """
        if self.spline is None:
            return "Spline has not been initialized."
        return self.spline.exit_message

    @property
    def fit_valid(self):
        """
        Return whether the spline successfully fit the provided kernel.

        If imperfect fits are permitted, will return `True` so long as a
        solution exists.  In cases where imperfect fits are not allowed, the
        fit will only be considered valid if a full rank solution has been
        determine, or the fit was successful within the provided user or
        default parameters.

        Returns
        -------
        valid : bool
            `True` if the fit provides a valid solution, and `False` otherwise.
        """
        if self.spline is None:
            return False
        code = self.exit_code
        if code > 5:
            return False
        if self.imperfect:
            return True

        if -3 <= code <= 0:
            return True
        return abs(code) == self.spline.rank

    @property
    def coefficients(self):
        """
        Return the spline coefficients.

        Returns
        -------
        numpy.ndarray (float)
            The spline coefficients of shape (n_coefficients,).  Here,
            n_coefficients = product(n_knots - degrees - 1) over all
            dimensions.
        """
        if self.spline is None:
            return None
        return self.spline.coefficients

    @property
    def knots(self):
        """
        Return the coordinates of the spline knots.

        Returns
        -------
        numpy.ndarray (float)
            An array of shape (n_dimensions, max(n_knots)).
        """
        if self.spline is None:
            return None
        return self.spline.knots

    @property
    def panel_mapping(self):
        """
        Return the 1-D to N-D panel mapping used for subsequent fitting.

        Each panel is bounded by the knot vertices.

        Returns
        -------
        numpy.ndarray (int)
            The 1-D to N-D panel map of shape (n_dimensions, n_panels).
        """
        if self.spline is None:
            return None
        return self.spline.panel_mapping

    @property
    def panel_steps(self):
        """
        Return the 1-D panel mapping steps for each dimension.

        Returns
        -------
        numpy.ndarray (int)
            The panel steps of shape (n_dimensions,)
        """
        if self.spline is None:
            return None
        return self.spline.panel_steps

    @property
    def knot_steps(self):
        """
        Return the 1-D panel mapping steps for each dimension.

        Returns
        -------
        numpy.ndarray (int)
            The knot steps of shape (n_dimensions,)
        """
        if self.spline is None:
            return None
        return self.spline.knot_steps

    @property
    def nk1(self):
        """
        Return the nk1 values for the knots.

        The returned value is n_knots - degrees - 1.

        Returns
        -------
        numpy.ndarray (int)
            The nk1 values of shape (n_dimensions,).
        """
        if self.spline is None:
            return None
        return self.spline.nk1

    @property
    def spline_mapping(self):
        """
        Return the 1-D to N-D index map for the spline.

        Returns
        -------
        numpy.ndarray (int)
            The spline map of shape (n_dimensions, product(degrees + 1)).
        """
        if self.spline is None:
            return None
        return self.spline.spline_mapping

    @property
    def n_knots(self):
        """
        Return the number of spline knots in each dimension.

        Returns
        -------
        numpy.ndarray (int)
            An array of shape (n_dimensions,).
        """
        if self.spline is None:
            return None
        return self.spline.n_knots

    @property
    def extent(self):
        """
        Return the extent of the kernel coordinates in each dimension.

        Returns
        -------
        extent : numpy.ndarray (float)
            An array of shape (n_dimensions, 2) where extent[0, 0] gives the
            minimum offset for dimension 0 and extent[0, 1] gives the maximum
            offset for dimension 0.
        """
        if not isinstance(self.kernel_coordinates, np.ndarray):
            return None
        result = np.empty((self.features, 2), dtype=float)
        result[:, 0] = np.nanmin(self.kernel_coordinates, axis=1)
        result[:, 1] = np.nanmax(self.kernel_coordinates, axis=1)
        return result

    @property
    def resampling_arguments(self):
        """
        Return the spline parameters necessary for resampling.

        Returns all of the arguments in the correct order necessary for
        :func:`perform_fit` aside from the coordinates (the first argument).

        Returns
        -------
        tuple
        """
        if self.spline is None:
            return tuple([None] * 9)
        return (self.knots, self.coefficients, self.degrees,
                self.panel_mapping, self.panel_steps, self.knot_steps,
                self.nk1, self.spline_mapping, self.n_knots)

    def set_kernel(self, kernel, kernel_spacing=1.0, kernel_offsets=None,
                   degrees=3, smoothing=0.0, imperfect=None, **spline_kwargs):
        """
        Setting the kernel automatically initializes a :class:`Spline` object
        that is solved and may be used for interpolating kernel values at
        locations away from the kernel vertices.  Both regular (grid) kernels
        and irregular kernels may be supplied.  Either `kernel_spacing` or
        `kernel_offsets` must be supplied in order to determine the coordinates
        of any interpolated points.  In the case of irregular kernels, these
        must be provided in `kernel_offsets`.  For regularly spaced grid,
        `kernel_spacing` may be provided instead.

        Generally, smoothing should be set to zero for interpolation.  However,
        in cases where a noisy irregular kernel is provided, smoothing may be
        set to `None` for a nominal fit, or provided if an optimal value is
        known.

        Parameters
        ----------
        kernel : numpy.ndarray (float)
            The kernel to set.  Must have n_features dimensions.
        kernel_spacing : float or numpy.ndarray (float), optional
            The spacing between each kernel element in units of the
            coordinates. Supply either as a single value for all features,
            or as an array of shape (n_features,) giving the kernel
            spacing for each feature.
        kernel_offsets : tuple or array_like, optional
            If the kernel is regular, should be an n-dimensional tuple
            containing the grid indices in each dimension.  Otherwise, should
            be an array of shape (n_dimensions, kernel.size).
        degrees : int or numpy.ndarray (int), optional
            The degree of spline to fit in each dimension.  Either a scalar can
            be supplied pertaining to all dimensions, or an array of shape
            (n_dimensions,) can be used.
        smoothing : float, optional
            Used to specify the smoothing factor.  If set to `None`, the
            smoothing will be determined based on user settings or input data.
            If `exact` is `True`, smoothing will be disabled (zero).  If
            `exact` is `False`, smoothing will be set to n - sqrt(2 * n)
            where n is the number of data values.  For interpolation,
            smoothing should be set to zero.  Smoothing must be >= 0.
        imperfect : bool, optional
            If a spline fit to the kernel is allowed to be imperfect (`True`),
            will only raise an error on spline fitting if a major error was
            encountered.  Otherwise, fits will be permitted so long as a
            solution was reached, even if that solution did not meet
            expectations.
        spline_kwargs : dict, optional
            Please see :class:`Spline` for a list of all spline initialization
            keyword arguments.  Note that `solve` will always be set to `True`.

        Returns
        -------
        None
        """
        if imperfect is not None:
            self.imperfect = imperfect
        self.parse_kernel(kernel, kernel_spacing=kernel_spacing,
                          kernel_offsets=kernel_offsets)
        self.fit_spline(degrees=degrees, smoothing=smoothing, **spline_kwargs)

    def parse_kernel(self, kernel, kernel_spacing=1.0, kernel_offsets=None):
        """
        Check and then convert the kernel arguments for spline fitting.

        Parameters
        ----------
        kernel : numpy.ndarray (float)
            The kernel to set.  Must have n_features dimensions.
        kernel_spacing : float or numpy.ndarray (float), optional
            The spacing between each kernel element in units of the
            coordinates. Supply either as a single value for all features,
            or as an array of shape (n_features,) giving the kernel spacing
            for each feature.
        kernel_offsets : tuple or array_like, optional
            If the kernel is regular, should be an n-dimensional tuple
            containing the grid indices in each dimension.  Otherwise, should
            be an array of shape (n_dimensions, kernel.size).

        Returns
        -------
        None
        """
        self.kernel = np.asarray(kernel, dtype=float)
        get_offsets = kernel_offsets is None

        if kernel_spacing is None and kernel_offsets is None:
            raise ValueError("Must supply either kernel_spacing or "
                             "kernel_offsets.")

        if get_offsets and self.kernel.ndim != self.features:
            raise ValueError(f"Kernel must have the same number of "
                             f"dimensions as the input data "
                             f"({kernel.ndim} != {self.features}).")

        irregular = self.kernel.ndim == 1 and self.features > 1

        if get_offsets:
            kernel_spacing = np.atleast_1d(
                np.asarray(kernel_spacing, dtype=float))
            # Expand to all dimensions if a single value was supplied.
            if kernel_spacing.size == 1 and self.features > 1:
                kernel_spacing = np.full(self.features, kernel_spacing[0])

            if kernel_spacing.size != self.features:
                raise ValueError(
                    f"Kernel spacing size does not equal the number of "
                    f"features ({kernel_spacing.size} != {self.features})")
            kernel_offsets = []
            for axis in range(self.features):
                numpy_axis = self.features - 1 - axis
                size = self.kernel.shape[numpy_axis]
                offset = 0.5 if size % 2 == 0 else 0.0
                axis_coordinates = np.arange(size) - (size // 2) + offset
                axis_coordinates *= kernel_spacing[axis]
                kernel_offsets.append(axis_coordinates)
            kernel_offsets = tuple(kernel_offsets)
            self.kernel_spacing = kernel_spacing
        else:
            self.kernel_spacing = None

        if self.features == 1:
            kernel_coordinates = np.atleast_2d(kernel_offsets).astype(float)
            if kernel_coordinates[0].size != kernel.size:
                raise ValueError("1-D kernel coordinates do not match kernel "
                                 "shape.")
        elif irregular:
            try:
                kernel_coordinates = np.asarray(kernel_offsets, dtype=float)
            except ValueError:
                raise ValueError("Irregular kernel offsets should be supplied "
                                 "as an (n_dimensions, kernel.size) array.")
            if kernel_coordinates.shape != (self.features, kernel.size):
                raise ValueError(f"Irregular kernel offsets do not match "
                                 f"expected shape "
                                 f"({self.features}, {kernel.size}). Received "
                                 f"{kernel_coordinates.shape}.")

        else:
            kernel_coordinates = np.vstack(
                [np.asarray(x, dtype=float).ravel() for x in
                 np.meshgrid(*kernel_offsets[::-1], indexing='ij')[::-1]])

        self.kernel_coordinates = kernel_coordinates

    def fit_spline(self, degrees=3, smoothing=0.0, **spline_kwargs):
        """
        Fit a spline to the tree kernel.

        Parameters
        ----------
        degrees : int or numpy.ndarray (int), optional
            The degree of spline to fit in each dimension.  Either a scalar can
            be supplied pertaining to all dimensions, or an array of shape
            (n_dimensions,) can be used.
        smoothing : float, optional
            Used to specify the smoothing factor.  If set to `None`, the
            smoothing will be determined based on user settings or input data.
            If `exact` is `True`, smoothing will be disabled (zero).  If
            `exact` is `False`, smoothing will be set to n - sqrt(2 * n)
            where n is the number of data values.  For interpolation,
            smoothing should be set to zero.  Smoothing must be >= 0.
        spline_kwargs : dict, optional
            Please see :class:`Spline` for a list of all spline initialization
            keyword arguments.  Note that `solve` will always be set to `True`.

        Raises
        ------
        RuntimeError
           If the fit to the kernel was not adequate for further fitting.  If
           no further tweaking of the input parameters is possible, set
           `imperfect`=`True` during initialization, or manually set the
           `imperfect` attribute to `True` manually.

        Returns
        -------
        None
        """
        if 'solve' in spline_kwargs:
            spline_kwargs['solve'] = True

        self.spline = Spline(*self.kernel_coordinates, self.kernel.ravel(),
                             degrees=degrees, smoothing=smoothing,
                             **spline_kwargs)

        if not self.fit_valid:
            raise RuntimeError(f"Unsuccessful fit: {self.exit_message}")
