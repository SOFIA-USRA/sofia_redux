# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree

from sofia_redux.toolkit.resampling.resample_utils import (
    polynomial_terms, polynomial_exponents, polynomial_derivative_map)

__all__ = ['Rtree']


class Rtree(object):

    def __init__(self, argument, shape=None, build_type='all',
                 order=None, fix_order=True, leaf_size=40,
                 **distance_kwargs):
        r"""
        Create a tree structure for use with the resampling algorithm.

        The resampling tree is primarily responsible for deriving and
        storing all independent variables necessary for polynomial fitting,
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

        POLYNOMIAL ORDER AND TERMS

        The resampling tree is also responsible for deriving all polynomial
        terms for the fit as well as mapping these terms to the derivative of
        the function.  Please see :func:`power_product`,
        :func:`polynomial_exponents`, and :func:`polynomial_derivative_map`
        for further details.  It is also possible to calculate terms for
        a range of orders in the case where order may vary to accommodate
        fits where insufficient samples are present.  To allow `order` to
        vary, set `fix_order` to `False`.

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
        distance_kwargs : dict, optional
            Optional keyword arguments passed into
            :func:`sklearn.neighbors.DistanceMetric`.  The default is to use
            the "minkowski" definition with `p=2`, i.e., the Euclidean
            definition.
        """

        self._shape = None
        self._n_features = None
        self._n_blocks = None
        self._search = None
        self._tree = None
        self._balltree = None
        self._ball_initialized = False
        self._hood_initialized = False
        self._order = None
        self._order_symmetry = None
        self._order_required = None
        self._order_varies = None
        self._exponents = None
        self._derivative_term_map = None
        self._phi_terms_precalculated = False
        self.block_offsets = None
        self.block_population = None
        self.populated = None
        self.hood_population = None
        self.coordinates = None
        self.phi_terms = None
        self.term_indices = None
        self.derivative_term_indices = None

        arg = np.asarray(argument)
        if np.asarray(arg).ndim > 1:
            self.build_tree(arg, shape=shape, method=build_type,
                            leaf_size=leaf_size, **distance_kwargs)
        else:
            self._set_shape(tuple(arg))

        if order is not None:
            self.set_order(order, fix_order=fix_order)

            if self.coordinates is not None:
                self.precalculate_phi_terms()

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def features(self):
        return self._n_features

    @property
    def order(self):
        return self._order

    @property
    def exponents(self):
        if self._exponents is not None:
            return self._exponents.copy()
        return None

    @property
    def derivative_term_map(self):
        if self._derivative_term_map is not None:
            return self._derivative_term_map.copy()
        return None

    @property
    def order_symmetry(self):
        return self._order_symmetry

    @property
    def order_varies(self):
        return self._order_varies

    @property
    def phi_terms_precalculated(self):
        return self._phi_terms_precalculated

    @property
    def n_blocks(self):
        return self._n_blocks

    @property
    def search(self):
        return self._search

    @property
    def balltree_initialized(self):
        return self._ball_initialized

    @property
    def hood_initialized(self):
        return self._hood_initialized

    @property
    def n_members(self):
        if self.coordinates is None:
            return 0
        else:
            return self.coordinates.shape[1]

    def _set_shape(self, shape):
        r"""
        Set the shape of the tree.

        The shape of the tree determines the number of n-dimensional coordinate
        bins.  Once defined, a number of useful factors are pre-calculated to
        allow for easy determination of neighboring bins to any given bin.

        The search kernel is every permutation of [-1, 0, 1] over all features,
        meaning each bin not on the edge of the tree bounds has :math:`3^n - 1`
        neighbors.

        Parameters
        ----------
        shape : n-tuple of int
            The size for each of the n features of the tree.

        Returns
        -------
        None
        """
        self._shape = np.asarray(shape).ravel().astype(int)
        self._n_features = len(self._shape)
        self._n_blocks = np.prod(self._shape)

        searches = np.array(list(
            itertools.product([-1, 0, 1], repeat=self.features)))

        self._search = searches.T

        # self._multipliers = abs(searches)
        # self._deltas = -1 * (searches < 0)
        # reverse_search = searches * -1
        # self._reverse_deltas = -1 * (reverse_search < 0)

        # Reset necessary attributes
        self._tree = None
        self.block_offsets = None
        self.block_population = None
        self.populated = None
        self.hood_population = None
        self.term_indices = None

    def set_order(self, order, fix_order=True):
        r"""
        Set the order(s) of polynomial fit for the resampling algorithm tree.

        In addition to declaring the desired order of polynomial fit, the user
        may also opt to select an algorithm to validate the desired order of
        fit is possible given a distribution of samples.  Please see
        :func:`resample_utils.check_orders` for a full description of available
        algorithms.

        Throughout the resampling algorithm, orders are classified as either
        "symmetrical" or "asymmetrical".  An order is termed as symmetrical
        if it is the same for all features of the polynomial fit.  i.e., the
        maximum exponent for each feature in a polynomial equation will be
        equal to the order.  For example, a 2-dimensional 2nd order polynomial
        is of the form:

        .. math::

            f(x, y) = c_0 + c_1 x + c_2 x^2 + c_3 y + c_4 x y + c_5 y^2

        where the maximum exponent for :math:`x` and :math:`y` in any
        term is 2.

        An order is considered "asymmetrical" if orders are not equal across
        features or dimensions.  For example, a 2-dimensional polynomial with
        an order of 1 in :math:`x`, and 2 in :math:`y` would have the form:

        .. math::

            f(x, y) = c_0 + c_1 x + c_2 y + c_3 x y + c_4 y^2

        where the maximum exponent of :math:`x` is 1, and the maximum exponent
        of :math:`y` is 2.

        If orders are symmetrical, the user can allow the order to vary such
        that it passes the order validation algorithm.  For instance, if only
        2 samples exist for a 1-dimensional polynomial fit, and the user
        requested a 2nd order polynomial fit, the resampling algorithm will
        lower the order to 1, so that a fit can be performed (if the algorithm
        worked by counting the number of available samples).  If orders are
        asymmetrical, this option is not available and fits will be aborted
        if they fail the order validation algorithm.

        Parameters
        ----------
        order : int or array_like (n_features,)
           The symmetrical or asymmetrical orders respectively.
        fix_order : bool, optional
            If `order` is symmetrical, allow for a varying order in an attempt
            to pass the order validation algorithm (the order can only
            decrease).

        Returns
        -------
        None
        """
        o = np.asarray(order)
        order_symmetry = o.shape == ()
        if not order_symmetry:
            if o.size != self._n_features:
                raise ValueError(
                    "Number of orders (%i) does not match number of "
                    "features (%i)" % (o.size, self.features))
        if order_symmetry:
            self._order = int(order)
        else:
            self._order = o.astype(int)
        self._order_symmetry = order_symmetry
        self._order_varies = not fix_order and order_symmetry
        # Note, it is possible to vary orders if order_symmetry is False, but
        # will require either a ton of either memory or computation time.
        # (plus I'll need to write the code)

    def precalculate_phi_terms(self):
        r"""
        Calculates polynomial terms for the tree coordinates and order.

        Calculates the :math:`\Phi` terms for the equation:

        .. math::

            f(\Phi) = \hat{c} \cdot \Phi

        The polynomial terms are dependent on the tree coordinates, and the
        order of polynomial fit.  Please see :func:`polynomial_exponents` for
        a description on how terms are derived.

        If the order is not fixed (i.e., is allowed to vary as marked by the
        `order_varies` attribute), a set of terms will be derived for all
        orders up to the order defined by the user.  Please note that orders
        may only vary if the order is "symmetrical" as defined in
        `Rtree.set_order`.  In addition, a mapping relating terms to
        derivatives of the fit is also created in the `derivative_term_map`
        attribute, details of which can be found in
        :func:`polynomial_derivative_map`.  When the order does vary, the
        terms for order k are given as:

            tree.phi_terms[tree.order_term_indices[k]:
                           tree.order_term_indices[k+1]]

        Returns
        -------
        None
        """
        if self.coordinates is None:
            raise ValueError("Tree has not been populated with coordinates.")

        if self.order is None:
            raise ValueError("Order has not been set.")

        self._phi_terms_precalculated = False

        if not self.order_varies:  # easy
            self.term_indices = None
            exponents = polynomial_exponents(self._order, ndim=self.features)
            self.phi_terms = polynomial_terms(self.coordinates, exponents)
            self._derivative_term_map = polynomial_derivative_map(exponents)
            self.derivative_term_indices = None

        else:
            # For each possible order generate phi terms and derivative maps.
            # Also, since we concatenate the results into a single large array,
            # create indices that reference correct terms for a given order.
            order_phi_indices = [0]
            order_phi = []
            order_derivative_indices = [0]
            order_derivative_maps = []
            exponents = None

            for o in range(self.order + 1):
                exponents = polynomial_exponents(o, ndim=self.features)
                phi = polynomial_terms(self.coordinates, exponents)
                order_phi.append(phi)
                order_phi_indices.append(phi.shape[0] + order_phi_indices[o])
                derivative_map = polynomial_derivative_map(exponents)
                order_derivative_maps.append(derivative_map)
                order_derivative_indices.append(derivative_map.shape[2]
                                                + order_derivative_indices[o])

            self.phi_terms = np.empty((order_phi_indices[-1], self.n_members),
                                      dtype=np.float64)
            self.term_indices = np.asarray(order_phi_indices,
                                           dtype=int)
            self._derivative_term_map = np.empty(
                (self.features, 3, order_derivative_indices[-1]), dtype=int)

            self.derivative_term_indices = np.asarray(
                order_derivative_indices, dtype=int)

            for o in range(self.order + 1):
                i0, i1 = order_phi_indices[o: o + 2]
                self.phi_terms[i0: i1] = order_phi[o]
                i0, i1 = order_derivative_indices[o: o + 2]
                self._derivative_term_map[..., i0: i1] = \
                    order_derivative_maps[o]

        self._phi_terms_precalculated = True
        self._exponents = exponents

    def to_index(self, coordinates):
        r"""
        Find the tree block indices of a coordinate or set of coordinates.

        Examples
        --------
        A tree of shape (5, 6) in 2 dimensions contains coordinates ranging
        from 0-5 in x, and 0-6 in y.  The coordinate (x, y) = (2.7, 3.1) would
        exist in tree block [2, 3] (floored values), but the tree block
        structure is one-dimensional.  The actual block index would be
        15 since it is calculated as (2 * 6) + 3, or (x * shape[1]) + y.

        Parameters
        ----------
        coordinates : numpy.ndarray (n_feature,) or (n_feature, n_coordinates)
            The coordinates for which to return tree block indices.
            Coordinates should be scaled such that they range from
            0 -> tree.shape[k] for the feature k.

        Returns
        -------
        index or indices : np.int64 or np.ndarray (n_coordinates,)
            The tree block indices for the supplied coordinates.
        """
        c = np.asarray(coordinates, dtype=int)
        single = c.ndim == 1
        if single:
            c = c[:, None]
        index = np.ravel_multi_index(c, self._shape, mode='clip')
        return index[0] if single else index

    def from_index(self, index):
        r"""
        Find the lower corner coordinate for a tree block index.

        Examples
        --------
        The lower corner coordinate in a tree block is given as the floored
        values of all coordinates that exist inside the block.  For example,
        the index 15 in a tree of shape (5, 6) corresponds to coordinates in
        the range x = (2->3), y = (3->4).  Therefore, the lower coordinate is
        (2, 3).

        Parameters
        ----------
        index : int or numpy.ndarray (n_indices,)
            The indices for which to return the lower corner coordinate(s).

        Returns
        -------
        lower_corner_coordinates : numpy.ndarray of int
           An array of shape (n_features,) or (n_features, n_indices)
           containing the lower corner coordinates of supplied tree block
           indices.
        """
        c = np.asarray(index, dtype=int)
        shape = c.shape
        single = shape == ()
        index = np.unravel_index(np.atleast_1d(c).ravel(), self._shape)
        coordinates = np.empty((self._n_features, c.size), dtype=int)
        coordinates[:] = index
        return coordinates.ravel() if single else coordinates

    def build_tree(self, coordinates, shape=None, method='all',
                   leaf_size=40, **distance_kwargs):
        r"""
        Create the ball and hood tree structures from coordinates.

        Populates the tree with coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray (n_features, n_coordinates)
            The coordinates from which to build the trees.
        shape : n-tuple of int, optional
            The shape of the tree.  If not supplied it is calculated as
            floor(max(coordinate[k])) + 1 for the feature k.
        method : str, optional
            Specifies the trees to build.  Available options are {'hood',
            'balltree', 'all', None}.  The default ('all') builds both
            the balltree and hood tree.
        leaf_size : int, optional
            An specifies the point at which the ball tree switches over to
            the brute force method.  See :func:`sklearn.neighbors.BallTree`
            for further information.
        distance_kwargs : dict, optional
            Optional keyword arguments passed into
            :func:`sklearn.neighbors.DistanceMetric`.  The default is to use
            the "minkowski" definition with `p=2`, i.e., the Euclidean
            definition.

        Returns
        -------
        None
        """
        if shape is None:
            self._set_shape(coordinates.astype(int).max(axis=1) + 1)
        else:
            self._set_shape(shape)
        method = str(method).lower().strip()

        self.coordinates = np.asarray(coordinates).astype(float, order='F')

        if method == 'hood':
            self._build_hood_tree()
        elif method == 'balltree':
            self._build_ball_tree(leaf_size=leaf_size, **distance_kwargs)
        elif method == 'none':
            pass
        elif method == 'all':
            self._build_hood_tree()
            self._build_ball_tree(leaf_size=leaf_size, **distance_kwargs)
        else:
            raise ValueError("Unknown tree building method: %s" % method)

    def _build_ball_tree(self, leaf_size=40, **distance_kwargs):
        r"""
        Build the balltree structure.

        The balltree is created using :func:`sklearn.neighbors.BallTree` and
        allows rapid calculation of relative distances from a supplied set
        of coordinates to those within the tree.

        Parameters
        ----------
        leaf_size : int, optional
            An specifies the point at which the ball tree switches over to
            the brute force method.  See :func:`sklearn.neighbors.BallTree`
            for further information.
        distance_kwargs : dict, optional
            Optional keyword arguments passed into
            :func:`sklearn.neighbors.DistanceMetric`.  The default is to use
            the "minkowski" definition with `p=2`, i.e., the Euclidean
            definition.

        Returns
        -------
        None
        """
        self._ball_initialized = False
        if leaf_size is None:
            self._balltree = BallTree(self.coordinates.T,
                                      **distance_kwargs)
        else:
            self._balltree = BallTree(self.coordinates.T,
                                      leaf_size=leaf_size,
                                      **distance_kwargs)
        self._ball_initialized = True

    def _build_hood_tree(self):
        r"""
        Build the hood tree structure.

        The hood tree defines a number of attributes allowing for fast access
        to all coordinates within a tree block, or the coordinates within
        a tree block and all neighboring blocks.  A single block and all its
        neighboring blocks are referred to as the "neighborhood" or "hood" of
        the block.  In addition, the population of each block and hood are also
        stored.

        Returns
        -------
        None
        """
        self._hood_initialized = False
        bins = self.to_index(self.coordinates)
        indices = np.arange(bins.size)
        n_bins = np.prod(self._shape)
        self._tree = csr_matrix((indices, [bins, indices]),
                                shape=(n_bins, bins.size))
        self._tree = np.split(self._tree.data, self._tree.indptr[1:-1])

        self.block_offsets = self.coordinates - self.coordinates.astype(int)
        self.block_population = np.array([s.size for s in self._tree])
        self.populated = np.nonzero(self.block_population > 0)[0]
        self.hood_population = np.zeros_like(self.block_population)
        self.max_in_hood = np.zeros_like(self.block_population)
        for block in range(self.n_blocks):
            hoods = self.neighborhood(block, cull=True)
            if hoods.size == 0:  # pragma: no cover
                self.hood_population[block] = 0
                self.max_in_hood[block] = 0
            else:
                self.hood_population[block] = np.sum(
                    self.block_population[hoods])
                self.max_in_hood[block] = np.max(
                    self.block_population[hoods])
        self._hood_initialized = True

    def query_radius(self, coordinates, radius=1.0, **kwargs):
        r"""
        Return all tree members within a certain distance of given points.

        Quickly retrieves all members of the tree population within the radius
        of some defined coordinate(s).  The default distance definition is
        "minkowski" with p=2, which is equivalent to the "euclidean"
        definition. For a list of available distance metric please see
        `sklearn.neighbors.BallTree.valid_metrics`.  The distance metric may
        be defined during Rtree initialization.

        Parameters
        ----------
        coordinates : numpy.ndarray
            An (n_features, n_coordinates) or (n_features,) array containing
            coordinates around which to determine tree members within a
            certain radius.
        radius : float, optional
            The search radius around each coordinate.
        kwargs : dict, optional
            Keywords for :func:`sklearn.neighbors.BallTree.query_radius`.

        Returns
        -------
        indices : numpy.ndarray of object (n_coordinates,)
            Each element is a numpy integer array containing the indices of
            tree members within `radius`.  This return value may change
            depending on the options in `kwargs`.  Please see
            :func:`sklearn.neighbors.BallTree.query_radius` for further
            information.
        """
        if not self._ball_initialized:
            raise RuntimeError("Ball tree not initialized")
        if coordinates.ndim == 1:
            c = coordinates[None]
        else:
            c = coordinates.T
        return self._balltree.query_radius(c, radius, **kwargs)

    def block_members(self, block, get_locations=False, get_terms=False):
        if not self._hood_initialized:
            raise RuntimeError("Neighborhood tree not initialized.")
        members = self._tree[block]
        if not get_locations and not get_terms:
            return members

        result = members,
        if get_locations:
            result += self.coordinates[:, members],
        if get_terms:
            if not self.phi_terms_precalculated:
                raise RuntimeError("Phi terms have not been calculated.")
            result += self.phi_terms[:, members],

        return result

    def neighborhood(self, index, cull=False, valid_neighbors=False):
        expanded = self.from_index(index)[:, None] + self.search
        bad = np.any(
            (expanded < 0) | (expanded >= self._shape[:, None]), axis=0)
        hood = self.to_index(expanded)
        keep = ~bad
        if cull:
            hood = hood[keep]
        else:
            hood[bad] = -1

        return (hood, keep) if valid_neighbors else hood

    def hood_members(self, center_block, get_locations=False, get_terms=False):
        r"""
        Find all members within the neighborhood of a single block.

        Parameters
        ----------
        center_block : int
            The index of the block at the center of the neighborhood.
        get_locations : bool, optional
            If `True`, return the coordinates of the hood members.
        get_terms : bool, optional
            If `True`, return the calculated "phi" terms of hood members.  Note
            that a polynomial order must have been set, and terms calculated.
            See :func:`Rtree.set_order`, and
            :func:`Rtree.precalculate_phi_terms` for further information.

        Returns
        -------
        members, [coordinates, terms]
            members is a numpy.ndarray of int and shape (found_members,).
            If `get_locations` was set to `True,` coordinates is of shape
            (n_features, found_members).  If `get_terms` was set to `True`,
            terms is of shape (n_terms, found_members).
        """
        if not self._hood_initialized:
            raise RuntimeError("Neighborhood tree not initialized.")

        blocks = self.neighborhood(center_block, cull=True)
        hood_members = np.empty(self.block_population[blocks].sum(), dtype=int)
        found = 0
        for block in blocks:
            block_members = self.block_members(block)
            hood_members[found: found + block_members.size] = block_members
            found += block_members.size

        hood_members = hood_members[:found]
        if not get_locations and not get_terms:
            return hood_members

        result = hood_members,
        if get_locations:
            result += self.coordinates[:, hood_members],

        if get_terms:
            if not self._phi_terms_precalculated:
                raise RuntimeError("Phi terms have not been calculated.")
            result += self.phi_terms[:, hood_members],

        return result

    def __call__(self, x, reverse=False):
        r"""
        Returns the block index of a coordinate

        Parameters
        ----------
        x : int or numpy.ndarray (features,) or (features, n_coordinates)
            The block for which to find the lower coordinate.  If `reverse` is
            `False`, returns the block index for a given coordinate.
        reverse : bool, optional
            Specifies if the return value should be the lower coordinate or
            block index.

        Returns
        -------
        block_index or coordinate : int or numpy.ndarray
        """
        if reverse:
            return self.from_index(x)
        else:
            return self.to_index(x)


# # The following functions are not used but available as an
# # alternative to BallTree.
#
# _fast_flags = {'nsz', 'nnan', 'ninf'}
#
# def hood_members(self, center_block, get_locations=False, get_terms=False):
#
#     if not self._hood_initialized:
#         raise RuntimeError("Neighborhood tree not initialized.")
#
#     hood, populated = self.neighborhood(
#         center_block, cull=True, valid_neighbors=True)
#
#     if hood.size == 0:  # pragma: no cover
#         if not get_locations and not get_terms:
#             return hood
#         result = hood,
#         if get_locations:
#             result += np.empty((self.features, 0), dtype=np.float64),
#         if get_terms:
#             if self._phi_terms_precalculated:
#                 result += np.empty((self.phi_terms.shape[0], 0),
#                                    dtype=np.float64),
#             else:
#                 result += None,
#         return result
#
#     members = np.empty(self.block_population[hood].sum(), dtype=int)
#     multipliers = self._multipliers[populated]
#     deltas = self._deltas[populated]
#
#     population = 0
#     for block, m, d in zip(hood, multipliers, deltas):
#
#         valid_members = self.cull_members(
#             self.block_offsets, self._tree[block], m, d, 0)
#         new_population = population + valid_members.size
#         members[population:new_population] = valid_members
#         population = new_population
#
#     members = members[:population]
#     if not get_terms and not get_locations:
#         return members
#
#     result = members,
#     if get_locations:
#         result += self.coordinates[:, members],
#
#     if get_terms:
#         if not self._phi_terms_precalculated:
#             raise RuntimeError("Phi terms have not been calculated.")
#         result += self.phi_terms[:, members],
#
#     return result
#
# @staticmethod
# @njit(nogil=True, cache=True, fastmath=True)
# def cull_members(offsets, members, multiplier,
#                  delta, return_indices):   # pragma: no cover
#
#     features = delta.size
#     for k in range(features):
#         if multiplier[k] != 0:
#             break
#         elif delta[k] != 0:
#             break
#     else:
#         if return_indices:
#             result = np.empty(members.size, dtype=nb.i8)
#             for i in range(members.size):
#                 result[i] = i
#             return result
#         else:
#             return members
#
#     n_members = members.size
#     keep = np.empty(n_members, dtype=nb.i8)
#     n_found = 0
#     for i in range(n_members):
#         d = 0.0
#         member = members[i]
#         offset = offsets[:features, member]
#         for k in range(features):
#             doff = multiplier[k] * offset[k] + delta[k]
#             doff *= doff
#             d += doff
#             if d > 1:
#                 break
#         else:
#             if return_indices:
#                 keep[n_found] = i
#             else:
#                 keep[n_found] = member
#
#             n_found += 1
#
#     return keep[:n_found]

# @njit(cache=True, fastmath=_fast_flags)
# def update_cross_distance(start, visitor_members, local_members,
#                           separations, out_visit, out_local,
#                           out_separation):  # pragma: no cover
#
#     n_visitor = visitor_members.size
#     n_local = local_members.size
#     n = start
#     for i in range(n_visitor):
#         visitor = visitor_members[i]
#         for j in range(n_local):
#             d = separations[i, j]
#             if d <= 1.0:
#                 out_visit[n] = visitor
#                 out_local[n] = local_members[j]
#                 out_separation[n] = d
#                 n += 1
#     return n
#
#
# @jit(cache=True, fastmath=_fast_flags)
# def block_intersection(block, visitor_tree, local_tree,
#                        get_distance=True, get_edges=True): # pragma: no cover
#
#     n_visitors = visitor_tree.block_population[block]
#     if n_visitors == 0:
#         return [], []
#     visitor_members, visitor_locations = visitor_tree.block_members(
#         block, get_locations=True)
#     visitor_offsets = visitor_tree.block_offsets
#
#     hood_population = local_tree.hood_population[block]
#     if hood_population == 0:
#         return [], []
#
#     hoods, populated = local_tree.neighborhood(
#         block, cull=True, cull_idx=True)
#     pop_idx = np.nonzero(populated)[0]
#     hood_members = [local_tree.block_members(hood) for hood in hoods]
#     visitor_deltas = visitor_tree.deltas[pop_idx]
#     local_deltas = local_tree.reverse_deltas[pop_idx]
#     multipliers = visitor_tree.multipliers[pop_idx]  # same for both
#
#     return hood_loop(
#         hoods, hood_population, block,
#         hood_members, visitor_members,
#         local_tree.block_offsets, visitor_offsets,
#         visitor_tree.coordinates, local_tree.coordinates,
#         multipliers, visitor_deltas, local_deltas,
#         get_edges, get_distance)
#
#
# @jit(parallel=False, cache=True, fastmath=_fast_flags)
# def hood_loop(hoods, hood_population, block, # max_populations (before block)
#               hood_members, visitor_members,
#               local_block_offsets, visitor_offsets,
#               visitor_coordinates, local_coordinates,
#               multipliers, visitor_deltas, local_deltas,
#               get_edges, get_distance):  # pragma: no cover
#
#     n_hoods = hoods.size
#     n_visitors = visitor_members.size
#     all_visitor_ids = np.arange(n_visitors)
#
#     if get_edges:
#         ndim = visitor_deltas.shape[1]
#         hood_left = np.zeros((ndim, n_visitors, hood_population), dtype=int)
#     else:
#         hood_left = None
#
#     if get_distance:
#         hood_dist = np.empty((n_visitors, hood_population))
#     else:
#         hood_dist = None
#
#     hood_counts = np.zeros(n_visitors, dtype=int)
#     hood_founds = np.empty((n_visitors, hood_population), dtype=int)
#
#     for i in range(n_hoods):
#         local_hood = hoods[i]
#         in_the_hood = local_hood == block
#         local_members = hood_members[i]
#         if local_members.size == 0:
#             continue
#
#         if in_the_hood:
#             locals_near_block = local_members
#             off_center = None
#         else:
#             off_center = multipliers[i]
#             locals_near_block = cull_members(
#                 local_block_offsets, local_members,
#                 off_center, visitor_deltas[i], 0)
#         if locals_near_block.size == 0:
#             continue
#
#         if in_the_hood:
#             visitor_ids = all_visitor_ids
#             visitors_near_hood = visitor_members
#         else:
#             visitor_ids = cull_members(
#                 visitor_offsets, visitor_members, off_center,
#                 local_deltas[i], 1)
#             if visitor_ids.size == 0:
#                 continue
#             visitors_near_hood = visitor_members[visitor_ids]
#
#         visitor_locations = visitor_coordinates[:, visitors_near_hood]
#
#         filter_coordinates(
#             visitor_locations, local_coordinates,  # coordinates
#             visitor_ids, locals_near_block,  # reduced and full indices
#             hood_founds, hood_counts,  # outputs
#             hood_dist, hood_left)
#
#     return hood_founds, hood_counts, hood_dist, hood_left
#
#
# @njit(cache=True, fastmath=_fast_flags)
# def filter_coordinates(visitor_locations,
#                        local_coordinates,
#                        visitor_ids,
#                        locals_near_block,
#                        hood_found,
#                        hood_count,
#                        hood_dist,
#                        hood_left):  # pragma: no cover
#
#     n_features, n_update = visitor_locations.shape
#     n_samples = locals_near_block.size
#     do_edge = hood_left is not None
#     do_distance = hood_dist is not None
#     for i in prange(n_update):
#         ind0 = visitor_ids[i]
#         for j in range(n_samples):
#             idx = hood_count[ind0]
#             ind1 = locals_near_block[j]
#             d2 = 0.0
#             for k in range(n_features):
#                 diff = visitor_locations[k, i] - local_coordinates[k, ind1]
#                 if do_edge:
#                     if diff < 0:
#                         hood_left[k, ind0, idx] += 1
#                 diff *= diff
#                 d2 += diff
#                 if d2 > 1:
#                     break
#             else:
#                 hood_count[ind0] += 1
#                 hood_found[ind0, idx] = ind1
#                 if do_distance:
#                     hood_dist[ind0, idx] = d2
#
#
#
#
# @njit(nogil=True, cache=True, fastmath=True)
# def access_members(array, indices):  # pragma: no cover
#     """
#     Return selected row elements of a 2-D array.
#
#     This is a utility function using Numba to allow for fast retrieval of
#     values from the second axis of a 2-D Numpy array.  This is roughly
#     twice as fast as using standard Numpy fancy indexing.
#
#     Parameters
#     ----------
#     array : numpy.ndarray (n_columns, n_rows)
#         The input Numpy array.
#     indices : numpy.ndarray (n_elements,)
#         The row indices to retrieve.
#
#     Returns
#     -------
#     indexed_array : numpy.ndarray (n_columns, n_elements)
#         The output array[:, indices].
#     """
#     return array[:, indices]
#
# @property
# def reverse_deltas(self):
#     return self._reverse_deltas
#
# @property
# def deltas(self):
#     return self._deltas
#
# @property
# def multipliers(self):
#     return self._multipliers
