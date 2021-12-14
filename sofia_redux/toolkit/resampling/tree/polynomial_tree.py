# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
from sofia_redux.toolkit.resampling.resample_utils import (
    polynomial_terms, polynomial_exponents, polynomial_derivative_map)

__all__ = ['PolynomialTree']


class PolynomialTree(BaseTree):

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
        super().__init__(argument, shape=shape, build_type=build_type,
                         leaf_size=leaf_size, **distance_kwargs)

        self._order = None
        self._order_symmetry = None
        self._order_required = None
        self._order_varies = None
        self._exponents = None
        self._derivative_term_map = None
        self._phi_terms_precalculated = False
        self.phi_terms = None
        self.term_indices = None
        self.derivative_term_indices = None

        if order is not None:
            self.set_order(order, fix_order=fix_order)

            if self.coordinates is not None:
                self.precalculate_phi_terms()

    @property
    def order(self):
        """
        Return the order of the polynomial for the tree.

        Returns
        -------
        int or numpy.ndarray (int)
            A scalar value if orders for each feature are identical, or an
            array of values of shape (n_features,) if values are not identical.
        """
        return self._order

    @property
    def exponents(self):
        """
        Return the polynomial exponents for the tree.

        The polynomial exponents describe how the polynomial terms should be
        calculated.  For example, the exponents [[0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [0, 2]] represent the equation:

          f(x, y) = c0 + c1.x + c2.x^2 + c3.y + c4.xy + c5.y^2

        Returns
        -------
        numpy.ndarray (int)
            An array of shape (n_exponents, n_features)
        """
        if self._exponents is not None:
            return self._exponents.copy()
        return None

    @property
    def derivative_term_map(self):
        """
        Returns mapping necessary to describe the 1st derivative of the tree.

        The terms describing the derivative of a polynomial function should all
        be present in the exponent terms of the tree, but need to be reordered
        in a specific way and may also contain an additional coefficient.
        Once the coefficients of the fit are known (c), the derivative of
        feature k may be be calculated by:

            sum(h[k, 0] * c[h[k, 1]] * p[h[k, 2], k])

        where p are the exponent terms, and h is the map returned by this
        property.

        Returns
        -------
        mapping : numpy.ndarray (int)
            An array of shape (n_features, 3, n_terms).
        """
        if self._derivative_term_map is not None:
            return self._derivative_term_map.copy()
        return None

    @property
    def order_symmetry(self):
        """
        Return whether the orders are symmetrical.

        Orders are considered symmetrical when the orders of each feature are
        identical.

        Returns
        -------
        bool
        """
        return self._order_symmetry

    @property
    def order_varies(self):
        """
        Return whether the order may be decreased.

        The order of the polynomial may be reduced if specifically set by the
        user (fix_order=False) and the order is symmetrical.  In this case,
        a subset of polynomial terms can be extracted from the original terms
        to describe a lower order polynomial.

        Returns
        -------
        bool
        """
        return self._order_varies

    @property
    def phi_terms_precalculated(self):
        """
        Return whether the polynomial terms have been calculated.

        Returns
        -------
        bool
        """
        return self._phi_terms_precalculated

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
        super()._set_shape(shape)
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
        `PolynomialTree.set_order`.  In addition, a mapping relating terms to
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

    def block_members(self, block, get_locations=False, get_terms=False):
        """
        Find all members within a single block.

        Parameters
        ----------
        block : int
            The index of the block.
        get_locations : bool, optional
            If `True`, return the coordinates of the hood members.
        get_terms : bool, optional
            If `True`, return the calculated "phi" terms of hood members.  Note
            that a polynomial order must have been set, and terms calculated.
            See :func:`PolynomialTree.set_order`, and
            :func:`PolynomialTree.precalculate_phi_terms` for further
            information.

        Returns
        -------
        members, [coordinates, terms]
            members is a numpy.ndarray of int and shape (found_members,).
            If `get_locations` was set to `True,` coordinates is of shape
            (n_features, found_members).  If `get_terms` was set to `True`,
            terms is of shape (n_terms, found_members).
        """
        result = super().block_members(block, get_locations=get_locations)
        if not get_terms:
            return result

        if get_terms:
            if not self._phi_terms_precalculated:
                raise RuntimeError("Phi terms have not been calculated.")
        if get_locations:
            result += self.phi_terms[:, result[0]],
        else:
            result = result, self.phi_terms[:, result]
        return result

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
            See :func:`PolynomialTree.set_order`, and
            :func:`PolynomialTree.precalculate_phi_terms` for further
            information.

        Returns
        -------
        members, [coordinates, terms]
            members is a numpy.ndarray of int and shape (found_members,).
            If `get_locations` was set to `True,` coordinates is of shape
            (n_features, found_members).  If `get_terms` was set to `True`,
            terms is of shape (n_terms, found_members).
        """
        result = super().hood_members(center_block,
                                      get_locations=get_locations)
        if not get_terms:
            return result

        if get_terms:
            if not self._phi_terms_precalculated:
                raise RuntimeError("Phi terms have not been calculated.")
        if get_locations:
            result += self.phi_terms[:, result[0]],
        else:
            result = result, self.phi_terms[:, result]
        return result
