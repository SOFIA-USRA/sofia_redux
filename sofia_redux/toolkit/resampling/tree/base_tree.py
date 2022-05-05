# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
import itertools
import inspect
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree
import sofia_redux.toolkit.resampling.tree as tree_module
from sofia_redux.toolkit.utilities.func import byte_size_of_object


__all__ = ['BaseTree']


class BaseTree(object):

    def __init__(self, argument, shape=None, build_type='all', leaf_size=40,
                 large_data=False, **distance_kwargs):
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
        large_data : bool, optional
            If `True`, indicates that this resampling algorithm will run on
            a large set of data, and the tree should be created on subsets
            of the data only when necessary.
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
        self.block_offsets = None
        self.block_population = None
        self.populated = None
        self.hood_population = None
        self.coordinates = None
        self.large_data = large_data
        self.ball_tree_block = None
        self.ball_tree_members = None
        self.leaf_size = leaf_size
        self.distance_kwargs = distance_kwargs

        arg = np.asarray(argument)
        if np.asarray(arg).ndim > 1:
            self.build_tree(arg, shape=shape, method=build_type,
                            leaf_size=self.leaf_size, **self.distance_kwargs)
        else:
            self._set_shape(tuple(arg))

    @property
    def shape(self):
        """
        Return the shape of the coordinate bins of the tree.

        The coordinate bins divide coordinates into bins where all values
        between 2 <= x_i < 3 for coordinate x in dimension i are in bin 0.
        Note that coordinates should have been scaled accordingly such that
        they range between 0 -> n, where n are the maximum number of bins.

        Returns
        -------
        shape : tuple (int)
            The number of bins in each dimension.
        """
        return tuple(self._shape)

    @property
    def features(self):
        """
        Returns the number of features (dimensions) in the tree.

        Returns
        -------
        int
        """
        return self._n_features

    @property
    def n_blocks(self):
        """
        Return the total number of blocks in the tree.

        The total number of blocks is simply the number of available coordinate
        bins.

        Returns
        -------
        int
        """
        return self._n_blocks

    @property
    def search(self):
        """
        Return a search array used to identify neighbouring blocks.

        Returns
        -------
        search_array : numpy.ndarray (int)
            An (n_features, n_permutations) containing values {-1, 0, 1} where
            each permutation gives a block offset in relation to a central
            block.
        """
        return self._search

    @property
    def balltree_initialized(self):
        """
        Return `True` if the BallTree has been initialized.

        Returns
        -------
        bool
        """
        return self._ball_initialized

    @property
    def hood_initialized(self):
        """
        Return `True` if the neighbourhood tree has been initialized.

        Returns
        -------
        bool
        """
        return self._hood_initialized

    @property
    def n_members(self):
        """
        Return the total number of coordinates stored in the tree.

        Returns
        -------
        int
        """
        if self.coordinates is None:
            return 0
        else:
            return self.coordinates.shape[1]

    @classmethod
    def estimate_ball_tree_bytes(cls, coordinates, leaf_size=40):
        """
        Estimate the maximum number of bytes used to construct the ball-tree.

        According to sklearn documentation, the memory needed to store the
        tree scales as::

             2 ** (1 + floor(log2((n-1) / leaf_size))) - 1

        However, empirically the data also scale as (ndim+1) * leaf_size.

        The estimated number of elements is given as::

            (ndim+1) * leaf_size * (2 ** (1+floor(log2((n-1) / leaf_size)))-1)

        This is then scaled by the number of bytes in a float (typically 8).

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates of shape (n_dimensions, n)
        leaf_size : int
            The number of leaves used to construct the ball-tree

        Returns
        -------
        max_size : int
            The maximum number of bytes used to construct the tree.
        """
        if leaf_size is None:
            leaf_size = 40
        float_bytes = np.empty(0, dtype=float).itemsize
        if coordinates.ndim == 1:
            coordinates = coordinates[None]
        n_dimensions = coordinates.shape[0]
        n = coordinates.size // n_dimensions

        scale = 2 ** (1 + np.floor(np.log2((n - 1) / leaf_size))) - 1
        correcting_factor = leaf_size * (n_dimensions + 1)
        n_elements = scale * correcting_factor
        return int(n_elements * float_bytes)

    @classmethod
    def estimate_split_tree_bytes(cls, coordinates,
                                  window=1, leaf_size=40):
        """
        Return the size of a ball tree for a single neighborhood.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates of shape (n_dimensions, n)
        window : int or float or numpy.ndarray, optional
            The window binning size.  If a numpy array is supplied, it should
            be of shape (n_dimensions,).
        leaf_size : int
            The number of leaves used to construct the ball-tree

        Returns
        -------
        max_size : int
            The maximum number of bytes used to construct the tree.
        """
        if leaf_size is None:
            leaf_size = 40
        n_bins = cls.estimate_n_bins(coordinates, window=window)
        if coordinates.ndim == 1:
            coordinates = coordinates[None]
        n_dimensions = coordinates.shape[0]
        n_samples = coordinates[0].size
        samples_per_bin = n_samples / n_bins
        bins_per_hood = 3 ** n_dimensions
        n = bins_per_hood * samples_per_bin  # samples in each hood tree
        n = int(np.clip(n, 0, n_samples))
        scale = 2 ** (1 + np.floor(np.log2((n - 1) / leaf_size))) - 1
        correcting_factor = leaf_size * (n_dimensions + 1)
        float_bytes = np.empty(0, dtype=float).itemsize
        n_elements = scale * correcting_factor
        return int(n_elements * float_bytes)

    @classmethod
    def estimate_n_bins(cls, coordinates, window=1):
        """
        Estimate the number of bins in the hood tree.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates of shape (n_dimensions, n).
        window : int or float or numpy.ndarray, optional
            The window binning size.  If a numpy array is supplied, it should
            be of shape (n_dimensions,).

        Returns
        -------
        bytes : int
        """
        if coordinates.ndim == 1:
            coordinates = coordinates[None]

        n_dimensions = coordinates.shape[0]
        span = np.empty(n_dimensions, dtype=float)
        for i in range(n_dimensions):
            span[i] = np.nanmax(coordinates[i]) - np.nanmin(coordinates[i])

        span /= window
        bins = np.ceil(span).astype(int)
        n_bins = np.clip(int(np.prod(bins)), 1, None)
        return n_bins

    @classmethod
    def estimate_hood_tree_bytes(cls, coordinates, window=1):
        """
        Estimate the maximum byte size for the neighborhood tree.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates of shape (n_dimensions, n).
        window : int or float or numpy.ndarray, optional
            The window binning size.  If a numpy array is supplied, it should
            be of shape (n_dimensions,).

        Returns
        -------
        bytes : int
        """
        if coordinates.ndim == 1:
            coordinates = coordinates[None]

        n_dimensions = coordinates.shape[0]
        n_bins = cls.estimate_n_bins(coordinates, window=window)
        n_samples = coordinates.size // n_dimensions

        test_array = np.zeros(0, dtype=int)
        item_byte_size = test_array.itemsize
        empty_byte_size = byte_size_of_object(test_array)
        average_samples = n_samples / n_bins
        array_byte_size = empty_byte_size + (item_byte_size * average_samples)
        # Add all sub arrays and the container array
        hood_tree_byte_size = (array_byte_size * n_bins) + empty_byte_size
        return int(np.ceil(hood_tree_byte_size))

    @classmethod
    def estimate_max_bytes(cls, coordinates, window=1, leaf_size=40,
                           full_tree=True, **kwargs):
        """
        Estimate the maximum number of bytes required by the tree.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates for the tree of shape (n_dimensions, n)
        window : int or float, optional
            The size of the tree windows
        leaf_size : int, optional
            The number of leaves used to construct the ball-tree
        full_tree : bool, optional
            Calculate the maximum number of bytes if the full ball-tree is
            pre-calculated.  Otherwise, calculates the size of a single
            neighborhood sized ball-tree.
        kwargs : dict, optional
            Additional keyword arguments to pass that may be used by subclasses
            of the BaseTree.

        Returns
        -------
        max_size : int
            The maximum number of bytes used to construct the tree.
        """
        float_bytes = np.empty(0, dtype=float).itemsize

        # 1 for coordinates, 1 for coordinate offsets.
        coordinate_bytes = 2 * coordinates.size * float_bytes
        if full_tree:
            ball_tree_bytes = cls.estimate_ball_tree_bytes(
                coordinates, leaf_size=leaf_size)
        else:
            ball_tree_bytes = cls.estimate_split_tree_bytes(
                coordinates, window=window, leaf_size=leaf_size)

        hood_tree_bytes = cls.estimate_hood_tree_bytes(
            coordinates, window=window)

        return int(ball_tree_bytes + hood_tree_bytes + coordinate_bytes)

    @staticmethod
    def get_class_for(thing):
        """
        Return a Tree class specific to a given grid, resampler, or name.

        Parameters
        ----------
        thing : BaseGrid or ResampleBase or str
            Either a sub-class of a BaseGrid, ResampleBase, or a string.

        Returns
        -------
        BaseTree subclass
        """
        if isinstance(thing, str):
            name = thing
        else:
            if inspect.isclass(thing):
                class_name = thing.__name__
            else:
                class_name = thing.__class__.__name__
            if 'Grid' in class_name:
                name = class_name.split('Grid')[0]
            elif 'Resample' in class_name:
                name = class_name.split('Resample')[-1]
            else:
                name = None

        return BaseTree.get_class_for_name(name)

    @staticmethod
    def get_class_for_name(name):
        """
        Return a Tree class of the given name.

        Parameters
        ----------
        name : str
            The name of the tree.

        Returns
        -------
        BaseTree subclass
        """
        tree_path = tree_module.__name__
        name = 'base' if name in ['', None] else name
        module_path = (tree_path + f'.{name}_tree').lower().strip()
        class_name = ''.join(
            [s[0].upper() + s[1:] for s in name.split('_')]) + 'Tree'
        try:
            module = importlib.import_module(module_path)  # Allow errors
            tree_class = getattr(module, class_name)
            if tree_class is None:  # pragma: no cover
                return BaseTree
            return tree_class
        except ModuleNotFoundError:
            return BaseTree

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
            raise ValueError(f"Unknown tree building method: {method}")

    def build_ball_tree_for_block(self, block):
        """
        Build a ball tree for a neighborhood around a single block.

        Parameters
        ----------
        block : int
            The center block of the neighborhood.

        Returns
        -------
        None
        """
        if not self.hood_initialized:
            self._build_hood_tree()
        self.ball_tree_block = int(block)
        self.ball_tree_members, coordinates = self.hood_members(
            self.ball_tree_block, get_locations=True)
        if self.leaf_size is None:
            self._balltree = BallTree(
                coordinates.T, **self.distance_kwargs)
        else:
            self._balltree = BallTree(
                coordinates.T, leaf_size=self.leaf_size,
                **self.distance_kwargs)

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
        self.ball_tree_block = None
        if self.large_data:
            # Do not attempt to build a full ball tree for large data
            return
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

    def query_radius(self, coordinates, radius=1.0, block=None, **kwargs):
        r"""
        Return all tree members within a certain distance of given points.

        Quickly retrieves all members of the tree population within the radius
        of some defined coordinate(s).  The default distance definition is
        "minkowski" with p=2, which is equivalent to the "euclidean"
        definition. For a list of available distance metric please see
        `sklearn.neighbors.BallTree.valid_metrics`.  The distance metric may
        be defined during PolynomialTree initialization.

        Parameters
        ----------
        coordinates : numpy.ndarray
            An (n_features, n_coordinates) or (n_features,) array containing
            coordinates around which to determine tree members within a
            certain radius.
        radius : float, optional
            The search radius around each coordinate.
        block : int, optional
            Used to create a temporary ball tree for the given block when
            operating on large data.
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
        if coordinates.ndim == 1:
            c = coordinates[None]
        else:
            c = coordinates.T

        if self.large_data:
            if block is None:
                raise ValueError("No block number for large data query has "
                                 "been supplied.")
            self.build_ball_tree_for_block(block)
            indices = self._balltree.query_radius(c, radius, **kwargs)
            if kwargs.get('return_distance'):
                return_distance = True
                indices, distances = indices
            else:
                return_distance = False
                distances = None

            for i, point_indices in enumerate(indices):
                indices[i] = self.ball_tree_members[point_indices]

            if return_distance:
                return indices, distances
            else:
                return indices

        elif not self.balltree_initialized:
            raise RuntimeError("Ball tree not initialized")
        else:
            return self._balltree.query_radius(c, radius, **kwargs)

    def block_members(self, block, get_locations=False):
        if not self._hood_initialized:
            raise RuntimeError("Neighborhood tree not initialized.")
        members = self._tree[block]
        if not get_locations:
            return members
        else:
            return members, self.coordinates[:, members]

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

    def hood_members(self, center_block, get_locations=False):
        r"""
        Find all members within the neighborhood of a single block.

        Parameters
        ----------
        center_block : int
            The index of the block at the center of the neighborhood.
        get_locations : bool, optional
            If `True`, return the coordinates of the hood members.

        Returns
        -------
        members, [coordinates]
            members is a numpy.ndarray of int and shape (found_members,).
            If `get_locations` was set to `True,` coordinates is of shape
            (n_features, found_members).
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
        if not get_locations:
            return hood_members
        else:
            return hood_members, self.coordinates[:, hood_members]

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
