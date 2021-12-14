# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
import numpy as np

from sofia_redux.toolkit.resampling.resample_utils import scale_coordinates
from sofia_redux.toolkit.resampling.tree.base_tree import BaseTree
import sofia_redux.toolkit.resampling.grid as grid_module

__all__ = ['BaseGrid']


class BaseGrid(object):

    def __init__(self, *grid, scale_factor=None, scale_offset=None,
                 build_tree=False, tree_shape=None, dtype=None, **kwargs):
        """
        Define and initialize a resampling grid.

        The resampling grid is used to specify a set of output coordinates for
        the resampling objects.  The grid coordinates may be either arranged
        as a typical grid or as a set of irregularly spaced coordinates.  Here,
        a typical grid implies that all output coordinates are aligned across
        all dimensions.

        The resampling grid contains its own Tree object that is used to
        quickly map arbitrary coordinates onto grid positions (see
        :class:`BaseTree` for further details).  A scaling factor and offset
        may exist between supplied coordinates and grid coordinates if
        necessary, which are applied according to :func:`scale_coordinates`.

        Parameters
        ----------
        grid : tuple or numpy.ndarray, optional
            Either an n-dimensional tuple containing the grid coordinates for
            each feature, or a single array of shape (n_features, n) that may
            be used to specify irregular grid coordinates.
        scale_factor : numpy.ndarray (float), optional
            The scaling factor between the actual coordinates and grid
            coordinates.  Must be supplied in conjunction with `scale_offset`.
            Should be an array of shape (n_features,).
        scale_offset : numpy.ndarray (float), optional
            The scaling offset between the actual coordinates and grid
            coordinates.  Must be supplied in conjunction with `scale_factor`.
            Should be an array of shape (n_features,).
        build_tree : bool, optional
            If `True`, build the associated tree for this grid.
        tree_shape : tuple or numpy.ndarray (int), optional
            The shape of the tree blocks.  If not supplied, defaults to maximum
            grid coordinates in each dimension.
        dtype : type, optional
            The type of the grid coordinates.  If not supplied, defaults to
            the type of the provided `grid` coordinates.  If neither grid nor
            dtype is provided, defaults to float.
        kwargs : dict, optional
            Optional keyword arguments passed into the `set_indexer` method.
            These parameters will be applied during tree initialization.
        """
        self._regular = None
        self._shape = None
        self._size = None
        self._nfeatures = None
        self._scaled = False
        self._scale_factor = None
        self._scale_offset = None
        self._last_scale_factor = None
        self._last_scale_offset = None
        self.tree = None
        self.grid = None

        if scale_factor is not None:
            self._scale_factor = np.asarray(scale_factor).astype(float)
            if scale_offset is None:
                raise ValueError("Specify both factor and offset to scale")

        if scale_offset is not None:
            self._scale_offset = np.asarray(scale_offset).astype(float)
            if scale_factor is None:
                raise ValueError("Specify both factor and offset to scale")

        if len(grid) == 1 and np.asarray(grid[0]).ndim == 2:
            # setup for irregular output
            self._regular = False
            self._singular_output = False
            self.grid = np.asarray(grid[0]).astype(dtype)
            self._shape = (self.grid.shape[1],)
            self._size = self.grid.shape[1]
        else:
            # setup for regular grid output
            self._shape = tuple([len(g) if hasattr(g, '__len__') else 1
                                for g in grid[::-1]])

            # Someone really should have said 'xy' indexing doesn't work
            # past 2 dimensions!!!... This is C vs. F array types, not what
            # the name implies.
            self.grid = np.vstack(
                [np.asarray(g, dtype=float).ravel() for g in
                 np.meshgrid(*grid[::-1], indexing='ij')[::-1]])

            self.grid = np.asarray(self.grid, dtype=dtype)
            self._size = self.grid.shape[1]
            self._singular_output = self._size == 1
            self._regular = not self._singular_output

        self._nfeatures = self.grid.shape[0]

        if self._scale_factor is not None:
            self.scale(self._scale_factor, self._scale_offset)
        self.set_indexer(tree_shape, build_tree=build_tree, **kwargs)

    @staticmethod
    def get_class_for(thing):
        """
        Return a Grid class specific to a given tree, resampler, or name.

        Parameters
        ----------
        thing : BaseTree or ResampleBase or str
            Either a sub-class of a BaseTree, ResampleBase, or a string.

        Returns
        -------
        BaseTree subclass
        """
        if isinstance(thing, str):
            name = thing
        else:
            if 'Tree' in thing.__class__.__name__:
                name = thing.__class__.__name__.split('Tree')[0]
            elif 'Resample' in thing.__class__.__name__:
                name = thing.__class__.__name__.split('Resample')[-1]
            else:
                name = None
        return BaseGrid.get_class_for_name(name)

    @staticmethod
    def get_class_for_name(name):
        """
        Return a Grid class of the given name.

        Parameters
        ----------
        name : str
            The name of the grid.

        Returns
        -------
        BaseGrid subclass
        """
        grid_path = grid_module.__name__
        name = 'base' if name in ['', None] else name
        module_path = (grid_path + f'.{name}_grid').lower().strip()
        class_name = ''.join(
            [s[0].upper() + s[1:] for s in name.split('_')]) + 'Grid'
        try:
            module = importlib.import_module(module_path)  # Allow errors
            grid_class = getattr(module, class_name)
            if grid_class is None:  # pragma: no cover
                return BaseGrid
            return grid_class
        except ModuleNotFoundError:
            return BaseGrid

    @property
    def regular(self):
        """
        Return whether the grid contains feature aligned grid coordinates.

        Returns
        -------
        bool
        """
        return self._regular

    @property
    def singular(self):
        """
        Return whether the grid consists of only a single output coordinate.

        Returns
        -------
        bool
        """
        return self._singular_output

    @property
    def shape(self):
        """
        Return the shape of the grid.

        Returns
        -------
        tuple (int)
        """
        return self._shape

    @property
    def size(self):
        """
        Return the number of grid vertices.

        Returns
        -------
        int
        """
        return self._size

    @property
    def features(self):
        """
        Return the number of grid dimensions.

        Returns
        -------
        int
        """
        return self._nfeatures

    @property
    def scale_factor(self):
        """
        Return the scaling factor between actual and grid coordinates.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self._scale_factor

    @property
    def scale_offset(self):
        """
        Return the scaling offset between actual and grid coordinates.

        Returns
        -------
        numpy.ndarray (float)
        """
        return self._scale_offset

    @property
    def tree_class(self):
        """
        Return the relevant tree class for this grid.

        Returns
        -------
        BaseTree subclass
        """
        return self.get_tree_class()

    def get_tree_class(self):
        """
        Return the relevant tree class for the grid.

        Returns
        -------
        BaseGrid subclass
        """
        return BaseTree.get_class_for(self)

    def reshape_data(self, data):
        """
        Reshape data to the grid dimensions.

        Note that this will only reshape data if the grid is regular.

        Parameters
        ----------
        data : numpy.ndarray
            Data of the shape (n_sets, grid.size) or (grid.size,).

        Returns
        -------
        reshaped_data : numpy.ndarray
            Data of shape (grid.shape).
        """
        ndim = data.ndim
        if self._regular:
            if ndim == 1:
                return data.reshape(self._shape)
            elif ndim == 2:
                return data.reshape((data.shape[0],) + self._shape)
            else:
                raise ValueError("Incompatible data dimensions. Data must be "
                                 "of shape (grid.size,) or (N, grid.size)")

        elif self._singular_output:
            if ndim == 1:
                return data.ravel()[0]
            elif ndim == 2:  # multi set
                return data[:, 0]
            else:
                raise ValueError("Incompatible data dimensions. Data must be "
                                 "of shape (1,) or (n_sets, 1)")
        else:
            return data

    def unscale(self):
        """
        Unscale the grid coordinates.

        Remove a scaling factor and offset if previously applied.

        Returns
        -------
        None
        """
        if not self._scaled:
            return
        self._last_scale_factor = self._scale_factor.copy()
        self._last_scale_offset = self._scale_offset.copy()
        self.grid = scale_coordinates(
            self.grid, self._scale_factor, self._scale_offset, reverse=True)
        self._scale_factor = None
        self._scale_offset = None
        self._scaled = False

    def scale(self, factor, offset):
        """
        Apply a scaling factor and offset to the grid coordinates.

        Parameters
        ----------
        factor : numpy.ndarray
            A scaling factor of shape (n_features,).
        offset : numpy.ndarray
            A scaling offset of shape (n_features,).

        Returns
        -------
        None
        """
        if self._scaled:
            self.unscale()
        self.grid = scale_coordinates(self.grid, factor, offset, reverse=False)
        self._scaled = True
        self._scale_factor = np.asarray(factor).astype(float)
        self._scale_offset = np.asarray(offset).astype(float)

    def rescale(self):
        """
        Re-apply the previous scaling factors and offsets if removed.

        Returns
        -------
        None
        """
        if self._scaled:
            return
        elif self._last_scale_factor is None:
            return
        self.scale(self._last_scale_factor, self._last_scale_offset)
        self._last_scale_factor = None
        self._last_scale_offset = None

    def set_indexer(self, shape=None, build_tree=False, build_type='hood',
                    **kwargs):
        """
        Calculate the indexing mapping the grid coordinates to the tree.

        Parameters
        ----------
        shape : tuple (int), optional
            The shape of the grid.  If not supplied, uses the maximum internal
            grid coordinates + 1 in each dimension.
        build_tree : bool, optional
            If `True`, builds a neighborhood tree of the type `build_type`.
        build_type : str, optional
            Must be one of {'hood', 'balltree', 'all', None}.  Please see
            :class:`BaseTree` for further details.  The default type necessary
            for resampling is 'hood'.
        kwargs : dict, optional
            Optional keyword arguments passed into the tree initialization.

        Returns
        -------
        None
        """
        if shape is None:
            shape = np.max(self.grid, axis=1) + 1
        if build_tree:
            self.tree = self.tree_class(self.grid, shape=shape,
                                        build_type=build_type, **kwargs)
        else:
            self.tree = self.tree_class(shape, **kwargs)

    def __call__(self):
        """
        Returns the grid.

        Returns
        -------
        numpy.ndarray
            The resampling grid of shape (n_features, m) where m is the number
            of grid vertices.
        """
        return self.grid  # not a copy
