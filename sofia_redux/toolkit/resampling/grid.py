# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from .resample_utils import scale_coordinates
from .tree import Rtree

__all__ = ['ResampleGrid']


class ResampleGrid(object):

    def __init__(self, *grid, scale_factor=None, scale_offset=None,
                 build_tree=False, tree_shape=None, dtype=None):
        """Define and update grid for resampling."""

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
        self.set_indexer(tree_shape, build_tree=build_tree)

    @property
    def regular(self):
        return self._regular

    @property
    def singular(self):
        return self._singular_output

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def features(self):
        return self._nfeatures

    @property
    def scale_factor(self):
        return self._scale_factor

    @property
    def scale_offset(self):
        return self._scale_offset

    def reshape_data(self, data):
        ndim = data.ndim
        if self._regular:
            if ndim == 1:
                return data.reshape(self._shape)
            elif ndim == 2:
                return data.reshape((data.shape[0],) + self._shape)
            else:
                raise ValueError("incompatible data dimensions. Data must be "
                                 "of shape (grid.size,) or (N, grid.size)")

        elif self._singular_output:
            if ndim == 1:
                return data.ravel()[0]
            elif ndim == 2:  # multi set
                return data[:, 0]
            else:
                raise ValueError("incompatible data dimensions. Data must be "
                                 "of shape (1,) or (n_sets, 1)")
        else:
            return data

    def unscale(self):
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
        if self._scaled:
            self.unscale()
        self.grid = scale_coordinates(self.grid, factor, offset, reverse=False)
        self._scaled = True
        self._scale_factor = np.asarray(factor).astype(float)
        self._scale_offset = np.asarray(offset).astype(float)

    def rescale(self):
        if self._scaled:
            return
        elif self._last_scale_factor is None:
            return
        self.scale(self._last_scale_factor, self._last_scale_offset)
        self._last_scale_factor = None
        self._last_scale_offset = None

    def set_indexer(self, shape, build_tree=False):
        if shape is None:
            shape = self.grid.max(axis=1) + 1
        if build_tree:
            self.tree = Rtree(self.grid, shape=shape, build_type='hood')
        else:
            self.tree = Rtree(shape)

    def __call__(self):
        return self.grid  # not a copy
