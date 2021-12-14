# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.grid.base_grid import BaseGrid

__all__ = ['PolynomialGrid']


class PolynomialGrid(BaseGrid):
    """
    The polynomial grid contains a PolynomialTree instead of the BaseTree
    object.
    """
    pass
