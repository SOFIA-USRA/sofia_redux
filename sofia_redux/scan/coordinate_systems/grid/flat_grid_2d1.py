# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.grid.grid_2d1 import Grid2D1
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.cartesian_system \
    import CartesianSystem

__all__ = ['FlatGrid2D1']


class FlatGrid2D1(Grid2D1):

    def __init__(self):
        """
        Initialize a flat 2-dimensional grid.

        The flat grid consists of two (x, y) Cartesian axes.  Grid projections
        occur in relation to a single reference coordinate (at the origin by
        default).

        The flat grid is used to convert from 2D Cartesian (x, y) coordinates
        to offsets in relation to a specified reference onto a regular grid,
        and the reverse operation.

        Forward transform: grid projection -> offsets -> coordinates
        Reverse transform: coordinates -> offsets -> grid projection
        """
        super().__init__()

    def copy(self):
        """
        Return a copy of the FlatGrid2D1.

        Returns
        -------
        FlatGrid2D1
        """
        return super().copy()

    def set_defaults(self):
        """
        Set the default values for the grid.

        The defaults for the FlatGrid2D are cartesian (x, y) axes and a
        DefaultProjection2D projection.

        Returns
        -------
        None
        """
        self.set_coordinate_system(CartesianSystem(axes=2))
        super().set_defaults()
        super().set_projection(DefaultProjection2D())

    @classmethod
    def get_coordinate_instance_for(cls, name):
        """
        Return a coordinate instance for the given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        Coordinate2D1
        """
        return Coordinate2D1()

    def set_projection(self, value):
        """
        Set the grid projection.

        Parameters
        ----------
        value : Projection2D

        Returns
        -------
        None
        """
        if not isinstance(value, DefaultProjection2D):
            raise ValueError("Generic projections are not allowed here.")
        super().set_projection(value)

    def parse_projection(self, header):
        """
        Parse the projection from the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        pass
