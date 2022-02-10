# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.cartesian_system \
    import CartesianSystem

__all__ = ['FlatGrid2D']


class FlatGrid2D(Grid2D):

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
        Return a copy of the FlatGrid2D.

        Returns
        -------
        FlatGrid2D
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
        Coordinate2D
        """
        return Coordinate2D()

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

    def parse_header(self, header):
        """
        Parse a FITS header and apply to the grid.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        super().parse_header(header)
        alt = self.get_fits_id()
        ctype1_key = f'CTYPE1{alt}'
        ctype2_key = f'CTYPE2{alt}'
        if ctype1_key in header:
            self.x_axis.short_label = str(header.get(ctype1_key))
        if ctype2_key in header:
            self.y_axis.short_label = str(header.get(ctype2_key))

    def edit_header(self, header):
        """
        Edit a FITS header with the grid information.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        super().edit_header(header)
        alt = self.get_fits_id()
        header[f'CTYPE1{alt}'] = self.x_axis.short_label, 'Axis-1 name'
        header[f'CTYPE2{alt}'] = self.y_axis.short_label, 'Axis-2 name'
