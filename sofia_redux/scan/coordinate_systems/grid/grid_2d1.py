# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units
from copy import deepcopy

from sofia_redux.scan.coordinate_systems.grid.grid_1d import Grid1D
from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid
from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.utilities.utils import round_values

__all__ = ['Grid2D1']


class Grid2D1(Grid2D):

    def __init__(self):
        """
        Initialize a 2-dimensional abstract grid with one orthogonal axis.

        The grid is used to convert from 2D coordinates to offsets in relation
        to a specified reference onto a regular grid, and the reverse
        operation.  There is also one additional axis typically in units
        different from the standard grid.

        Forward transform: grid projection -> offsets -> coordinates
        Reverse transform: coordinates -> offsets -> grid projection
        """
        self.z = Grid1D(first_axis=3)
        super().__init__()

    def copy(self):
        """
        Return a copy of the Grid2D.

        Returns
        -------
        Grid2D1
        """
        new = super().copy()
        new.z = self.z.copy()
        return new

    def copy_2d_from(self, grid_2d):
        """
        Copy the 2D attributes from another grid to this one.

        Parameters
        ----------
        grid_2d : Grid2D

        Returns
        -------
        Grid2D1
        """
        self.set_projection(grid_2d.projection)
        self.m = deepcopy(grid_2d.m)
        self.i = deepcopy(grid_2d.i)

        self.reference_index.xy_coordinates = grid_2d.reference_index
        self.coordinate_system = grid_2d.coordinate_system
        self.variant = grid_2d.variant

    @property
    def reference(self):
        """
        Return the reference value for the grid.

        Returns
        -------
        Coordinate2D1
        """
        return super().reference

    @reference.setter
    def reference(self, value):
        """
        Return the reference value for the grid.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        self.set_reference(value)

    @property
    def reference_index(self):
        """
        Return the reference index for the grid.

        Returns
        -------
        Coordinate2D1
        """
        return self.get_reference_index()

    @reference_index.setter
    def reference_index(self, value):
        """
        Set the reference index for the grid.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        self.set_reference_index(value)

    @property
    def resolution(self):
        """
        Return the grid resolution.

        Returns
        -------
        Coordinate2D1
        """
        return self.get_resolution()

    @resolution.setter
    def resolution(self, value):
        """
        Set the grid resolution.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        self.set_resolution(value)

    @property
    def z_axis(self):
        """
        Return the z-axis of the grid.

        Returns
        -------
        CoordinateAxis
        """
        return self.z.coordinate_system.axes[0]

    @property
    def fits_z_unit(self):
        """
        Return the unit for the z-axis data.

        Returns
        -------
        units.Unit
        """
        return self.z_axis.unit

    def __eq__(self, other):
        """
        Check if this grid is equal to another.

        Parameters
        ----------
        other : Grid2D1

        Returns
        -------
        equal : bool
        """
        if not super().__eq__(other):
            return False
        return self.z == other.z

    @classmethod
    def to_coordinate2d1(cls, value):
        """
        Convert a value to a Coordinate2D1 object.

        Parameters
        ----------
        value : iterable

        Returns
        -------
        Coordinate2D1
        """
        if isinstance(value, Coordinate2D1):
            return value
        try:
            n = len(value)
            if n == 1:
                value = value[0]
        except TypeError:
            n = 1

        if n == 0:
            if isinstance(value, units.Quantity):
                unit = value.unit
            else:
                unit = None
            return Coordinate2D1(xy_unit=unit)

        if n == 1:  # Just the xy coordinates
            return Coordinate2D1(xy=[value] * 2)

        if n == 2:
            xy = cls.to_coordinate2d(value[0])
            return Coordinate2D1(xy=xy, z=value[1])

        if n == 3:
            return Coordinate2D1(xy=[value[0], value[1]], z=value[2])

        return Coordinate2D1()

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
        xy = super().get_coordinate_instance_for(name)
        return Coordinate2D1(xy=xy)

    @classmethod
    def get_grid_2d1_instance_for(cls, ctype_1, ctype_2):
        """
        Return a Grid2D instance for the given FITS coordinate types.

        A suitable grid instance will be returned depending on the provided
        CTYPE values.  If any CTYPE value is not recognized, a FlatGrid2D
        instance will be returned.  In cases where a spherical system is
        recognized, a SphericalGrid will be returned.

        CTYPE values should be 8 characters in length, divided into 2 segments
        of 4 characters each.  The first set should give the type of world
        coordinates, and the second set should provide the type of projection
        geometry.  The first set is left justified, with hyphens filling in any
        blank characters to the right, and the second set should be right
        justified with hyphens filling in any blank characters to the left.
        For example, 'RA---TAN' or 'GLON-AIT'.

        Note that in addition to the standard FITS CTYPE standards, the LON and
        LAT coordinate systems are also recognized.  E.g., 'LAT--TAN'.
        xLON, xyLN, xLAT, xyLT coordinate systems are also recognized, where
        x and y indicate arbitrary (but matching across longitude and latitude)
        characters.

        Parameters
        ----------
        ctype_1 : str or None
            The type of the first world coordinate axis.
        ctype_2 : str or None
            The type of the second world coordinate axis.

        Returns
        -------
        Grid2D
        """
        grid_2d = super().get_grid_2d_instance_for(ctype_1, ctype_2)
        if isinstance(grid_2d, SphericalGrid):
            new = grid_2d.get_grid_instance('spherical_grid_2d1')
        else:
            new = grid_2d.get_grid_instance('flat_grid_2d1')
        new.copy_2d_from(grid_2d)
        return new

    @classmethod
    def from_header(cls, header, alt=''):
        """
        Create a grid instance from a header.

        Parameters
        ----------
        header : fits.Header
            The FITS header from which to create the grid.
        alt : str, optional
            The alternate coordinate system designation.

        Returns
        -------
        Grid2D1
        """
        grid_2d = super().from_header(header, alt=alt)
        if isinstance(grid_2d, SphericalGrid):
            grid_2d1 = grid_2d.get_grid_instance('spherical_grid_2d1')
        else:
            grid_2d1 = grid_2d.get_grid_instance('flat_grid_2d1')
        grid_2d1.copy_2d_from(grid_2d)
        if alt:
            grid_2d1.z.variant = ord(alt) - ord('A')
        grid_2d1.parse_header(header)
        return grid_2d1

    def get_default_coordinate_instance(self):
        """
        Return a coordinate for the dimensions of the grid.

        Returns
        -------
        Coordinate2D1
        """
        return Coordinate2D1()

    def to_string(self):
        """
        Return a string representation of the grid.

        Returns
        -------
        str
        """
        projection_name = self.reference.xy_coordinates.__class__.__name__
        projection_name = projection_name.split('Coordinates')[0]
        reference_str = f'{projection_name}: {self.reference}'
        projection_str = (f'{self.projection.get_full_name()} '
                          f'({self.projection.get_fits_id()})')

        r = self.resolution
        dx, dy, dz = r.x, r.y, r.z
        if isinstance(dx, units.Quantity):
            dx = dx.value
        spacing_str = f'({dx:.3f} x {dy:.3f}) x {dz:.5f}'
        crpix_str = f'{self.reference_index} C-style, 0-based'
        return (f'{reference_str}\n'
                f'Projection: {projection_str}\n'
                f'Grid Spacing: {spacing_str}\n'
                f'Reference Pixel: {crpix_str}')

    def for_resolution(self, resolution):
        """
        Return a Grid2D1 for a given resolution.

        Parameters
        ----------
        resolution : astropy.units.Quantity or numpy.ndarray or Coordinate2D1

        Returns
        -------
        grid : Grid2D1
        """
        resolution = self.to_coordinate2d1(resolution)
        grid = self.copy()
        grid.set_resolution(resolution)

        x_factor = self.resolution.x / resolution.x
        y_factor = self.resolution.y / resolution.y
        z_factor = self.resolution.z / resolution.z
        if isinstance(x_factor, units.Quantity):
            x_factor = x_factor.decompose().value
            y_factor = y_factor.decompose().value
            z_factor = z_factor.decompose().value

        reference_index = grid.reference_index
        reference_index.scale_x(x_factor)
        reference_index.scale_y(y_factor)
        reference_index.scale_z(z_factor)
        grid.reference_index = reference_index
        return grid

    def get_pixel_volume(self):
        """
        Return the volume of one pixel on the grid.

        Returns
        -------
        volume : float or units.Quantity
        """
        return self.get_pixel_area() * self.resolution.z

    def get_resolution(self):
        """
        Return the grid resolution in (x, y) and z.

        Returns
        -------
        resolution : Coordinate2D1
        """
        return Coordinate2D1(super().get_resolution(), z=self.z.resolution)

    def set_resolution(self, resolution):
        """
        Set the grid resolution.

        Parameters
        ----------
        resolution : Coordinate2D1 or float or numpy.ndarray or Quantity
            The resolution to set.  Must be a Coordinate2D1 in order to
            set the z-axis resolution.

        Returns
        -------
        None
        """
        resolution = self.to_coordinate2d1(resolution)
        super().set_resolution(resolution.xy_coordinates)
        self.z.set_resolution(resolution.z)

    def get_pixel_size(self):
        """
        Return the pixel size in (x, y) and z.

        Returns
        -------
        resolution : Coordinate2D1
        """
        return self.get_resolution()

    def get_pixel_size_z(self):
        """
        Return the pixel size in the z-direction.

        Returns
        -------
        size : float or units.Quantity
        """
        return self.z.resolution

    def is_reverse_z(self):
        """
        Returns if the z-axis is reversed.

        Returns
        -------
        bool
        """
        return self.z_axis.reverse

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
        self.z.parse_header(header)

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
        self.z.edit_header(header)

    def coordinates_at(self, grid_indices, coordinates=None):
        """
        Return the coordinates at given grid indices.

        Parameters
        ----------
        grid_indices : Coordinate2D1
            The grid indices for which to find coordinates.
        coordinates : Coordinate2D1, optional
            Optional output coordinates to hold the result.

        Returns
        -------
        Coordinate2D1
        """
        xy = super().coordinates_at(grid_indices.xy_coordinates)
        z = self.z.coordinates_at(grid_indices.z_coordinates)
        if coordinates is None:
            return Coordinate2D1(xy=xy, z=z)

        coordinates.xy_coordinates = xy
        coordinates.z_coordinates = z
        return coordinates

    def index_of(self, coordinates, grid_indices=None):
        """
        Return the index of a coordinate on the grid.

        Parameters
        ----------
        coordinates : Coordinate2D1
            The "true" coordinates.
        grid_indices : Coordinate2D1, optional
            The output coordinates that will hold the result.

        Returns
        -------
        grid_indices : Coordinate2D1
        """
        xy = super().index_of(coordinates.xy_coordinates)
        z = self.z.index_of(coordinates.z_coordinates)
        if grid_indices is None:
            return Coordinate2D1(xy=xy, z=z)

        grid_indices.xy_coordinates = xy
        grid_indices.z_coordinates = z
        return grid_indices

    def offset_to_index(self, offset, in_place=False):
        """
        Convert an offset to a grid location.

        Parameters
        ----------
        offset : Coordinate2D1 or Coordinate2D
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        index : Coordinate2D1 or Coordinate2D
        """
        # Cannot do in-place due to class change
        if isinstance(offset, Coordinate2D):
            return super().offset_to_index(offset)
        ixy = super().offset_to_index(offset.xy_coordinates, in_place=in_place)
        iz = self.z.offset_to_index(offset.z_coordinates, in_place=in_place)
        if in_place:
            return offset

        return Coordinate2D1(xy=ixy, z=iz)

    def index_to_offset(self, grid_indices, in_place=False):
        """
        Convert grid indices to offset positions.

        Parameters
        ----------
        grid_indices : Coordinate2D1 or Coordinate2D
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        offset : Coordinate2D1 or Coordinate2D
        """
        if isinstance(grid_indices, Coordinate2D):
            return super().index_to_offset(grid_indices)

        xy = super().index_to_offset(grid_indices.xy_coordinates,
                                     in_place=in_place)
        z = self.z.index_to_offset(grid_indices.z_coordinates,
                                   in_place=in_place)
        if in_place:
            return grid_indices
        return Coordinate2D1(xy=xy, z=z)

    def get_reference(self):
        """
        Return the reference position of the grid.

        Returns
        -------
        reference : Coordinate2D1
        """
        return Coordinate2D1(xy=super().get_reference(),
                             z=self.z.get_reference())

    def set_reference(self, value):
        """
        Set the reference position of the grid.

        Parameters
        ----------
        value : Coordinate2D1
            The reference coordinate to set.

        Returns
        -------
        None
        """
        super().set_reference(value.xy_coordinates)
        self.z.set_reference(value.z_coordinates)

    def get_reference_index(self):
        """
        Return the reference index of the reference position on the grid.

        Returns
        -------
        index : Coordinate2D1
        """
        return Coordinate2D1(xy=super().get_reference_index(),
                             z=self.z.get_reference_index())

    def set_reference_index(self, value):
        """
        Set the reference index of the reference position on the grid.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        super().set_reference_index(value.xy_coordinates)
        self.z.set_reference_index(value.z_coordinates)

    def get_coordinate_index(self, coordinates, indices=None):
        """
        Get the indices of the given coordinates.

        Parameters
        ----------
        coordinates : Coordinate2D1
            The coordinates for which to find grid indices.
        indices : Coordinate2D1, optional
            Optional output coordinates to return.  If not created, a fresh
            Coordinate2D1 instance will be used.

        Returns
        -------
        indices : Coordinate2D1
            The grid indices.
        """
        xy = super().get_coordinate_index(coordinates.xy_coordinates)
        z = self.z.index_of(coordinates.z_coordinates)
        if indices is None:
            return Coordinate2D1(xy=xy, z=z)

        indices.xy_coordinates = xy
        indices.z_coordinates = z
        return indices

    def get_offset_index(self, offsets, indices=None):
        """
        Get the grid indices of the given offsets.

        Parameters
        ----------
        offsets : Coordinate2D1
            The offsets for which to find indices.
        indices : Coordinate2D1
            Optional coordinates in which to place the results.

        Returns
        -------
        indices : Coordinate2D1
            The grid indices.
        """
        xy = super().get_offset_index(offsets.xy_coordinates)
        z = self.z.offset_to_index(offsets.z_coordinates, in_place=False)
        z.set_x(round_values(z.x))
        if indices is None:
            return Coordinate2D1(xy=xy, z=z)
        indices.xy_coordinates = xy
        indices.z_coordinates = z
        return indices

    def get_coordinates(self, indices, coordinates=None):
        """
        Get the coordinates at the given grid indices.

        Parameters
        ----------
        indices : Coordinate2D1
        coordinates : Coordinate2D1, optional
            Optional coordinates in which to store the output coordinates.

        Returns
        -------
        coordinates : Coordinates2D1
        """
        xy = super().get_coordinates(indices.xy_coordinates)
        z = self.z.coordinates_at(indices.z_coordinates)
        if coordinates is None:
            return Coordinate2D1(xy=xy, z=z)

        coordinates.xy_coordinates = xy
        coordinates.z_coordinates = z
        return coordinates

    def get_offset(self, indices, offset=None):
        """
        Get coordinate offsets from the given grid indices.

        Parameters
        ----------
        indices : Coordinate2D1 or Coordinate2D
            The grid indices.
        offset : Coordinate2D1, optional
            Optional coordinates in which to store the results.

        Returns
        -------
        offset : Coordinate2D1 or Coordinate2D
        """
        if not isinstance(indices, (Coordinate2D1, Coordinate3D)):
            return super().get_offset(indices, offset=offset)

        if isinstance(indices, Coordinate2D1):
            xy = super().get_offset(indices.xy_coordinates)
            z = indices.z_coordinates
        else:
            xy = super().get_offset(Coordinate2D(indices))
            z = Coordinate1D(indices.z)

        z = self.z.index_to_offset(z)
        if offset is None:
            return Coordinate2D1(xy=xy, z=z)

        offset.xy_coordinates = xy
        offset.z_coordinates = z
        return offset

    def toggle_native(self, offset, in_place=True):
        """
        Reverse the x and y axis of native offsets if axes are reversed.

        Scales reversed axis by -1.

        Parameters
        ----------
        offset : Coordinate2D1
        in_place : bool, optional
            If `True`, toggle in-place, otherwise return a copy.

        Returns
        -------
        Coordinate2D1
        """
        if not in_place:
            offset = offset.copy()

        super().toggle_native(offset.xy_coordinates, in_place=True)
        if self.is_reverse_z():
            offset.scale_z(-1)

        return offset

    @abstractmethod
    def parse_projection(self, header):  # pragma: no cover
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
